import torch
import torch.nn.functional as F
from transformers import (AutoModelForSeq2SeqLM,
                        AutoTokenizer,
                        PreTrainedTokenizer,
                        PreTrainedTokenizerFast)
import evaluate

from fire import Fire
import pandas as pd
from tqdm import tqdm
import json

from typing import List, Dict, Union
from collections import defaultdict
from functools import partial
from pprint import pprint

from ipdb import set_trace

class Harimplus_Scorer:
    def __init__(self,
                    pretrained_name:str='none',
                    tokenizer:Union[PreTrainedTokenizer, PreTrainedTokenizerFast]=None,
                    mixing_factor:float=7., # same as lambda in the paper
                    device:str='cuda',

                    src_maxlen=1024,
                    tgt_maxlen=110,
                ):
        self._pretrained_name = pretrained_name
        self._lambda = mixing_factor

        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._encdec_model = AutoModelForSeq2SeqLM.from_pretrained(self._pretrained_name)
        if tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained_name)
        else:
            self._tokenizer = tokenizer
        self._encdec_model.to(self._device)
        self._encdec_model.eval()

        self._src_maxlen = src_maxlen
        self._tgt_maxlen = tgt_maxlen



    def _prep_input(self, src_tgt_txts, src_or_tgt='src'):
        L = self._src_maxlen if src_or_tgt=='src' else self._tgt_maxlen
        if isinstance(src_tgt_txts, pd.Series):
            src_tgt_txts=src_tgt_txts.tolist()
            if src_or_tgt == 'src':
                src_tgt_txts = [ s.replace("\n", " ") for s in src_tgt_txts ]
        return self._tokenizer(src_tgt_txts, padding=True, truncation=True, max_length=L, return_tensors='pt') # ModelInput dataclass


    '''below are helper functions w/o dependency to the self, but included inside the class for ease of use'''
    def likelihoods(self, logits, force_decode_indices, tgt_mask):
        probs = F.softmax(logits, dim=-1)
        probs_force_decode_ = probs.gather(-1, force_decode_indices.unsqueeze(-1)).squeeze()
        probs_force_decode= probs_force_decode_ * tgt_mask
        assert probs_force_decode.shape == force_decode_indices.shape
        return probs_force_decode

    def log_likelihoods(self, logits, force_decode_indices, tgt_mask):
        ll = F.log_softmax(logits, dim=-1)
        ll_force_decode_ = ll.gather(-1, force_decode_indices.unsqueeze(-1)).squeeze()
        ll_force_decode = ll_force_decode_ * tgt_mask

        return ll_force_decode

    def harim(self, s2s_logits, lm_logits, force_decode_indices, tgt_mask ):
        p_s2s, p_lm = self.likelihoods(s2s_logits, force_decode_indices, tgt_mask), \
                        self.likelihoods(lm_logits, force_decode_indices, tgt_mask)

        delta = p_s2s - p_lm
        margin_linear = (1-delta) / 2
        harim = -(1-p_s2s) * margin_linear + 1
        return harim # this is -1 * hallucination risk

    def make_minibatches(self, exs:List[str], bsz:int=32):
        idx=0
        minibatches = []
        while True:
            start = idx
            end = idx+bsz
            if start >= len(exs):
                break

            minibatches.append( exs[start:end] )
            idx += bsz
        return minibatches

    def make_empty_minibatches(self, minibatches:List[List[str]]):
        e_minibatches = minibatches.copy()
        for i, mb in enumerate(e_minibatches):
            e_minibatches[i] = ['' for ex in mb]
        return e_minibatches


    def compute(self, predictions:List[str],
                        references:List[str],
                        bsz:int=32,
                        use_aggregator:bool=False,
                        return_details:bool=False,
                        tokenwise_score:bool=False,
                        ):
        '''
        returns harim+ score (List[float]) for predictions (summaries) and references (articles)
        **Note**
            - here, predictions = generated summaries to be evaluated, references = article to be summarized (but to follow the convention of the evaluate, we named kwarg as "references")
            - log_ppl equals to bartscore (yuan et al., neurips 2021)

        if tokenwise_score:
            returns minibatch chunks of harim+ scores and log-likelihoods with tokenized predictions (List[str])
        if use_aggregator:
            returning scores are aggregated (mean) over given test set
        '''

        # tokenize/prep src/tgts
        make_minibatches_bsz = partial(self.make_minibatches, bsz=bsz)
        summaries = predictions
        articles = references
        b_srcs, b_tgts = map(make_minibatches_bsz, [articles, summaries])
        b_emps = self.make_empty_minibatches(b_srcs)

        scores=defaultdict(list)
        for mini_s, mini_e, mini_t in tqdm(zip(b_srcs, b_emps, b_tgts), total=len(b_tgts), desc=f"computing HaRiM+ {bsz=}, core={self._pretrained_name}"):
            src_in = self._prep_input(mini_s, src_or_tgt='src')
            emp_in = self._prep_input(mini_e, src_or_tgt='src')
            tgt_in = self._prep_input(mini_t, src_or_tgt='tgt')
            if emp_in.input_ids.shape[-1]==0: # emp_in.input_ids.shape == (32,0)
                boseos = f"{self._tokenizer.bos_token}{self._tokenizer.eos_token}"
                mini_e_ = [boseos for _ in range(len(mini_e))]
                emp_in = self._prep_input( mini_e_, src_or_tgt='src' )


            tgt_mask = tgt_in.attention_mask

            src_in = src_in.to(self._device)
            emp_in = emp_in.to(self._device)
            tgt_in = tgt_in.to(self._device)
            tgt_mask = tgt_mask.to(self._device)
            fill_ignore_mask = ~(tgt_mask.bool())

            with torch.no_grad():
                # token_type_ids attribute causes error
                s2s_logits = self._encdec_model.forward(
                                                    input_ids = src_in.input_ids,
                                                    attention_mask = src_in.attention_mask,
                                                    labels = tgt_in.input_ids.masked_fill(fill_ignore_mask, -100),
                                                    return_dict=True).logits
                lm_logits = self._encdec_model.forward(
                                                    input_ids = emp_in.input_ids,
                                                    attention_mask = emp_in.attention_mask,
                                                    labels = tgt_in.input_ids.masked_fill(fill_ignore_mask, -100),
                                                    return_dict=True).logits
                sent_lengths = tgt_mask.sum(-1)
                ll_tok = self.log_likelihoods(s2s_logits, tgt_in.input_ids, tgt_mask)
                ll = ll_tok.sum(-1) / sent_lengths

                harim_tok = self.harim(s2s_logits, lm_logits, tgt_in.input_ids, tgt_mask)
                harim = harim_tok.sum(-1) / sent_lengths

                harim_plus_normalized = ll + self._lambda * harim # loglikelihood + lambda * negative_harim (negative harim=-1* risk)

                scores['harim+'].extend(harim_plus_normalized.tolist())
                scores['harim'].extend(harim.tolist())
                scores['log_ppl'].extend(ll.tolist())

                if tokenwise_score:
                    scores['tok_harim+'].append(harim_tok*self._lambda + ll_tok)
                    scores['tok_predictions'].append( [self._tokenizer.convert_ids_to_token(idxs) for idxs in src_in.labels] )

        if use_aggregator: # after
            for k, v in scores.items():
                if not k.startswith('tok_'):
                    scores[k] = sum(v)/len(v) # aggregate (mean)
        scores['lambda'] = self._lambda
        if not return_details:
            scores = scores['harim+']
        return scores
