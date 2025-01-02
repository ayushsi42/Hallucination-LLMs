import datasets
import evaluate
# from harim_scorer import Harimplus_Scorer #no plan to package it to pip
import torch
import torch.nn.functional as F
from transformers import (AutoModelForSeq2SeqLM,
                        AutoTokenizer,
                        PreTrainedTokenizer,
                        PreTrainedTokenizerFast,
                        )
from transformers.tokenization_utils_base import BatchEncoding # for custom tokenizer other than huggingface
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Union
from collections import defaultdict
from functools import partial

logger = evaluate.logging.get_logger(__name__)

CODEBASE_URL='https://huggingface.co/spaces/NCSOFT/harim_plus'
PAPER_URL='https://arxiv.org/abs/2211.12118'

_CITATION = """\
@inproceedings{son-etal-2022-harim,
    title = "{H}a{R}i{M}$^+$: Evaluating Summary Quality with Hallucination Risk",
    author = "Son, Seonil (Simon)  and
      Park, Junsoo  and
      Hwang, Jeong-in  and
      Lee, Junghwa  and
      Noh, Hyungjong  and
      Lee, Yeonsoo",
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing",
    month = nov,
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-main.66",
    pages = "895--924",
    abstract = "One of the challenges of developing a summarization model arises from the difficulty in measuring the factual inconsistency of the generated text. In this study, we reinterpret the decoder overconfidence-regularizing objective suggested in (Miao et al., 2021) as a hallucination risk measurement to better estimate the quality of generated summaries. We propose a reference-free metric, HaRiM+, which only requires an off-the-shelf summarization model to compute the hallucination risk based on token likelihoods. Deploying it requires no additional training of models or ad-hoc modules, which usually need alignment to human judgments. For summary-quality estimation, HaRiM+ records state-of-the-art correlation to human judgment on three summary-quality annotation sets: FRANK, QAGS, and SummEval. We hope that our work, which merits the use of summarization models, facilitates the progress of both automated evaluation and generation of summary.",
}
"""

_DESCRIPTION = f"""HaRiM+ is a reference-less evaluation metric (i.e. requires only article-summary pair, no reference summary) for summarization which leverages the power of summarization model.
Summarization model inside the HaRiM+ will read and evaluate how good the quality of a summary given the paired article.
It will work great for ranking the summary-article pairs according to its quality.

HaRiM+ is proved effective for benchmarking summarization systems (system-level performance) as well as ranking the article-summary pairs (segment-level performance) in comprehensive aspect such as factuality, consistency, coherency, fluency, and relevance. For details, refer to our [paper]({PAPER_URL}) published in AACL2022.

NOTE that for HaRiM+...
* predictions = summaries (List[str])
* references = articles (List[str])

Also Note that
* higher score = better quality
"""

_KWARGS_DESCRIPTION = """
HaRiM+ score.
Args:
    For scorer = evaluate.load():
    `pretrained_name` (str or pathlib.Path): summarization model checkpoint or path, loaded by transformers.AutoModelForSeq2SeqLM.from_pretrained(). Defaults to Yale-LILY/brio-cnndm-uncased.
    `tokenizer`: (use when your tokenizer cannot be loaded by from_pretrained)Tokenizer function compatible with transformers.PreTrainedTokenizer. It requires tokenizer.pad_token|eos_token|bos_token and tokenizer.__call__() method for HaRiM+ score computation.

    For scorer.compute():
    `predictions` (list of str): generated summaries
    `references` (list of str): source articles to be summarized
    `use_aggregator` (bool=False): if True, average of the scores are returned
    `bsz` (int=32): batch size for harim to iterate through the given pairs
    `return_details` (bool=False): whether to show more than harim+ score (returns logppl, harim term. refer to the paper for detail)
        `tokenwise_score` (bool=False): whether to show tokenwise scores for input pairs (if return_details=False, this is ignored)

Returns:
    'results' (list of float): harim+ score for each summary-article pair

Examples:
    >>> summaries = ["hello there", "hello there"]
    >>> articles = ["hello, this is the article to be summarized", "hello, this is the article to be summarized"]
    >>> scorer = evaluate.load("NCSOFT/harim_plus") #, pretrained_name='PRETRAINEDNAME', tokenizer=TOKENIZER # optional
    >>> results = scorer.compute(predictions=summaries, references=articles) # use_aggregator=True # optional
    >>> print([round(v, 2) for v in results["harim+"]])
    [float, float]
"""

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
                        # tokenwise_score:bool=False,
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


            tgt_mask = tgt_in.attention_mask # torch.Tensor
            # if not tokenizer loaded from huggingface, this might cause some problem (.to(device))
            if not isinstance(src_in, BatchEncoding):
                src_in = BatchEncoding(src_in)
            if not isinstance(emp_in, BatchEncoding):
                emp_in = BatchEncoding(emp_in)
            if not isinstance(tgt_in, BatchEncoding):
                tgt_in = BatchEncoding(tgt_in)
                
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

                harim_plus_normalized = (ll + self._lambda * harim) # loglikelihood + lambda * negative_harim (negative harim=-1* risk)

                scores['harim+'].extend(harim_plus_normalized.tolist())
                scores['harim'].extend(harim.tolist())
                scores['log_ppl'].extend(ll.tolist())

                # if tokenwise_score:
                #     scores['tok_harim+'].append(harim_tok*self._lambda + ll_tok)
                #     scores['tok_predictions'].append( [self._tokenizer.convert_ids_to_token(idxs) for idxs in src_in.labels] )

        if use_aggregator: # after
            for k, v in scores.items():
                if not k.startswith('tok_'):
                    scores[k] = sum(v)/len(v) # aggregate (mean)
        scores['lambda'] = self._lambda
        if not return_details:
            scores = scores['harim+']
        return scores





@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Harimplus(evaluate.Metric):
    def __init__(self,
                    pretrained_name='facebook/bart-large-cnn',
                    tokenizer=None,
                    device='cuda',
                    **kwargs
                    ):
        super().__init__(**kwargs)
        self.myconfig = dict(
                            pretrained_name=pretrained_name,
                            tokenizer=tokenizer,
                            device=device,
                            )

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=CODEBASE_URL,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=[CODEBASE_URL],
            reference_urls=[CODEBASE_URL, PAPER_URL],
        )

    def _download_and_prepare(self, dl_manager):
        pretrained_name = self.myconfig['pretrained_name']
        is_custom_tokenizer = self.myconfig['tokenizer'] is not None
        logger.warning(
            "Loading HaRiM+ score"
            f"\tpretrained_name = {pretrained_name}"
        )
        if is_custom_tokenizer:
            logger.warning(
                f"tokenizer is overriden by \n\tself.myconfig['tokenizer']"
            )
        logger.warning(
            "You can change checkpoints with `pretrained_name` kwarg in evaluate.load. Strongly recommend to use *-large or larger ones."
            "Refrain from using checkpoints trained on noisy corpus such as bbc-XSUM.")

        # download the model checkpoint specified by self.myconfig_name and set up the scorer
        self.scorer = Harimplus_Scorer(**self.myconfig)

    def _compute(self, predictions=None,
                        references=None,
                        use_aggregator=False,
                        bsz=32,
                        return_details=False):       
                        # tokenwise_score=False,
                        
        summaries = predictions
        articles = references
        scores = self.scorer.compute(predictions=summaries,
                                    references=articles,
                                    use_aggregator=use_aggregator,
                                    bsz=bsz, #tokenwise_score=tokenwise_score,
                                    return_details=return_details)
        return scores
