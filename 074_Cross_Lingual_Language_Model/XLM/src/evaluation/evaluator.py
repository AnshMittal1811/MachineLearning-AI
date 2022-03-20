# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import subprocess
from collections import OrderedDict
import numpy as np
import torch

from ..utils import to_cuda, restore_segmentation, concat_batches
from ..model.memory import HashingMemory

BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi-bleu.perl')
assert os.path.isfile(BLEU_SCRIPT_PATH)

# our
import copy
import itertools

logger = getLogger()


def kl_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    _x = x.copy()
    _x[x == 0] = 1
    return np.log(len(x)) + (x * np.log(_x)).sum()


def gini_score(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    B = np.cumsum(np.sort(x)).mean()
    return 1 - 2 * B


def tops(x):
    # assert np.abs(np.sum(x) - 1) < 1e-5
    y = np.cumsum(np.sort(x))
    top50, top90, top99 = y.shape[0] - np.searchsorted(y, [0.5, 0.1, 0.01])
    return top50, top90, top99


def eval_memory_usage(scores, name, mem_att, mem_size):
    """
    Evaluate memory usage (HashingMemory / FFN).
    """
    # memory slot scores
    assert mem_size > 0
    mem_scores_w = np.zeros(mem_size, dtype=np.float32)  # weighted scores
    mem_scores_u = np.zeros(mem_size, dtype=np.float32)  # unweighted scores

    # sum each slot usage
    for indices, weights in mem_att:
        np.add.at(mem_scores_w, indices, weights)
        np.add.at(mem_scores_u, indices, 1)

    # compute the KL distance to the uniform distribution
    mem_scores_w = mem_scores_w / mem_scores_w.sum()
    mem_scores_u = mem_scores_u / mem_scores_u.sum()

    # store stats
    scores['%s_mem_used' % name] = float(100 * (mem_scores_w != 0).sum() / len(mem_scores_w))

    scores['%s_mem_kl_w' % name] = float(kl_score(mem_scores_w))
    scores['%s_mem_kl_u' % name] = float(kl_score(mem_scores_u))

    scores['%s_mem_gini_w' % name] = float(gini_score(mem_scores_w))
    scores['%s_mem_gini_u' % name] = float(gini_score(mem_scores_u))

    top50, top90, top99 = tops(mem_scores_w)
    scores['%s_mem_top50_w' % name] = float(top50)
    scores['%s_mem_top90_w' % name] = float(top90)
    scores['%s_mem_top99_w' % name] = float(top99)

    top50, top90, top99 = tops(mem_scores_u)
    scores['%s_mem_top50_u' % name] = float(top50)
    scores['%s_mem_top90_u' % name] = float(top90)
    scores['%s_mem_top99_u' % name] = float(top99)


class Evaluator(object):

    def __init__(self, trainer, data, params):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.data = data
        self.dico = data['dico']
        self.params = params
        self.memory_list = trainer.memory_list

        # create directory to store hypotheses, and reference files for BLEU evaluation
        if self.params.is_master:
            params.hyp_path = os.path.join(params.dump_path, 'hypotheses')
            subprocess.Popen('mkdir -p %s' % params.hyp_path, shell=True).wait()
            self.create_reference_files()

    def get_iterator(self, data_set, lang1, lang2=None, stream=False, data_key = None):
        """
        Create a new iterator for a dataset.
        """
        
        # our
        #params = copy.deepcopy(self.params.meta_params[data_key]) if data_key else self.params
        params = self.params.meta_params[data_key] if data_key else self.params
        
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 is None or lang2 in params.langs
        assert stream is False or lang2 is None

        # hacks to reduce evaluation time when using many languages
        if len(params.langs) > 30:
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh", "ab", "ay", "bug", "ha", "ko", "ln", "min", "nds", "pap", "pt", "tg", "to", "udm", "uk", "zh_classical"])
            eval_lgs = set(["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"])
            subsample = 10 if (data_set == 'test' or lang1 not in eval_lgs) else 5
            n_sentences = 600 if (data_set == 'test' or lang1 not in eval_lgs) else 1500
        elif len(params.langs) > 5:
            subsample = 10 if data_set == 'test' else 5
            n_sentences = 300 if data_set == 'test' else 1500
        else:
            # n_sentences = -1 if data_set == 'valid' else 100
            n_sentences = -1
            subsample = 1
            
        if lang2 is None:
            if stream:
                if data_key :
                    iterator = self.data[data_key]['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
                else :
                    iterator = self.data['mono_stream'][lang1][data_set].get_iterator(shuffle=False, subsample=subsample)
            else:
                if data_key :
                    iterator = self.data[data_key]['mono'][lang1][data_set].get_iterator(
                        shuffle=False,
                        group_by_size=True,
                        n_sentences=n_sentences,
                    )
                else :
                    iterator = self.data['mono'][lang1][data_set].get_iterator(
                        shuffle=False,
                        group_by_size=True,
                        n_sentences=n_sentences,
                    )            
        else:
            assert stream is False
            _lang1, _lang2 = (lang1, lang2) if lang1 < lang2 else (lang2, lang1)
            if data_key :
                iterator = self.data[data_key]['para'][(_lang1, _lang2)][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences
                )
            else :
                iterator = self.data['para'][(_lang1, _lang2)][data_set].get_iterator(
                    shuffle=False,
                    group_by_size=True,
                    n_sentences=n_sentences
                )

        for batch in iterator:
            yield batch if lang2 is None or lang1 < lang2 else batch[::-1]

    def create_reference_files(self):
        """
        Create reference files for BLEU evaluation.
        """
        params = self.params
        params.ref_paths = {}
        
        if not params.meta_learning :
                
            for (lang1, lang2), v in self.data['para'].items():

                assert lang1 < lang2

                for data_set in ['valid', 'test']:

                    # define data paths
                    lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                    lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))

                    # store data paths
                    params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                    params.ref_paths[(lang1, lang2, data_set)] = lang2_path

                    # text sentences
                    lang1_txt = []
                    lang2_txt = []

                    # convert to text
                    for (sent1, len1), (sent2, len2) in self.get_iterator(data_set, lang1, lang2):
                        lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                        lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))

                    # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                    lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                    lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                    # export hypothesis
                    with open(lang1_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lang1_txt) + '\n')
                    with open(lang2_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lang2_txt) + '\n')

                    # restore original segmentation
                    restore_segmentation(lang1_path)
                    restore_segmentation(lang2_path)
        else :
            # our 
            
            for lgs, data in self.data.items():
                try :
                    for (lang1, lang2), v in data['para'].items():
                       
                        assert lang1 < lang2
                          
                        for data_set in ['valid', 'test']:

                            # define data paths
                            lang1_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang2, lang1, data_set))
                            lang2_path = os.path.join(params.hyp_path, 'ref.{0}-{1}.{2}.txt'.format(lang1, lang2, data_set))
                            
                            # store data paths
                            params.ref_paths[(lang2, lang1, data_set)] = lang1_path
                            params.ref_paths[(lang1, lang2, data_set)] = lang2_path
                            
                            # text sentences
                            lang1_txt = []
                            lang2_txt = []
                            
                            # convert to text : 
                            for (sent1, len1), (sent2, len2) in self.get_iterator(data_set, lang1, lang2, data_key = lgs):
                                lang1_txt.extend(convert_to_text(sent1, len1, self.dico, params))
                                lang2_txt.extend(convert_to_text(sent2, len2, self.dico, params))
                                
                            # replace <unk> by <<unk>> as these tokens cannot be counted in BLEU
                            lang1_txt = [x.replace('<unk>', '<<unk>>') for x in lang1_txt]
                            lang2_txt = [x.replace('<unk>', '<<unk>>') for x in lang2_txt]

                            # export hypothesis
                            with open(lang1_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(lang1_txt) + '\n')
                            with open(lang2_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(lang2_txt) + '\n')

                            # restore original segmentation
                            restore_segmentation(lang1_path)
                            restore_segmentation(lang2_path)
                except Exception as e:
                    pass
            
    def mask_out(self, x, lengths, rng, data_key = None):
        """
        Decide of random words to mask out.
        We specify the random generator to ensure that the test is the same at each epoch.
        """
        # our
        params = self.params
        if data_key :
            #params = copy.deepcopy(self.params.meta_params[data_key])
            params = self.params.meta_params[data_key]
            # todo
            params.pred_probs = self.params.pred_probs
            
        slen, bs = x.size()

        # words to predict - be sure there is at least one word per sentence
        to_predict = rng.rand(slen, bs) <= params.word_pred
        to_predict[0] = 0
        for i in range(bs):
            to_predict[lengths[i] - 1:, i] = 0
            if not np.any(to_predict[:lengths[i] - 1, i]):
                v = rng.randint(1, lengths[i] - 1)
                to_predict[v, i] = 1
        pred_mask = torch.from_numpy(to_predict.astype(np.uint8))

        # generate possible targets / update x input
        pred_mask = pred_mask.bool() # MODIFIED BY JAMES TO GET RID OF THE WARNING
        _x_real = x[pred_mask]
        _x_mask = _x_real.clone().fill_(params.mask_index)
        x = x.masked_scatter(pred_mask, _x_mask)

        assert 0 <= x.min() <= x.max() < params.n_words
        assert x.size() == (slen, bs)
        assert pred_mask.size() == (slen, bs)

        return x, _x_real, pred_mask

    def run_all_evals(self, trainer):
        """
        Run all evaluations.
        """
        
        params = self.params
        
        # our
        evaluate_mlm = False
        evaluate_clm = False
        evaluate_mt = False
        
        
        scores = OrderedDict({'epoch': trainer.epoch})

        with torch.no_grad():

            for data_set in ['valid', 'test']:
                
                if not params.meta_learning :

                    # causal prediction task (evaluate perplexity and accuracy)
                    for lang1, lang2 in params.clm_steps:
                        self.evaluate_clm(scores, data_set, lang1, lang2)

                    # prediction task (evaluate perplexity and accuracy)
                    for lang1, lang2 in params.mlm_steps:
                        self.evaluate_mlm(scores, data_set, lang1, lang2)

                    # machine translation task (evaluate perplexity and accuracy)
                    for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps]):
                        eval_bleu = params.eval_bleu and params.is_master
                        self.evaluate_mt(scores, data_set, lang1, lang2, eval_bleu)

                    # report average metrics per language
                    _clm_mono = [l1 for (l1, l2) in params.clm_steps if l2 is None]
                    if len(_clm_mono) > 0:
                        scores['%s_clm_ppl' % data_set] = np.mean([scores['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                        scores['%s_clm_acc' % data_set] = np.mean([scores['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                    _mlm_mono = [l1 for (l1, l2) in params.mlm_steps if l2 is None]
                    if len(_mlm_mono) > 0:
                        scores['%s_mlm_ppl' % data_set] = np.mean([scores['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                        scores['%s_mlm_acc' % data_set] = np.mean([scores['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])

                else :
                    # our 
                    
                    # equivalent to "for task in list of task" in the original algorithm
                    for lgs, params_ in params.meta_params.items() :
                        # todo ?
                        params_.is_master = params.is_master
                        
                        data_key=lgs
                        
                        try :
                            scores[data_key]
                        except KeyError :
                            scores[data_key] = {}
                            scores[data_key]['epoch'] = scores['epoch']
                        
                        # causal prediction task (evaluate perplexity and accuracy)
                        for lang1, lang2 in params_.clm_steps:
                            evaluate_clm = True
                            self.evaluate_clm(scores[data_key], data_set, lang1, lang2, data_key = data_key)
               
                        # prediction task (evaluate perplexity and accuracy)
                        for lang1, lang2 in params_.mlm_steps:
                            evaluate_mlm = True
                            self.evaluate_mlm(scores[data_key], data_set, lang1, lang2, data_key = data_key)
                        
                        # machine translation task (evaluate perplexity and accuracy)
                        for lang1, lang2 in set(params_.mt_steps + [(l2, l3) for _, l2, l3 in params_.bt_steps]):
                            evaluate_mt = True
                            eval_bleu = params_.eval_bleu and params_.is_master
                            self.evaluate_mt(scores[data_key], data_set, lang1, lang2, eval_bleu, data_key = data_key)
                        """
                        # report average metrics per language
                        _clm_mono = [l1 for (l1, l2) in params_.clm_steps if l2 is None]
                        if len(_clm_mono) > 0:
                            scores[data_key]['%s_clm_ppl' % data_set] = np.mean([scores[data_key]['%s_%s_clm_ppl' % (data_set, lang)] for lang in _clm_mono])
                            scores[data_key]['%s_clm_acc' % data_set] = np.mean([scores[data_key]['%s_%s_clm_acc' % (data_set, lang)] for lang in _clm_mono])
                        _mlm_mono = [l1 for (l1, l2) in params_.mlm_steps if l2 is None]
                        if len(_mlm_mono) > 0:
                            scores[data_key]['%s_mlm_ppl' % data_set] = np.mean([scores[data_key]['%s_%s_mlm_ppl' % (data_set, lang)] for lang in _mlm_mono])
                            scores[data_key]['%s_mlm_acc' % data_set] = np.mean([scores[data_key]['%s_%s_mlm_acc' % (data_set, lang)] for lang in _mlm_mono])
                        """
        
        # our : report average metrics per task
        # if params.meta_learning 
        ## clm and mlm
        def get_step(steps):
            langs = []
            for (l1, l2) in steps :
                if (l1 is None) and (not (l2 is None)) :
                    langs.append(l2)
                elif (not (l1 is None)) and (l2 is None) :
                    langs.append(l1)
                elif (not (l1 is None)) and (not (l2 is None)) :
                    langs.append(l1+"_"+l2)
            langs = list(set(langs))
            return langs

        for objectif in ((['clm'] if evaluate_clm else []) + (['mlm'] if evaluate_mlm else [])) :
            values = ['ppl', 'acc']
            for data_key, params_ in params.meta_params.items() :
                steps = params_.mlm_steps if objectif=='mlm' else params_.clm_steps
                """
                langs = []
                for (l1, l2) in steps :
                    if (l1 is None) and (not (l2 is None)) :
                        langs.append(l2)
                    elif (not (l1 is None)) and (l2 is None) :
                        langs.append(l1)
                    elif (not (l1 is None)) and (not (l2 is None)) :
                        langs.append(l1+"_"+l2)   
                # removes duplicates
                langs = list(set(langs))
                """
                langs = get_step(steps)
                for data_set in ['valid', 'test'] :
                    for value in values :
                        a = [scores[data_key]['%s_%s_%s_%s' % (data_set, lang, objectif, value)] 
                             for lang in langs
                             if '%s_%s_%s_%s' % (data_set, lang, objectif, value) in scores[data_key].keys()
                            ]
                        if a != [] :
                            scores[data_key]['task_(%s)_%s_%s_%s' % (data_key, data_set, objectif, value)] = np.mean(a)
                        
            for data_set in ['valid', 'test'] :
                # Right now we are averaging the values for each task. 
                # todo : Do a study to see how to weight the coefficients of this average (average weighted by the coefficients not all equal). 
                for value in values :
                    scores['%s_%s_%s' % (data_set, objectif, value)] =  np.mean(
                        [
                            scores[data_key]['task_(%s)_%s_%s_%s' % (data_key, data_set, objectif, value)] 
                            for data_key in params.meta_params.keys()  
                            if 'task_(%s)_%s_%s_%s' % (data_key, data_set, objectif, value) in scores[data_key].keys()
                        ]
                    )
         
        ## mt
        if evaluate_mt :
                
            eval_bleu = params.eval_bleu and params.is_master
            values = ['ppl', 'acc'] + (['bleu'] if eval_bleu else [])
                
            for data_key, params_ in params.meta_params.items() :
                 
                langs = [lang1+"-"+lang2 for lang1, lang2 in set(params_.mt_steps + [(l2, l3) for _, l2, l3 in params_.bt_steps])]
                                      
                for data_set in ['valid', 'test'] :
                    for value in values :
                        #scores[data_key]['%s_mt_%s' % (data_set, value)] = scores[data_key]['task_(%s)_%s_mt_%s' % (data_key, data_set, value)]
                        a = [scores[data_key]['%s_%s_mt_%s' % (data_set, lang, value)] 
                             for lang in langs 
                             if '%s_%s_mt_%s' % (data_set, lang, value) in scores[data_key].keys()
                            ]
                        if a != [] :
                            scores[data_key]['task_(%s)_%s_mt_%s' % (data_key, data_set, value)] = np.mean(a)
                    
            for data_set in ['valid', 'test'] :
                # Right now we are averaging the values for each task. 
                # todo : Do a study later to see how to weight the coefficients of this average (average weighted by the coefficients not all equal). 
                for value in values :
                    scores['%s_mt_%s' % (data_set, value)] =  np.mean(
                        [
                            #scores[data_key]['%s_mt_%s' % (data_set, value)] for data_key in params.meta_params.keys()
                            scores[data_key]['task_(%s)_%s_mt_%s' % (data_key, data_set, value)] 
                            for data_key in params.meta_params.keys()   
                            if 'task_(%s)_%s_mt_%s' % (data_key, data_set, value) in scores[data_key].keys()
                        ]
                    )

        ## valid_mt_bleu
        ### aggregation metrics : "name_metric1=mean(m1,m2);name_metric2=sum(m4,m5)..."

        score_keys = scores.keys()

        def off_stars(metrics_list):
            eval_bleu = params.eval_bleu and params.is_master
            result_list = []
            for metric in metrics_list :
                if "*" in metric :
                    parts = metric.split("_")
                    assert len(parts) == 4 
                    re_metrics = []
                    for i, part in enumerate(parts):
                        if "*" == part :
                            if i == 0 :
                                re_metrics.append(["valid", "test"])
                            elif i == 1 :
                                langs = [lang1+"-"+lang2 for lang1, lang2 in set(params.mt_steps + [(l2, l3) for _, l2, l3 in params.bt_steps])]
                                langs_mlm = get_step(params.mlm_steps)
                                langs_clm = get_step(params.clm_steps)
                                langs = langs + langs_mlm + langs_clm
                                re_metrics.append(langs) 
                            elif i == 2 :
                                ap = []
                                if params.clm_steps :
                                    ap.append("clm")
                                if params.mlm_steps :
                                    ap.append("mlm")
                                if eval_bleu :
                                    ap.append("mt")
                                if ap :
                                   re_metrics.append(ap) 
                            elif i == 3 :
                                re_metrics.append(['ppl', 'acc'] + (['bleu'] if eval_bleu else []))  
                        else :
                            re_metrics.append([part])
                    # produit cartesien
                    re_metrics = itertools.product(*re_metrics)
                    re_metrics = ["_".join(re_metric) for re_metric in re_metrics]
                    re_metrics = [re_metric for re_metric in re_metrics if re_metric in score_keys]
                    result_list += re_metrics
                else :
                    result_list.append(metric)
            return result_list

        if params.aggregation_metrics :
            for ag_metrics in params.aggregation_metrics.split(";") :
                s = ag_metrics.split("=")
                name = s[0]
                assert (not name in score_keys) and (not "*" in name)
                s = s[1].split("(")
                reductor = s[0] 
                assert reductor in ["mean", "sum"]
                reductor = np.mean if reductor == "mean" else np.sum
                metrics_list = s[1][:-1].split(",") # withdraw the last parathesis and split
                metrics_list = off_stars(metrics_list)
                scores[name] = reductor([scores[metric] for metric in metrics_list])
                
        ##############

        return scores

    def evaluate_clm(self, scores, data_set, lang1, lang2, data_key = None):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        # our
        params = self.params
        if data_key :
            params = copy.deepcopy(self.params.meta_params[data_key])
            # todo ??
            params.multi_gpu = self.params.multi_gpu
        
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.decoder
        model.eval()
        model = model.module if params.multi_gpu else model

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else f"{lang1}-{lang2}"

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None), data_key = data_key):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            alen = torch.arange(lengths.max(), dtype=torch.long, device=lengths.device)
            pred_mask = alen[:, None] < lengths[None] - 1
            y = x[1:].masked_select(pred_mask[:-1])
            assert pred_mask.sum().item() == y.size(0)

            # cuda
            x, lengths, positions, langs, pred_mask, y = to_cuda(x, lengths, positions, langs, pred_mask, y)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=True)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

        # log
        logger.info("Found %i words in %s. %i were predicted correctly." % (n_words, data_set, n_valid))

        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_clm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_clm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words)
        scores[acc_name] = 100. * n_valid / n_words

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s_%s' % (data_set, l1l2, mem_name), mem_att, params.mem_size)

    def evaluate_mlm(self, scores, data_set, lang1, lang2, data_key = None):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        params = self.params
        if data_key :
            #params = copy.deepcopy(self.params.meta_params[data_key])
            params = self.params.meta_params[data_key]
            # todo ??
            params.multi_gpu = self.params.multi_gpu
            
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs or lang2 is None

        model = self.model if params.encoder_only else self.encoder
        model.eval()
        model = model.module if params.multi_gpu else model

        rng = np.random.RandomState(0)

        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2] if lang2 is not None else None
        l1l2 = lang1 if lang2 is None else f"{lang1}_{lang2}"

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        i = 0
        
        for batch in self.get_iterator(data_set, lang1, lang2, stream=(lang2 is None), data_key = data_key):

            # batch
            if lang2 is None:
                x, lengths = batch
                positions = None
                langs = x.clone().fill_(lang1_id) if params.n_langs > 1 else None
            else:
                (sent1, len1), (sent2, len2) = batch
                x, lengths, positions, langs = concat_batches(sent1, len1, lang1_id, sent2, len2, lang2_id, params.pad_index, params.eos_index, reset_positions=True)

            # words to predict
            x, y, pred_mask = self.mask_out(x, lengths, rng, data_key = data_key)

            # cuda
            x, y, pred_mask, lengths, positions, langs = to_cuda(x, y, pred_mask, lengths, positions, langs)

            # forward / loss
            tensor = model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
            word_scores, loss = model('predict', tensor=tensor, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += len(y)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

            i = i + 1
            if i == 10 :
                #break
                pass
        # compute perplexity and prediction accuracy
        ppl_name = '%s_%s_mlm_ppl' % (data_set, l1l2)
        acc_name = '%s_%s_mlm_acc' % (data_set, l1l2)
        scores[ppl_name] = np.exp(xe_loss / n_words) if n_words > 0 else 1e9
        scores[acc_name] = 100. * n_valid / n_words if n_words > 0 else 0.
        
        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s_%s' % (data_set, l1l2, mem_name), mem_att, params.mem_size)


class SingleEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build language model evaluator.
        """
        super().__init__(trainer, data, params)
        self.model = trainer.model


class EncDecEvaluator(Evaluator):

    def __init__(self, trainer, data, params):
        """
        Build encoder / decoder evaluator.
        """
        super().__init__(trainer, data, params)
        self.encoder = trainer.encoder
        self.decoder = trainer.decoder

    def evaluate_mt(self, scores, data_set, lang1, lang2, eval_bleu, data_key = None):
        """
        Evaluate perplexity and next word prediction accuracy.
        """
        # our
        params = self.params
        if data_key :
            #params = copy.deepcopy(self.params.meta_params[data_key])
            params = self.params.meta_params[data_key]
            # todo ??
            params.multi_gpu = self.params.multi_gpu
            params.hyp_path = self.params.hyp_path
            params.ref_paths = self.params.ref_paths
            
        assert data_set in ['valid', 'test']
        assert lang1 in params.langs
        assert lang2 in params.langs

        self.encoder.eval()
        self.decoder.eval()
        encoder = self.encoder.module if params.multi_gpu else self.encoder
        decoder = self.decoder.module if params.multi_gpu else self.decoder

        params = params
        lang1_id = params.lang2id[lang1]
        lang2_id = params.lang2id[lang2]

        n_words = 0
        xe_loss = 0
        n_valid = 0

        # only save states / evaluate usage on the validation set
        eval_memory = params.use_memory and data_set == 'valid' and self.params.is_master
        HashingMemory.EVAL_MEMORY = eval_memory
        if eval_memory:
            all_mem_att = {k: [] for k, _ in self.memory_list}

        # store hypothesis to compute BLEU score
        if eval_bleu:
            hypothesis = []

        for batch in self.get_iterator(data_set, lang1, lang2, data_key = data_key):

            # generate batch
            (x1, len1), (x2, len2) = batch
            langs1 = x1.clone().fill_(lang1_id)
            langs2 = x2.clone().fill_(lang2_id)

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = alen[:, None] < len2[None] - 1  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1, len1, langs1, x2, len2, langs2, y = to_cuda(x1, len1, langs1, x2, len2, langs2, y)

            # encode source sentence
            enc1 = encoder('fwd', x=x1, lengths=len1, langs=langs1, causal=False)
            enc1 = enc1.transpose(0, 1)
            enc1 = enc1.half() if params.fp16 else enc1

            # decode target sentence
            dec2 = decoder('fwd', x=x2, lengths=len2, langs=langs2, causal=True, src_enc=enc1, src_len=len1)

            # loss
            word_scores, loss = decoder('predict', tensor=dec2, pred_mask=pred_mask, y=y, get_scores=True)

            # update stats
            n_words += y.size(0)
            xe_loss += loss.item() * len(y)
            n_valid += (word_scores.max(1)[1] == y).sum().item()
            if eval_memory:
                for k, v in self.memory_list:
                    all_mem_att[k].append((v.last_indices, v.last_scores))

            # generate translation - translate / convert to text
            if eval_bleu:
                max_len = int(1.5 * len1.max().item() + 10)
                if params.beam_size == 1:
                    generated, lengths = decoder.generate(enc1, len1, lang2_id, max_len=max_len)
                else:
                    generated, lengths = decoder.generate_beam(
                        enc1, len1, lang2_id, beam_size=params.beam_size,
                        length_penalty=params.length_penalty,
                        early_stopping=params.early_stopping,
                        max_len=max_len
                    )
                hypothesis.extend(convert_to_text(generated, lengths, self.dico, params))

        # compute perplexity and prediction accuracy
        scores['%s_%s-%s_mt_ppl' % (data_set, lang1, lang2)] = np.exp(xe_loss / n_words)
        scores['%s_%s-%s_mt_acc' % (data_set, lang1, lang2)] = 100. * n_valid / n_words

        # compute memory usage
        if eval_memory:
            for mem_name, mem_att in all_mem_att.items():
                eval_memory_usage(scores, '%s_%s-%s_%s' % (data_set, lang1, lang2, mem_name), mem_att, params.mem_size)

        # compute BLEU
        if eval_bleu:

            # hypothesis / reference paths
            hyp_name = 'hyp{0}.{1}-{2}.{3}.txt'.format(scores['epoch'], lang1, lang2, data_set)
            hyp_path = os.path.join(params.hyp_path, hyp_name)
            ref_path = params.ref_paths[(lang1, lang2, data_set)]

            # export sentences to hypothesis file / restore BPE segmentation
            with open(hyp_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(hypothesis) + '\n')
            restore_segmentation(hyp_path)

            # evaluate BLEU score
            bleu = eval_moses_bleu(ref_path, hyp_path)
            logger.info("BLEU %s %s : %f" % (hyp_path, ref_path, bleu))
            scores['%s_%s-%s_mt_bleu' % (data_set, lang1, lang2)] = bleu


def convert_to_text(batch, lengths, dico, params):
    """
    Convert a batch of sentences to a list of text sentences.
    """
    batch = batch.cpu().numpy()
    lengths = lengths.cpu().numpy()

    slen, bs = batch.shape
    assert lengths.max() == slen and lengths.shape[0] == bs
    assert (batch[0] == params.eos_index).sum() == bs
    assert (batch == params.eos_index).sum() == 2 * bs
    sentences = []

    for j in range(bs):
        words = []
        for k in range(1, lengths[j]):
            if batch[k, j] == params.eos_index:
                break
            words.append(dico[batch[k, j]])
        sentences.append(" ".join(words))
    return sentences

"""
import threading
import time, math
stop_threads = False
def thread_target(cmd : str, wait : int = 0, timeout : int = None):
    #128 = programme non lancé, 0 = programme était lancé, bien fermé
    os.system(cmd)
    if not timeout :
        timeout = math.inf
    t = 0
    while t < timeout :
        os.system(cmd)
        time.sleep(wait)
        t += wait
        global stop_threads
        if stop_threads :
            break
"""

def eval_moses_bleu(ref, hyp):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    assert os.path.isfile(hyp)
    assert os.path.isfile(ref) or os.path.isfile(ref + '0')
    assert os.path.isfile(BLEU_SCRIPT_PATH)
    command = BLEU_SCRIPT_PATH + ' %s < %s'
    # our
    if os.name == 'nt' : # windows os
        command = "perl " + command

    p = subprocess.Popen(command % (ref, hyp), stdout=subprocess.PIPE, shell=True)

    #global stop_threads
    #stop_threads = False
    #threading.Thread(target = thread_target,  kwargs={"cmd" : "taskkill /f /im notepad.exe", "wait" : 5, "timeout" : None}).start()
    result = p.communicate()[0].decode("utf-8")
    #stop_threads = True
    
    if result.startswith('BLEU'):
        return float(result[7:result.index(',')])
    else:
        logger.warning('Impossible to parse BLEU score! "%s"' % result)
        return -1
