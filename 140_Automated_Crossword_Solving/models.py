# This file contains the inference code for loading and running the closed-book and open-book QA models
import os
import csv
import glob
import gzip
import string
import sys
from typing import List, Tuple, Dict
import re
import math
import collections

import numpy as np
import unicodedata
import torch
from torch import Tensor as T
from torch import nn

from DPR.dpr.models import init_biencoder_components
from DPR.dpr.options import setup_args_gpu, print_args, set_encoder_params_from_state 
from DPR.dpr.indexer.faiss_indexers import DenseIndexer, DenseFlatIndexer
#from DPR.dpr.data.reader_data import ReaderSample, ReaderPassage, get_best_spans
#from DPR.dpr.models import init_reader_components
from DPR.dpr.utils.data_utils import Tensorizer
from DPR.dpr.utils.model_utils import load_states_from_checkpoint, get_model_obj

from transformers import GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, AutoTokenizer
from segment_fill import segment_fill

SEGMENTER_CACHE = {}
RERANKER_CACHE = {}

def setup_closedbook(process_id):
    dpr = DPRForCrossword(
        "checkpoints/biencoder/dpr_biencoder.bin",
        "checkpoints/biencoder/wordlist.tsv",
        "checkpoints/biencoder/embeddings/embeddings.json_*",
        retrievalmodel=False,
        process_id=process_id
    )
    return dpr

def setup_t5_reranker(process_id):
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
    model = T5ForConditionalGeneration.from_pretrained('checkpoints/byt5_reranker/')
    model.eval().to('cuda:'+str(process_id % torch.cuda.device_count()))
    return model, tokenizer

def t5_reranker_score_with_clue(model, tokenizer, clues, possibly_ungrammatical_fills):
    global RERANKER_CACHE
    results = []
    for clue, possibly_ungrammatical_fill in zip(clues, possibly_ungrammatical_fills):
        if not possibly_ungrammatical_fill.islower():
            possibly_ungrammatical_fill = possibly_ungrammatical_fill.lower()
        clue = preprocess_clue_fn(clue)
        if clue[-3:] == '. .':
            clue = clue[:-3]
        elif clue[-3:] == ' ..':
            clue = clue[:-3]
        elif clue[-2:] == '..':
            clue = clue[:-2]
        elif clue[-1] == '.':
            clue = clue[:-1]

        if clue + possibly_ungrammatical_fill in RERANKER_CACHE:
            results.append(RERANKER_CACHE[clue + possibly_ungrammatical_fill])
            continue
        else:
            with torch.inference_mode():
                inputs = tokenizer(['Q: ' + clue], return_tensors='pt')['input_ids'].to(model.device)
                labels = tokenizer([possibly_ungrammatical_fill], return_tensors='pt')['input_ids'].to(model.device)
                loss = model(inputs, labels=labels)
                answer_length = labels.shape[1]
                logprob = -loss[0].item() * answer_length
                results.append(logprob)
                RERANKER_CACHE[clue + possibly_ungrammatical_fill] = logprob

    return results

def preprocess_clue_fn(clue):
    clue = str(clue)

    # https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    clue = ''.join(c for c in unicodedata.normalize('NFD', clue) if unicodedata.category(c) != 'Mn')

    clue = re.sub("\x17|\x18|\x93|\x94|“|”|''|\"\"", "\"", clue)
    clue = re.sub("\x85|…", "...", clue)
    clue = re.sub("\x91|\x92|‘|’", "'", clue)

    clue = re.sub("‚", ",", clue)
    clue = re.sub("—|–", "-", clue)
    clue = re.sub("¢", " cents", clue)
    clue = re.sub("¿|¡|^;|\{|\}", "", clue)
    clue = re.sub("÷", "division", clue)
    clue = re.sub("°", " degrees", clue)

    euro = re.search("^£[0-9]+(,*[0-9]*){0,}| £[0-9]+(,*[0-9]*){0,}", clue)
    if euro:
        num = clue[:euro.end()]
        rest_clue = clue[euro.end():]
        clue = num + " Euros" + rest_clue
        clue = re.sub(", Euros", " Euros", clue)
        clue = re.sub("Euros [Mm]illion", "million Euros", clue)
        clue = re.sub("Euros [Bb]illion", "billion Euros", clue)
        clue = re.sub("Euros[Kk]", "K Euros", clue)
        clue = re.sub(" K Euros", "K Euros", clue)
        clue = re.sub("£", "", clue)

    clue = re.sub(" *\(\d{1,},*\)$| *\(\d{1,},* \d{1,}\)$", "", clue)

    clue = re.sub("&amp;", "&", clue)
    clue = re.sub("&lt;", "<", clue)
    clue = re.sub("&gt;", ">", clue)

    clue = re.sub("e\.g\.|for ex\.", "for example", clue)
    clue = re.sub(": [Aa]bbreviat\.|: [Aa]bbrev\.|: [Aa]bbrv\.|: [Aa]bbrv|: [Aa]bbr\.|: [Aa]bbr", " abbreviation", clue)
    clue = re.sub("abbr\.|abbrv\.", "abbreviation", clue)
    clue = re.sub("Abbr\.|Abbrv\.", "Abbreviation", clue)
    clue = re.sub("\(anag\.\)|\(anag\)", "(anagram)", clue)
    clue = re.sub("org\.", "organization", clue)
    clue = re.sub("Org\.", "Organization", clue)
    clue = re.sub("Grp\.|Gp\.", "Group", clue)
    clue = re.sub("grp\.|gp\.", "group", clue)
    clue = re.sub(": Sp\.", " (Spanish)", clue)
    clue = re.sub("\(Sp\.\)|Sp\.", "(Spanish)", clue)
    clue = re.sub("Ave\.", "Avenue", clue)
    clue = re.sub("Sch\.", "School", clue)
    clue = re.sub("sch\.", "school", clue)
    clue = re.sub("Agcy\.", "Agency", clue)
    clue = re.sub("agcy\.", "agency", clue)
    clue = re.sub("Co\.", "Company", clue)
    clue = re.sub("co\.", "company", clue)
    clue = re.sub("No\.", "Number", clue)
    clue = re.sub("no\.", "number", clue)
    clue = re.sub(": [Vv]ar\.", " variable", clue)
    clue = re.sub("Subj\.", "Subject", clue)
    clue = re.sub("subj\.", "subject", clue)
    clue = re.sub("Subjs\.", "Subjects", clue)
    clue = re.sub("subjs\.", "subjects", clue)

    theme_clue = re.search("^.+\|[A-Z]{1,}", clue)
    if theme_clue:
        clue = re.sub("\|", " | ", clue)

    if "Partner of" in clue:
        clue = re.sub("Partner of", "", clue)
        clue = clue + " and ___"

    link = re.search("^.+-.+ [Ll]ink$", clue)
    if link:
        no_link = re.search("^.+-.+ ", clue)
        x_y = clue[no_link.start():no_link.end() - 1]
        x_y_lst = x_y.split("-")
        clue = x_y_lst[0] + " ___ " + x_y_lst[1]

    follower = re.search("^.+ [Ff]ollower$", clue)
    if follower:
        no_follower = re.search("^.+ ", clue)
        x = clue[:no_follower.end() - 1]
        clue = x + " ___"

    preceder = re.search("^.+ [Pp]receder$", clue)
    if preceder:
        no_preceder = re.search("^.+ ", clue)
        x = clue[:no_preceder.end() - 1]
        clue = "___ " + x

    if re.search("--[^A-Za-z]|--$", clue):
        clue = re.sub("--", "__", clue)
    if not re.search("_-[A-Za-z]|_-$", clue):
        clue = re.sub("_-", "__", clue)

    clue = re.sub("_{2,}", "___", clue)

    clue = re.sub("\?$", " (wordplay)", clue)

    nonverbal = re.search("\[[^0-9]+,* *[^0-9]*\]", clue)
    if nonverbal:
        clue = re.sub("\[|\]", "", clue)
        clue = clue + " (nonverbal)"

    if clue[:4] == "\"\"\" " and clue[-4:] == " \"\"\"":
        clue = "\"" + clue[4:-4] + "\""
    if clue[:4] == "''' " and clue[-4:] == " '''":
        clue = "'" + clue[4:-4] + "'"
    if clue[:3] == "\"\"\"" and clue[-3:] == "\"\"\"":
        clue = "\"" + clue[3:-3] + "\""
    if clue[:3] == "'''" and clue[-3:] == "'''":
        clue = "'" + clue[3:-3] + "'"

    return clue


def answer_clues(dpr, clues, max_answers, output_strings=False):
    clues = [preprocess_clue_fn(c.rstrip()) for c in clues]
    outputs = dpr.answer_clues_closedbook(clues, max_answers, output_strings=output_strings)
    return outputs

class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """
    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
        device=None,
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index
        self.device = device

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []
        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):
                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q)
                    for q in questions[batch_start : batch_start + bsz]
                ]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).to(self.device)
                q_seg_batch = torch.zeros_like(q_ids_batch).to(self.device)
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))
        query_tensor = torch.cat(query_vectors, dim=0)
        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(self, query_vectors: np.array, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        results = self.index.search_knn(query_vectors, top_docs)
        return results

class FakeRetrieverArgs:
    """Used to surpress the existing argparse inside DPR so we can have our own argparse"""
    def __init__(self):
        self.do_lower_case = False
        self.pretrained_model_cfg = None
        self.encoder_model_type = None
        self.model_file = None
        self.projection_dim = 0
        self.sequence_length = 512
        self.do_fill_lower_case = False
        self.desegment_valid_fill = False
        self.no_cuda = False
        self.local_rank = -1
        self.fp16 = False
        self.fp16_opt_level = "O1"


class DPRForCrossword(object):
    """Closedbook model for Crossword clue answering"""

    def __init__(
        self,
        model_file,
        ctx_file,
        encoded_ctx_file,
        batch_size=6000,
        retrievalmodel=False,
        process_id=0
    ):
        self.retrievalmodel = retrievalmodel  # am I a wikipedia retrieval model or a closed-book model
        args = FakeRetrieverArgs()
        args.model_file = model_file
        args.ctx_file = ctx_file
        args.encoded_ctx_file = encoded_ctx_file
        args.batch_size = batch_size
        self.device = torch.device("cuda:"+str(process_id%torch.cuda.device_count()))

        setup_args_gpu(args)
        print_args(args)

        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)

        tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

        question_encoder = encoder.question_model
        question_encoder = question_encoder.to(self.device)
        question_encoder.eval()

        # load weights from the model file
        model_to_load = get_model_obj(question_encoder)

        prefix_len = len("question_model.")
        question_encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith("question_model.")
        }
        model_to_load.load_state_dict(question_encoder_state)
        vector_size = model_to_load.get_out_size()

        index = DenseFlatIndexer(vector_size, 50000)

        self.retriever = DenseRetriever(
            question_encoder,
            args.batch_size,
            tensorizer,
            index,
            self.device,
        )

        # index all passages
        ctx_files_pattern = args.encoded_ctx_file
        input_paths = glob.glob(ctx_files_pattern)
        self.retriever.index.index_data(input_paths)

        self.all_passages = self.load_passages(args.ctx_file)
        self.fill2id = {}
        for key in self.all_passages.keys():
            self.fill2id[
                "".join(
                    [
                        letter
                        for letter in self.all_passages[key][1].upper()
                        if letter in string.ascii_uppercase
                    ]
                )
            ] = key

        # might as well uppercase and remove non-alphas from the fills before we start to save time later
        if not retrievalmodel:
            temp = {}
            for my_id in self.all_passages.keys():
                temp[my_id] = "".join([c.upper() for c in self.all_passages[my_id][1] if c.upper() in string.ascii_uppercase])
            self.len_all_passages = len(list(self.all_passages.values()))
            self.all_passages = temp
            

    @staticmethod
    def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
        docs = {}
        if ctx_file.endswith(".gz"):
            with gzip.open(ctx_file, "rt") as tsvfile:
                reader = csv.reader(
                    tsvfile,
                    delimiter="\t",
                )
                # file format: doc_id, doc_text, title
                for row in reader:
                    if row[0] != "id":
                        docs[row[0]] = (row[1], row[2])
        else:
            with open(ctx_file) as tsvfile:
                reader = csv.reader(
                    tsvfile,
                    delimiter="\t",
                )
                # file format: doc_id, doc_text, title
                for row in reader:
                    if row[0] != "id":
                        docs[row[0]] = (row[1], row[2])
        return docs

    def answer_clues_closedbook(self, questions, max_answers, output_strings=False):
        # assumes clues are preprocessed
        assert self.retrievalmodel == False
        questions_tensor = self.retriever.generate_question_vectors(questions)

        if max_answers > self.len_all_passages:
            max_answers = self.len_all_passages

        # get top k results
        top_ids_and_scores = self.retriever.get_top_docs(questions_tensor.numpy(), max_answers)
        
        if not output_strings:
            return top_ids_and_scores
        else:
            # get the string forms
            all_answers = []
            all_scores = []
            for ans in top_ids_and_scores: 
                all_answers.append(list(map(self.all_passages.get, ans[0])))
                all_scores.append(ans[1])
            return all_answers, all_scores

    def get_wikipedia_docs(self, questions, max_docs):
        # assumes clues are preprocessed
        assert self.retrievalmodel  
        questions_tensor = self.retriever.generate_question_vectors(questions)

        # get top k results. add 2 in case of duplicates (see below
        top_ids_and_scores = self.retriever.get_top_docs(questions_tensor.numpy(), max_docs + 2)  

        all_paragraphs = []
        for ans in top_ids_and_scores:
            paragraphs = []
            for i in range(len(ans[0])):
                id_ = ans[0][i]
                id_ = id_.replace("wiki:", "")
                mydocument = self.all_passages[id_]
                if mydocument in paragraphs:
                    print("woah, duplicate!!!")
                    continue
                paragraphs.append(mydocument)
            all_paragraphs.append(paragraphs[0:max_docs])
        
        return all_paragraphs
