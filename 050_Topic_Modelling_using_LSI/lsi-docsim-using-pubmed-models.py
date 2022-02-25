#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#    Copyright (C) Jatin Golani 2018 <jeetu.golani@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import pubmed_parser as parser
import logging
import glob
import argparse
import pdftotext
from collections import Counter, OrderedDict
from gensim.test.utils import get_tmpfile
from gensim import corpora, models, similarities, utils
from gensim.parsing import preprocessing
from operator import itemgetter


class PDFDocument:
    """
    Processes PDF documents using pdftotext.
    self.wordlist holds the individual word tokens.
    """
    def __init__(self, id, filename):
        self.id = id
        self.filename = filename
        self.wordlist = self.extract_text_from_pdf()
        if self.wordlist:
            self.wordlist = self.preprocess()            

    def __str__(self):
        return str(self.filename)

    def preprocess(self):
        """
        Strips away tags, punctuations,whitespaces,numbers,stopwords,words shorter than three chars. 
        """
        CUSTOM_FILTERS = [lambda x: x.lower(),preprocessing.strip_tags,preprocessing.strip_punctuation,\
preprocessing.strip_multiple_whitespaces,preprocessing.strip_numeric,preprocessing.remove_stopwords,\
preprocessing.strip_short]
        return preprocessing.preprocess_string(self.wordlist,CUSTOM_FILTERS)


    def extract_text_from_pdf(self, encoding='utf-8'):
        """
        Extracts the text of a PDF
        """
        with open(self.filename,'rb') as fp:
            pdf = pdftotext.PDF(fp)        
        self.text = "".join(pdf)
        return self.text


class NXMLDocument:
    """
    Processes Pubmed OA nxml documents. pubmed_parser is used to parse the nxml files.
    self.wordlist holds the individual word tokens.
    """
    def __init__(self,id,filename):
        self.id = id
        self.filename = filename
        self.wordlist = self.get_words()
        if self.wordlist:
            self.wordlist = self.preprocess()

    def __str__(self):
        return str(self.filename)

    def get_words(self):
        words = []
        pubmed_dict = parser.parse_pubmed_xml(self.filename)
        text = pubmed_dict['full_title'] + ' ' + pubmed_dict['abstract']
        pubmed_paras_dict = parser.parse_pubmed_paragraph(self.filename)        
        for paras in pubmed_paras_dict:
            text = text + paras['text']
      
        # encodes the unicode string to ascii and replaces the xml entity character references
        # with '?' symbols. decode() then converts this byte string to a regular string for later
        # processing - strip(punctuation) fails otherwise. replace() gets rid of all '?' symbols and      
        # replaces with a space. Later the text is split into words. 
        text = text.encode('ascii','replace').decode('ascii').replace('?',' ')
        return text

    def preprocess(self):
        """
        Strips away tags, punctuations,whitespaces,numbers,stopwords,words shorter than three chars. 
        """
        CUSTOM_FILTERS = [lambda x: x.lower(),preprocessing.strip_tags,preprocessing.strip_punctuation,\
preprocessing.strip_multiple_whitespaces,preprocessing.strip_numeric,preprocessing.remove_stopwords,\
preprocessing.strip_short]
        return preprocessing.preprocess_string(self.wordlist,CUSTOM_FILTERS)



def load_pdf_docs(docpath,min_words=256):
    idx = 1
    flname = docpath + '/**/*.pdf'
    files = glob.glob(flname,recursive=True)
    for file in files:
        pdfdoc = PDFDocument(idx,file)
        if len(pdfdoc.wordlist) >= min_words:
            print('\t{0:03d} -> {1}'.format(pdfdoc.id,pdfdoc.filename))                    
            yield (idx,pdfdoc.filename,pdfdoc.wordlist)
            idx += 1


def load_xml_docs(docpath,min_words=256):
    idx = 1
    path_xml = parser.list_xml_path(docpath)
    for filename in path_xml:
        document = NXMLDocument(idx,filename)
        if len(document.wordlist) >= min_words:
            print('\t{0:03d} -> {1}'.format(document.id,document.filename))                    
            yield (idx,document.filename,document.wordlist)
            idx += 1


def load_docs(docpath,doctype,min_words=256):
    if doctype.lower() == 'nxml':
        return load_xml_docs(docpath,min_words)
    elif doctype.lower() == 'pdf':
        return load_pdf_docs(docpath,min_words)


def get_corpus(path,doctype,dictionary,catalog,min_words=256):
    for idx,filename,wordlist in load_docs(path,doctype,min_words):
        catalog.update({idx:filename})
        yield dictionary.doc2bow(wordlist)


def main(modelpath,doctype,docpath):
    num_topics = 500
    min_words = 256
    threshold = 0.30
    catalog = {}    

    # load pubmed pre-trained models
    print('Loading pubmed pre-trained model')
    path = modelpath + '/pubmed-xml.dict'
    dictionary = corpora.Dictionary.load(path)
    print("\ndictionary = ",dictionary)
    path = modelpath + '/pubmed-xml-tfidf-model.tfidf'
    model_tfidf = models.TfidfModel.load(path)
    path = modelpath + '/pubmed-xml-lsi-model.lsi'
    lsi = models.LsiModel.load(path)

    # build a new corpus against which we will be using the pre-trained lsi model
    corpus = get_corpus(docpath,doctype,dictionary,catalog,min_words)
    # transform new corpus using the pre-trained tfidf model
    corpus_tfidf = model_tfidf[corpus]
    # create an index with the pre-trained lsi model used against the above tfidf corpus transformation.
    index_tmpfile = get_tmpfile("index")
    index = similarities.Similarity(index_tmpfile,lsi[corpus_tfidf],num_features=num_topics)    
    

    # Generate similarities for each document against the other documents.
    # Similarities are not sorted against the whole corpus but.
    # Here for each document, the most similar other document is shown followed by the next in
    # descending order. 
    # This method is low on memory utilisation.
    id = 1
    for idx,filename,wordlist in load_docs(docpath,doctype,min_words):
        pairs = OrderedDict()
        pub_id = id
        vec_bow = dictionary.doc2bow(wordlist)
        vec_lsi = lsi[vec_bow]
        sims = index[vec_lsi]
        sim_list = list(enumerate(sims,1))
        sim_list.sort(key=lambda x: x[1],reverse=True)
        for idx,similarity in sim_list:
            sim_id = idx
            if sim_id != pub_id and similarity > threshold:
                if (sim_id,pub_id) not in pairs.keys():
                    pairs.update({(pub_id,sim_id): similarity})
        pairs = OrderedDict(sorted(pairs.items(),key=itemgetter(1),reverse=True))
        for pair, similarity in pairs.items():
            print('Doc: {0} is {2:0.3f} similar to Doc: {1}'.format(catalog[pair[0]],catalog[pair[1]],similarity))
        id += 1


if __name__ == "__main__":
    cmdparser = argparse.ArgumentParser()
    cmdparser.add_argument('modelpath',help="path at which pubmed pre-trained models are located")
    cmdparser.add_argument('doctype',choices=['pdf','nxml'],help="type of documents to process (pdf or nxml)")
    cmdparser.add_argument('docpath',help="path to documents to be processed")    
    args = cmdparser.parse_args()
    main(args.modelpath,args.doctype,args.docpath)
