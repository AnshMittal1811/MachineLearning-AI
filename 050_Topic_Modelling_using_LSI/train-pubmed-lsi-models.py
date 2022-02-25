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


import os
import pubmed_parser as parser
import logging
import argparse
from collections import Counter, OrderedDict
from gensim.test.utils import get_tmpfile
from gensim import corpora, models, similarities, utils
from gensim.parsing import preprocessing
from operator import itemgetter

class Document(object):
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


def load_xml_docs(docpath,min_words=256):
    """
    Generator function that goes through all nxml documents in the path and returns
    a tuple of document index, filename and word list.
    """
    idx = 1
    path_xml = parser.list_xml_path(docpath)
    for filename in path_xml:
        document = Document(idx,filename)
        if len(document.wordlist) >= min_words:
            print('\t{0:03d} -> {1}'.format(document.id,document.filename))                    
            yield (idx,document.filename,document.wordlist)
            idx += 1


def get_dictionary(docpath,min_words=256):
    return corpora.Dictionary(wordlist for idx,filename,wordlist in load_xml_docs(docpath,min_words))    

def get_corpus(docpath,dictionary,catalog,min_words=256):
    for idx,filename,wordlist in load_xml_docs(docpath,min_words):
        catalog.update({idx:filename})
        yield dictionary.doc2bow(wordlist)


def main(docpath,sortsims):
    num_topics = 500
    min_words = 256
    threshold = 0.30
    catalog = {}

    logging.basicConfig(filename='./pubmed-nxml-lsa.log',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
       
    if (os.path.exists("./pubmed-xml.dict")):
        print('Loading previously saved dictionary')
        dictionary = corpora.Dictionary.load('./pubmed-xml.dict')
    else:
        print('Loading documents and creating dictionary')
        dictionary = get_dictionary(docpath,min_words)    
        dictionary.filter_extremes(no_below=30,no_above=0.5,keep_n=100000)
        dictionary.compactify()
        dictionary.save('./pubmed-xml.dict')
        dictionary.save_as_text('./pubmed-xml-dict.txt')
    print("\ndictionary = ",dictionary)

    corpus = get_corpus(docpath,dictionary,catalog,min_words)
    corpora.MmCorpus.serialize('./pubmed-xml-corpus.mm', corpus)
    stored_corpus = corpora.MmCorpus('./pubmed-xml-corpus.mm')

    model_tfidf = models.TfidfModel(stored_corpus)
    model_tfidf.save('./pubmed-xml-tfidf-model.tfidf')

    corpus_tfidf = model_tfidf[stored_corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    lsi.save('./pubmed-xml-lsi-model.lsi')
    print("\n---------------------------\n")
    print('\n\nTopics:\n',lsi.print_topics())
    print("\n---------------------------\n")
    corpora.MmCorpus.serialize('./pubmed-xml-lsi-corpus_tfidf.mm',lsi[corpus_tfidf])
    

    # The below sections generate similarities between the pubmed documents. 
    # Given the huge size of the pubmed corpus, generating similarities takes considerable time and memory
    
    # This option generates similarities sorted against the whole corpus as a whole based on highest    
    # similarity i.e. document pairs with the highest to lowest similarity in the entire corpus are
    # shown. 
    # This method requires the most memory considering the almost 1M documents in the 
    # corpus. 16GB RAM was not good enough when I ran this on approx 1M pubmed docs. You have been
    # warned :) 
    if sortsims == 'full':
        # creates a similarity index for the entire corpus.
        index_tmpfile = get_tmpfile("index")
        index = similarities.Similarity(index_tmpfile,lsi[corpus_tfidf],num_features=num_topics)
        pairs = OrderedDict()
        id = 1
        # use above index to create similarities for each document
        for idx,filename,wordlist in load_xml_docs(docpath,min_words):
            catalog.update({idx:filename})
            pub_id = id
            vec_bow = dictionary.doc2bow(wordlist)
            vec_lsi = lsi[vec_bow]
            sims = index[vec_lsi]
            sim_list = list(enumerate(sims,1))
            sim_list.sort(key=lambda x: x[1],reverse=True)
            # updates the pairs OrderedDict with document pairs as the key and their similarity 
            # as the dictionary value. Since pairs holds this mapping for each document over
            # all others and keeps this all in memory, this is extremely memory intensive.            
            for idx,similarity in sim_list:
                sim_id = idx
                if sim_id != pub_id and similarity > threshold:
                    if (sim_id,pub_id) not in pairs.keys():
                        pairs.update({(pub_id,sim_id): similarity})
            id += 1
        
        pairs = OrderedDict(sorted(pairs.items(),key=itemgetter(1),reverse=True))
        for pair, similarity in pairs.items():
            print('Doc: {0} is {2:0.3f} similar to Doc {1}'.format(catalog[pair[0]],catalog[pair[1]],similarity))
    
  
    # This option generates similarities for each document against the other documents.
    # Similarities are not sorted against the whole corpus unlike the 'full' option.
    # Here for each document, the most similar other document is shown followed by the next in
    # descending order. 
    # This method does not require as much memory. This method is also implemented in the
    # lsi-docsim-using-pubmed-models.py file so you could also run that on your models.
    if sortsims == 'perdoc':
        index_tmpfile = get_tmpfile("index")
        index = similarities.Similarity(index_tmpfile,lsi[corpus_tfidf],num_features=num_topics)
        id = 1
        for idx,filename,wordlist in load_xml_docs(docpath,min_words):
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
    cmdparser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    cmdparser.add_argument('docpath',help="path at which pubmed nxml documents are located")
    cmdparser.add_argument('--sortsims',choices=['full','perdoc'],help="ouput sorted similarities of pubmed docs.\nfull - shows the most similar docs across the full collection (high memory).\nperdoc - shows how similar is each doc to others in the collection. (less memory).")
    args = cmdparser.parse_args()
    main(args.docpath,args.sortsims)
