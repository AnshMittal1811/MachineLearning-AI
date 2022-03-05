import numpy as np
from collections import Counter

class SentenceIndexer:
    """
    create word-level encoded sentences from space-separated inputs
    
    Arguments
    ---------
    max_len : int
        maximum sentence length (settable, & callable after fit())
    max_mode : str
        max sent mode, if not fixed: 'max' or 'std' (for mean + 2*stdev)
    max_vocab : int
        maximum word vocabulary (settable, & callable after fit())
    padding : str
        'pre' or 'post' for zero-padding
    pad : str
        string for PAD elements
    unk : str
        string for OOV elements
    """
    def __init__(self, max_len=None, max_mode='max', max_vocab=None, padding='post', pad='_PAD_', unk='_UNK_'):
        self.max_len = max_len
        self.max_mode = max_mode
        self.max_vocab = max_vocab
        self.padding = padding
        self.pad = pad
        self.unk = unk
        self.word2idx = None
        self.idx2word = None
        self.VOCAB_SIZE = 0
    
    
    def fit(self, sent_toks, verbose=False):
        """
        create the initial vocabularies
        
        Arguments
        ---------
        sent_toks : list
            list of lists of tokenized sentences
        verbose : bool
            flag to print execution log
        """
        
        # word- and char-tokenize
        if verbose:
            print('fit(): splitting...')
        
        # max_sent_len
        if self.max_len is None:
            sent_lens = [len(s) for s in sent_toks]
            if self.max_mode == 'std':
                self.max_len = int(np.round(np.mean(sent_lens) + (2*np.std(sent_lens))))
            else:
                self.max_len = max(sent_lens)
            if verbose:
                print('fit(): max sent len set to', self.max_len)
                
        sent_vocab = list(set([w for s in sent_toks for w in s]))
        if self.max_vocab is None:
            self.max_vocab = len(sent_vocab)+2
        elif len(sent_vocab)+2 < self.max_vocab:
            self.max_vocab = len(sent_vocab)+2
        if verbose:
            print('fit(): max vocab sz set to', self.max_vocab)
            
        # for max
        def gettopn(lst, maxcount):
            flst = [(self.pad, 0)]
            tops = [t[0] for t in Counter(lst).most_common()][:self.max_vocab-2]
            for idx, item in enumerate(tops):
                flst.append((item, idx+1))
            flst.append((self.unk, len(flst))) # because could be smaller than max
            l2i = dict(flst)
            i2l = dict([(v, k) for (k, v) in l2i.items()])
            return l2i, i2l

        if verbose:
            print('fit(): creating conversion dictionaries...')
        # word tokens
        self.word2idx, self.idx2word = gettopn([w for s in sent_toks for w in s], self.max_vocab)
        
        # diagnostics
        if verbose:
            print('fit(): done!')
            
        return
    
    
    def transform(self, sentences, verbose=False):
        """
        tokenize inputs
        
        Arguments
        ---------
        sentences : list
            list of lists of tokenized sentences
        verbose : bool
            flag to print execution log
            
        Returns
        -------
        np.array(sent_idx) : numpy.ndarray
            indexed sentences
        """
        
        sent_toks = sentences[:]
        # test for fit() first
        if self.word2idx is None or self.idx2word is None:
            raise AttributeError('call fit() first!')
            return
            
        if verbose:
            print('transform(): indexing...')
            
        # index sents
        sent_idx = []
        for osent in sent_toks:
            sent = osent[:]
            sent_idx.append(self._index(sent, self.word2idx, self.max_len, self.pad))
        
        if verbose:
            print('transform(): done!')
            
        return np.array(sent_idx)
    
    
    def _index(self, lst, dct, maxlen, padloc):
        """helper function to index single list element"""
        
        while len(lst) < maxlen:
            if padloc == 'pre':
                lst.insert(0, self.pad)
            else:
                lst.append(self.pad)
        
        lst = lst[:maxlen]
        enc = [dct.get(c, dct[self.unk]) for c in lst]
        
        return enc
    
    
    def inverse_transform(self, lst):
        """
        transform indexed sentences to lists of tokens
        
        Arguments
        ---------
        sentences : list
            list of lists (or 2D np.array) of indexed sentences
        verbose : bool
            flag to print execution log
            
        Returns
        -------
        sent_dec : list
            list of strings of tokenized sentences
        """
        
        sent_dec = []
        for sent in lst:
            dec = [self.idx2word.get(c, self.unk) for c in list(np.trim_zeros(sent, 'b'))]
        sent_dec.append(dec)
        
        return sent_dec


class CharacterIndexer:
    """
    create character and word-level encoded sentences from space-separated inputs
    
    Arguments
    ---------
    max_sent_len : int
        maximum allowed sentence length
    max_sent_mode : str
        mode for automatically determining max sent len ('max' or 'std'=mean+2*std)
    max_word_len : int
        maximum allowed word length
    max_word_mode : str
        mode for automatically determining max word len ('max' or 'std'=mean+2*std)
    max_word_vocab : int
        maximum word vocabulary (default: 16000)
    max_char_vocab : int
        maximum character vocab (default: 100)
    padding : str
        'pre' or 'post' for zero-padding
    pad : str
        string for PADDING index
    unk : str
        string for OOV elements
    """
    def __init__(self, max_sent_len=None, max_sent_mode='max', 
                 max_word_len=None, max_word_mode='std', 
                 max_word_vocab=10000, max_char_vocab=100, 
                 padding='post', pad='_PAD_', unk='_UNK_'):
        self.max_sent_len = max_sent_len
        self.max_sent_mode = max_sent_mode
        self.max_word_len = max_word_len
        self.max_word_mode = max_word_mode
        self.max_word_vocab = max_word_vocab
        self.max_char_vocab = max_char_vocab
        self.padding = padding
        self.pad = pad
        self.unk = unk
        self.char2idx = None
        self.word2idx = None
        self.idx2char = None
        self.idx2word = None
    
    
    def fit(self, sent_toks, verbose=False):
        """
        create the initial vocabularies
        
        Arguments
        ---------
        x_data : list
            list of sentence strings separated by spaces
        y_data : list
            list of list of strings indicating word labels
        verbose : bool
            flag to print execution log
        """
        # word- and char-tokenize
        if verbose:
            print('fit(): splitting...')
        word_list = []
        char_toks = []
        char_lens = []
        char_list = []
        for s in sent_toks:
            this_sent = []
            for w in s:
                word_list.append(w)
                this_sent.append(list(w))
                char_lens.append(len(list(w)))
                for c in w:
                    char_list.append(c)
            char_toks.append(this_sent)
        
        # max_sent_len
        if self.max_sent_len is None:
            sent_lens = [len(s) for s in sent_toks]
            if self.max_sent_mode == 'std':
                self.max_sent_len = int(np.round(np.mean(sent_lens) + (2*np.std(sent_lens))))
            else:
                self.max_sent_len = max(sent_lens)
            if verbose:
                print('fit(): max sent len set to', self.max_sent_len)
        
        # max_word_len
        if self.max_word_len is None:
            if self.max_word_mode == 'std':
                self.max_word_len = int(np.round(np.mean(char_lens) + (2*np.std(char_lens))))
            else:
                self.max_word_len = max(char_lens)
            if verbose:
                print('fit(): max word len set to', self.max_word_len)

        # for max
        def gettopn(lst, maxcount):
            flst = [(self.pad, 0)]
            tops = [t[0] for t in Counter(lst).most_common()][:maxcount-2] # <- bc UNK and PAD
            for idx, item in enumerate(tops):
                flst.append((item, idx+1))
            flst.append((self.unk, len(flst))) # because could be smaller than max
            l2i = dict(flst)
            i2l = dict([(v, k) for (k, v) in l2i.items()])
            return l2i, i2l

        if verbose:
            print('fit(): creating conversion dictionaries...')
        # word tokens
        self.word2idx, self.idx2word = gettopn(word_list, self.max_word_vocab)
        self.max_word_vocab = len(self.word2idx.keys())
        # char tokens
        self.char2idx, self.idx2char = gettopn(char_list, self.max_char_vocab)
        self.max_char_vocab = len(self.char2idx.keys())
        
        # diagnostics
        if verbose:
            print('fit(): tru word vocab:', self.max_word_vocab)
            print('fit(): tru char vocab:', self.max_char_vocab)
            print('fit(): done!')
            
        return
    
    
    def transform(self, sent_toks, verbose=False):
        """
        tokenize inputs
        
        Arguments
        ---------
        x_data : list
            list of sentence strings separated by spaces
        y_data : list
            list of list of strings indicating word labels
        verbose : bool
            flag to print execution log
            
        Returns
        -------
        x : y
            zzz
        """
        
        # test for fit() first
        if self.char2idx is None or self.word2idx is None or self.idx2char is None or self.idx2word is None:
            raise AttributeError('call fit() first!')
            return

        sent_idx = []
        char_idx = []
        
        # split sents
        if verbose:
            print('transform(): splitting...')
        char_toks = []
        for s in sent_toks:
            this_sent = []
            for w in s:
                this_sent.append(list(w))
            char_toks.append(this_sent)
            
        if verbose:
            print('transform(): indexing...')
            
        # index sents
        sent_idx = [self._index(sent, self.word2idx, self.max_sent_len, self.pad) for sent in sent_toks]
        
        # index chars
        for sent in char_toks:
            this_sent_enc = []
            for word in sent[:self.max_sent_len]:
                this_sent_enc.append(self._index(word, self.char2idx, self.max_word_len, 'mid'))
            while len(this_sent_enc) < self.max_sent_len:
                this_sent_enc.append([0 for _ in range(self.max_word_len)])
            char_idx.append(this_sent_enc)
        
        if verbose:
            print('transform(): done!')
            
        return np.array(sent_idx), np.array(char_idx)
    

    def _index(self, lst, dct, maxlen, padloc):
        """helper function to index single list element"""
        
        while len(lst) < maxlen:
            if padloc == 'pre':
                lst.insert(0, self.pad)
            elif padloc == 'mid':
                lst.insert(0, self.pad)
                lst.append(self.pad)
            else:
                lst.append(self.pad)
        
        lst = lst[:maxlen]
        enc = [dct.get(c, dct[self.unk]) for c in lst]
        
        return enc
    
    
    def inverse_transform(self, lst):
        """transform indexed sentences to text"""
        
        sent_dec = []
        for sent in lst:
            dec = [self.idx2word.get(c, self.unk) for c in list(np.trim_zeros(sent, 'b'))]
        sent_dec.append(dec)
        
        return sent_dec


class SlotIndexer:
    """index per-token labels e.g. for NER, slot-filling"""
    
    def __init__(self, max_len=10, padding='post', pad='_PAD_', unk='_UNK_'):
        self.max_len = max_len
        self.padding = padding
        self.pad = pad
        self.unk = unk
        self.label2idx = None
        self.idx2label = None
        self.labelsize = None
        
    def fit(self, lst, verbose=False):
        
        lst = lst[:]
        flst = [(self.pad, 0)]
        tops = [t[0] for t in Counter([w for s in lst for w in s]).most_common()]
        for idx, item in enumerate(tops):
            flst.append((item, idx+1))
        l2i = dict(flst)
        i2l = dict([(v, k) for (k, v) in l2i.items()])
        self.label2idx = l2i
        self.idx2label = i2l
        self.labelsize = len(self.label2idx.keys())
        if verbose:
            print('fit(): labels set to size:', self.labelsize)
        
        return
    
    
    def transform(self, data):
        
        lst = data[:]
        sent_enc = []
        for osent in lst:
            sent = osent[:]
            while len(sent) < self.max_len:
                if self.padding == 'pre':
                    sent.insert(0, self.pad)
                else:
                    sent.append(self.pad)
            sent = sent[:self.max_len]
            enc = [[self.label2idx.get(c, 0)] for c in sent]
            sent_enc.append(enc)
        
        return np.array(sent_enc)
    
    
    def inverse_transform(self, data):
        
        lst = data[:]
        sent_dec = []
        for osent in lst:
            sent = osent[:]
            sent = np.squeeze(sent)
            dec = [self.idx2label.get(c, self.unk) for c in list(np.trim_zeros(sent, 'b'))]
            sent_dec.append(dec)
        
        return sent_dec
    

class IntentIndexer:
    """index a list of labels. outputs as a 2D array of 1D indices"""
    
    def __init__(self, unk='_UNK_'):
        self.unk = unk
        self.label2idx = None
        self.idx2label = None
        self.labelsize = None
        
    def fit(self, data, verbose=False):
        
        lst = data[:]
        flst = [('UNK', 0)] + [(t[0], i+1) for i, t in enumerate(Counter(lst).most_common())]
        l2i = dict(flst)
        i2l = dict([(v, k) for (k, v) in l2i.items()])
        self.label2idx = l2i
        self.idx2label = i2l
        self.labelsize = len(self.label2idx.keys())
        if verbose:
            print('fit(): labels set to size:', self.labelsize)
        
        return
    
    
    def transform(self, data):
        
        lst = data[:]
        sent_enc = [[self.label2idx.get(c, 0)] for c in lst]
        
        return np.array(sent_enc)
    
    
    def inverse_transform(self, data):
        
        lst = data[:]
        sent = np.squeeze(lst)
        sent_dec = [self.idx2label.get(c, self.unk) for c in sent]
        
        return sent_dec