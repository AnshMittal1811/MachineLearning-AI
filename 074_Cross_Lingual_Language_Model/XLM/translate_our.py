import os
import io
import sys
import argparse
import torch

from src.utils import AttrDict
from src.utils import bool_flag, initialize_exp
from src.data.dictionary import Dictionary
from src.model.transformer import TransformerModel

class Translate():
    def __init__(self, model_path, tgt_lang, src_lang,dump_path = "./dumped/", exp_name="translate", exp_id="test", batch_size=32):
        
        # parse parameters
        parser = argparse.ArgumentParser(description="Translate sentences")
        
        # main parameters
        parser.add_argument("--dump_path", type=str, default=dump_path, help="Experiment dump path")
        parser.add_argument("--exp_name", type=str, default=exp_name, help="Experiment name")
        parser.add_argument("--exp_id", type=str, default=exp_id, help="Experiment ID")
        parser.add_argument("--batch_size", type=int, default=batch_size, help="Number of sentences per batch")
        # model / output paths
        parser.add_argument("--model_path", type=str, default=model_path, help="Model path")
        # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
        # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")
        # source language / target language
        parser.add_argument("--src_lang", type=str, default=src_lang, help="Source language")
        parser.add_argument("--tgt_lang", type=str, default=tgt_lang, help="Target language")
        parser.add_argument('-d', "--text", type=str, default="", nargs='+', help="Text to be translated")

        params = parser.parse_args()
        assert params.src_lang != '' and params.tgt_lang != '' and params.src_lang != params.tgt_lang

        # initialize the experiment
        logger = initialize_exp(params)
        
        # On a pas de GPU
        #reloaded = torch.load(params.model_path)
        reloaded = torch.load(params.model_path, map_location=torch.device('cpu'))
        model_params = AttrDict(reloaded['params'])
        self.supported_languages = model_params.lang2id.keys() 
        logger.info("Supported languages: %s" % ", ".join(self.supported_languages))

        # update dictionary parameters
        for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
            try :    
                setattr(params, name, getattr(model_params, name))    
            except AttributeError :
                key = list(model_params.meta_params.keys())[0]
                attr = getattr(model_params.meta_params[key], name)
                setattr(params, name, attr)
                setattr(model_params, name, attr)
                 
        # build dictionary / build encoder / build decoder / reload weights
        self.dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
        #self.encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
        self.encoder = TransformerModel(model_params, self.dico, is_encoder=True, with_output=True).eval()
        #self.decoder = TransformerModel(model_params, dico, is_encoder=False, with_output=True).cuda().eval()
        self.decoder = TransformerModel(model_params, self.dico, is_encoder=False, with_output=True).eval()
        self.encoder.load_state_dict(reloaded['encoder'])
        self.decoder.load_state_dict(reloaded['decoder'])
        params.src_id = model_params.lang2id[params.src_lang]
        params.tgt_id = model_params.lang2id[params.tgt_lang]
        self.model_params = model_params
        self.params = params
  
    def translate(self, src_sent=[]):
        flag = False
        if type(src_sent) == str :
            src_sent = [src_sent]
            flag = True
        tgt_sent = []
        for i in range(0, len(src_sent), self.params.batch_size):
            # prepare batch
            word_ids = [torch.LongTensor([self.dico.index(w) for w in s.strip().split()])
                        for s in src_sent[i:i + self.params.batch_size]]
            lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
            batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.params.pad_index)
            batch[0] = self.params.eos_index
            for j, s in enumerate(word_ids):
                if lengths[j] > 2:  # if sentence not empty
                    batch[1:lengths[j] - 1, j].copy_(s)
                batch[lengths[j] - 1, j] = self.params.eos_index
            langs = batch.clone().fill_(self.params.src_id)

            # encode source batch and translate it
            #encoded = self.encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
            encoded = self.encoder('fwd', x=batch, lengths=lengths, langs=langs, causal=False)
            encoded = encoded.transpose(0, 1)
            #decoded, dec_lengths = self.decoder.generate(encoded, lengths.cuda(), self.params.tgt_id, max_len=int(1.5 * lengths.max().item() + 10))
            decoded, dec_lengths = self.decoder.generate(encoded, lengths, self.params.tgt_id, max_len=int(1.5 * lengths.max().item() + 10))

            # convert sentences to words
            for j in range(decoded.size(1)):

                # remove delimiters
                sent = decoded[:, j]
                delimiters = (sent == self.params.eos_index).nonzero().view(-1)
                assert len(delimiters) >= 1 and delimiters[0].item() == 0
                sent = sent[1:] if len(delimiters) == 1 else sent[1:delimiters[1]]

                # output translation
                source = src_sent[i + j].strip()
                target = " ".join([self.dico[sent[k].item()] for k in range(len(sent))])
                sys.stderr.write("%i / %i: %s -> %s\n" % (i + j, len(src_sent), source, target))
                tgt_sent.append(target )
        
        if flag :
            return tgt_sent[0]
        return tgt_sent

# Ici on permet d'appeler le modele en faisant :

#python translate_our.py --exp_name translate \
#     --src_lang en --tgt_lang fr \
#     --model_path trained_model.pth --output_path output

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/", help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of sentences per batch")

    # model / output paths
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument("--output_path", type=str, default="", help="Output path")

    # parser.add_argument("--max_vocab", type=int, default=-1, help="Maximum vocabulary size (-1 to disable)")
    # parser.add_argument("--min_count", type=int, default=0, help="Minimum vocabulary count")

    # source language / target language
    parser.add_argument("--src_lang", type=str, default="", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="", help="Target language")

    parser.add_argument('-d', "--text", type=str, default="", nargs='+', help="Text to be translated")
    
    return parser

def main(params):
    # initialize the experiment
    logger = initialize_exp(params)

    # generate parser / parse parameters
    #parser = get_parser()
    #params = parser.parse_args()
    trainer = Translate(model_path = params.model_path , tgt_lang = params.tgt_lang, src_lang = params.src_lang,
                    dump_path = params.dump_path, exp_name=params.exp_name, 
                    exp_id=params.exp_id, batch_size=params.batch_size)
    
    print(trainer.translate(params.text.split('[SEP]')))

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    
    if isinstance(params.text, list) :
        params.text = ' '.join(params.text)
    if params.text.startswith("\""):
        params.text = params.text[1:] 
    if params.text.endswith("\""):
        params.text = params.text[:len(params.text)-1]
    
    # translate
    with torch.no_grad():
        main(params)


