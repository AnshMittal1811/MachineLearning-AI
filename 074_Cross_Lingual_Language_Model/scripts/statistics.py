# usage : python statistics.py --logFilePath $logFilePath ...
# Display the statistics on the evolution of a given training, and therefore the path to the log file is specified in parameter.
# You have to pass the same values for the parameters as during training.

import argparse
import os
import copy

from XLM.src.utils import bool_flag
from XLM.src.data.loader import check_data_params

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Statistics display")

    # main parameters
    parser.add_argument("--logFilePath", type=str, default="", help="Path to log file")
    parser.add_argument("--lgs", type=str, default="", help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    
    # training steps
    parser.add_argument("--clm_steps", type=str, default="", help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="", help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--mt_steps", type=str, default="", help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="", help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="", help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="", help="Parallel classification steps")
    
    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False, help="Evaluate BLEU score during MT training")
    
    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=True, help="Only use an encoder")
    
    return parser


def get_error_value(stat, line, objectif, step_value):
    if objectif in line :
        a = objectif+"-"+step_value+":"
        try :
            a = line.split(a)[1].strip()
        except IndexError as ie:
            return
    
        if "||" in a :
            a = a.split("||")[0].strip()
        else :
            a = a.split(" ")[0].strip()
      
        try :
            stat[objectif]
            try :
                stat[objectif][step_value]
            except :
                 stat[objectif][step_value] = []
        except :
            stat[objectif] = {}
            stat[objectif][step_value] = []
        
        stat[objectif][step_value].append(a) 

def get_steps_value(objectif, params)  :
  # faire ca avec un switch case
  if objectif in ["CLM","MLM"] :
    langs = []
    steps = params.mlm_steps if objectif=='MLM' else params.clm_steps
    for (l1, l2) in steps :
        if (l1 is None) and (not (l2 is None)) :
            langs.append(l2)
        elif (not (l1 is None)) and (l2 is None) :
            langs.append(l1)
        elif (not (l1 is None)) and (not (l2 is None)) :
            langs.append(l1+"-"+l2)
    # removes duplicates
    langs = list(set(langs))
    return langs

  elif objectif in ["PC", "MT", "BT"] :
    steps = params.pc_steps if objectif=='PC' else (params.mt_steps if objectif=='MT' else params.bt_steps)
    langs = []
    for liste in steps:
        s = liste[0]
        for l in liste[1:] :
            s = s + "-" + l
        langs.append(s)
    return langs

  elif objectif == "AE" :
    return params.ae_steps
      
  return []

def read_log(logFilePath, params, error = False, evaluation = False):
  
  if (not evaluation) and (not error) :
    return

  stat = {}
  if error :
    stat['error'] = {}
  if evaluation :
    stat['evaluation'] = {}

  with open(logFilePath, 'r', encoding="utf-8") as logFile :
    
    for line in logFile.readlines():
    
      if error :
        if "model LR" in line :
          
          for objectif in ["CLM", "MLM", "PC", "AE", "MT", "BT"] :
            for step_value in get_steps_value(objectif = objectif, params = params) :
              get_error_value(stat['error'], line, objectif, step_value)

      if evaluation :
        if (("valid" in line) or ("test" in line)) and ("->" in line) :
          a = line.split("->")[1].strip()
          # todo : put it where ?  
  return stat
          

def plot(stat):
   print(stat)

def main(params):
  if not params.meta_learning :
     stat = read_log(logFilePath = params.logFilePath, params = params, error = True, evaluation = False)  
  else :
     stat = {}
     for k, v in params.meta_params.items() :
        stat[k] = read_log(logFilePath = params.logFilePath, params = v, error = True, evaluation = False)
        
        
  plot(stat)

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    
    # check parameters
    assert os.path.isfile(params.logFilePath), "log file path file not found"
    
    
    # Check to see if we need to do metalearning.
    meta_lgs = params.lgs.split("|")
    params.meta_learning = False
    params.n_task = len(meta_lgs)
    
    if params.n_task != 1 :
        
        params.meta_learning = True
    
        params.meta_params = {}
        
        meta_tmp = ["" for _ in range(params.n_task)]

        meta_clm = params.clm_steps.split("|")
        if meta_clm[0] == "" :
            meta_clm = meta_tmp

        meta_mlm = params.mlm_steps.split("|")
        if meta_mlm[0] == "" :
            meta_mlm = meta_tmp

        meta_pc = params.pc_steps.split("|")
        if meta_pc[0] == "" :
            meta_pc = meta_tmp

        meta_mt = params.mt_steps.split("|")
        if meta_mt[0] == "" :
            meta_mt = meta_tmp

        meta_ae = params.ae_steps.split("|")
        if meta_ae[0] == "" :
            meta_ae = meta_tmp

        meta_bt = params.bt_steps.split("|")
        if meta_bt[0] == "" :
            meta_bt = meta_tmp

        langs, clms, mlms, pcs, mts, aes, bts = [], [], [], [], [], [], []


        # check parameters
        for meta_objectif in [meta_clm, meta_mlm, meta_pc, meta_mt, meta_ae, meta_bt] :
            assert len(meta_objectif) == params.n_task

        for lgs, clm, mlm, pc, mt, ae, bt in zip(meta_lgs, meta_clm, meta_mlm, meta_pc, meta_mt, meta_ae, meta_bt) :

            params.lgs = lgs 
            params.clm_steps = clm 
            params.mlm_steps = mlm 
            params.pc_steps = pc 
            params.mt_steps = mt 
            params.ae_steps = ae    
            params.bt_steps = bt 

            check_data_params(params, check_only_objectifs = True)
            
            params.meta_params[lgs] = copy.deepcopy(params)

            langs.append(params.langs)
            clms.append(params.clm_steps)
            mlms.append(params.mlm_steps)
            pcs.append(params.pc_steps)
            mts.append(params.mt_steps)
            aes.append(params.ae_steps)
            bts.append(params.bt_steps)

        if params.meta_learning :
            params.langs = langs
            params.clm_steps = clms
            params.mlm_steps = mlms
            params.pc_steps = pcs
            params.mt_steps = mts
            params.ae_steps = aes
            params.bt_steps = bts

        params.lgs = meta_lgs
    
    else :
        check_data_params(params, check_only_objectifs = True)
    
    print(params)
     
    # run experiment
    main(params)

