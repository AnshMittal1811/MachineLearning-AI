# anky.py

# usage : python anky.py --lgs $lgs --srcFilePath $srcFilePath --targetFilesPath $targetFilesPath
# Separates a file downloaded under http://www.manythings.org/anki/ into two files (see anky.sh)

import argparse
import os

def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="http://www.manythings.org/anki/")

    # main parameters
    parser.add_argument("--srcFilePath", type=str, default="", help="Path of the file containing the data")
    parser.add_argument("--targetFilesPath", type=str, default="", help="Path to the folder in which to put the results")
    parser.add_argument("--lgs", type=str, default="", help="Languages concerned in alphabetical order and separated by a comma : e.g. 'de,en'.")
    return parser

def opus(srcFile, targetFile1, targetFile2):
  with open(srcFile, 'r', encoding="utf-8") as srcTxtFile :
    with open(targetFile1, 'w', encoding="utf-8") as targetTxtFile1 :
      with open(targetFile2, 'w', encoding="utf-8") as targetTxtFile2 :
        for line in srcTxtFile.readlines():
          samples = line.split("CC-BY")[0].strip().split("\t")
          targetTxtFile1.writelines(samples[0]+"\n")
          targetTxtFile2.writelines(samples[1]+"\n")

def main(params):
  targetFile1 = params.targetFilesPath+"/"+params.lgs[0]+"-"+params.lgs[1]+"."+params.lgs[0]+".txt" 
  targetFile2 = params.targetFilesPath+"/"+params.lgs[0]+"-"+params.lgs[1]+"."+params.lgs[1]+".txt"
  if params.lgs[0] == "en" :
    opus(srcFile = params.srcFilePath, targetFile1 = targetFile1, targetFile2 = targetFile2)
  else :
    opus(srcFile = params.srcFilePath, targetFile1 = targetFile2, targetFile2 = targetFile1)

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    
    # check parameters
    assert os.path.isfile(params.srcFilePath), "Source file not found"
    if not os.path.exists(params.targetFilesPath):
      os.makedirs(params.targetFilesPath)
    lgs = params.lgs.split(",")
    assert len(lgs)==2 and lgs[1] > lgs[0], "It is mandatory to specify two languages in alphabetical order."
    assert "en" in lgs
    params.lgs = lgs

    # run experiment
    main(params)

