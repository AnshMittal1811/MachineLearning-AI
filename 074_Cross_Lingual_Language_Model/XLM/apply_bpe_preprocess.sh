#!/bin/bash

pair=en-fr
OUTPATH=/content/XLM/data/processed/XLM_en_fr/50k
FASTBPE=tools/fastBPE/fast

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    $FASTBPE applybpe $OUTPATH/$pair.$lg.$split data/para/$pair.$lg.$split $OUTPATH/codes
    python /content/XLM/preprocess.py $OUTPATH/vocab $OUTPATH/$pair.$lg.$split
  done
done