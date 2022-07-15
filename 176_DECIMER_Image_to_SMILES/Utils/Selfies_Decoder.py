'''
 * This Software is under the MIT License
 * Refer to LICENSE or https://opensource.org/licenses/MIT for more information
 * Written by ©Kohulan Rajan 2019
'''
#Original source: https://github.com/nextmovesoftware/deepsmiles
#Deepsmiles decoding implementation for my work 

import numpy as np
import deepsmiles
import argparse
from selfies import encoder, decoder, selfies_alphabet  

parser = argparse.ArgumentParser(description="SELFIES to SMILES")
# Input Arguments
parser.add_argument(
	'--input',
	help = 'Enter the input filename',
	required = True
	)
parser.add_argument(
	'--output',
	help = 'Enter the output filename as desired',
	required = True
)
args= parser.parse_args()

print("Conversion Started!")

f = open(args.output,'w')

with open(args.input,"r") as fp:
	for i,line in enumerate(fp):
		id =(line.strip().split("\t")[1])
		smiles = (line.strip().split("\t")[0])
		
		try:
		    decoded = decoder(smiles)
		    f.write(decoded+"\t"+id+"\n")
		except Exception as e:
		    decoded = None
		    f.write(smiles+"DecodeError! Error! \n")
		except IndexError as e2:
		    decoded = None
		    f.write(smiles+"DecodeError! Error message was Indexerror"+"\n")
		#if decoded:
		    #print("Decoded: %s" % decoded)

f.close()
