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

parser = argparse.ArgumentParser(description="DeppSMILES to SMILES")
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

print("DeepSMILES version: %s" % deepsmiles.__version__)
converter = deepsmiles.Converter(rings=True, branches=True)
print(converter) # record the options used

f = open(args.output,'w')

with open(args.input,"r") as fp:
	for i,line in enumerate(fp):
		id =(line.strip().split("\t")[1])
		smiles = (line.strip().split("\t")[0])
		
		try:
		    decoded = converter.decode(smiles)
		    f.write(decoded+"\t\t"+id+"\n")
		except deepsmiles.DecodeError as e:
		    decoded = None
		    f.write(smiles+"DecodeError! Error message was '%s'" % e.message+"\n")
		except IndexError as e2:
		    decoded = None
		    f.write(smiles+"DecodeError! Error message was Indexerror"+"\n")
		#if decoded:
		    #print("Decoded: %s" % decoded)

f.close()
