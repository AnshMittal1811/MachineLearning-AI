import numpy as np
from selfies import encoder, decoder, selfies_alphabet  
from rdkit import Chem

print("Smiles to Selfies")


f = open('SELFIES_file.txt','w')



with open("SMILES_file.txt","r") as fp:
	for i,line in enumerate(fp):
		chembl =(line.strip().split("\t\t")[0])
		smiles = (line.strip().split("\t\t")[1])

		try:
			can_smiles=Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
			encoded = encoder(can_smiles.upper())
			#print(encoded)
			#print("Encoded: %s" % encoded)
			f.write(chembl+","+encoded+"\n")
		except Exception as e:
			print(chembl)

		#if decoded:
		    #print("Decoded: %s" % decoded)

f.close()
