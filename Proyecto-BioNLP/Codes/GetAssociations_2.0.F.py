# -*- encoding: utf-8 -*-

#############################################################################################################
				#Code_GetAsociations_2.0.F.py
#############################################################################################################

"""DESCRIPTION

Author= LuisFernando

Version= 2.0

Date: 11/10/17

Description: This code searchs diseases and rs numbers through the sentences to establish relationships

Parameters: Path to the file to get the associations and output file for the chart

Output:
 1) Chart with associations between rs numbers and diseases

Environment:
  a) DicFenosFile: file with dictionary of diseases
  b) DicFenos: list with diseases
  c) SentencesFile: initial file to get associations
  d) Sentences: list of sentences
  e) Sent: every sentence from the variable Sentences
  f) Feno: every disease from the variable DicFenos
  g) SearchRs: list with rs numbers found
  h) AssociationsFile: ouputfile

Usage: python3 Code_GetAsociations_2.0.F.py [input file] [output file]

"""

import re 
from sys import argv

with open('/export/storage/users/luciarn/BI_project/Mineria/Data/DiccionarioFinal_1.0.F.txt','r') as DicFenosFile, open(argv[1],'r') as SentencesFile:
    DicFenos=DicFenosFile.readlines()
    Sentences=SentencesFile.readlines()
print ('Getting Associations...')
with open(argv[2],'w') as AsociationsFile:
    for Sent in Sentences:
        for Feno in DicFenos:
            if re.search(re.compile('\W'+Feno.rstrip()+'\W', re.IGNORECASE),Sent):
                SearchRs=re.findall(re.compile('rs\s?\d{2,}', re.IGNORECASE), Sent)
                if len(SearchRs):
                    for Rs in set(SearchRs):
                        print(Rs + '\t' + Feno.rstrip() + '\n')
                        AsociationsFile.write(Rs + '\t' + Feno.rstrip() + '\n')
print('Associations... Done')
