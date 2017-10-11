# -*- encoding: utf-8 -*-

#############################################################################################################
				#FilterDic_0.0.F.py
#############################################################################################################

"""DESCRIPTION

Author= LuisFernando_Lucia

Version= 0.0

Date: 11/10/17

Description: This code filters the dictionary that contains diseases from the files and the NIH, from the all the sentences of the abstracts

Parameters: None

Output:
 1) Dictionary filtered

Environment:
  a) DicFenosFile: file with dictionary of diseases
  b) DicFenos: list with diseases
  c) SentencesFile: initial file to get associations
  d) Sentences: list of sentences
  e) FilterDicFile: file with the dictionary of diseases filtered
  f) Feno: every disease from the variable DicFenos

Usage: python3 FilterDic_0.0.FL.py

"""

import re

with open('/export/storage/users/luciarn/BI_project/Mineria/Data/DiccionarioFinal_0.1.F.txt','r') as DicFenosFile, open('/export/storage/users/luciarn/BI_project/Mineria/Data/PubMedID-Sentence.txt','r') as SentencesFile:
    DicFenos=DicFenosFile.readlines()
    Sentences=SentencesFile.read()
print('Filtering dictionary...')
with open('/export/storage/users/luciarn/BI_project/Mineria/Data/DiccionarioFinal_1.0.FL.txt','w') as FilterDicFile:    
    for Feno in DicFenos:
        if re.search(re.compile('\W'+Feno.rstrip()+'\W', re.IGNORECASE), Sentences):
           FilterDicFile.write(Feno)
print('Filtering... Done')
