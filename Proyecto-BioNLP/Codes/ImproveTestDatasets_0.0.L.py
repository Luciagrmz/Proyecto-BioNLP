# -*- encoding: utf-8 -*-

#############################################################################################################
				#ImproveTestDatasets_0.0.L.py
#############################################################################################################

"""DESCRIPTION

Author= Lucia

Version= 1.0

Date: 11/10/17

Description: This code improves the initial training sets by improve the tagging of diseases and/or rs numbers

Parameters: None

Output:
 1) Training sentences with lemmas and tagging of diseases and rs numbers in a txt file
 2) Training sentences with lemmas, postgas and tagging of diseases and rs numbers in a txt file

Environment:
  a) SentencesFile: initial file with training sentences
  b) DicFenosFile: file with dictionary of diseases
  c) Sentences: list with the training sentences
  d) DicFenos: list with diseases
  e) LemmaRsDisFile: output file with training sentences that contains lemmas and tagging of diseases and rs numbers
  f) Tag: contains the list of tags for each sentence
  g) Sent: every sentence from the variable Sentences
  h) Feno: every disease in the DicFenos
  i) Search: contains a list with the matches found in the sentence for the disease
  j) LemmaPostagRsDisFile: output file with training sentences that contains lemmas, postags and tagging of diseases and rs numbers
  k) i: index of Sentences
  l) Sentsep: list with the line of text and their corresponding postags
  m) Lines: list of words from every sentence
  o) Linesubs: list of words from every sentence, but with the phenotypes that have more than one word joined with a '-'
  p) LenFeno: number of words that contains the phenotype
  q) LemmaPostagRsDisFile: output file with training sentences that contains lemmas, postags and tagging of diseases and rs numbers

Usage: python3 ImproveDatasets_1.0.FL.py

"""

import re
from sys import argv


with open('/export/storage/users/luciarn/BI_project/Mineria/Data/DiccionarioFinal_1.0.F.txt','r') as DicFenosFile:
    DicFenos=DicFenosFile.readlines()

#Dictionary with the start of some keywords
DicKeyws={'associat\w*':'KEYWD','relat\w*':'KEYWD','affect\w*':'KEYWD','susceptib\w*':'KEYWD','risk':'KEYWD','correlat\w*':'KEYWD','link\w*':'KEYWD','marker':'KEYWD','evidence\w*':'KEYWD','influenc\w*':'KEYWD','no ':'NEGKEYWD','not ':'NEGKEYWD','non ':'NEGKEYWD','don\S*\w*':'NEGKEYWD','negative':'NEGKEYWD','doesn\S*\w*':'NEGKEYWD','didn\S*\w*':'NEGKEYWD','wasn\S*\w*':'NEGKEYWD','couldn\S*\w*':'NEGKEYWD','isn\S*\w*':'NEGKEYWD','aren\S\w*':'NEGKEYWD'}


with open(argv[1],'r') as SentencesFile:
    Sentences=SentencesFile.readlines()

print ('Preprocessing started ...')
with open(argv[2],'w') as LemmaPostagRsDisFile:
    for Sent in Sentences:
        Sent=Sent.rstrip()
        for Feno in DicFenos:
            Feno=Feno.rstrip()
            if re.search(re.compile('\W'+Feno+'\W', re.IGNORECASE),Sent):
                Sentsep=Sent.split()
                Lines=Sentsep[:int(len(Sentsep)/2)]
                Tag=Sentsep[int(len(Sentsep)/2):]
                Linesubs=re.sub(re.compile('(\W)('+Feno+')(\W)', re.IGNORECASE),r'\g<1>'+re.sub(' ','-',Feno)+r'\g<3>', ' '.join(Lines))
                LenFeno=len(Feno.split())
                k=0
                for j,word in enumerate(Linesubs.split()):
                    if (re.sub(' ','-',Feno)).lower() == word.lower():
                        Linesubs+=' DIS'
                        k+=(LenFeno-1)
                    else:
                        Linesubs+=(' '+Tag[j+k])
                Sent=Linesubs
        for keyw in DicKeyws:
            Sentsep=Sent.split()
        Lines=Sentsep[:int(len(Sentsep)/2)]
        Tag=Sentsep[int(len(Sentsep)/2):]
            if re.search(re.compile(keyw, re.IGNORECASE),Sent):
                for j,word in enumerate(Linessubs.split()):
                    if re.search(re.compile('\\b'+keyw.rstrip(), re.IGNORECASE),word):
                        Tag[j]=DicKeyws[keyw]
        LemmaPosRsDisKwsFile.write(' '.join(Lines)+ ' '+ ' '.join(Tag)+ ' \n')
print ('Preprocessing done...')

