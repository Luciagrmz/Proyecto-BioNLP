# -*- encoding: utf-8 -*-

#############################################################################################################
				#ImproveDatasets_0.0.FL.py
#############################################################################################################

"""DESCRIPTION

Author= Lucia_LuisFernando

Version= 1.0

Date: 11/10/17

Description: This code improves the initial training sets by improve the tagging of diseases and/or rs numbers

Parameters: 
 1) Input file with sentences that contains lemmas, diseases and rs numbers (initial)
 2) Output file to write the improved training sentences with lemmas and tagging of diseases and rs numbers
 3) Input file with sentences that contains lemmas, postags, diseases and rs numbers (initial)
 2) Output file to write the improved training sentences with lemmas, postags, diseases and rs numbers

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

Usage: python3 ImproveDatasets_1.0.FL.py [Input Lemmas file] [Output Lemmas file] [Input Postags file] [Output Postags file]

"""

import re
from sys import argv

with open('/export/storage/users/luciarn/BI_project/Mineria/Data/DiccionarioFinal_1.0.F.txt','r') as DicFenosFile:
    DicFenos=DicFenosFile.readlines()

#CODE FOR IMPROVING TRAINING SENTENCES WITH LEMMAS, DISEASES AND RS NUMBER
print('Improvig training sentences with lemmas..')
with open(argv[1],'r') as SentencesFile:
    Sentences=SentencesFile.readlines()
with open(argv[2], 'w') as LemmaRsDisFile:
    for Sent in Sentences: 
        Tag=' '
        for Feno in DicFenos: 
            Search=re.findall(re.compile('\W'+Feno.rstrip()+'\W', re.IGNORECASE), Sent) 
            if len(Search) > 0:
                Tag+='DIS ' * len(Search)
        Search=re.findall(re.compile('rs\d{2,}', re.IGNORECASE), Sent)
        if len(Search) > 0:
            Tag+='RSNUM ' * len(Search)
        LemmaRsDisFile.write(Sent.rstrip() + Tag + ' \n')
print ('Tagging of diseases and rs numbers... Done')


#CODE FOR IMPROVING TRAINING SENTENCES WITH LEMMAS, POSTAGS, DISEASES AND RS NUMBERS
print('Improvig training sentences with lemmas and postags...')
with open(argv[3],'r') as SentencesFile:
    Sentences=SentencesFile.readlines()
with open(argv[4],'w') as LemmaPostagRsDisFile:
    for i,Sent in enumerate(Sentences):
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
        LemmaPostagRsDisFile.write(Sent+' \n')
print ('Tagging of diseases and rs numbers... Done')

