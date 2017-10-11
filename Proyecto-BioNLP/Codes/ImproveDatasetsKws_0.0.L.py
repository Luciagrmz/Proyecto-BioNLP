# -*- encoding: utf-8 -*-

#############################################################################################################
				#ImproveDatasetsKws_0.0.L.py
#############################################################################################################

"""DESCRIPTION

Author= Lucia

Version= 0.0

Date: 11/10/17

Description: This codes tags the training sentences with keywords

Parameters:
 1) Input file with improved training sentences that contains lemmas, diseases and rs numbers
 2) Output file to write the sentences with lemmas, diseases, rs numbers and keywords
 3) Input file with improved training sentences that contains lemmas, postags, diseases and rs numbers
 2) Output file to write the sentences with lemmas, postags, diseases, rs numbers and keywords

Output:
 1) Training sentences with lemmas, diseases, rs numbers, diseases and tagging of keywords in a txt file
 2) Training sentences with lemmas, postags, rs numbers, diseases and tagging of keywords in a txt file

Environment:
  a) DicKeyws: dictionary with keywords and their respective tag
  b) SentencesFile: initial file with training sentences
  c) Sentences: list with the training sentences
  e) LemmaRsDisKwsFile: output file with training sentences that contains lemmas, rs numbers, diseases and tagging of keywords
  f) Tag: contains the list of tags (RS NUM, DIS or their corresponding postags) for each sentence
  g) Sent: every sentence from the variable Sentences
  h) keyw: every keyword in the DicKeyws
  i) Search: contains a list with the matches found in the sentence for the disease
  j) LemmaPostagRsDisKwsFile: output file with training sentences, lemmas, postags and tagging of diseases and rs
  k) i: index of Sentences
  l) Sentsep: list with the line of text and their corresponding postags
  m) Lines: list of words from every sentence
  n) word: word in Lines
  
Usage: python3 ImproveDatasetsKws_0.0.L.py [inputfile Lemmas] [outputfile Lemmas] [Inputfile Postags] [Outputfile Postags]

"""

import re
from sys import argv

#Dictionary with the start of some keywords
DicKeyws={'associat\w*':'KEYWD','relat\w*':'KEYWD','affect\w*':'KEYWD','susceptib\w*':'KEYWD','risk':'KEYWD','correlat\w*':'KEYWD','link\w*':'KEYWD','marker':'KEYWD','evidence\w*':'KEYWD','influenc\w*':'KEYWD','no ':'NEGKEYWD','not ':'NEGKEYWD','non ':'NEGKEYWD','don\S*\w*':'NEGKEYWD','negative':'NEGKEYWD','doesn\S*\w*':'NEGKEYWD','didn\S*\w*':'NEGKEYWD','wasn\S*\w*':'NEGKEYWD','couldn\S*\w*':'NEGKEYWD','isn\S*\w*':'NEGKEYWD','aren\S\w*':'NEGKEYWD'}

#CODE FOR IMPROVING TAGGING KEYWORDS IN THE TRAINING SENTENCES THAT CONTAINS LEMMAS, DISEASES AND RS NUMBERS
print('Starting tagging of keyword in training sentences with lemmas, rs numbers and diseases ..')
with open(argv[1],'r') as SentencesFile:
    Sentences=SentencesFile.readlines()
with open(argv[2], 'w') as LemmaRsDisKwsFile:
    for Sent in Sentences: 
        Tag=' '
        for keyw in DicKeyws: 
            Search=re.findall(re.compile('\\b'+keyw, re.IGNORECASE), Sent) 
            if len(Search) > 0:
                Tag+=DicKeyws[keyw] + ' ' * len(Search)
        LemmaRsDisKwsFile.write(Sent.rstrip() + ' ' + Tag + ' \n')
print ('Tagging done..')

##CODE FOR IMPROVING TAGGING KEYWORDS IN THE TRAINING SENTENCES THAT CONTAINS LEMMAS, POSTAGS, DISEASES AND RS NUMBERS
print('Starting tagging of keyword in training sentences with lemmas, postags, rs numbers and diseases ..')
with open(argv[3],'r') as SentencesFile:
     Sentences=SentencesFile.readlines()
with open(argv[4],'w') as LemmaPosRsDisKwsFile:
    for i,Sent in enumerate(Sentences):
        Sentsep=Sent.rstrip().split()
        Lines=Sentsep[:int(len(Sentsep)/2)]
        Tag=Sentsep[int(len(Sentsep)/2):]
        for keyw in DicKeyws:
            if re.search(re.compile(keyw, re.IGNORECASE),Sent):
                for i,word in enumerate(Lines):
                    if re.search(re.compile('\\b'+keyw, re.IGNORECASE),word):
                        Tag[i]=DicKeyws[keyw]
        LemmaPosRsDisKwsFile.write(' '.join(Lines)+ ' '+ ' '.join(Tag)+ ' \n')
print ('Tagging done..')

