# -*- encoding: utf-8 -*-

#############################################################################################################
				#ProcessSentences.0.0.L.py
#############################################################################################################

"""DESCRIPTION

Author: Lucia

Version: 0.0

Date: 13/10/17

Description: This code preprocess the abstracts in order to contain lemmas and postags

Parameters: 
 1) Path to the file you want to process
 2) Path to the outputfile

Output:
 1) Preprocessed Abstracts

Environment:
  a) SentencesFile: output from the lemmatization and postagging of corenlp
  b) Sentences: list of lines in SentencesFile
  c) ListSentences: list with the lemmas of a sentences
  d) ListTags: list with the postags
  e) pubmedfound: flag to indicate if a pubmed was found
  f) pubmed: pubmed id
  g) search: list with the lemma and postag of every word
  h) outputfile: preprocessed abstracts

Usage: python3 ProcessSentences.0.0.L.py [input] [output]

"""




import re
from sys import argv
with open(argv[1],'r') as SentencesFile:
    Sentences=SentencesFile.readlines()
ListSentences=[]
ListTags=[]
pubmedfound=0
with open (argv[2],'w') as outputfile:
    for i,sent in enumerate(Sentences):
        if re.match(re.compile('Sentence\s\#\d+'), sent):
            if i is not 0:
                if pubmedfound:
                    ListSentences=ListSentences[1:]
                    ListTags=ListTags[1:]
                    pubmedfound=0
                outputfile.write(pubmed + '\t' + ' '.join(ListSentences) + ' ' + ' '.join(ListTags) + '\n')
                ListSentences=[]
                ListTags=[] 
        search=re.search(r'PartOfSpeech=(.+)\sLemma=(.+)\]',sent, re.I)
        if search:
            ListSentences.append(search.group(2))
            ListTags.append(search.group(1))
        if re.match(r'\d+\t',sent):
            pubmed=re.match(r'(\d+)\t',sent).group(1)
            pubmedfound=1
            
