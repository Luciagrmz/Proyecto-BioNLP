# -*- encoding: utf-8 -*-

#############################################################################################################
				#CalculateF1_1.0.FL.py
#############################################################################################################

"""DESCRIPTION

Author: Lucia_LuisFernando

Version: 1.0

Date: 11/10/17

Description: This code calculates F1 with similarity of chain.To be considered correct it needs to have more than 85% of similarity

Parameters: 
 1) Path to the file that you want to compare
 2) Path to the reference file  

Output:
 1) F-1 Score

Environment:
  a) File: file that you want to compare
  b) RefFile: reference file
  c) Sentences: list with the lines of File
  d) RefSentences: list with the lines of the RefFile
  e) NumReference: number of lines in the reference
  f) NumSentences: number of elements extracted
  g) NumCorrect: number of correct extractions
  h) line: iterator for the variable Sentences
  i) refline: iterator for RefSentences

Usage: python3 CalculateF1_1.0.L.py [path to file] [path to reference file]

"""

from sys import argv

with open(agv[1],'r') as File, open(argv[2],'r') as RefFile:
    Sentences=File.readlines()
    RefSentences=RefFile.readlines()

NumReference=len(RefSentences)
NumSentences=len(Sentences)
NumCorrect=0

for line in Sentences:
    for refline in RefSentences:
        if (refline.split('\t')[0].lower() == line.split('\t')[0].lower() ) and (fuzz.ratio (refline.split('\t')[1].lower(), line.split('\t')[1].lower() ) > 85): # checks that the rsnum is equal and the disease is at least 85% similar
            NumCorrect+=1
            break #only needs to make a match    

print ('Reference: ', NumReference)
print ('Extractions: ', NumSentences)
print ('Corrected extractions: ', NumCorrect)
print ('Precision: ', NumCorrect / NumSentences)
print ('Recall: ', NumCorrect / NumReference)
print ('F-1: ', 2 * ( ((NumCorrect / NumSentences)* (NumCorrect / NumReference)) / ((NumCorrect / NumSentences) + (NumCorrect / NumReference)) )  ) #2*(Precision*Recall/Precision + Recall)

