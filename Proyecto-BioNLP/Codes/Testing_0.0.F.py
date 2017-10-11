# -*- encoding: utf-8 -*-

#############################################################################################################
				#Testing_0.0.F.py
#############################################################################################################

"""DESCRIPTION

Author= LuisFernando

Version= 0.0

Date: 11/10/17

Description: Tests and gets F1-scores based on the classified classes

Parameters: 
 1) --inputPath Path to read input files.
 2) --inputTestClasses File of the correct classes
 3) --inputPredictedClasses File of the predicted classes by the classifier
 4) --outputPath Path for output files.
 5) --outputFile File for validation report.

Output:
 1) Testing report 

Usage: python3 Code_Testing_0.0.F.py --inputPath [] --inputTestClasses [] --inputPredictedClasses [] --outputPath [] --outputFile []

"""


import os
from time import time
from optparse import OptionParser
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, \
    classification_report, make_scorer, precision_score, recall_score
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import scipy.stats

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path to read input files", metavar="PATH")
    parser.add_option("--inputTestClasses", dest="inputTestClasses",
                      help="File to read test true classes", metavar="FILE")
    parser.add_option("--inputPredictedClasses", dest="inputPredictedClasses",
                      help="File to read predicted classes by the classifier", metavar="FILE")
    parser.add_option("--outputPath", dest="outputPath",
                          help="Path for output files", metavar="PATH")
    parser.add_option("--outputFile", dest="outputFile",
                      help="File for validation report", metavar="FILE")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)



    with open(os.path.join(options.inputPath, options.inputTestClasses), encoding='utf8', mode='r') \
            as classFile:
        y_true = [line.strip('\n') for line in classFile]

    with open(os.path.join(options.inputPath, options.inputPredictedClasses), encoding='utf8', mode='r') \
            as classPredictedFile:
        y_predicted = [line.strip('\n') for line in classPredictedFile]


    pre = precision_score(y_true, y_predicted, average='weighted')
    rec = recall_score(y_true, y_predicted, average='weighted')
    f1 = f1_score(y_true, y_predicted, average='weighted')


    print('**********        TESTING REPORT     **********\n')
    print('Precision: {}'.format(pre))
    print('Recall: {}'.format(rec))
    print('F-score: {}'.format(f1))
    print('Confusion matrix:')
    print(str(confusion_matrix(y_true, y_predicted)))
    print('Classification report:')
    print(classification_report(y_true, y_predicted))
