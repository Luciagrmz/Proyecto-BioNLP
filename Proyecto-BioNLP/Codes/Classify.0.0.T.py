# -*- encoding: utf-8 -*-

#############################################################################################################
				#Classify.0.0.T.py
#############################################################################################################

"""DESCRIPTION

Author: CMendezC

Version: 0.0

Date: 11/10/17

Description: classify text files by using trained SVM model and vectorizer.

Parameters: 
 1) --inputPath: Path to read TXT files to classify.
 2) --inputFile: File to read text to classify (one per line).
 3) --outputPath: Path to place classified TXT files.
 4) --outputFile: Name of the output file
 5) --modelPath: Parent path to read trained model and vectorizer.
 6) --modelName: Name of model and vectorizer to load.
 7) --clasePos: Positive class of the classification
 8) --claseNeg: Negative class of the classification

Output:
 1) Classified text files

Usage: python3 Classify.0.0.T.py --inputPath [] --inputFile [] --outputPath [] --outputFile [] --modelPath [] --modelName [] --clasePos [] --claseNeg []

"""

import os
from time import time
from optparse import OptionParser
from nltk import word_tokenize
import sys
from scipy.sparse import csr_matrix
from sklearn.externals import joblib
from sklearn.svm import SVC
import scipy.stats
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection

__author__ = 'CMendezC'


###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path to read file with features extracted to classify", metavar="PATH")
    parser.add_option("--inputFile", dest="inputFile",
                      help="File to read text to classify (one per line)", metavar="FILE")
    parser.add_option("--outputPath", dest="outputPath",
                      help="Path to place classified text", metavar="PATH")
    parser.add_option("--outputFile", dest="outputFile",
                      help="Output file name to write classified text", metavar="FILE")
    parser.add_option("--modelPath", dest="modelPath",
                      help="Path to read trained model", metavar="PATH")
    parser.add_option("--modelName", dest="modelName",
                      help="Name of model and vectorizer to load", metavar="NAME")
    # Clase positiva para clasificación
    parser.add_option("--clasePos", dest="clasePos",
                      help="Clase positiva del corpus", metavar="CLAS")
    # Clase negativa para clasificación
    parser.add_option("--claseNeg", dest="claseNeg",
                      help="Clase negativa del corpus", metavar="CLAS")

    (options, args) = parser.parse_args()
    if len(args) > 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read file with features extracted to classify: " + str(options.inputPath))
    print("File to read text to classify (one per line): " + str(options.inputFile))
    print("Path to place classified TXT file: " + str(options.outputPath))
    print("Output file name to write classified text: " + str(options.outputFile))
    print("Path to read trained model, vectorizer, and dimensionality reduction: " + str(options.modelPath))
    print("Name of model, vectorizer, and dimensionality reduction to load: " + str(options.modelName))
    print("Positive class: " + str(options.claseNeg))
    print("Negative class: " + str(options.clasePos))

    t0 = time()

    listSentences = []

    with open(os.path.join(options.inputPath, options.inputFile), 'r', encoding='utf8', errors='replace') as iFile:
        for line in iFile.readlines():
            line = line.strip('\n')
            listSentences.append(line)

    print("Classifying texts...")
    print("   Loading model and vectorizer: " + options.modelName)
    if options.modelName.find('.SVM.'):
        classifier = SVC()
    classifier = joblib.load(os.path.join(options.modelPath, 'models', options.modelName + '.mod'), mmap_mode=None)
    vectorizer = joblib.load(os.path.join(options.modelPath, 'vectorizers', options.modelName + '.vec'))

    matrix = csr_matrix(vectorizer.transform(listSentences), dtype='double')
    print("   matrix.shape " + str(matrix.shape))

    # Classify corpus list
    y_predicted = classifier.predict(matrix)

    print("   Predicted class list length: " + str(len(y_predicted)))

    with open(os.path.join(options.outputPath, options.outputFile), "w", encoding="utf-8") as oFile:
        oFile.write('SENTENCE\tPREDICTED_CLASS\n')
        for s, pc in zip(listSentences, y_predicted):
            oFile.write(s + '\t' + pc + '\n')

    print("Texts classified in : %fs" % (time() - t0))

