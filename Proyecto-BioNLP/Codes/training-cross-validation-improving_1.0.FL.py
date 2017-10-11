# -*- encoding: utf-8 -*-

#############################################################################################################
				#training-cross-validation-improving_1.0.FL.py
#############################################################################################################

"""DESCRIPTION

Author= CMendezC and Lucia_LuisFernando

Version= 1.0

Date: 11/10/17

Description: training and cross-validation

Parameters:
 1) --inputPath: Path to read input files.
 2) --inputTrainingSentences: File to read training data.
 3) --inputTrainingClasses: File to read training true classes.
 4) --outputPath: Path for output files.
 5) --outputFile: File for validation report.
 6) --classifier: SVM or MLP 
 7) --removeStopWords:  Remove stop words
 8) --vectype Vectorizer type: TFIDF or BINARY
 9) --positiveClass: Positive class
 10) --kernel: SVM Kernel

Output:
 1) Testing report 

Usage: python3 training-cross-validation-improving_1.0.FL.py --inputPath [] --inputTrainingSentences [] --inputTrainingClasses [] --outputPath [] --outputFile [] --classifier [] --removeStopWords [] --vectype [] --positiveClass [] --kernel []

"""

import os
from time import time
from optparse import OptionParser
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report, make_scorer, precision_score, recall_score
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
import scipy.stats
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier # 1.0.F
from numpy.random import uniform, randint # 1.0.F
from random import sample # 1.0.F

__author__ = 'CMendezC'

# Execution:

# python3.4 training-cross-validation-improving.py
# --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets
# --inputTrainingSentences training-sentences.rs.word.txt
# --inputTrainingClasses training-classes.txt
# --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports
# --outputFile report_SVM_linear_word.txt
# --classifier SVM 
# --vectype TFIDF
# --kernel linear
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.word.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_linear_word.txt --classifier SVM --vectype TFIDF --positiveClass DISEASE --kernel linear
# --kernel rbf
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.word.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_rbf_word.txt --classifier SVM --vectype TFIDF --positiveClass DISEASE --kernel rbf
# --kernel poly
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.word.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_poly_word.txt --classifier SVM --vectype TFIDF --positiveClass DISEASE --kernel poly

# python3.4 training-cross-validation-improving.py
# --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets
# --inputTrainingSentences training-sentences.rs.word.txt
# --inputTrainingClasses training-classes.txt
# --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports
# --outputFile report_SVM_word.txt
# --classifier SVM
# --vectype BINARY
# --positiveClass DISEASE
# --kernel linear
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.word.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_word_binary_linear.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel linear
# --kernel rbf
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.word.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_word_binary_rbf.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbf
# --kernel poly
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.word.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_word_binary_poly.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel poly

# python3.4 training-cross-validation-improving.py
# --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets
# --inputTrainingSentences training-sentences.rs.lemma.txt
# --inputTrainingClasses training-classes.txt
# --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports
# --outputFile report_SVM_lemma_binary.txt
# --classifier SVM
# --vectype BINARY
# --positiveClass DISEASE
# --kernel linear
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_binary_linear.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel linear
# --kernel rbf
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_binary_rbf.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbf
# --kernel poly
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_binary_poly.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel poly

# python3.4 training-cross-validation-improving.py
# --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets
# --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt
# --inputTrainingClasses training-classes.txt
# --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports
# --outputFile report_SVM_lemma_lemma_pos_pos_binary.txt
# --classifier SVM
# --vectype BINARY
# --positiveClass DISEASE
# --kernel linear
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_binary_linear.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel linear
# --kernel rbf
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_binary_rbf.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbf
# --kernel poly
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_binary_poly.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbpoly

# python3.4 training-cross-validation-improving.py
# --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets
# --inputTrainingSentences training-sentences.rs.lemma_lemma_tag_tag.txt
# --inputTrainingClasses training-classes.txt
# --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports
# --outputFile report_SVM_lemma_lemma_tag_tag_binary.txt
# --classifier SVM
# --vectype BINARY
# --positiveClass DISEASE
# --kernel linear
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_tag_tag.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_tag_tag_binary_linear.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel linear
# --kernel rbf
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_tag_tag.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_tag_tag_binary_rbf.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbf
# --kernel poly
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_tag_tag.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_tag_tag_binary_poly.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbpoly

# python3.4 training-cross-validation-improving.py
# --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets
# --inputTrainingSentences training-sentences.rs.tag4lemma.txt
# --inputTrainingClasses training-classes.txt
# --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports
# --outputFile report_SVM_tag4lemma_binary.txt
# --classifier SVM
# --vectype BINARY
# --positiveClass DISEASE
# --kernel linear
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.tag4lemma.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_tag4lemma_binary_linear.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel linear
# --kernel rbf
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.tag4lemma.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_tag4lemma_binary_rbf.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbf
# --kernel poly
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.tag4lemma.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_tag4lemma_binary_poly.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel poly

# python3.4 training-cross-validation-improving.py
# --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets
# --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt
# --inputTrainingClasses training-classes.txt
# --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports
# --outputFile report_SVM_lemma_lemma_pos_pos_binary.txt
# --classifier SVM
# --vectype BINARY
# --positiveClass DISEASE
# --kernel linear
# --sngram 1
# --fngram 2
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_binary_linear.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel linear --sngram 1 --fngram 2
# --kernel rbf
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_binary_rbf.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbf --sngram 1 --fngram 2
# --kernel poly
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_binary_poly.txt --classifier SVM --vectype BINARY --positiveClass DISEASE --kernel rbpoly --sngram 1 --fngram 2

# python3.4 training-cross-validation-improving.py
# --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets
# --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt
# --inputTrainingClasses training-classes.txt
# --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports
# --outputFile report_SVM_lemma_lemma_pos_pos_binary.txt
# --classifier SVM
# --vectype TFIDFBINARY
# --positiveClass DISEASE
# --kernel linear
# --sngram 1
# --fngram 2
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_tfidfbinary_linear.txt --classifier SVM --vectype TFIDFBINARY --positiveClass DISEASE --kernel linear --sngram 1 --fngram 2
# --kernel rbf
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_tfidfbinary_rbf.txt --classifier SVM --vectype TFIDFBINARY --positiveClass DISEASE --kernel rbf --sngram 1 --fngram 2
# --kernel poly
# python3.4 training-cross-validation-improving.py --inputPath /home/compu2/gene-disease-relation-extraction/training-testing-data-sets --inputTrainingSentences training-sentences.rs.lemma_lemma_pos_pos.txt --inputTrainingClasses training-classes.txt --outputPath /home/compu2/gene-disease-relation-extraction/training-testing-reports --outputFile report_SVM_lemma_lemma_pos_pos_tfidfbinary_poly.txt --classifier SVM --vectype TFIDFBINARY --positiveClass DISEASE --kernel poly --sngram 1 --fngram 2

###########################################################
#                       MAIN PROGRAM                      #
###########################################################

if __name__ == "__main__":
    # Parameter definition
    parser = OptionParser()
    parser.add_option("--inputPath", dest="inputPath",
                      help="Path to read input files", metavar="PATH")
    parser.add_option("--inputTrainingSentences", dest="inputTrainingSentences",
                      help="File to read training data", metavar="FILE")
    parser.add_option("--inputTrainingClasses", dest="inputTrainingClasses",
                      help="File to read training true classes", metavar="FILE")
    parser.add_option("--outputPath", dest="outputPath",
                          help="Path for output files", metavar="PATH")
    parser.add_option("--outputFile", dest="outputFile",
                      help="File for validation report", metavar="FILE")
    parser.add_option("--classifier", dest="classifier",
                      help="Classifier", metavar="CLASSIFIER")
    parser.add_option("--vectype", dest="vectype",
                      help="Vectorizer type: TFIDF, TFIDFBINARY, BINARY", metavar="TEXT")
    parser.add_option("--positiveClass", dest="positiveClass",
                      help="Positive class", metavar="TEXT")
    parser.add_option("--removeStopWords", default=False,
                      action="store_true", dest="removeStopWords",
                      help="Remove stop words")
    parser.add_option("--kernel", dest="kernel", default='linear',
                      choices=('rbf', 'linear', 'poly'),
                      help="Kernel", metavar="TEXT")
    parser.add_option("--sngram", type="int",
                      dest="sngram", default=1,
                      help="Start n-gram", metavar="N")
    parser.add_option("--fngram", type="int",
                      dest="fngram", default=1,
                      help="Final n-gram", metavar="N")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.error("None parameters indicated.")
        sys.exit(1)

    # Printing parameter values
    print('-------------------------------- PARAMETERS --------------------------------')
    print("Path to read input files: " + str(options.inputPath))
    print("File to read training data: " + str(options.inputTrainingSentences))
    print("File to read training true classes: " + str(options.inputTrainingClasses))
    print("Path for output files: " + str(options.outputPath))
    print("File to write report: " + str(options.outputFile))
    print("Classifier: " + str(options.outputFile))
    print("Vectorizer type: " + str(options.vectype))
    print("Positive class: " + str(options.positiveClass))
    print("Remove stop words: " + str(options.removeStopWords))
    print("Kernel: " + str(options.kernel))
    print("Start ngram: " + str(options.sngram))
    print("Final ngram: " + str(options.fngram))

    # Start time
    t0 = time()

    t1 = time()
    print("Reading training and true classes...")
    trueTrainingClasses = []
    with open(os.path.join(options.inputPath, options.inputTrainingClasses), encoding='utf8', mode='r') \
            as classFile:
        for line in classFile:
            line = line.strip('\n')
            trueTrainingClasses.append(line)

    trainingSentences = []
    with open(os.path.join(options.inputPath, options.inputTrainingSentences), encoding='utf8', mode='r') \
            as dataFile:
        for line in dataFile:
            line = line.strip('\n')
            trainingSentences.append(line)
    # print(trainingSentences)

    print("     Reading training and true classes done in {:.2} seg".format((time() - t1)))

    if options.removeStopWords:
        print("   Removing stop words")
        pf = stopwords.words('english')
    else:
        pf = None

    t1 = time()
    print('     Creating training vectorizers...')
    if options.vectype == "TFIDF":
        vectorizer = TfidfVectorizer(ngram_range=(options.sngram, options.fngram), stop_words=pf)
    elif options.vectype == "TFIDFBINARY":
        vectorizer = TfidfVectorizer(ngram_range=(options.sngram, options.fngram), stop_words=pf, binary=True)
    elif options.vectype == "BINARY":
        vectorizer = CountVectorizer(ngram_range=(options.sngram, options.fngram), stop_words=pf, binary=True)

    matrixTraining = csr_matrix(vectorizer.fit_transform(trainingSentences), dtype='double')
    print('     matrixTraining.shape: ', matrixTraining.shape)
    print("        Creating training vectorizer done in {:.2} seg".format((time() - t1)))

    scoring = make_scorer(f1_score, pos_label=options.positiveClass)

    if options.classifier == "SVM":
        classifier = SVC()
        if options.kernel == 'rbf':
            paramGrid = {'C': scipy.stats.expon(scale=10), 'gamma': scipy.stats.expon(scale=.1),
                         'kernel': ['rbf'], 'class_weight': ['balanced']}
        elif options.kernel == 'linear':
            paramGrid = {'C': scipy.stats.expon(scale=10), 'kernel': ['linear'],
                         'class_weight': ['balanced']}
        elif options.kernel == 'poly':
            paramGrid = {'C': scipy.stats.expon(scale=10), 'gamma': scipy.stats.expon(scale=.1),
                         'degree': [2, 3],
                         'kernel': ['poly'], 'class_weight': ['balanced']}
        classifier_cv = model_selection.RandomizedSearchCV(classifier, paramGrid,
                                                          cv=10, n_jobs=30, verbose=3,
                                                          scoring=scoring, random_state=42)

    if options.classifier == "MLP":
        classifier = MLPClassifier()
        paramGrid= {'hidden_layer_sizes':[(x,) for x in sample(range(30,101),2)],'max_iter': sample(range(80, 201),2)} 
        classifier_cv = model_selection.GridSearchCV(classifier, paramGrid,
                                                          cv=10, n_jobs=30, verbose=3,
                                                          scoring=scoring)
    t1 = time()
    print("   Training and cross validation...")
    classifier_cv.fit(matrixTraining, trueTrainingClasses)
    best_score = classifier_cv.best_score_
    best_parameters = classifier_cv.best_estimator_.get_params()
    print("     Training and cross validation done in {:.2} seg".format((time() - t1)))

    print("   Saving validation report...")
    with open(os.path.join(options.outputPath, options.outputFile), mode='w', encoding='utf8') as oFile:
        oFile.write('**********        VALIDATION REPORT     **********\n')
        oFile.write('Classifier: {}\n'.format(options.classifier))
        oFile.write('Kernel: {}\n'.format(options.kernel))
        oFile.write('Best score{}: {}\n'.format(scoring, best_score))
        oFile.write('Best parameters:\n')
        for param in sorted(best_parameters.keys()):
            oFile.write("\t%s: %r\n" % (param, best_parameters[param]))

    print("     Saving validation report done!")

    print('**********        VALIDATION REPORT     **********')
    print('Classifier: {}'.format(options.classifier))
    print('Kernel: {}'.format(options.kernel))
    print('Best score{}: {}'.format(scoring, best_score))
    print('Best parameters:')
    for param in sorted(best_parameters.keys()):
        print("\t%s: %r" % (param, best_parameters[param]))

    print("     Saving best model and vectorizer...")
    t1 = time()
    joblib.dump(classifier_cv.best_estimator_,
                    os.path.join(options.outputPath, 'models', options.outputFile.replace(".txt", ".mod")))
    joblib.dump(vectorizer,
                    os.path.join(options.outputPath, 'vectorizers', options.outputFile.replace(".txt", ".vec")))
    print("        Saving training model and vectorizer done in: %fs" % (time() - t1))

    print("Training and cross validation done in: %fs" % (time() - t0))
