# -*- encoding: utf-8 -*-

#############################################################################################################
				#GenerateGrid_0.0.L.py
#############################################################################################################

"""DESCRIPTION

Author= Lucia

Version= 0.0

Date: 11/10/17

Description: This codes generates the grid of experiments for runing the training of a SVM classifier and a MLP classifier

Parameters: None

Output:
 1) Bash with grid of experiments for SVM
 2) Bash with grid of experiments for MLP

Environment:
  a) InputFiles: input files with specific features for the training
  b) Vectors: type of vectors that the classifier can use
  c) StopWords: wheter or not to remove stop words
  d) Kernels: options for kernels
  e) Ngrams: combinations for initial and final n-gram in form of a tuple (initial,final)
  f) BashFile: output file with all the combinations
  g) List_Executionline: string with the execution line

Usage: python3 GenerateGrid_0.0.L.py

"""

InputFiles=['training-sentences.rs.lemma.txt','training-sentences.rs.lemma_postag_rs_dis.txt','training sentences.rs.lemma_rs_dis.txt', 'training-sentences.rs.lemma_postag_rs_dis_kws.txt','training sentences.rs.lemma_rs_dis_kws.txt']
Vectors=['TFIDF','BINARY', 'TFIDFBINARY']
StopWords=['True.--removeStopWords','False.']
Kernels=['linear', 'rbf','poly']
Ngrams=['1-1','2-2','3-3','1-2','1-3']

#CODE TO GENERATE COMBINATIONS FOR SVM
with open('/export/storage/users/luciarn/BI_project/Mineria/Codes/RunSvm_0.0.L.bash','w') as f:
    f.write('source activate python3 \n')
    List_Executionline=[]
    for inputfile in InputFiles:
        for vector in Vectors:
            for stopword in StopWords:
                for ngram in Ngrams:
                    for kernel in Kernels:
                        f.write('python3 /export/storage/users/luciarn/BI_project/Mineria/Codes/training-cross-validation-improving_1.0.FL.py --inputPath /export/storage/users/luciarn/BI_project/Mineria/Data --inputTrainingSentences ' + inputfile + ' --inputTrainingClasses training-classes.txt --outputPath /export/storage/users/luciarn/BI_project/Mineria/Reports --outputFile report.SVM_' + inputfile.split('.')[2] + '_'+ vector + '_sw' + stopword.split('.')[0] + '_' + kernel + '_ng' + ngram + '.txt --classifier SVM --vectype ' + vector + ' --positiveClass DISEASE '+ stopword.split('.')[1] + ' --kernel ' + kernel + ' --sngram ' + ngram.split('-')[0] + ' --fngram ' + ngram.split('-')[1] + ' \n') 
print('Bash for SVM... Done')

#CODE TO GENERATE COMBINATIONS FOR MLP
with open('/export/storage/users/luciarn/BI_project/Mineria/Codes/RunMlp_0.0.L.bash','w') as f:
    f.write('source activate python3 \n')
    List_Executionline=[]
    for inputfile in InputFiles:
        for vector in Vectors:
            for stopword in StopWords:
                for ngrama in Ngrams:
                    f.write('python3 /export/storage/users/luciarn/BI_project/Mineria/Codes/training-cross-validation-improving_1.0.FL.py --inputPath /export/storage/users/luciarn/BI_project/Mineria/Data --inputTrainingSentences ' + inputfile + ' --inputTrainingClasses training-classes.txt --outputPath /export/storage/users/luciarn/BI_project/Mineria/Reports --outputFile report.MLP_' + inputfile.split('.')[2] + '_'+ vector +'_sw' + stopword.split('.')[0] + '_ng' + ngram + '.txt --classifier MLP --vectype ' + vector + ' --positiveClass DISEASE ' + stopword.split('.')[1] + ' --sngram ' + ngram.split('-')[0] + ' --fngram ' + ngram.split('-')[1] + ' \n') 
print('Bash for MLP... Done')
