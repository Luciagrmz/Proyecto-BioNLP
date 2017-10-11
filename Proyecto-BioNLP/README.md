### Proyecto-BioNLP
This Repository contains the necessary information to compare the classificators: SVM (Support Vector Machine) and MLP (Multilayer Perceptron) within a given dataset that contains a list of abstracts from the biomedicine literature. It contains scripts implemented in python, one-liner perl commands and linux-shell. 

##Getting Started
-Principal Dependencies needed: scikit-learn 0.19.0, fuzzywuzzy 0.15.1 and  re 7.2
- The repository is divided in three parts:Data, Codes and Reports

##Data 
- DiccionarioFinal_0.1.F: dictionary of diseases obtained from several sources
- DiccionarioFinal_1.0.FL: filtered dictionary
- PM-Sentences: List of all sentences extracted from abstracts with their respective pubmed-id
- Reference-nonredundant_0.0.E: reference to compare with the initial approach
- Rs-Dise?
- training-classes: classes curated manually from the training sentences
- training-sentences.rs.lemma: training sentences with lemmas
- training-sentences.rs.lemma_lemma_tag_tag: initial training sentences with lemmas and some tags of diseases and rs-numbers
- training-sentences.rs.lemma_lemma_pos_pos: initial training sentences with lemmas, postags and some tags of diseases and rs-numbers
- training-sentences.rs.lemma_rs_dis : training sentences with lemmas and tagging of diseases and rs-numbers based on our dictionary obtained
- training-sentences.rs.lemma_rs_dis_kws : training-sentences.rs.lemma_rs_dis with tagging of keywords
- training-sentences.rs.lemma_postags_rs_dis: training sentences with lemmas, postags and tagging of diseases and rs-numbers based on our dictionary
- training-sentences.rs.lemma_postags_rs_dis_kws: training-sentences.rs.lemma_postags_rs_dis with tagging of keywords
- test-classes: classes curated manually from the test sentences
- test-sentences.rs.lemma_lemma_pos_pos.txt: initial test sentences with lemmas, postags and some tags of diseases and rs-numbers
- test-sentences.rs.lemma_postags_rs_dis_kws: test sentences processed according to the best parameters of the classfier (sentences with lemmas, postags, rs numbers, diseases and keywords)

##Codes
- FilterDic_0.0.FL.py: Filters the dictionary 
- GetAssociations: Code for the initial approach of getting associations between rs-numbers nad diseases
- CalculateF1_1.0.L: Calculates F1 score for the initial approach
- ImproveDatasets_1.0.FL: Improves tagging of diseases and rs-numbers
- ImproveDatasetsKws_0.0.L: Tags sentences with keywords
- GenerateGrid_0.0.L: Generates the scripts to run the necessary combinations of parameters for the training part
- RunMlp_0.0.L , RunSvm_0.0.L : scripts for running the combination of parametersfor every classfier
- training-cross-validation-improving_1.0.FL: training and cross validation to get the best hyperparameters
- classiy.0.0.T: classify sentences that have or have not an associations
- Testing_0.0.F: generates testing report with the F1 score
- GetF1Scores.0.0.F: generates a table with all the F1-scores of all the combination of parameters obtained

##Reports
- F1Results: Table with all the F1-scores of all the combinations
- report.SVM_lemma_postag_rs_dis_kws_BINARY_swTrue_rbf_ng1-1.txt: report of the training with the combination that yields the highest score
- report.SVM_lemma_postag_rs_dis_kws_BINARY_swTrue_rbf_ng1-1.mod: model of the highest score
- report.SVM_lemma_postag_rs_dis_kws_BINARY_swTrue_rbf_ng1-1.vec: vectorizer of the highest 

##Authors:
- Lucia Ramirez Navarro
- Gilberto Durán Bishop
- Luis Fernando Altamirano
- Luis Enrique Ramirez Serrano
