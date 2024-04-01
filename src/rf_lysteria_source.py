# IMPORT PACKAGE 

# Stats
import numpy as np
import random 
from collections import Counter
import itertools
from scipy import interp
from itertools import cycle
import sys
import math

# pandas
import pandas as pd

# visualisation
import matplotlib.pyplot as plt
#import seaborn as sns
#from IPython.display import Image
#from subprocess import call
#from sklearn.tree import export_graphviz
#import pydot
#from yellowbrick.model_selection import FeatureImportances
#from sklearn.linear_model import Lasso

# modelling 
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import xgboost
from xgboost import XGBClassifier
#from bayes_opt import BayesianOptimization


# model evaluation
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import roc_curve, auc,  RocCurveDisplay
from yellowbrick.classifier import ClassificationReport, ROCAUC, ClassBalance,  ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# cross validation 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import CVScores
from yellowbrick.model_selection import RFECV
from sklearn.model_selection import cross_val_score

#Upsampling data
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

#Saving models
import joblib
from joblib import dump, load


import shap


def usage():

    sys.exit()

def get_opts():
    if len(sys.argv) != 9:
        usage()
    else:
        return sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8]

def load_features(feature_file):

    features = pd.read_csv(feature_file, sep='\t', header=0, index_col=0)
    print("load feature file")

    return features


def tune_model(model_name, model_input, sampling, block_strategy, cv_strategy, experiment_name):
    RSEED = 50

    train_labels = np.array(model_input['labels'])
    train= model_input.iloc[:,1:]

    #Set Model
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=RSEED)
        scoring_strategy=['accuracy']



    #SET UP GRID VALUES FOR HYPER-PARAMETER TUNNING
    if cv_strategy == 'random':
        #===RF-SPECIFIC
        # number of features at every split
        max_features = ['log2', 'sqrt']
        # number of trees
        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]

        # max depth
        max_depth = [int(x) for x in np.linspace(100, 500, num=10)]


    #Create a imblearn Pipeline to tune hyper-parameters with oversampling included

    # Oversampling strategy, random grid and Pipeline
    if sampling == 'random':
        oversampler = RandomOverSampler(random_state=RSEED)
        # create random grid
        random_grid = {
            'model__n_estimators': n_estimators,
            'model__max_features': max_features,
            'model__max_depth': max_depth
        }

        tunning_pipeline = Pipeline([
            ('oversampler', oversampler),
            ('model', model)
        ])

    if sampling == 'none':
        if model_name == 'RF':
            # create random grid
            random_grid = {
                'model__n_estimators': n_estimators,
                'model__max_features': max_features,
                'model__max_depth': max_depth
        }
            tunning_pipeline = Pipeline([
                ('model', model)
            ])


    if block_strategy=='none':
        #groups = np.array(model_input['t5'])
        rskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RSEED)
        cv_iterator = list(rskf.split(train, train_labels))


    # Fitting the pipeline and obtain the best parameters using random-search
    #GridSearch
    if cv_strategy == 'random':
        model_tunning = RandomizedSearchCV(estimator=tunning_pipeline, param_distributions = random_grid, n_iter = 100, cv = cv_iterator, scoring=scoring_strategy, refit='accuracy', verbose=2, random_state=RSEED, n_jobs = -1)


    if model_name == 'RF':
        model_tunning.fit(train, train_labels)


    # Select the best parameters and get all CV results
    best_param= model_tunning.best_params_
    best_score=model_tunning.best_score_
    cv_results=pd.DataFrame(model_tunning.cv_results_)
    best_model=model_tunning.best_estimator_

    #Filter the results of the classifier
    best_index = model_tunning.best_index_
    best_cv_results=pd.DataFrame(cv_results.iloc[best_index,:])
    best_cv_results=best_cv_results.transpose()
    best_cv_results = best_cv_results.assign(experiment=experiment_name)

    #EXPORT THE ML MODEL
    model_file_name = f'../results/02_RF_models/RF_model_{experiment_name}.joblib'  # Using f-string formatting
    joblib.dump(best_model, model_file_name)

    return best_param,best_cv_results,best_model

def build_confusion_matrix(model_input, best_params, sampling,block_strategy, model_name, experiment_name):

    #1. Wrange the data for the models
    all_labels = np.array(model_input['labels'])
    features= model_input.iloc[:,1:]

    #2. Create a list of all labels
    label_names = sorted(set(all_labels))
    label_names_array = np.array(label_names)



    #set up the random seed
    RSEED = 50

    if block_strategy=='none':
        #groups = np.array(model_input['t5'])
        rskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RSEED)
        cv_iterator = list(rskf.split(features, all_labels))

    acc_muvr_max = []

    # data frame for storing probabilities of classification
    final_res = pd.DataFrame()

    # Test-set true lables
    list_test_labels = []
    # Test-set predictions
    list_test_pred = []

    for x in range(len(cv_iterator)):
        test = model_input.iloc[cv_iterator[x][1]]
        samples = test.index.values.tolist()
        test_features = test.iloc[:,1:]
        test_labels = test['labels'].values.ravel()

        train = model_input.iloc[cv_iterator[x][0]]
        train_features = train.iloc[:,1:]
        train_labels = train['labels'].values.ravel()

        if model_name == 'RF':
            if sampling=='random':
                features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features,train_labels)
                model = RandomForestClassifier(n_estimators=best_params['model__n_estimators'],
                                               random_state=RSEED,
                                               max_features=best_params['model__max_features'],
                                               max_depth=best_params['model__max_depth'],
                                               n_jobs=-1, verbose=1)
            if sampling=='none':
                model = RandomForestClassifier(n_estimators=best_params['model__n_estimators'],
                                               random_state=RSEED,
                                               max_features=best_params['model__max_features'],
                                               max_depth=best_params['model__max_depth'],
                                               n_jobs=-1, verbose=1)


        if sampling == 'none':
            model.fit(train_features,train_labels)

        else:
            model.fit(features_resampled, labels_resampled)

        # test the model on test data
        test_model = model.predict(test_features)
        test_rf_probs = model.predict_proba(test_features)

        list_test_labels = list_test_labels + list(test_labels)
        list_test_pred = list_test_pred + list(test_model)

        #Create a DF containing the labels, predictions and probabilities
        res_df = pd.DataFrame(
            {'samples': samples, 'labels': test_labels, 'predictions': test_model})

        res_df = pd.merge(res_df, pd.DataFrame(test_rf_probs), how='left', left_index=True, right_index=True)

        final_res = pd.concat([final_res, res_df])


        acc_muvr_max += [model.score(test_features, test_labels)]
    print(sum(acc_muvr_max) / len(acc_muvr_max))


    #CREATE PROBABILITIES REPORT
    ##Change columns names
    col_names_initial=np.array(['samples','labels','predictions'])
    final_res_names = np.concatenate((col_names_initial,label_names_array))
    final_res.columns=final_res_names

    ##Add experiment names to the probabilities DF
    final_res=final_res.assign(experiment=experiment_name)

    #Export probabilities of the training set
    final_res_file_name = f'../results/01_model_training/Probabilities_{experiment_name}.tsv'  # Using f-string formatting
    final_res.to_csv(final_res_file_name, sep='\t')


    #CREATE CLASSIFICATION REPORT BASED ON ALL ITERATIONS
    report_ = classification_report(
        digits=6,
        y_true=list_test_labels,
        y_pred=list_test_pred,
        output_dict=True)

    classification_report_df = pd.DataFrame(report_).transpose()
    classification_report_df = classification_report_df.assign(experiment=experiment_name)

    #Export classification report
    classification_report_file_name = f'../results/01_model_training/classification_report_{experiment_name}.tsv'  # Using f-string formatting
    classification_report_df.to_csv(classification_report_file_name, sep='\t')

    # CREATE CONFUSION MATRIX
    cm = confusion_matrix(list_test_labels, list_test_pred, labels=label_names_array)
    cm_df=pd.DataFrame(cm, index=label_names_array, columns=label_names_array)

    # Export DataFrame to CSV
    confusion_matrix_file_name=f'../results/01_model_training/confusion_matrix_{experiment_name}.tsv'
    cm_df.to_csv(confusion_matrix_file_name, sep='\t')

    return final_res,cm_df

def predict_human(model,test_features,experiment_name):
    RSEED = 50

    #Get the order of training features names in the model
    features_names= model._final_estimator.feature_names_in_
    #Re-order test features as training features
    test_features=test_features[features_names]

    #Get the sample names into a list
    samples = test_features.index.values.tolist()

    #Make the predictions and calculate the probabilities
    rf_human_predictions = model.predict(test_features)
    rf_human_probabilities = model.predict_proba(test_features)

    #Combine everything to a dataframe
    human_predictions_df = pd.DataFrame(
        {'samples': samples, 'predictions': rf_human_predictions})
    human_predictions_df = pd.merge(human_predictions_df, pd.DataFrame(rf_human_probabilities), how='left', left_index=True, right_index=True)


    ##Change columns names
    all_labels = model._final_estimator.classes_
    label_names_array = np.array(all_labels)

    col_names_initial = np.array(['samples', 'predictions'])
    human_predictions_df_names = np.concatenate((col_names_initial, label_names_array))
    human_predictions_df.columns = human_predictions_df_names

    ##Add experiment names to the probabilities DF
    human_predictions_df=human_predictions_df.assign(experiment=experiment_name)

    #Export probabilities of the training set
    human_predictions_file_name = f'../results/03_human_predictions/Probabilities_{experiment_name}.tsv'  # Using f-string formatting
    human_predictions_df.to_csv(human_predictions_file_name, sep='\t')

    print('Human Samples correctly predicted')



################ MAIN ##############

#1. Load Training & Test files files
train_file_s1,train_file_s2,train_file_s2n1, train_file_s3, test_file_s1,test_file_s2, test_file_s2n1,test_file_s3= get_opts()
#training
train_data_s1 = load_features(train_file_s1)
train_data_s2 = load_features(train_file_s2)
train_data_s2n1 = load_features(train_file_s2n1)
train_data_s3 = load_features(train_file_s3)
#testing
test_data_s1 = load_features(test_file_s1)
test_data_s2 = load_features(test_file_s2)
test_data_s2n1 = load_features(test_file_s2n1)
test_data_s3 = load_features(test_file_s3)

print('All data has been imported')

#2. RF HYPER-PARAMETER OPTIMIZATION & ACCURACY CALCULATION
##2.1 Source_1
rf_best_params_source_1,best_cv_results_source_1,rf_model_source_1 = tune_model('RF',train_data_s1, 'none', 'none', 'random','Source_1')
rf_best_params_source_1_up,best_cv_results_source_1_up,rf_model_source_1_up = tune_model('RF',train_data_s1, 'random', 'none', 'random','Source_1_up')
##2.2 Source_2
rf_best_params_source_2,best_cv_results_source_2,rf_model_source_2 = tune_model('RF',train_data_s2, 'none', 'none', 'random','Source_2')
rf_best_params_source_2_up,best_cv_results_source_2_up, rf_model_source_2_up = tune_model('RF',train_data_s2, 'random', 'none', 'random','Source_2_up')
##2.3 Source_2n1
rf_best_params_source_2n1,best_cv_results_source_2n1,rf_model_source_2n1 = tune_model('RF',train_data_s2n1, 'none', 'none', 'random','Source_2n1')
rf_best_params_source_2n1_up,best_cv_results_source_2n1_up,rf_model_source_2n1_up = tune_model('RF',train_data_s2n1, 'random', 'none', 'random','Source_2n1_up')
##2.4 Source_3
rf_best_params_source_3,best_cv_results_source_3,rf_model_source_3 = tune_model('RF',train_data_s3, 'none', 'none', 'random','Source_3')
rf_best_params_source_3_up,best_cv_results_source_3_up,rf_model_source_3_up = tune_model('RF',train_data_s3, 'random', 'none', 'random','Source_3_up')
##2.5 concatenate and export results
final_cv_results = pd.concat([best_cv_results_source_1, best_cv_results_source_1_up,
                              best_cv_results_source_2, best_cv_results_source_2_up,
                              best_cv_results_source_2n1, best_cv_results_source_2n1_up,
                              best_cv_results_source_3, best_cv_results_source_3_up,])

final_cv_results.to_csv('../results/01_model_training/summary_accuracy.tsv', sep='\t')

#3. CONFUSION MATRIX CONSTRUCTION
##3.1 WITH TRAINING DATA - NO UP-SAMPLING
final_res_s1,cm_s1= build_confusion_matrix(train_data_s1, rf_best_params_source_1, 'none','none','RF', 'Source_1')
final_res_s2,cm_s2= build_confusion_matrix(train_data_s2, rf_best_params_source_2, 'none','none','RF', 'Source_2')
final_res_s2n1,cm_s2n1= build_confusion_matrix(train_data_s2n1, rf_best_params_source_2n1, 'none','none','RF', 'Source_2n1')
final_res_s3,cm_s3= build_confusion_matrix(train_data_s3, rf_best_params_source_3, 'none','none','RF', 'Source_3')

##3.2 TRAINING DATA - RANDOM UP-SAMPLING
final_res_s1_up,cm_s1_up= build_confusion_matrix(train_data_s1, rf_best_params_source_1_up, 'random','none','RF', 'Source_1_up')
final_res_s2_up,cm_s2_up= build_confusion_matrix(train_data_s2, rf_best_params_source_2_up, 'random','none','RF', 'Source_2_up')
final_res_s2n1_up,cm_s2n1_up= build_confusion_matrix(train_data_s2n1, rf_best_params_source_2n1_up, 'random','none','RF', 'Source_2n1_up')
final_res_s3_up,cm_s3_up= build_confusion_matrix(train_data_s3, rf_best_params_source_3_up, 'random','none','RF', 'Source_3_up')


#4. PREDICT HUMAN SAMPLES
##4.1 Models with no up-sampling
predict_human(rf_model_source_1,test_data_s1,'Source_1')
predict_human(rf_model_source_2,test_data_s2,'Source_2')
predict_human(rf_model_source_2n1,test_data_s2n1,'Source_2n1')
predict_human(rf_model_source_3,test_data_s3,'Source_3')

##4.2 Models with random up-sampling
predict_human(rf_model_source_1_up,test_data_s1,'Source_1_up')
predict_human(rf_model_source_2_up,test_data_s2,'Source_2_up')
predict_human(rf_model_source_2n1_up,test_data_s2n1,'Source_2n1_up')
predict_human(rf_model_source_3_up,test_data_s3,'Source_3_up')





