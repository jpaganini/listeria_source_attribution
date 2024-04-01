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

###### ------- MUVR ------- ######
#from py_muvr.feature_selector import FeatureSelector
from concurrent.futures import ProcessPoolExecutor

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

def prepare_data_muvr(train_data):
    # THIS MIGHT HAVE TO BE REMOVED - OR PUT IN A DIFFERENT SCRIPT
    train_data_df = pd.read_csv(train_data, sep='\t', header=0, index_col=0)

    train_data_muvr = train_data_df.sort_index().drop_duplicates(subset=['t5', 'SYMP'],
                                                       keep='last')  # remove samples that contain a +

    train_data_muvr.to_csv(r'2023_train_data_filtered.tsv', sep='\t')

    return train_data_muvr

def feature_reduction(train_data_muvr,chisq_file, model):
    # THIS MIGHT HAVE TO BE REMOVED - OR PUT IN A DIFFERENT SCRIPT
    train_data_muvr = pd.read_csv(train_data_muvr, sep='\t', header=0)
    train_data_muvr= train_data_muvr.set_index('SRA')
    columns_to_drop = ['MOLIS', 'LINEAGE','STX','SNP ADDRESS','t5','SYMP H/L']  # Replace with the actual column names
    train_data_muvr = train_data_muvr.drop(columns=columns_to_drop)

    # Create an iterator for reading chisq_features line by line
    reader_chisq = pd.read_csv(chisq_file, sep='\t', header=0, iterator=True, chunksize=1)

    # Create a dataframe to hold the results
    model_input = pd.DataFrame()

    # Get the first line of chisq_features
    try:
        chunk_chisq = next(reader_chisq)
    except StopIteration:
        chunk_chisq = pd.DataFrame()

    while not chunk_chisq.empty:
        # Set the index as the first column
        chunk_chisq.set_index(chunk_chisq.columns[0], inplace=True)
        chunk_chisq = chunk_chisq.astype("int8")

        # Merge the current line with isolate_metadata based on your desired criteria
        merged_line = pd.merge(train_data_muvr, chunk_chisq, left_index=True, right_index=True, how='inner')
        #print(merged_line)
        model_input = pd.concat([model_input, merged_line], ignore_index=False)

        #Get the following lines of the dataframe
        try:
            chunk_chisq = next(reader_chisq)
        except StopIteration:
            chunk_chisq = pd.DataFrame()

    to_predict = ['SYMP']

    X_muvr = model_input.drop('SYMP', axis = 1).to_numpy()
    y_muvr = model_input['SYMP'].values.ravel()

    if model=='XGBC':
        encoder = OneHotEncoder(sparse=False)

    # Reshape y to a 2D array as fit_transform expects a 2D array
        y_encoded = encoder.fit_transform(np.array(y_muvr).reshape(-1, 1))
        y_variable = y_encoded

    elif model=='RFC':
        y_variable = y_muvr

    else:
        print ("Select a valid model: RFC or XBGC")
        SystemExit



    feature_names = model_input.drop(columns=["SYMP"]).columns


    feature_selector = FeatureSelector(
        n_repetitions=10,
        n_outer=5,
        n_inner=4,
        estimator=model,
        metric="MISS",
        features_dropout_rate=0.9
    )

    feature_selector.fit(X_muvr, y_variable)
    selected_features = feature_selector.get_selected_features(feature_names=feature_names)

    # Obtain a dataframe containing MUVR selected features
    df_muvr_min = model_input[to_predict+list(selected_features.min)]
    df_muvr_mid = model_input[to_predict+list(selected_features.mid)]
    df_muvr_max = model_input[to_predict+list(selected_features.max)]

    print('something')

    #df_muvr_min.to_csv(r'2023_jp_muvr_min.tsv', sep='\t')
    #df_muvr_mid.to_csv(r'2023_jp_muvr_mid.tsv', sep='\t')
    #df_muvr_max.to_csv(r'2023_jp_muvr_max.tsv', sep='\t')

    return df_muvr_max

def feature_extraction(min_muvr_filtered_file, mid_muvr_filtered_file, max_muvr_filtered_file, chisq_file):
    # THIS MIGHT HAVE TO BE REMOVED - OR PUT IN A DIFFERENT SCRIPT
    min_features_columns = pd.read_csv(min_muvr_filtered_file, sep='\t', header=0, index_col=0).columns[1:].tolist()
    mid_features_columns = pd.read_csv(mid_muvr_filtered_file, sep='\t', header=0, index_col=0).columns[1:].tolist()
    max_features_columns = pd.read_csv(max_muvr_filtered_file, sep='\t', header=0, index_col=0).columns[1:].tolist()

    #Get column names
    min_features_columns = ['Unnamed: 0'] + min_features_columns
    mid_features_columns = ['Unnamed: 0'] + mid_features_columns
    max_features_columns = ['Unnamed: 0'] + max_features_columns

    #Get data from chisq file
    min_chisq_data = pd.read_csv(chisq_file, sep='\t', header=0, index_col=0, usecols=min_features_columns)
    mid_chisq_data = pd.read_csv(chisq_file, sep='\t', header=0, index_col=0, usecols=mid_features_columns)
    max_chisq_data = pd.read_csv(chisq_file, sep='\t', header=0, index_col=0, usecols=max_features_columns)


    min_chisq_data.to_csv(r'2023_jp_complete_muvr_min.tsv', sep='\t')
    mid_chisq_data.to_csv(r'2023_jp_complete_muvr_mid.tsv', sep='\t')
    max_chisq_data.to_csv(r'2023_jp_complete_muvr_max.tsv', sep='\t')

def tune_model(model_name, model_input, sampling, block_strategy, cv_strategy, experiment_name):
    RSEED = 50

    train_labels = np.array(model_input['labels'])
    train= model_input.iloc[:,1:]

    #Set Model
    if model_name == 'RF':
        model = RandomForestClassifier(random_state=RSEED)
        scoring_strategy=['accuracy','balanced_accuracy']

    if model_name == 'XGBC':
        # Encode the labels
        encoder = LabelEncoder()
        train_labels = encoder.fit_transform(train_labels)
        #Calculate sample weights
        sample_weights = compute_sample_weight(
            class_weight='balanced',
            y=train_labels
        )
        # Calculate sample weights
        model = XGBClassifier(objective='multi:softmax')



    #SET UP GRID VALUES FOR HYPER-PARAMETER TUNNING
    if cv_strategy == 'random':
        #===RF-SPECIFIC
        # number of features at every split
        max_features = ['log2', 'sqrt']

        # max depth
        max_depth = [int(x) for x in np.linspace(100, 500, num=10)]

        #===SMOTE
        #K-neighbors for smote
        k_neighbors = [1,2,3,4]

        #==XGBC
        eta = np.linspace(0.01, 0.2, 10)
        gamma = [0,3,5,7,9]
        max_depth_xgbc = [3,4,5,6,7,8,9,10]
        min_child_weight = [1,2,3,4,5]
        subsample = [0.6, 0.7, 0.8, 0.9, 1]
        #scale_pos_weight = sample_weights #only for binary classification
        colsample_bytree = [0.7, 0.8, 0.9, 1]

        #==COMMON MODEL PARAMETERS
        # number of trees
        n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]

    if cv_strategy == 'grid':
         max_features=['sqrt']
         best_n_estimators=644
         lower_bound_estimators = best_n_estimators - 5
         upper_bound_estimators = best_n_estimators + 5
         n_estimators = list(range(lower_bound_estimators, upper_bound_estimators + 1))

         best_max_depth=340
         lower_bound_depth = best_max_depth - 5
         upper_bound_depth = best_max_depth + 5
         max_depth = list(range(lower_bound_estimators, upper_bound_estimators + 1))

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

    if sampling == 'smote':
        oversampler = SMOTE(random_state=RSEED)
        # create random grid
        random_grid = {
            'oversampler__k_neighbors': k_neighbors,
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

        elif model_name == 'XGBC':
            random_grid = {
                'eta': eta,
                'gamma': gamma,
                'max_depth': max_depth_xgbc,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                #'model__sample_weight': sample_weights, #pass directly to fit object
                'colsample_bytree': colsample_bytree,
                'n_estimators': n_estimators
            }

            tunning_pipeline = model
            #tunning_pipeline = Pipeline([
            #    ('model', model)
            #])


    #create iterator list according to the blocking strategy
    if block_strategy=='t5':
        groups = np.array(model_input['t5'])
        sfgs = StratifiedGroupKFold(n_splits=10)
        cv_iterator = list(sfgs.split(train, train_labels, groups=groups))

    if block_strategy=='lineage':
        groups = np.array(model_input['LINEAGE'])
        logo = LeaveOneGroupOut()
        cv_iterator = list(logo.split(train, train_labels, groups=groups))

    if block_strategy=='none':
        #groups = np.array(model_input['t5'])
        rskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RSEED)
        cv_iterator = list(rskf.split(train, train_labels))


    # Fitting the pipeline and obtain the best parameters using random-search
    #GridSearch
    if cv_strategy == 'random':
        model_tunning = RandomizedSearchCV(estimator=tunning_pipeline, param_distributions = random_grid, n_iter = 100, cv = cv_iterator, scoring=scoring_strategy, refit='accuracy', verbose=2, random_state=RSEED, n_jobs = -1)

    if cv_strategy == 'grid':
        model_tunning = GridSearchCV(estimator=tunning_pipeline, param_grid = random_grid, cv = cv_iterator, verbose=2, n_jobs = -1)

    if model_name == 'RF':
        model_tunning.fit(train, train_labels)

    #if model_name == 'XGBC':
    #    model_tunning.fit(train, train_labels, sample_weight=sample_weights, verbose=False)


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

    #Create an iterator to separate files in groups
    if block_strategy=='t5':
        groups = np.array(model_input['t5'])
        sgkf=StratifiedGroupKFold(n_splits=10)
        cv_iterator = list(sgkf.split(features, all_labels, groups=groups))

    if block_strategy=='lineage':
        groups = np.array(model_input['LINEAGE'])
        logo = LeaveOneGroupOut()
        cv_iterator = list(logo.split(features, all_labels, groups=lineages))

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

        if model_name =='XGBC':
            encoder = LabelEncoder()
            test_labels = encoder.fit_transform(test_labels)
            train_labels = encoder.fit_transform(train_labels)

        if model_name == 'RF':
            if sampling=='random':
                features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features,train_labels)
                model = RandomForestClassifier(n_estimators=best_params['model__n_estimators'],
                                               random_state=RSEED,
                                               max_features=best_params['model__max_features'],
                                               max_depth=best_params['model__max_depth'],
                                               n_jobs=-1, verbose=1)
            if sampling=='smote':
                features_resampled, labels_resampled = SMOTE(random_state=RSEED,k_neighbors=1).fit_resample(train_features,train_labels)
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

        if model_name == 'XGBC':

            if sampling=='none':
                sample_weights = compute_sample_weight(
                    class_weight='balanced',
                    y=train_labels
                )
                model = XGBClassifier(objective='multi:softmax',
                                      subsample=0.7,
                                      n_estimators=822,
                                      min_child_weight=1,
                                      max_depth=9,
                                      gamma=0,
                                      eta=0.178,
                                      colsample_bytree=0.8,
                                      random_state=RSEED,
                                      n_jobs=-1)

            if sampling=='random':
                features_resampled, labels_resampled = RandomOverSampler(random_state=RSEED).fit_resample(train_features,train_labels)
                model = XGBClassifier(n_estimators=822,
                                               random_state=RSEED,
                                               max_features='sqrt',
                                               n_jobs=-1, verbose=1, max_depth=220)
            if sampling=='smote':
                features_resampled, labels_resampled = SMOTE(random_state=RSEED,k_neighbors=1).fit_resample(train_features,train_labels)
                model = XGBClassifier(n_estimators=377,
                                               random_state=RSEED,
                                               max_features='sqrt',
                                               n_jobs=-1, verbose=1, max_depth=300)

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
#rf_best_params_source_1,best_cv_results_source_1,rf_model_source_1 = tune_model('RF',train_data_s1, 'none', 'none', 'random','Source_1')
#rf_best_params_source_1_up,best_cv_results_source_1_up,rf_model_source_1_up = tune_model('RF',train_data_s1, 'random', 'none', 'random','Source_1_up')
##2.2 Source_2
#rf_best_params_source_2,best_cv_results_source_2,rf_model_source_2 = tune_model('RF',train_data_s2, 'none', 'none', 'random','Source_2')
#rf_best_params_source_2_up,best_cv_results_source_2_up, rf_model_source_2_up = tune_model('RF',train_data_s2, 'random', 'none', 'random','Source_2_up')
##2.3 Source_2n1
#rf_best_params_source_2n1,best_cv_results_source_2n1,rf_model_source_2n1 = tune_model('RF',train_data_s2n1, 'none', 'none', 'random','Source_2n1')
#rf_best_params_source_2n1_up,best_cv_results_source_2n1_up,rf_model_source_2n1_up = tune_model('RF',train_data_s2n1, 'random', 'none', 'random','Source_2n1_up')
##2.4 Source_3
rf_best_params_source_3,best_cv_results_source_3,rf_model_source_3 = tune_model('RF',train_data_s3, 'none', 'none', 'random','Source_3')
rf_best_params_source_3_up,best_cv_results_source_3_up,rf_model_source_3_up = tune_model('RF',train_data_s3, 'random', 'none', 'random','Source_3_up')

#FOR TESTING - THIS WILL BE REMOVED ======
#best_cv_results_source_1.to_csv('../results/tmp/tmp_best_model_source_1.tsv', sep='\t')
#best_cv_results_source_1_up.to_csv('../results/tmp/tmp_best_model_source_1_up.tsv', sep='\t')
#=============

##2.5 concatenate and export results
#final_cv_results = pd.concat([best_cv_results_source_1, best_cv_results_source_1_up,
#                              best_cv_results_source_2, best_cv_results_source_2_up,
#                              best_cv_results_source_2n1, best_cv_results_source_2n1_up,
#                              best_cv_results_source_3, best_cv_results_source_3_up,])

#final_cv_results.to_csv('../results/01_model_training/summary_accuracy.tsv', sep='\t')



#3. CONFUSION MATRIX CONSTRUCTION
#3.1 WITH TRAINING DATA - NO UP-SAMPLING
#final_res_s1,cm_s1= build_confusion_matrix(train_data_s1, rf_best_params_source_1, 'none','none','RF', 'Source_1')
#final_res_s2,cm_s2= build_confusion_matrix(train_data_s2, rf_best_params_source_2, 'none','none','RF', 'Source_2')
#final_res_s2n1,cm_s2n1= build_confusion_matrix(train_data_s2n1, rf_best_params_source_2n1, 'none','none','RF', 'Source_2n1')
#final_res_s3,cm_s3= build_confusion_matrix(train_data_s3, rf_best_params_source_3, 'none','none','RF', 'Source_3')

##3.2 TRAINING DATA - RANDOM UP-SAMPLING
#final_res_s1_up,cm_s1_up= build_confusion_matrix(train_data_s1, rf_best_params_source_1_up, 'random','none','RF', 'Source_1_up')
#final_res_s2_up,cm_s2_up= build_confusion_matrix(train_data_s2, rf_best_params_source_2_up, 'random','none','RF', 'Source_2_up')
#final_res_s2n1_up,cm_s2n1_up= build_confusion_matrix(train_data_s2n1, rf_best_params_source_2n1_up, 'random','none','RF', 'Source_2n1_up')
#final_res_s3_up,cm_s3_up= build_confusion_matrix(train_data_s3, rf_best_params_source_3_up, 'random','none','RF', 'Source_3_up')

#4. LOAD MODELS - OPTIONAL
#rf_model_source_2_up = joblib.load("../results/02_RF_models/RF_model_Source_2_up.joblib") #this will be removed


#5. PREDICT HUMAN SAMPLES
#5.1 Models with no up-sampling
#predict_human(rf_model_source_1,test_data_s1,'Source_1')
#predict_human(rf_model_source_2,test_data_s2,'Source_2')
#predict_human(rf_model_source_2n1,test_data_s2n1,'Source_2n1')
#predict_human(rf_model_source_3,test_data_s3,'Source_3')

#5.2 Models with random up-sampling
#predict_human(rf_model_source_1_up,test_data_s1,'Source_1_up')
#predict_human(rf_model_source_2_up,test_data_s2,'Source_2_up')
#predict_human(rf_model_source_2n1_up,test_data_s2n1,'Source_2n1_up')
#predict_human(rf_model_source_3_up,test_data_s3,'Source_3_up')



#




