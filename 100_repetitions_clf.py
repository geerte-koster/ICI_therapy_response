#!/usr/bin/python3

"""
DESCRIPTION:
    Code to perform 100 repetitions of training and validation of a random forest classifier. Input data and further details
    can be specified via Linux command line.

AUTHOR:
    Geerte Koster
    email: geertekoster@hotmail.com
"""

# Import all necessary modules, libraries and functions
import argparse
import sys
import os
import pickle

# import the libraries
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import math
import scipy.stats as stats

# import the needed functions from sklearn and imblearn
import sklearn.model_selection as ms
import sklearn.feature_selection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTENC, SMOTE

# disable run time warnings (MCC) and UserWarnings (ROCPLOT)
import warnings
warnings.filterwarnings("ignore")

def import_data(data_file, drop):
    """
    Function to import data from the given data file. If the data contains the column ANONPT_ID (Mariathasan data),
    the 25 samples that are used as a withheld test set are dropped from the data. Furthermore, if drop is specified,
    the overlap of correctly or incorrectly samples (determined as overlap of 15 and 31 gene signature including
    TMB and immune phenotypes) is dropped from the data as well.

    As input, the function takes the path to the datafile and the drop parameter. As output, dataframes containing
    all features (X), the response (y) and the patient IDs (ID) is returned.
    """

    data = pd.read_csv(data_file)

    # drop 25 samples to withheld as test set, do this only if the column ANONPT_ID is present in the data
    # IDs to be dropped:
    if 'ANONPT_ID' in data.columns:
        dropID = [10050, 10215, 10093, 10212, 10181, 10055, 10239, 10071, 10209, 10103, 10325, 10321, 10189, 10076, 10058, 10119, 10115, 10187, 10224, 10279, 10049, 10371, 10299, 10277, 10084]
        test_set = data[data["ANONPT_ID"].isin(dropID)]
        data = data.drop(test_set.index, axis=0)

        if drop == "c":
            # optional, drop always correct samples (overlap 15 and 31 genes)
            drop_correct = [10122, 10259, 10007, 10139, 10274, 10155, 10157, 10162, 10296, 10174, 10191, 10196, 10201, 10335, 10341,
                            10344, 10349, 10355, 10230, 10233, 10363, 10242, 10116, 10120, 10249, 10262, 10008, 10141, 10270, 10016,
                            10024, 10154, 10287, 10033, 10291, 10164, 10036, 10043, 10300, 10173, 10171, 10052, 10180, 10182, 10054,
                            10056, 10317, 10064, 10329, 10207, 10080, 10088, 10345, 10346, 10097, 10227, 10105, 10106, 10236]
            correct_set = data[data["ANONPT_ID"].isin(drop_correct)]
            data = data.drop(correct_set.index, axis=0)
        if drop == "i":
            # optional, drop always incorrect samples (overlap 15 and 31 genes)
            drop_incorrect = [10364, 10244, 10178, 10085, 10087, 10280, 10350, 10223, 10129, 10047, 10166, 10040, 10172, 10367]
            incorrect_set = data[data["ANONPT_ID"].isin(drop_incorrect)]
            data = data.drop(incorrect_set.index, axis=0)

        # split in variable to be predicted (y), features (X) and ID
        y = data.loc[:,"binaryResponse"]
        ID = data.loc[:,"ANONPT_ID"]
        X = data.drop(["binaryResponse", "ANONPT_ID"], axis = 1)

    else:
        # split in variable to be predicted (y) and features (X) and ID
        y = data.loc[:, "binaryResponse"]
        ID = data.index
        X = data.drop(["binaryResponse"], axis=1)

    return X, y, ID

def balanced_data(X_train, y_train):
    """
    Function to balance train data by undersampling of non-responders. The function takes the complete X_train and y_train
    dataframes and returns a list of relatively balanced dataframes. If the responders form more than 22.5% of the data,
    then divide X_train non-responders in 3 parts, otherwise divide the non-responders in 4 parts. To create relatively balanced
    dataframes, each time all responders are combined with the groups of non-responders.
    """

    # specifiy the responders in the data
    Xtrain_res = X_train[y_train == 1]
    ytrain_res = y_train[y_train == 1]


    ## balance dataset into 3 or 4 different Xtrain nres groups, depending on number of nonresponders
    if len(Xtrain_res) / (len(X_train[y_train == 0]) + len(Xtrain_res)) >= 0.225:
        i1 = math.trunc(len(X_train[y_train == 0]) / 3)
        i2 = 2 * i1
        i3 = 3 * i1 + 1

        Xtrain_nres = [X_train[y_train == 0][0:i1], X_train[y_train == 0][i1:i2], X_train[y_train == 0][i2:i3]]
        ytrain_nres = [y_train.loc[list(Xtrain_nres[0].index)], y_train.loc[list(Xtrain_nres[1].index)],
                       y_train.loc[list(Xtrain_nres[2].index)]]

    else:
        i1 = math.trunc(len(X_train[y_train == 0]) / 4)
        i2 = 2 * i1
        i3 = 3 * i1
        i4 = 4 * i1 + 1

        Xtrain_nres = [X_train[y_train == 0][0:i1], X_train[y_train == 0][i1:i2], X_train[y_train == 0][i2:i3],
                       X_train[y_train == 0][i3:i4]]
        ytrain_nres = [y_train.loc[list(Xtrain_nres[0].index)], y_train.loc[list(Xtrain_nres[1].index)],
                       y_train.loc[list(Xtrain_nres[2].index)], y_train.loc[list(Xtrain_nres[3].index)]]

    # create balanced dataframes
    X_train_bal = []
    y_train_bal = []

    # append each balanced dataframe to the list
    for i in range(len(Xtrain_nres)):

        X_train_bal.append(pd.concat([Xtrain_res, Xtrain_nres[i]]))
        y_train_bal.append(pd.concat([ytrain_res, ytrain_nres[i]]))

    return X_train_bal, y_train_bal

def SMOTE_balancing(X_train, y_train, n_jobs, max_idx, randomstate):
    """
    Function to balance train data by using SMOTE. The function takes the complete X_train and y_train
    dataframes and returns a list containing the balanced X_train and y_train dataframes. Furthermore, a max_idx parameter
    is specified (automatically determined from input data) as the highest patient ID present in the data. The newly synthesized
    samples will be assigned a new patient ID that is higher than the IDs already present in the data.
    """

    # do SMOTE for combi of categorical and numeric data (SMOTENC) if immune phenotype is present, otherwise do normal SMOTE
    if "('desert',)" in X_train.columns:
        idx_cat = [X_train.columns.get_loc(i) for i in ["('desert',)", "('excluded',)", "('inflamed',)"]]
        oversample = SMOTENC(categorical_features=idx_cat, n_jobs=n_jobs, random_state=randomstate)

    else:
        oversample = SMOTE(n_jobs=n_jobs, random_state=randomstate)

    X_over, y_over = oversample.fit_resample(X_train, y_train)

    # set the original indices for the real samples and create new IDs for the synthesized samples
    start_idx = max_idx + 1
    SMOTE_idx = pd.Index(range(start_idx, (start_idx + (len(X_over)-len(X_train)))))
    ori_idx = X_train.index
    idx = ori_idx.append(SMOTE_idx)

    X_over.index = idx
    y_over.index = idx

    return [X_over], [y_over]

def mutual_information(X_train, y_train, X_test, n_features, randomstate):
    """
    Function to do feature selection based on the train data using mutual information. The required input is the X_train and y_train
    dataframes. The n_features parameter specifies the number of features the original amount is reduced to. The X_test dataframe is
    necessary as input to also reduce the number of features in that validation data to the same set as determined on the train data.

    The function return the X_train_small and X_test_small dataframes that only contain the features as determined with mutual information.
    If TMB is present in the data, this feature will be kept automatically in the data.
    """

    if 'FMOne mutation burden per MB' in X_train.columns:
        TMB = X_train.loc[:, 'FMOne mutation burden per MB']
        X_count = X_train.drop('FMOne mutation burden per MB', axis=1)

        feat_ranking = fs.mutual_info_classif(X_count, y_train, random_state=randomstate)
        # select the top features and keep only those in df
        feat_and_imp = pd.DataFrame(columns=X_count.columns)
        feat_and_imp.loc[0] = feat_ranking
        features = feat_and_imp.sort_values(by=0, axis=1, ascending=False).columns[0:n_features]

        X_train_small = X_count.loc[:,features]
        X_train_small["TMB"] = TMB

        TMB_test = X_test.loc[:, 'FMOne mutation burden per MB']
        X_test_small = X_test.loc[:, features]
        X_test_small["TMB"] = TMB_test

    else:
        feat_ranking = fs.mutual_info_classif(X_train, y_train, random_state=randomstate)
        # select the top features and keep only those in df
        feat_and_imp = pd.DataFrame(columns=X_train.columns)
        feat_and_imp.loc[0] = feat_ranking
        features = feat_and_imp.sort_values(by=0, axis=1, ascending=False).columns[0:n_features]

        X_train_small = X_train.loc[:, features]
        X_test_small = X_test.loc[:, features]

    return X_train_small, X_test_small

def RFE(X_train, y_train, X_test, n_features, n_jobs, randomstate):
    """
    Function to do feature selection based on the train data using recursive feature elimination (RFE) using a random forest classifier
    and a step size of 500. The required input is the X_train and y_train dataframes, the n_features parameter specifies the number of
    features the original amount is reduced to. The X_test dataframe is necessary as input to also reduce the number of features in
    that validation data to the same set as determined on the train data.

    The function returns the X_train_small and X_test_small dataframes that only contain the features as determined with RFE.
    If TMB is present in the data, this feature will be kept automatically in the data.
    """

    if 'FMOne mutation burden per MB' in X_train.columns:
        TMB = X_train.loc[:, 'FMOne mutation burden per MB']
        X_count = X_train.drop('FMOne mutation burden per MB', axis=1)

        clf = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced_subsample", random_state=randomstate)
        feat_selector = fs.RFE(clf, n_features_to_select=n_features, step=500)
        feat_selector = feat_selector.fit(X_count, y_train)

        idx = feat_selector.get_support()
        X_train_small = X_count.loc[:, idx]
        X_train_small["TMB"] = TMB

        features = X_train_small.columns

        TMB_test = X_test.loc[:, 'FMOne mutation burden per MB']
        X_test_small = X_test.loc[:, features]
        X_test_small["TMB"] = TMB_test

    else:
        clf = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced_subsample", random_state=randomstate)
        feat_selector = fs.RFE(clf, n_features_to_select=n_features, step=500)
        feat_selector = feat_selector.fit(X_train, y_train)

        idx = feat_selector.get_support()
        X_train_small = X_train.loc[:, idx]

        features = X_train.columns[idx]

        X_test_small = X_test.loc[:, features]

    return X_train_small, X_test_small

def PCA_transformation(X_train, X_test, n_features, randomstate):
    """
    Function to do reduce the number of input features by doing PCA. The required input is the X_train dataframe,
    the n_features parameter specifies the number of principles components (PCs) the original data is projected on.
    After determining these PCs, the dimensions of the X_test dataframe are reduced by projecting the samples on the
    same principal components.

    The function returns the df_trainPCA and df_testPCA dataframes that only contain the features as reduced by PCA.
    """

    # do PCA on train data
    # do a z-score transformation of the count features
    X_count_tr = X_train.apply(stats.zscore)
    # remove the columns that contain NaNs
    X_count_tr= X_count_tr.dropna(axis=1, how='all')

    # fit and transform the X_train data
    pca = PCA(n_components=n_features, random_state=randomstate)
    X_train_pca = pca.fit_transform(X_count_tr)

    # apply same transformation (PCA fitted on train) to test data
    PCA_train = pca.fit(X_count_tr)
    # do a z-score transformation of the count features

    X_count_te = X_test.apply(stats.zscore)
    # remove the columns that contain NaNs
    X_count_te = X_count_te.dropna(axis=1, how='all')

    X_test_pca = PCA_train.transform(X_count_te)

    # PC names
    PC_col = []
    for i in list(range(1, n_features + 1)):
        PC_col.append("PC" + str(i))

    # create a pandas df from the transformed data
    df_trainPCA = pd.DataFrame(X_train_pca)
    df_trainPCA.columns = PC_col
    df_trainPCA.index = X_train.index

    df_testPCA = pd.DataFrame(X_test_pca)
    df_testPCA.columns = PC_col
    df_testPCA.index = X_test.index

    return df_trainPCA, df_testPCA

def PCA_TMB(X_train, X_test, n_features, randomstate):
    """
    Almost same function as PCA_transformation, only with this function, the TMB is not used as input for the PCA but is
    added again once the dimensionality reduction is done. The TMB is therefore not modified but it is present in the X_test data.
    This function should be used if TMB is present as feature in the input data.
    """

    # do PCA on train data without TMB
    TMB_tr = X_train.loc[:,'FMOne mutation burden per MB']
    X_count_tr = X_train.drop('FMOne mutation burden per MB', axis=1)
    # do a z-score transformation of the count features
    X_count_tr = X_count_tr.apply(stats.zscore)
    # remove the columns that contain NaNs
    X_count_tr= X_count_tr.dropna(axis=1, how='all')

    # fit and transform the X_train data
    pca = PCA(n_components=n_features, random_state=randomstate)
    X_train_pca = pca.fit_transform(X_count_tr)

    # apply same transformation (PCA fitted on train) to test data
    PCA_train = pca.fit(X_count_tr)
    TMB_te = X_test.loc[:, 'FMOne mutation burden per MB']
    X_count_te = X_test.drop('FMOne mutation burden per MB', axis=1)
    # do a z-score transformation of the count features
    X_count_te = X_count_te.apply(stats.zscore)
    # remove the columns that contain NaNs
    X_count_te = X_count_te.dropna(axis=1, how='all')
    X_test_pca = PCA_train.transform(X_count_te)

    # PC names
    PC_col = []
    for i in list(range(1, n_features + 1)):
        PC_col.append("PC" + str(i))

    # create a pandas df from the transformed data
    df_trainPCA = pd.DataFrame(X_train_pca)
    df_trainPCA.columns = PC_col
    df_trainPCA.index = TMB_tr.index
    X_train_PCA = pd.concat([df_trainPCA, TMB_tr], axis=1)

    df_testPCA = pd.DataFrame(X_test_pca)
    df_testPCA.columns = PC_col
    df_testPCA.index = TMB_te.index
    X_test_PCA = pd.concat([df_testPCA, TMB_te], axis=1)

    return X_train_PCA, X_test_PCA

def cross_validation(p_grid, X_train_small, y_train, clf, X_test, y_test, ID, n_jobs):
    """
    Function to do 5-fold cross validation to determine the best hyper parameter settings for the random forest. To find settings of
    the classifier that has the best performance, the GridSearchCV function of sklearn is used. This function needs a specified grid,
    which is provided with p_grid, a classifier to test the performances of, provided via clf, and a scoring function. As a scoring
    function, the Matthews correlation coefficient is used.

    This function returns the following:
     Best classifier: best_clf
     The hyper parameter settings of the best classifier: hyper_ps
     Performance of best classifier on the train and validation data: train_mcc, test_mcc
     The highest cross validation score as calculated during GridSearchCV: CV_mmc
     The out-of-bag (oob) score of best classifier: oob
     Permutation importance of all input features: feat_imps
     Pandas files with true y, predicted y and patient ID for both train and validation data: train_df, test_df
     Pandas files with true y, class probabilities of predictions and patient ID for both train and validation data: train_proba_df, test_proba_df

    """

    # do 5 fold cross validation to select best hyper parameter settings
    MCC_scorer = make_scorer(matthews_corrcoef)

    CV_clf = ms.GridSearchCV(estimator=clf, param_grid=p_grid, scoring=MCC_scorer, cv=5, n_jobs=n_jobs)
    CV_clf.fit(X_train_small, y_train)

    # select classifier with best hyper parameter settings
    best_clf = CV_clf.best_estimator_

    # get the hyperparameter settigns of best clf
    hyper_ps = CV_clf.best_params_

    # calculate MCC over training data and over validation data, calculate accuracy over validation data
    y_pred_train = best_clf.predict(X_train_small)
    y_pred_test = best_clf.predict(X_test)
    train_mcc = matthews_corrcoef(y_train, y_pred_train)
    test_mcc = matthews_corrcoef(y_test, y_pred_test)
    CV_mmc =  CV_clf.best_score_
    acc = accuracy_score(y_test, y_pred_test)

    # Calculate oob score if this is provided by the given classifier. If not provided, oob score is set to 0.
    try:
        oob = best_clf.oob_score_
    except:
        oob = 0

    # create pandas file with true y, predicted y and patient ID -- for both train and validation data
    # train data
    train_df = pd.merge(y_train, ID.loc[X_train_small.index], left_index=True, right_index=True)
    train_df["y_pred"] = y_pred_train
    corr_pred_tr = train_df["y_pred"] == train_df["binaryResponse"]
    train_df["correct_prediction"] = corr_pred_tr

    # validation data
    test_df = pd.merge(y_test, ID.loc[X_test.index], left_index=True, right_index=True)
    test_df["y_pred"] = y_pred_test
    corr_pred_tr = test_df["y_pred"] == test_df["binaryResponse"]
    test_df["correct_prediction"] = corr_pred_tr

    # create pandas file with true y, class probabilites of prediction and patient ID -- for both train and test data
    # train data
    train_proba_df = pd.merge(y_train, ID.loc[X_train_small.index], left_index=True, right_index=True)
    train_proba_df["y_pred0"] = [item[0] for item in best_clf.predict_proba(X_train_small)]
    train_proba_df["y_pred1"] = [item[1] for item in best_clf.predict_proba(X_train_small)]

    # validation data
    test_proba_df = pd.merge(y_test, ID.loc[X_test.index], left_index=True, right_index=True)
    test_proba_df["y_pred0"] = [item[0] for item in best_clf.predict_proba(X_test)]
    test_proba_df["y_pred1"] = [item[1] for item in best_clf.predict_proba(X_test)]

    # get permutation feature importance
    feat_imps = permutation_importance(best_clf, X_test, y_test)

    return best_clf, hyper_ps, train_mcc, test_mcc, CV_mmc, oob, acc, feat_imps, train_df, test_df, train_proba_df, test_proba_df

def save_results(directory, l_train_MCC, l_test_MCC, l_AUC, l_CV_MCC, l_OOB, l_acc, ID_train, ID_test, ID_train_proba, ID_test_proba,
                 df_features, d_classifiers, d_hyper_ps, n_rep, split_df):
    """
    Function to save the following results in directory (as specified by input directory):
        - csv file containing all MCCs, accuracies and AUCs calculated over the different runs, saved as "scores_100.csv"
        - boxplot made from the "scores_100" dataframe, saved as "box_plot_100_runs.png"
        - text file containing the mean/stdev of the scores in the "scores_100" df, saved as "scores_100.txt"
        - csv file made from the feature importances over all runs, saved as "feature_importances.csv"
        - boxplot containing the top 10 features with highest mean feature importances, saved as "box_plot_top10_features.png"
        - csv file made from the pd dataframes containing the predicted classes for each of the samples over all runs, one
        csv file for the predictions in the train data "predicted_classes_train.csv" and one for the predictions in the
        validation data "predicted_classes_test.csv"
            - from these files, also distribution plots are made showing the fraction of times a sample is predicted correctly
            or incorrectly in train or validation data when considering all runs, saved as:
                "fraction_correct_predicted_train.png" and "fraction_correct_predicted_test.png"
            - this is also done for the predicted probabilities, resulting in the files "predicted_probabilities_train.csv" and
            "predicted_probabilities_test.csv" and in the plots that contain the distribution of average prediction probability
            of each sample when it is part of the train or validation data, saved as:
            "predicted_probabilites_0_train.png", "predicted_probabilites_1_train.png", "predicted_probabilites_0_test.png"
            and "predicted_probabilites_1_test.png"
       - dictionaries containing classifiers and hyperp settings are saved as pkl files: classifiers.pkl and hyperp_settings.pkl
       - csv files containing the ordered feature importances and feature counts, indicating the number of times a feature
       is used over all runs, saved as: "feature_counts.csv" and "features_ordered.csv"
       - csv file containing which samples are used as train and validation samples for each run, saved as: "split_test_train.csv"
    """

    # create directory for the run
    os.mkdir(directory)

    # save pandas dataframe containing the train MCCs, test MCCs and AUCs for the 100 runs
    scores_100 = pd.DataFrame(data={"train_MCC":l_train_MCC, "test_MCC":l_test_MCC, "CV score":l_CV_MCC, "OOB_score":l_OOB, "AUC":l_AUC,
                                    "Acc":l_acc})

    # make boxplots for the MCCs and AUCs
    plt.figure()
    scores_100.boxplot()
    plt.savefig("{}/{}".format(directory, "box_plot_100_runs.png"))

    # save a file with the average scores and standard deviations
    scores_100.loc["mean"] = scores_100.mean()
    scores_100.loc["st_dev"] = scores_100.std()
    scores_100.to_csv("{}/scores_100.csv".format(directory))

    # save a separate file with means and stdvs
    scores_100.loc[["mean", "st_dev"]].round(5).to_csv("{}/scores_100.txt".format(directory), sep= "\t")

    # save pandas dataframe with all feature importances
    df_features.loc["mean"] = df_features.mean()
    df_features.loc["st_dev"] = df_features.std()
    df_features.to_csv("{}/feature_importances.csv".format(directory))

    # sort dataframe based on mean and plot the top 10 features as boxplot
    df_features_plot = df_features.sort_values(by="mean", axis=1, ascending=False).drop(["mean", "st_dev"], axis=0)

    if len(df_features_plot.columns) <= 10:
        features_plot = df_features_plot
    else:
        features_plot = df_features_plot.iloc[:, :10]

    plt.figure()
    features_plot.boxplot(rot=90)
    plt.savefig("{}/box_plot_top10_features.png".format(directory))

    # save pd dataframe containing the predicted classes for the 100 runs
    ID_train.to_csv("{}/predicted_classes_train.csv".format(directory))
    ID_test.to_csv("{}/predicted_classes_test.csv".format(directory))

    # create plots of the predictions and save them
    # train data
    ID_train["total_pred"] = ID_train.iloc[:, 3:].notnull().sum(axis=1)
    ID_train["frac_pred"] = ID_train["total_correct"] / ID_train["total_pred"]

    fig = sns.displot(x=ID_train.loc[:, "frac_pred"], hue=ID_train["binaryResponse"], binwidth=0.05, multiple="stack")
    fig.savefig(("{}/fraction_correct_predicted_train.png").format(directory))

    # test data
    ID_test["total_pred"] = ID_test.iloc[:, 3:].notnull().sum(axis=1)
    ID_test["frac_pred"] = ID_test["total_correct"] / ID_test["total_pred"]

    fig = sns.displot(x=ID_test.loc[:, "frac_pred"], hue=ID_test["binaryResponse"], binwidth=0.05, multiple="stack")
    fig.savefig(("{}/fraction_correct_predicted_test.png").format(directory))

    # save files with the prediction probabilities
    ID_test_proba["total_pred"] = ID_test_proba.iloc[:, 3:].notnull().sum(axis=1)
    ID_test_proba["mean_0"] = ID_test_proba.filter(like="y_pred0").mean(axis=1, skipna=True)
    ID_test_proba["mean_1"] = ID_test_proba.filter(like="y_pred1").mean(axis=1, skipna=True)

    ID_train_proba["total_pred"] = ID_train_proba.iloc[:, 3:].notnull().sum(axis=1)
    ID_train_proba["mean_0"] = ID_train_proba.filter(like="y_pred0").mean(axis=1, skipna=True)
    ID_train_proba["mean_1"] = ID_train_proba.filter(like="y_pred1").mean(axis=1, skipna=True)

    ID_train_proba.to_csv("{}/predicted_probabilities_train.csv".format(directory))
    ID_test_proba.to_csv("{}/predicted_probabilities_test.csv".format(directory))

    # make distribution plots for the prediction probabilities
    fig = sns.displot(x=ID_train_proba.loc[:, "mean_0"], hue=ID_train_proba["binaryResponse"], binwidth=0.05, multiple="stack")
    fig.savefig(("{}/predicted_probabilites_0_train.png").format(directory))
    fig = sns.displot(x=ID_train_proba.loc[:, "mean_1"], hue=ID_train_proba["binaryResponse"], binwidth=0.05, multiple="stack")
    fig.savefig(("{}/predicted_probabilites_1_train.png").format(directory))

    fig = sns.displot(x=ID_test_proba.loc[:, "mean_0"], hue=ID_test_proba["binaryResponse"], binwidth=0.05, multiple="stack")
    fig.savefig(("{}/predicted_probabilites_0_test.png").format(directory))
    fig = sns.displot(x=ID_test_proba.loc[:, "mean_1"], hue=ID_test_proba["binaryResponse"], binwidth=0.05, multiple="stack")
    fig.savefig(("{}/predicted_probabilites_1_test.png").format(directory))

    # save dictionaries containing classifiers and hyperp settings as pkl files
    with open("{}/{}".format(directory, 'classifiers.pkl'), 'wb') as handle:
        pickle.dump(d_classifiers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("{}/{}".format(directory, 'hyperp_settings.pkl'), 'wb') as handle:
        pickle.dump(d_hyper_ps, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the ordered features per run as wel
    # get ordered feature importances for each of the 100 runs
    feat_ordered = {}
    feat_counts = {}

    for i in range(n_rep):
        imp_feat = df_features.iloc[i, :][df_features.iloc[i, :] > 0]
        ordered = imp_feat.sort_values(ascending=False)
        feat_ordered[i] = list(ordered.index.values)

        for i in ordered.index.values:
            if i in feat_counts.keys():
                feat_counts[i] +=1
            else:
                feat_counts[i] = 1

    # save feature counts to pandas df and to .csv file
    pd.DataFrame.from_dict(feat_counts, orient='index').to_csv("{}/feature_counts.csv".format(directory))

    # write the dictionary feat_ordered to a csv file (on each row the key and value)
    import csv
    # open file for writing, "w" is writing
    with open("{}/features_ordered.csv".format(directory), "w") as outfile:
        writer = csv.writer(outfile)
        for key, val in feat_ordered.items():
            # write value (list of features) to output file
            writer.writerow([key, val])

    # save the dataframe containing which samples are used as test and train for each run
    split_df.to_csv("{}/split_test_train.csv".format(directory))

def main():
    # different arguments that can be supplied via the command line
    parser = argparse.ArgumentParser(description='Classification of responders and non-responders')
    parser.add_argument('-i', '--input', required=True, dest='data_file',
                        metavar='data.csv', type=str,
                        help='Path to the data file')
    parser.add_argument('-p', '--directory', required=True, dest='directory',
                        type=str, help='directory for output files')
    parser.add_argument('-g', '--grid', required=True, dest='grid',
                        metavar='grid.pkl', type=str,
                        help='File containing hyper parameter grid for CV')
    parser.add_argument('-c', '--clf', required=True, dest='clf',
                        type=str,
                        help='Classifier, option are Random Forest (RF) or Logistic Regression (LR)')
    parser.add_argument('-j', '--n_jobs', required=False, default=-1, dest='number_jobs',
                        type=int, help='Number of jobs to run in parallel. Default = -1')
    parser.add_argument('-n', '--number', required=False, default=0, dest='number_features',
                        type=int,
                        help='Number of features to select. By default, no feature selection is done')
    parser.add_argument('-f', '--feature_selection', required=False, default=0, dest='feat_sel',
                        type=str,
                        help='Feature selection method. By default, no feature selection is done. Options are'
                             'MI, RFE, PCA and PCA+')
    parser.add_argument('-bal', '--balancing', required=False, default='y', dest='balancing',
                        type=str,
                        help='Specify this parameter as n for no balancing, and SMOTE for using SMOTE to '
                             'oversample the minority class. By default, undersampling is done')
    parser.add_argument('-per', '--permutation', required=False, default='n', dest='permutation',
                        type=str,
                        help='Specify this parameter as y if class labels should be permutated. By default,'
                             'no permutation is done')
    parser.add_argument('-rep', '--repetitions', required=False, default=100, dest='n_rep',
                        type=int,
                        help='Specify number of repetitions. By default, 100 repetitions are done.')
    parser.add_argument('-dr', '--drop', required=False, default="n", dest='drop',
                        type=str,
                        help='Specify if specific samples need to be dropped. Default is no dropping of samples. If c, '
                             'samples always correct are dropped, if i, samples always incorrect.')

    # Parse options
    args = parser.parse_args()

    if args.data_file is None:
        sys.exit('Input data is missing.')

    if args.directory is None:
        sys.exit('Specification of output directory is missing.')

    if args.number_features is None:
        sys.exit('Specification of number of features is missing.')

    if args.grid is None:
        sys.exit('Grid for hyper parameter search is missing.')

    # load all arguments from command line into variables
    data_file = args.data_file
    n_jobs = args.number_jobs
    n_features = args.number_features
    directory = args.directory
    clf = args.clf
    feat_sel = args.feat_sel
    balancing = args.balancing
    permutation = args.permutation
    n_rep = args.n_rep
    drop = args.drop

    # hyperparameters to optimize
    with open(args.grid, 'rb') as handle:
        p_grid = pickle.load(handle)

    # create lists to save the MCCs, AUCs, oob socres, accuracies and cross validation MCCs for all runs
    l_train_MCC = []
    l_test_MCC = []
    l_OOB = []
    l_AUC = []
    l_CV_MCC = []
    l_acc = []

    # load the data and extract X, y and patient IDs (25 IDs are dropped as test_set)
    X, y, ID = import_data(data_file, drop)

    # find the highest ID present in the dataframe, this is used if SMOTE balancing is done
    max_idx = X.index.max()

    # if balancing is SMOTE, create new ID dataframe with additional samples
    if balancing == "SMOTE":
        # check the difference in responders and non-responders in test data for the 1/5 4/5 split
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=45, stratify=y, random_state=1)
        imbalance = sum(y_train == 0) - sum(y_train == 1)

        for pt_id in range(max_idx + 1, max_idx + 1 + imbalance):
            ID.loc[pt_id] = pt_id
    else:
        ID = ID

    # create df to save the split of train / test data per run
    split_df = pd.DataFrame(data=ID)

    # set up two dataframes to save the predictions for the different samples (train / test)
    ID_test = pd.merge(ID, y, how="left", left_index=True, right_index=True)
    ID_test["total_correct"] = 0

    ID_train = pd.merge(ID, y, how="left", left_index=True, right_index=True)
    ID_train["total_correct"] = 0

    # create two dfs to save the probabilities of the classifications
    ID_test_proba = pd.merge(ID, y, left_index=True, right_index=True)
    ID_train_proba = pd.merge(ID, y, left_index=True, right_index=True)

    # create directories to save classifiers and hyper parameter settings
    d_classifiers = {}
    d_hyper_ps = {}

    # repeat training and validation of the model n_rep times (default is 100)
    for i in list(range(n_rep)):
        # split the data in train and validation sets
        X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=45, stratify=y, random_state=i)

        # do balancing (only skip when specified):
        if balancing == 'y':
            X_train_b, y_train_b = balanced_data(X_train, y_train)

        elif balancing == "SMOTE":
            X_train_b, y_train_b = SMOTE_balancing(X_train, y_train, n_jobs, max_idx, randomstate=i)
            # update ID_train with the SMOTE classes
            for pt_id in range(max_idx + 1, max_idx + 1 + imbalance):
                ID_train.loc[pt_id, "ANONPT_ID"] = pt_id
                ID_train.loc[pt_id,"binaryResponse"] = y_train_b[0].loc[pt_id]

        else:
            X_train_b, y_train_b = [X_train], [y_train]

        # do once if no balancing is done, otherwise repeat as many times as the number of subsets of non-responders
        for j in range(len(X_train_b)):
            X_train = X_train_b[j]
            y_train = y_train_b[j]

            # save which IDs are used as test and which as train
            split_df["run_{}.{}".format(i,j)] = "X"
            split_df.loc[X_train.index, "run_{}.{}".format(i,j)] = "train"
            split_df.loc[X_test.index, "run_{}.{}".format(i,j)] = "test"

            if permutation == 'y':
                # permute the class labels if this is specified
                y_train = shuffle(y_train, random_state=(j+(i*len(X_train_b))))
            else: # default
                y_train = y_train

            # do feature selection if n_features != 0
            if n_features == 0:
                # do no feature selection (default)
                X_train_small = X_train
                X_test_rep = X_test
                if i == 0 and j == 0:
                    df_features = pd.DataFrame(columns=X_train.columns)

            else:
                # check which feature selection method is specified and pick that one
                if feat_sel == "MI":
                    X_train_small, X_test_rep = mutual_information(X_train, y_train, X_test, n_features, randomstate=(j+(i*len(X_train_b))))
                    # for the first run, create a dataframe to save feature importance
                    if i == 0 and j == 0:
                        df_features = pd.DataFrame(columns=X_train_small.columns)

                elif feat_sel == "RFE":
                    X_train_small, X_test_rep = RFE(X_train, y_train, X_test, n_features, n_jobs, randomstate=(j+(i*len(X_train_b))))
                    # for the first run, create a dataframe to save feature importance
                    if i == 0 and j == 0:
                        df_features = pd.DataFrame(columns=X_train_small.columns)

                elif feat_sel == "PCA":
                    X_train_small, X_test_rep = PCA_transformation(X_train, X_test, n_features, randomstate=(j+(i*len(X_train_b))))
                    # for the first run, create a dataframe to save feature importance
                    if i == 0 and j == 0:
                        df_features = pd.DataFrame(columns=X_train_small.columns)

                elif feat_sel == "PCA+":
                    X_train_small, X_test_rep = PCA_TMB(X_train, X_test, n_features, randomstate=(j+(i*len(X_train_b))))
                    # for the first run, create a dataframe to save feature importance
                    if i == 0 and j == 0:
                        df_features = pd.DataFrame(columns=X_train_small.columns)

                # if no method is specified, do no feature selection
                else:
                    X_train_small = X_train
                    X_test_rep = X_test
                    if i == 0 and j == 0:
                        df_features = pd.DataFrame(columns=X_train.columns)

            # do cross validation
            # define classifier for cross validation
            if clf == 'RF':
                cvclf = RandomForestClassifier(class_weight="balanced_subsample", random_state=(j+(i*len(X_train_b))), oob_score=True)
            else:
                cvclf = LogisticRegression(solver= 'liblinear', random_state=(j+(i*len(X_train_b))))

            best_clf, hyper_ps, train_mcc, test_mcc, CV_mcc, oob, acc, feat_imps, train_df, test_df, train_proba_df, test_proba_df =\
                cross_validation(p_grid, X_train_small, y_train, cvclf, X_test_rep, y_test, ID, n_jobs)

            # save results of the cross validation for specific run
            l_train_MCC.append(train_mcc)
            l_test_MCC.append(test_mcc)
            l_CV_MCC.append(CV_mcc)
            l_OOB.append(oob)
            l_acc.append(acc)
            l_AUC.append(roc_auc_score(y_test, best_clf.predict(X_test_rep)))

            # save classifier + hyperparameter settings
            d_classifiers[(j+(i*len(X_train_b)))] = best_clf
            d_hyper_ps[(j+(i*len(X_train_b)))] = hyper_ps

            # add the predicted scores to the dataframes
            ID_train = ID_train.join(train_df.loc[:,"y_pred"], rsuffix=(str(i)+"."+str(j)))
            ID_test = ID_test.join(test_df.loc[:, "y_pred"], rsuffix=(str(i) + "." + str(j)))

            ID_train_proba = ID_train_proba.join(train_proba_df.loc[:, ["y_pred0", "y_pred1"]], rsuffix=("_" + str(i) + "." + str(j)))
            ID_test_proba = ID_test_proba.join(test_proba_df.loc[:, ["y_pred0", "y_pred1"]], rsuffix=("_" + str(i) + "." + str(j)))

            list_ID_dfs = [ID_train, ID_test]

            for x, ID_df in enumerate([train_df, test_df]):
                for idx in ID_df.index:
                    if ID_df.loc[idx,"correct_prediction"] == True:
                        list_ID_dfs[x].loc[idx,"total_correct"] += 1
                    else:
                        list_ID_dfs[x].loc[idx, "total_correct"] += 0

            # add feature importances to dataframe
            df_features.loc[(j+(i*len(X_train_b))),:] = feat_imps["importances_mean"]

    save_results(directory, l_train_MCC, l_test_MCC, l_AUC, l_CV_MCC, l_OOB, l_acc, ID_train, ID_test, ID_train_proba, ID_test_proba,
                 df_features, d_classifiers, d_hyper_ps, n_rep, split_df)

if __name__ == '__main__':
    main()