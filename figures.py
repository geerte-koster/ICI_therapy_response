"""
DESCRIPTION:
    Code to reproduce figure 1, 3, 4, 5 and 7 from the report: .... Moreover, with the script of figure 7, the validation of the
    best classifiers as produced by the script '100_repetitions_clf' on the witheld Mariathasan data and Riaz data
    is done.

AUTHOR:
    Geerte Koster
    email: geertekoster@hotmail.com
"""
# import necessary modules and functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.stats import wilcoxon, ranksums, chisquare
from statsmodels.stats.multitest import fdrcorrection
from statistics import mode
from sklearn.metrics import matthews_corrcoef, RocCurveDisplay, ConfusionMatrixDisplay
from venn import venn

# list of functions needed to create different subplots used inside the specified figure functions
def create_boxplot(path_dict, palette, output, figure):
    # create one df with all train MCC, test MCC and AUC values
    dict_scores = {}
    list_scores = []

    for method in path_dict.keys():
        # create df with the train MCC, test MCC and AUC values for each of the runs
        path = path_dict[method]
        scores = pd.read_csv(f"{path}/scores_100.csv", index_col=0)
        scores = scores.drop(["mean", "st_dev"], axis=0)
        scores = scores.drop(["CV score", "OOB_score"], axis=1)
        scores["Run"] = method

        # save score df in list_scores
        list_scores.append(scores)
        dict_scores[method] = scores

    all_scores = pd.concat(list_scores)
    scores_t = pd.melt(all_scores, id_vars="Run", value_vars=["train_MCC", "test_MCC", "AUC", "Acc"], var_name='Score',
                       value_name='Value')

    # create boxplot
    flierprops = dict(marker='o', markerfacecolor='None', markersize=3, markeredgecolor='black')
    sns.catplot(x="Score", y="Value", hue="Run", data=scores_t, kind="box",
                palette=palette,
                flierprops=flierprops)

    plt.savefig(f'figures/{figure}/{output}.eps', format='eps')

    return dict_scores
def sign_df(path_dict, dict_scores):
    # check if differences are significant
    # validation MCC and AUC
    sign_df_MCC = pd.DataFrame(columns=path_dict.keys(), index=path_dict.keys())
    sign_df_AUC = pd.DataFrame(columns=path_dict.keys(), index=path_dict.keys())
    sign_df_acc = pd.DataFrame(columns=path_dict.keys(), index=path_dict.keys())

    for method_1 in sign_df_MCC.columns:
        for method_2 in sign_df_MCC.columns:
            if method_1 != method_2:
                w_MCC, p_MCC = wilcoxon(dict_scores[method_1]["test_MCC"], dict_scores[method_2]["test_MCC"])
                # multiply the p value by the number of comparisons made
                p_adj_MCC = p_MCC * (len(sign_df_MCC.columns) - 1)
                sign_df_MCC.loc[method_1, method_2] = p_adj_MCC

                w_AUC, p_AUC = wilcoxon(dict_scores[method_1]["AUC"], dict_scores[method_2]["AUC"])
                p_adj_AUC = p_AUC * (len(sign_df_MCC.columns) - 1)
                sign_df_AUC.loc[method_1, method_2] = p_adj_AUC

                w_acc, p_acc = wilcoxon(dict_scores[method_1]["Acc"], dict_scores[method_2]["Acc"])
                p_adj_acc = p_acc * (len(sign_df_MCC.columns) - 1)
                sign_df_acc.loc[method_1, method_2] = p_adj_acc
            else:
                sign_df_MCC.loc[method_1, method_2] = "X"
                sign_df_AUC.loc[method_1, method_2] = "X"
                sign_df_acc.loc[method_1, method_2] = "X"

    return sign_df_MCC, sign_df_AUC, sign_df_acc
def remove_X(x):
    for i, a in enumerate(x):
        if "X" in a:
            x[i] = int(a.replace("X", ""))
        else:
            x[i] = a
    return x
def entrez_to_symbol(x, gene_an):
    for i, a in enumerate(x):
        if a in gene_an.entrez_id.values:
            x[i] = list(gene_an['symbol'][gene_an['entrez_id'] == a])[0]
        else:
            x[i] = a
    return x
def create_heatmap(df, gene_an, ID_incorrect, ID_correct):
    df = df.copy()
    df_response = df.loc[:, 'binaryResponse']
    df_ID = df.loc[:, 'ANONPT_ID']

    if "('desert',)" in df.columns:
        df_immphen = df.loc[:, ["('desert',)", "('excluded',)", "('inflamed',)"]].idxmax(1)
        df_gex = df.drop(['ANONPT_ID', 'binaryResponse', "('desert',)", "('excluded',)", "('inflamed',)"], axis=1)
        color_dict = {1: "cornflowerblue", 0: "crimson", "('desert',)": 'lemonchiffon', "('excluded',)": 'khaki',
                      "('inflamed',)": 'gold', "correct":'limegreen', "incorrect":"red"}
        color_df = pd.DataFrame()
        color_df["Response"] = df_response
        color_df["Immune phenotype"] = df_immphen

    if "desert" in df.columns:
        df_immphen = df.loc[:, ["desert", "excluded", "inflamed"]].idxmax(1)
        df_gex = df.drop(['ANONPT_ID', 'binaryResponse', "desert", "excluded", "inflamed"], axis=1)
        color_dict = {1: "cornflowerblue", 0: "crimson", "desert": 'lemonchiffon', "excluded": 'khaki',
                      "inflamed": 'gold', "correct":'limegreen', "incorrect":"red"}
        color_df = pd.DataFrame()
        color_df["Response"] = df_response
        color_df["Immune phenotype"] = df_immphen

    else:
        df_gex = df.drop(['ANONPT_ID', 'binaryResponse'], axis=1)
        color_dict = {1: "cornflowerblue", 0: "crimson", "correct":'limegreen', "incorrect":"red"}
        color_df = pd.DataFrame()
        color_df["Response"] = df_response

    gene_symbol = entrez_to_symbol(remove_X(list(df_gex.columns)), gene_an)
    df_gex.columns = gene_symbol
    df_gex.index = df_ID

    # create heatmap with sns
    color_df.index = df_ID
    color_df.insert(1, "Prediction", "correct")
    color_df.loc[ID_incorrect,"Prediction"] = "incorrect"

    # add discretised TMB to the color df
    TMB = df_gex["FMOne mutation burden per MB"]
    df_gex = df_gex.drop("FMOne mutation burden per MB", axis=1)
    TMB_labels = ['thistle', 'mediumorchid', 'darkorchid', 'indigo']
    TMB['TMB'] = pd.qcut(TMB, q=4, labels=TMB_labels)
    color_df["TMB"] = TMB['TMB']

    row_colors = color_df.replace(color_dict)

    IDs = ID_correct+ID_incorrect

    sns.clustermap(df_gex.loc[IDs,:], row_colors=row_colors.loc[IDs,:], cmap='viridis', z_score=1)
    plt.savefig(("figures/figure4/heatmap_def_{}.eps").format(len(df.columns)-6), format='eps')
def gex_plots(df, overlap, genes, gene_an):
    # add a column which indicates if a sample is always predicted correctly/incorrectly or if that is NA
    df = df.copy()
    df.loc[:,"Group"] = 'Responder'
    df.loc[df["binaryResponse"] == 0,"Group"] = "Non-responder"

    for key in overlap.keys():
        idx = df["ANONPT_ID"].isin(overlap[key])
        df.loc[idx,"Group"] = key

    df_boxplot = df.loc[:, df.columns.str.startswith('X')]
    df_boxplot.columns = entrez_to_symbol(remove_X(list(df_boxplot.columns)), gene_an)
    df_boxplot["Group"] = df["Group"]

    df_sign = pd.DataFrame(columns=genes)

    # set the properties for outlier values
    flierprops = dict(marker='o', markerfacecolor='None', markersize=5, markeredgecolor='black')

    for column in genes:

        plt.figure()
        sns.boxplot(x=list(df_boxplot["Group"]), y=list(df_boxplot.loc[:, column]), flierprops=flierprops, fliersize=0.1,
                      order=["Responder", "Non-responder", "res_cor", "nres_cor", "res_incor", "nres_incor"],
                      palette=["cornflowerblue", "crimson", "royalblue", "red", "lightsteelblue", "pink"]).set_title(column)

        sns.swarmplot(x=list(df_boxplot["Group"]), y=list(df_boxplot.loc[:,column]),
                    order=["Responder", "Non-responder", "res_cor", "nres_cor", "res_incor", "nres_incor"],
                    color=".2")

        plt.savefig(("figures/figure4/gex_def_{}.eps").format(column), format='eps')

        ## calculate significance between the three groups
        s_remaining, p_remaining = ranksums(df_boxplot.loc[df_boxplot["Group"]=="Responder", column],
                                      df_boxplot.loc[df_boxplot["Group"]=="Non-responder", column])
        df_sign.loc["remaining",column] = p_remaining

        s_cor, p_cor = ranksums(df_boxplot.loc[df_boxplot["Group"] == "res_cor", column],
                                      df_boxplot.loc[df_boxplot["Group"] == "nres_cor", column])
        df_sign.loc["correct", column] = p_cor

        s_incor, p_incor = ranksums(df_boxplot.loc[df_boxplot["Group"] == "res_incor", column],
                                      df_boxplot.loc[df_boxplot["Group"] == "nres_incor", column])
        df_sign.loc["incorrect", column] = p_incor

    return df_sign

# functions to create figures 1-7 and suppl figure 1-4
def figure_1():
    # import Mariathasan data
    gene_ex = pd.read_csv("csv_files/met_ID/count_vst_ID.csv")
    clin_feat = pd.read_csv("csv_files/clinical.csv")

    # we only consider the patients that have a binaryResponse
    response = clin_feat.loc[:,['binaryResponse','ANONPT_ID']].dropna()
    ID_res = list(response["ANONPT_ID"])

    IP = clin_feat.loc[:,['Immune phenotype','ANONPT_ID', 'binaryResponse']].dropna()
    IP = IP[IP["ANONPT_ID"].isin(ID_res)]
    TMB = pd.read_csv("csv_files/met_ID/TMB_ID.csv")

    data = {"Gene expression":set(gene_ex["ANONPT_ID"]), "Tumor mutation burden":set(TMB["ANONPT_ID"]), "Immune phenotype":set(IP["ANONPT_ID"])}
    venn(data, cmap = ["coral", "mediumorchid", "gold"])
    #plt.savefig('figures/figure1/venn_plot_data.eps', format='eps')

    # plot TMB distributions
    TMB.columns = ["TMB", "Response", "Patient ID"]
    TMB["Response"] = TMB["Response"].replace({0:"Non-responder", 1:"Responder"})
    my_pal = {"Responder": "cornflowerblue", "Non-responder": "crimson"}
    my_pal2 = {"Responder": "lightblue", "Non-responder": "pink"}

    plt.figure()
    sns.boxplot(x="Response", y="TMB", data=TMB, palette=my_pal2, fliersize=0.01, width=0.7, saturation=1)
    sns.swarmplot(x="Response", y="TMB", data=TMB, palette=my_pal)
    #plt.savefig('figures/figure1/TMB_distribution.eps', format='eps')

    ## check if the difference is significant
    TMB_res = list(TMB["TMB"][TMB["Response"] == "Responder"])
    TMB_nonres = list(TMB["TMB"][TMB["Response"] == "Non-responder"])
    s_tmb, p_tmb = ranksums(TMB_res, TMB_nonres)

    print("Significance TMB: \n", p_tmb)

    # plot immune phenotype distributions
    IP.columns = ["Immune phenotype", "Patient ID", "Response"]
    IP["Response"] = IP["Response"].replace({"SD/PD":"Non-responder", "CR/PR":"Responder"})
    IP_count = IP.drop("Patient ID", axis=1)

    IP_count = IP_count.groupby(['Response', 'Immune phenotype']).size().reset_index().pivot(columns='Response', index='Immune phenotype', values=0).transpose()
    IP_plot = IP_count.div(IP_count.sum(axis=1), axis=0)

    IP_plot.plot(kind='bar', stacked=True, color=['lemonchiffon', 'khaki', 'gold'])
    #plt.savefig('figures/figure1/IP_distribution.eps', format='eps')

    # check of the distributions of counts differ
    freq_res = list(IP_count.loc["Responder",:])
    freq_nonres = list(IP_count.loc["Non-responder",:])

    chisq, p = chisquare(freq_res, freq_nonres)
    print("Significance IP distribution: \n", p)
def figure_3(path_dicts, palettes, outputs, figure):
    for i, out in enumerate(outputs):
        scores = create_boxplot(path_dicts[i], palettes[i], outputs[i], figure)
        sign_MCC, sign_AUC, sign_acc = sign_df(path_dicts[i], scores)
        print(out, "\n Significance df MCC: \n", sign_MCC, "\n Significance df AUC: \n", sign_AUC,
              "\n Significance df acc: \n", sign_acc)
def figure_4(path_dict, palette, output, figure):
    #a
    # boxplots from the scores for TMB, 15 gene signature & 31 gene signature
    scores_3 = create_boxplot(path_dict, palette, output, figure)
    sign_3_MCC, sign_3_AUC, sign_3_acc = sign_df(path_dict, scores_3)

    print("Significance df MCC: \n", sign_3_MCC, "\n Significance df AUC: \n", sign_3_AUC,
          "\n Significance df acc: \n", sign_3_acc)

    # b-d
    # save pd dataframe containing the predicted classes for the 100 runs
    bin_pal = {1: 'cornflowerblue', 0: 'crimson'}

    for run in path_dict.keys(): # or for path_dict to create figure 3 b-d
        path = path_dict[run]
        ID_train = pd.read_csv("{}/predicted_classes_train.csv".format(path))
        ID_test = pd.read_csv("{}/predicted_classes_test.csv".format(path))

        # create plots of the predictions and save them
        # train data
        ID_train["total_pred"] = ID_train.iloc[:, 3:].notnull().sum(axis=1)
        ID_train["frac_pred"] = ID_train["total_correct"] / ID_train["total_pred"]

        #fig = sns.displot(x=ID_train.loc[:, "frac_pred"], hue=ID_train["binaryResponse"], binwidth=0.05, multiple="stack", palette=bin_pal)
        #fig.savefig(("figures/figure3/fraction_correct_predicted_train_{}.eps").format(run), format='eps')

        # test data
        ID_test["total_pred"] = ID_test.iloc[:, 3:].notnull().sum(axis=1)
        ID_test["frac_pred"] = ID_test["total_correct"] / ID_test["total_pred"]

        fig = sns.displot(x=ID_test.loc[:, "frac_pred"], hue=ID_test["binaryResponse"], binwidth=0.05, multiple="stack", palette=bin_pal)
        #fig.savefig(("figures/figure3/fraction_correct_predicted_val_{}.eps").format(run), format='eps')
def figure_5(overlap1, overlap2, gene_an, df_1, df_2, input_genes):
    ### Figure 4
    ## heatmap of overlap always correct / incorrectly predicted responders/non-responders is

    ID_correct = list(overlap1['res_cor']) + list(overlap1['nres_cor'])
    ID_incorrect = list(overlap1['res_incor']) + list(overlap1['nres_incor'])

    create_heatmap(df_1, gene_an, ID_incorrect, ID_correct)
    create_heatmap(df_2, gene_an, ID_incorrect, ID_correct)

    ## figure 4b, gene expression values for all four groups
    sign_31 = gex_plots(df_2, overlap2, input_genes, gene_an)

    # do fdr correction for all p values
    p_adj = []
    for signf in [sign_31]:
        p_array = signf.to_numpy()
        p_overig = p_array[0]
        p_correct = p_array[1]
        p_incorrect = p_array[2]
        for p_values in p_overig, p_correct, p_incorrect:
          p_fdr = fdrcorrection(p_values)
          p_adj.append(p_fdr[1])

        df_sign_adj = pd.DataFrame(columns=signf.columns, index=signf.index, data=p_adj)

    return df_sign_adj
    # do FDR correction for all p values together (less conservative)
    # for signf in [sign_31]:
    #     p_array = signf.to_numpy()
    #     p_values = p_array.flatten()
    #
    #     p_adj = multipletests(p_values, method="fdr_bh")[1]
    #     p_adj = np.reshape(p_adj, signf.shape)
    #
    #     df_sign_adj_2 = pd.DataFrame(columns=signf.columns, index=signf.index, data=p_adj)
def figure_7(path_dict_1, val_dfs, palette, path_dict_2, riaz_dfs, no_IP_dfs, paths_riaz):
    ## Figure 6 -- Validation of top classifiers via majority voting
    # a -- train, validation MCC and AUC of classifiers after addition of extra genes to signature
    scores_6a = create_boxplot(path_dict_1, palette, "15-35_genes", "figure6")
    sign_6a_MCC, sign_6a_AUC, sign_6a_acc = sign_df(path_dict_1, scores_6a)
    print("Significance df MCC: \n", sign_6a_MCC, "\n Significance df AUC: \n", sign_6a_AUC,
          "\n Significance df acc: \n", sign_6a_acc)

    # b -- confusion matrices of performance on test and validation data
    ## load the classifiers and the data
    # Mariathasan (test & train)
    mar_sign1 = val_dfs[0]
    mar_sign2 = val_dfs[1]
    TMB = val_dfs[2]

    # drop the 25 test samples
    dropID = [10050, 10215, 10093, 10212, 10181, 10055, 10239, 10071, 10209, 10103, 10325, 10321, 10189, 10076, 10058, 10119, 10115,
              10187, 10224, 10279, 10049, 10371, 10299, 10277, 10084]

    test_set_sign1 = mar_sign1[mar_sign1["ANONPT_ID"].isin(dropID)]
    mar_sign1 = mar_sign1.drop(test_set_sign1.index, axis=0)

    test_set_sign2 = mar_sign2[mar_sign2["ANONPT_ID"].isin(test_set_sign1["ANONPT_ID"])]
    mar_sign2 = mar_sign2.drop(test_set_sign2.index, axis=0)

    test_set_TMB = TMB[TMB["ANONPT_ID"].isin(test_set_sign1["ANONPT_ID"])]
    TMB = TMB.drop(test_set_TMB.index, axis=0)

    # classifiers for final signatures
    path_sign1 = path_dict_2["sign1"]
    path_sign2 = path_dict_2["sign2"]
    path_TMB = path_dict_2["TMB"]

    with open(f"{path_sign2}/classifiers.pkl", 'rb') as handle1:
        clf_sign2 = pickle.load(handle1)
    with open(f"{path_sign1}/classifiers.pkl", 'rb') as handle2:
        clf_sign1 = pickle.load(handle2)
    with open(f"{path_TMB}/classifiers.pkl", 'rb') as handle3:
        clf_TMB = pickle.load(handle3)

    ## select the top 5 classifiers from both signatures and retrain on complete Mariathasan data (without 25 test samples)
    # find the numbers of the 5 runs with highest MCC
    scores_sign2 = pd.read_csv(f"{path_sign2}/scores_100.csv", index_col=0)
    scores_sign2 = scores_sign2.drop(["mean", "st_dev"], axis=0)

    scores_sign1 = pd.read_csv(f"{path_sign1}/scores_100.csv", index_col=0)
    scores_sign1 = scores_sign1.drop(["mean", "st_dev"], axis=0)

    scores_TMB = pd.read_csv(f"{path_TMB}/scores_100.csv", index_col=0)
    scores_TMB = scores_TMB.drop(["mean", "st_dev"], axis=0)

    # find indices that correspond to 5 highest test MCC
    idx_sign2 = np.argsort(scores_sign2.loc[:,"test_MCC"])[::-1][0:5]
    idx_sign1 = np.argsort(scores_sign1.loc[:,"test_MCC"])[::-1][0:5]
    idx_TMB = np.argsort(scores_TMB.loc[:,"test_MCC"])[::-1][0:5]

    # select the classifiers corresponding to those indices
    top_clf_sign2 = []
    for idx in idx_sign2:
        top_clf_sign2.append(clf_sign2[idx])

    top_clf_sign1 = []
    for idx in idx_sign1:
        top_clf_sign1.append(clf_sign1[idx])

    top_clf_TMB = []
    for idx in idx_TMB:
        top_clf_TMB.append(clf_TMB[idx])

    # retrain the classifiers on the complete Mariathasan data
    def_clf_sign2 = []
    preds_sign2 = pd.DataFrame()
    X_sign2 = mar_sign2.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_sign2 = mar_sign2["binaryResponse"]
    X_test_sign2 = test_set_sign2.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_test_sign2 = test_set_sign2["binaryResponse"]

    def_clf_sign1 = []
    preds_sign1 = pd.DataFrame()
    X_sign1 = mar_sign1.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_sign1 = mar_sign1["binaryResponse"]
    X_test_sign1 = test_set_sign1.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_test_sign1 = test_set_sign1["binaryResponse"]

    def_clf_TMB = []
    preds_TMB = pd.DataFrame()
    X_TMB = TMB.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_TMB = TMB["binaryResponse"]
    X_test_TMB = test_set_TMB.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_test_TMB = test_set_TMB["binaryResponse"]

    for i, clf in enumerate(top_clf_sign2):
        fitted_clf = clf.fit(X_sign2, y_sign2)
        def_clf_sign2.append(fitted_clf)
        ## make a prediciton for each sample in test and Riaz data with each classifier
        pred_test = fitted_clf.predict(X_test_sign2)
        preds_sign2[i] = pred_test

    for i, clf in enumerate(top_clf_sign1):
        fitted_clf = clf.fit(X_sign1, y_sign1)
        def_clf_sign1.append(fitted_clf)
        ## make a prediciton for each sample in test and Riaz data with each classifier
        pred_test = fitted_clf.predict(X_test_sign1)
        preds_sign1[i] = pred_test

    for i, clf in enumerate(top_clf_TMB):
        fitted_clf = clf.fit(X_TMB, y_TMB)
        def_clf_TMB.append(fitted_clf)
        ## make a prediciton for each sample in test and Riaz data with each classifier
        pred_test = fitted_clf.predict(X_test_TMB)
        preds_TMB[i] = pred_test

    # assign a final prediction as the prediction chosen by most classifiers
    for idx in preds_sign2.index:
        preds_sign2.loc[idx,"final"] = int(mode(preds_sign2.iloc[idx,:]))

    MCC_sign2_test = matthews_corrcoef(y_test_sign2, preds_sign2["final"])
    print("MCC signature 2: ", MCC_sign2_test)

    # create ROC curve
    #RocCurveDisplay.from_predictions(y_test_sign2, preds_sign2["final"])
    ConfusionMatrixDisplay.from_predictions(y_test_sign2, preds_sign2["final"])
    plt.savefig("figures/figure6/confusion_matrix_35.eps", format='eps')

    # assign a final prediction as the prediction chosen by most classifiers
    for idx in preds_sign1.index:
        preds_sign1.loc[idx,"final"] = int(mode(preds_sign1.iloc[idx,:]))

    MCC_sign1_test = matthews_corrcoef(y_test_sign1, preds_sign1["final"])
    print("MCC signature 1: ", MCC_sign1_test)

    # create ROC curve
    #RocCurveDisplay.from_predictions(y_test_sign1, preds_sign1["final"])
    ConfusionMatrixDisplay.from_predictions(y_test_sign1, preds_sign1["final"])
    plt.savefig("figures/figure6/confusion_matrix_19.eps", format='eps')

    # assign a final prediction as the prediction chosen by most classifiers
    for idx in preds_TMB.index:
        preds_TMB.loc[idx,"final"] = int(mode(preds_TMB.iloc[idx,:]))

    MCC_TMB_test = matthews_corrcoef(y_test_TMB, preds_TMB["final"])
    print("MCC TMB: ", MCC_TMB_test)

    # create ROC curve
    #RocCurveDisplay.from_predictions(y_test_TMB, preds_TMB["final"])
    ConfusionMatrixDisplay.from_predictions(y_test_TMB, preds_TMB["final"])
    plt.savefig("figures/figure6/confusion_matrix_TMB.eps", format='eps')
    # create one ROC curve with all performances of sign1, sign2 and TMB
    # palette = {"15 genes + TMB":"coral", "19 genes + TMB":"orangered", "35 genes + TMB":"red"}
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test_sign1, preds_sign1["final"], ax=ax, name="19 genes signature")
    RocCurveDisplay.from_predictions(y_test_sign2, preds_sign2["final"], ax=ax, name="35 genes signature")
    RocCurveDisplay.from_predictions(y_test_TMB, preds_TMB["final"], ax=ax, name="TMB")
    plt.show()
    plt.savefig("figures/figure6/ROC_test_urothelial.eps", format='eps')

    ### Riaz validation
    riaz_15 = riaz_dfs[0]
    riaz_31 = riaz_dfs[1]
    riaz_TMB = riaz_dfs[2]

    no_IP_15 = no_IP_dfs[0]
    no_IP_31 = no_IP_dfs[1]
    no_IP_TMB = no_IP_dfs[2]

    # the Riaz data does not contain immune phenotypes, so the classifiers trained on the signatures without immune phenotypes are needed
    # classifiers for 15 & 31 gene signature
    path_31 = paths_riaz["path_riaz2"]
    path_15 = paths_riaz["path_riaz1"]
    path_TMB = paths_riaz["path_riazTMB"]

    with open(f"{path_31}/classifiers.pkl", 'rb') as handle1:
        clf_31 = pickle.load(handle1)
    with open(f"{path_15}/classifiers.pkl", 'rb') as handle2:
        clf_15 = pickle.load(handle2)
    with open(f"{path_TMB}/classifiers.pkl", 'rb') as handle3:
        clf_TMB = pickle.load(handle3)

    ## select the top 5 classifiers from both signatures and retrain on complete Mariathasan data (without 25 test samples)
    # find the numbers of the 5 runs with highest MCC
    scores_31 = pd.read_csv(f"{path_31}/scores_100.csv", index_col=0)
    scores_31 = scores_31.drop(["mean", "st_dev"], axis=0)

    scores_15 = pd.read_csv(f"{path_15}/scores_100.csv", index_col=0)
    scores_15 = scores_15.drop(["mean", "st_dev"], axis=0)

    scores_TMB = pd.read_csv(f"{path_TMB}/scores_100.csv", index_col=0)
    scores_TMB = scores_TMB.drop(["mean", "st_dev"], axis=0)

    # find indices that correspond to 5 highest test MCC
    idx_31 = np.argsort(scores_31.loc[:,"test_MCC"])[::-1][0:5]
    idx_15 = np.argsort(scores_15.loc[:,"test_MCC"])[::-1][0:5]
    idx_TMB = np.argsort(scores_TMB.loc[:,"test_MCC"])[::-1][0:5]

    # select the classifiers corresponding to those indices
    top_clf_31 = []
    for idx in idx_31:
        top_clf_31.append(clf_31[idx])

    top_clf_15 = []
    for idx in idx_15:
        top_clf_15.append(clf_15[idx])

    top_clf_TMB = []
    for idx in idx_TMB:
        top_clf_TMB.append(clf_TMB[idx])

    # retrain the classifiers on the complete Mariathasan data
    def_clf_31 = []
    preds_31 = pd.DataFrame()
    X_31 = no_IP_31.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_31 = no_IP_31["binaryResponse"]
    X_riaz_31 = riaz_31.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_riaz_31 = riaz_31["binaryResponse"]

    def_clf_15 = []
    preds_15 = pd.DataFrame()
    X_15 = no_IP_15.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_15 = no_IP_15["binaryResponse"]
    X_riaz_15 = riaz_15.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_riaz_15 = riaz_15["binaryResponse"]

    def_clf_TMB = []
    preds_TMB = pd.DataFrame()
    X_TMB = no_IP_TMB.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_TMB = no_IP_TMB["binaryResponse"]
    X_riaz_TMB = riaz_TMB.drop(["binaryResponse", "ANONPT_ID"], axis=1)
    y_riaz_TMB = riaz_TMB["binaryResponse"]

    for i, clf in enumerate(top_clf_31):
        fitted_clf = clf.fit(X_31, y_31)
        def_clf_31.append(fitted_clf)
        ## make a prediciton for each sample in test and Riaz data with each classifier
        pred_test = fitted_clf.predict(X_riaz_31)
        preds_31[i] = pred_test

    for i, clf in enumerate(top_clf_15):
        fitted_clf = clf.fit(X_15, y_15)
        def_clf_15.append(fitted_clf)
        ## make a prediciton for each sample in test and Riaz data with each classifier
        pred_test = fitted_clf.predict(X_riaz_15)
        preds_15[i] = pred_test

    for i, clf in enumerate(top_clf_TMB):
        fitted_clf = clf.fit(X_TMB, y_TMB)
        def_clf_TMB.append(fitted_clf)
        ## make a prediciton for each sample in test and Riaz data with each classifier
        pred_test = fitted_clf.predict(X_riaz_TMB)
        preds_TMB[i] = pred_test

    # assign a final prediction as the prediction chosen by most classifiers
    for idx in preds_31.index:
        preds_31.loc[idx,"final"] = int(mode(preds_31.iloc[idx,:]))

    MCC_31_riaz = matthews_corrcoef(y_riaz_31, preds_31["final"])
    print("MCC 35 signature: ", MCC_31_riaz)
    # create ROC curve
    #RocCurveDisplay.from_predictions(y_riaz_31, preds_31["final"])
    ConfusionMatrixDisplay.from_predictions(y_riaz_31, preds_31["final"])
    plt.savefig("figures/figure6/confusion_matrix_riaz_34.eps", format='eps')

    # assign a final prediction as the prediction chosen by most classifiers
    for idx in preds_15.index:
        preds_15.loc[idx,"final"] = int(mode(preds_15.iloc[idx,:]))

    MCC_15_riaz = matthews_corrcoef(y_riaz_15, preds_15["final"])
    print("MCC 19 signature: ", MCC_15_riaz)
    # create ROC curve
    #RocCurveDisplay.from_predictions(y_riaz_15, preds_15["final"])
    ConfusionMatrixDisplay.from_predictions(y_riaz_15, preds_15["final"])
    plt.savefig("figures/figure6/confusion_matrix_riaz_18.eps", format='eps')

    # assign a final prediction as the prediction chosen by most classifiers
    for idx in preds_TMB.index:
        preds_TMB.loc[idx,"final"] = int(mode(preds_TMB.iloc[idx,:]))

    MCC_TMB_riaz = matthews_corrcoef(y_riaz_TMB, preds_TMB["final"])
    print("MCC TMB: ", MCC_TMB_riaz)
    # create ROC curve
    #RocCurveDisplay.from_predictions(y_riaz_TMB, preds_TMB["final"])
    ConfusionMatrixDisplay.from_predictions(y_riaz_TMB, preds_TMB["final"])
    plt.savefig("figures/figure6/confusion_matrix_riaz_TMB.eps", format='eps')

    # one figure with all ROC curves
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_riaz_15, preds_15["final"], ax=ax, name="19 genes signature")
    RocCurveDisplay.from_predictions(y_riaz_31, preds_31["final"], ax=ax, name="35 genes signature")
    RocCurveDisplay.from_predictions(y_riaz_TMB, preds_TMB["final"], ax=ax, name="TMB")
    plt.show()
    plt.savefig("figures/figure6/ROC_riaz_melanoma.eps", format='eps')
def sup_figure_1(path_dict):
    # & Supplemental figure 1
        # save pd dataframe containing the predicted classes for the 100 runs
    bin_pal = {1: 'cornflowerblue', 0: 'crimson'}

    for run in path_dict.keys():  # or for path_dict to create figure 3 b-d
        path = path_dict[run]
        ID_train = pd.read_csv("{}/predicted_classes_train.csv".format(path))
        ID_test = pd.read_csv("{}/predicted_classes_test.csv".format(path))

        # create plots of the predictions and save them
        # train data
        ID_train["total_pred"] = ID_train.iloc[:, 3:].notnull().sum(axis=1)
        ID_train["frac_pred"] = ID_train["total_correct"] / ID_train["total_pred"]

        #fig = sns.displot(x=ID_train.loc[:, "frac_pred"], hue=ID_train["binaryResponse"], binwidth=0.05,
        #                  multiple="stack", palette=bin_pal)
        #fig.savefig(("figures/figure3/fraction_correct_predicted_train_{}.eps").format(run), format='eps')

        # test data
        ID_test["total_pred"] = ID_test.iloc[:, 3:].notnull().sum(axis=1)
        ID_test["frac_pred"] = ID_test["total_correct"] / ID_test["total_pred"]

        fig = sns.displot(x=ID_test.loc[:, "frac_pred"], hue=ID_test["binaryResponse"], binwidth=0.05, multiple="stack",
                          palette=bin_pal)
        #fig.savefig(("figures/figure3/fraction_correct_predicted_test_{}.eps").format(run), format='eps')
def sup_figure_2(path_dict):
    ## Suplemental figure 2
    # boxplots of feature importances
    for file in path_dict.keys():
        df_features = pd.read_csv(f"{path_dict[file]}/feature_importances.csv", index_col=0)

        df_features_plot = df_features.sort_values(by="mean", axis=1, ascending=False).drop(["mean", "st_dev"], axis=0)
        if len(df_features_plot.columns) <= 10:
            features_plot = df_features_plot
        else:
            features_plot = df_features_plot.iloc[:, :10]

        plt.figure()
        plt.gcf().subplots_adjust(bottom=0.3)
        features_plot.boxplot(rot=90)
        plt.savefig(("figures/supl_figure2/boxplot_def_{}.eps").format(file), format='eps')

# before running functions to create figures, load all paths and dfs:
path_val45 = "runs/Random_Forest/balanced/final_run/45_validation/"
# paths to different run outputs to be used for plotting
path_TMB = f"{path_val45}TMB"
# feature selection methods
path_MI = f"{path_val45}MI_50"
path_MI_TMB = f"{path_val45}MI_TMB_50"
path_RFE = f"{path_val45}RFE_50"
path_RFE_TMB = f"{path_val45}RFE_TMB_50"
path_PCA = f"{path_val45}PCA_50"
path_PCA_TMB = f"{path_val45}PCA_TMB_50"
# clinical and TMB
path_clin = f"{path_val45}clin_CTL"
path_TMB_clin = f"{path_val45}clin_CTL_TMB"
# count data signatures
path_15_count = f"{path_val45}15_count"
path_16_count = f"{path_val45}16_count"
path_31_count = f"{path_val45}31_count"
# 15 genes and IP/TMB
path_15 = f"{path_val45}15_count"
path_15_IP = f"{path_val45}15_IP"
path_15_TMB = f"{path_val45}15_TMB"
path_15_TMB_IP = f"{path_val45}15_TMB_IP"
# 31 genes and IP/TMB
path_31 = f"{path_val45}31_count"
path_31_IP = f"{path_val45}31_IP"
path_31_TMB = f"{path_val45}31_TMB"
path_31_TMB_IP = f"{path_val45}31_TMB_IP"
# paths to the final classifiers that will be validated on Mariathasan urothelial data
path_sign1 = f"{path_val45}19_TMB_IP"
path_sign2 = f"{path_val45}35_count"
# paths to the final classifiers that will be validated on Riaz melanoma data
path_riaz1 = f"{path_val45}18_zscore_TMB"
path_riaz2 = f"{path_val45}34_zscore_count"
path_riazTMB = f"{path_val45}TMB_zscore"
# dataframes used to create heatmaps of gene expression values
df_31 = pd.read_csv("csv_files/final_run/31_TMB_IP.csv")
df_15 = pd.read_csv("csv_files/final_run/15_TMB_IP.csv")
# dataframes used in validation of classifiers to retrain them on complete data
# Mariathasan data
mar_sign1 = pd.read_csv("csv_files/final_run/19_TMB_IP.csv")
mar_sign2 = pd.read_csv("csv_files/final_run/35_count.csv")
TMB = pd.read_csv("csv_files/final_run/TMB.csv")
# Mariathasan data without immunophenotypes
no_IP_1 = pd.read_csv("csv_files/final_run/18_zscore_TMB.csv")
no_IP_2 = pd.read_csv("csv_files/final_run/34_zscore_count.csv")
no_IP_TMB = pd.read_csv("csv_files/final_run/TMB_zscore.csv")
# Riaz data
df_riaz1 = pd.read_csv("csv_files/RIAZ/signatures/riaz_18_zscore.csv")
df_riaz2 = pd.read_csv("csv_files/RIAZ/signatures/riaz_34_zscore.csv")
# TMB that is included in the 34 gene signature file should be removed
df_riaz2 = df_riaz2.drop("FMOne mutation burden per MB", axis=1)
df_riazTMB = df_riaz1.loc[:,["ANONPT_ID", "binaryResponse", "FMOne mutation burden per MB"]]

# data file with info to convert X+entrez ID to normal symbol
gene_an = pd.read_csv("csv_files/gene_anotation.csv", index_col=0)

## run one of functions below to create all subplots of the different figures
### Figure 1
figure_1()

### Figure 3
paths = [{"MI":path_MI, "MI+TMB":path_MI_TMB, "RFE":path_RFE,
            "RFE+TMB":path_RFE_TMB, "PCA":path_PCA, "PCA+TMB":path_PCA_TMB, "TMB":path_TMB},
         {"clinical":path_clin, "clinical + TMB":path_TMB_clin, "TMB":path_TMB},
         {"15":path_15_count, "16":path_16_count, "31":path_31_count, "TMB":path_TMB},
         {"15":path_15, "15 + IP":path_15_IP, "15 + TMB":path_15_TMB, "15 + TMB + IP": path_15_TMB_IP, "TMB":path_TMB}]
palettes = [{"TMB": "mediumorchid", "MI": "mediumseagreen", "MI+TMB": "green", "RFE": "mediumturquoise",
              "RFE+TMB": "blue",  "PCA": "yellow", "PCA+TMB": "orange"},
            {"TMB": "mediumorchid", "clinical":"indigo", "clinical + TMB":"purple"},
            {"15":"coral", "16":"orange", "TMB":"mediumorchid", "31":"gold"},
            {"TMB":"mediumorchid", "15":"coral", "15 + IP":"pink", "15 + TMB":"hotpink", "15 + TMB + IP":"deeppink"}]
outputs = ["feature selection", "clinical", "15_16_31_count", "15_IP_TMB"]
figure_3(paths, palettes, outputs, "figure3")

### Figure 4
paths_4 = {"15_TMB_IP": path_15_TMB_IP, "31": path_31_count, "TMB": path_TMB}
palette_4 = {"15_TMB_IP": "coral", "TMB": "mediumorchid", "31": "gold"}
figure_4(paths_4, palette_4, "best_two", "figure4")

### Figure 5
# overlap when 15 + TMB + IP is compared with 31 + TMB + IP (45 val set)
# overlap samples always predicted correctly or incorrectly in validation data
overlap_always = {'res_cor': {10274, 10363, 10349, 10157, 10191, 10162, 10355, 10196, 10007, 10296, 10201, 10139, 10233, 10335},
                'res_incor': {10364}, 'nres_cor':
                    {10242, 10116, 10120, 10249, 10254, 10262, 10008, 10270, 10016, 10024, 10154, 10287, 10033, 10291, 10036, 10171,
                     10043, 10300, 10173, 10052, 10180, 10182, 10054, 10056, 10317, 10064, 10070, 10329, 10207, 10080, 10088, 10345,
                     10346, 10097, 10227, 10105, 10106, 10236},
                'nres_incor': {10178, 10306, 10085, 10280, 10223, 10129, 10065, 10047, 10166, 10040, 10172, 10367}}
# overlap samples predicted (in)correctly >= 85% of the runs in validation data
overlap_85 = {'res_cor': {10117, 10122, 10007, 10139, 10274, 10155, 10157, 10162, 10296, 10191, 10194, 10196, 10201, 10335,
                            10341, 10344, 10349, 10355, 10230, 10233, 10363},
                'res_incor': {10244, 10213, 10150, 10029, 10061, 10234, 10364, 10238},
                'nres_cor': {10242, 10116, 10120, 10249, 10253, 10254, 10127, 10262, 10006, 10008, 10012, 10013, 10268, 10141, 10016,
                             10270, 10275, 10023, 10024, 10151, 10026, 10154, 10028, 10287, 10033, 10291, 10164, 10293, 10036, 10298,
                             10043, 10300, 10173, 10171, 10305, 10052, 10180, 10182, 10054, 10056, 10059, 10315, 10317, 10063, 10064,
                             10070, 10199, 10329, 10206, 10207, 10080, 10337, 10078, 10088, 10345, 10346, 10222, 10095, 10097, 10226,
                             10099, 10227, 10104, 10105, 10106, 10236, 10365},
                'nres_incor': {10369, 10374, 10251, 10129, 10017, 10280, 10166, 10040, 10172, 10047, 10306, 10178, 10185,
                               10065, 10339, 10085, 10087, 10217, 10350, 10223, 10367}}

genes_fig5 = ["BLM", "DAPL1", "GADD45A"]
## to get all genes from df_31:
genes_all = list(df_31.columns[df_31.columns.str.startswith('X')])
genes_all = entrez_to_symbol(remove_X(genes_all), gene_an)
sign_fig5 = figure_5(overlap_always, overlap_85, gene_an, df_1=df_15, df_2=df_31, input_genes=genes_fig5)

### Figure 7
paths_7a = {"15 genes + TMB + IP": path_15_TMB_IP, "19 genes + TMB + IP": path_sign1, "31 genes": path_31_count,
            "35 genes": path_sign2}
palette_7a = {"15 genes + TMB + IP": "coral", "19 genes + TMB + IP": "orangered", "31 genes": "orange",
              "35 genes": "red"}

# input for validation on Mariathasan data
val_dfs = [mar_sign1, mar_sign2, TMB]
paths_val = {"sign1": path_sign1, "sign2": path_sign2, "TMB":path_TMB}

# input for validation on Riaz data
paths_riaz = {"path_riaz1":path_riaz1, "path_riaz2":path_riaz2, "path_riazTMB":path_riazTMB}
riaz_dfs = [df_riaz1, df_riaz2, df_riazTMB]
no_IP_dfs = [no_IP_1, no_IP_2, no_IP_TMB]

figure_7(paths_7a, val_dfs, palette_7a, paths_val, riaz_dfs, no_IP_dfs, paths_riaz)

### Suplemental figures
### Supl. figure 1 with distribution of predictions of responders/non-responders using no balancing or SMOTE
path_balancing = {"no_balancing": "runs/Random_Forest/balanced/signatures/15_TMB_sign_n",
                      "SMOTE": "runs/Random_Forest/balanced/signatures/15_TMB_sign_SMOTE"}
sup_figure_1(path_balancing)

### Supl. figure 2 to plot top 10 feature importances
feature_files = {"TMB_clin": path_TMB_clin,
                 "PCA_50": path_PCA,
                 "DNA_REP": "runs/Random_Forest/balanced/signatures/CTL_signature_DNArep_genes",
                 "CD8": "runs/Random_Forest/balanced/signatures/CTL_signature_CD8_genes"}
sup_figure_2(feature_files)

### Supl. figure 3 to plot gene expression values of 4 added genes
df_extra = pd.read_csv("csv_files/final_run/35_count.csv")
genes_extra = ["B2M", "CLDN3", "HLA-DRA", "FCGR1A"]
overlap_15_31_always = {'res_cor': {10274, 10341, 10344, 10122, 10363, 10349, 10191, 10162, 10196, 10007,
                                    10296, 10201, 10139, 10233, 10335},
                        'res_incor': {10364, 10244},
                        'nres_cor': {10242, 10116, 10120, 10249, 10262, 10008, 10141, 10270, 10016, 10024,
                                     10154, 10287, 10033, 10291, 10164, 10036, 10043, 10300, 10173, 10171,
                                     10052, 10180, 10182, 10054, 10056, 10317, 10064, 10329, 10207, 10080,
                                     10088, 10345, 10346, 10097, 10227, 10105, 10106, 10236},
                        'nres_incor': {10178, 10085, 10087, 10280, 10350, 10223, 10129, 10047, 10166, 10040, 10172, 10367}}
sign_extra = gex_plots(df_extra, overlap_15_31_always, genes_extra, gene_an)

## Suppl. figure 4: addition of IP/TMB to 31 gene signature
path_dict_sup4 = {"31":path_31, "31 + IP":path_31_IP, "31 + TMB":path_31_TMB, "31 + TMB + IP": path_31_TMB_IP, "TMB":path_TMB}
palette_sup4 =  {"TMB":"mediumorchid", "31":"red", "31 + IP":"pink", "31 + TMB":"hotpink", "31 + TMB + IP":"deeppink"}
figure_3(path_dict_sup4, palette_sup4, "31_TMB_IP", "supl_figure4")