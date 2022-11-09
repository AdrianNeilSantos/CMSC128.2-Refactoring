import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import EDA_Tools
import Preprocessing_Tools
import Feature_Selection_Tools
import Evaluate_Performance_Tools

# -*- coding: utf-8 -*-
"""
Created on Sun May  1 19:07:32 2022

@author: Sheila
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 23:54:42 2022

@author: Sheila
"""


def main():
    df = pd.read_csv("Depression.csv")
    df_nodupLE = data_preprocessing(df)
    _ = perform_EDA(df_nodupLE)
    x_train, x_test, y_train, y_test = split_df(df_nodupLE)
    model_training(x_train, x_test, y_train, y_test)


def data_preprocessing(df):
    ## REMOVE DUPLICATES ##
    EDA_Tools.describe_df(df, head=True, info=True, describe=True)
    EDA_Tools.count_col_value(df, "DEPRESSED", withPercentage=True)

    df.isnull().any()
    # Selecting duplicate rows except first
    # occurrence based on all columns
    # saves the duplicates in the dataframe called dup
    dup = df[df.duplicated()]
    print("These are the duplicate rows (there are 10 duplicates):")
    dup
    dup.shape

    # save a copy of the records with no duplicates
    # in a new dataframe named df_nodup
    df_nodup = df.drop_duplicates()
    df_nodup.to_csv("Depression_nodup.csv")
    df_nodup.shape

    df.duplicated()
    print("Shape of original dataframe from csv: ", df.shape)
    print("Shape of dataframe containing the duplicates: ", dup.shape)
    print("Shape of dataframe with no more duplicates: ", df_nodup.shape)

    #############################

    EDA_Tools.describe_df(df, head=True, info=True, describe=True)

    # to count how many have and do not have DEPRESSION
    df_nodup["DEPRESSED"].value_counts()

    # To get the percentage distribution of values of a column we use
    # df['colname'].value_counts(normalize=True)*100

    categorical_cols = df_nodup.columns
    categorical_cols = categorical_cols.drop("DEPRESSED")

    df_nodup.isnull().any()
    EDA_Tools.describe_df(df, info=True, describe=True)

    depressed_group = df_nodup.groupby("DEPRESSED")
    depressed_group_size = depressed_group.size()
    print(depressed_group_size)

    df_nodup = df_nodup.reset_index(drop=True)

    # Use LabelEncoder to recode the categorical values
    df_nodupLE = df_nodup.copy(deep=True)
    df_nodupLE = Preprocessing_Tools.recode_categorical(
        df_nodupLE,
        categorical_cols
    )

    ##################
    df_nodupLE = Preprocessing_Tools.standard_scale(
        df_nodupLE,
        df_nodup,
        "DEPRESSED"
    )
    ##################

    return df_nodupLE


def perform_EDA(df):
    # create whitish correlation matrix
    EDA_Tools.correlation_matrix(df)

    # Select features whose correlation with target is > 0.2
    corr_columns = Feature_Selection_Tools.get_features_by_corr(
        df, target="DEPRESSED", corr_threshold=0.2
    )

    # We create a new dataframe with the selected fields as columns
    df_corr = df[corr_columns].copy()
    return df_corr


def split_df(df):
    y = df["DEPRESSED"]
    x = df.drop("DEPRESSED", axis=1)

    xx_train, x_test, yy_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=14
    )

    sampler = SMOTE(random_state=0)

    x_train, y_train = sampler.fit_resample(xx_train, yy_train)

    # to count how many have and do not have DEPRESSION
    # in the SMOTEN Training dataset
    yy_train.value_counts()
    y_train.value_counts()  # we have 271 for 0  and 271 for 1
    y_test.value_counts()

    return x_train, x_test, y_train, y_test


def model_training(x_train, x_test, y_train, y_test):
    # DECISION TREE Hyperparameter Tuning
    model = DecisionTreeClassifier()
    parameters = {
        "splitter": ["best", "random"],
        "criterion": ["gini", "entropy"],
        "max_features": ["log2", "sqrt"],  # removed auto
        "max_depth": [2, 3, 5, 10, 17],
        "min_samples_split": [2, 3, 5, 7, 9],
        "min_samples_leaf": [1, 5, 8, 11],
        "random_state": [0, 1, 2, 3, 4, 5],
    }

    grid_search_dt = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="accuracy",
        cv=5,
        verbose=1
    )

    grid_search_dt.fit(x_train, y_train)
    print(grid_search_dt.best_estimator_)
    print(grid_search_dt.score(x_test, y_test))
    # Best Decision Tree parameters are
    # DecisionTreeClassifier(
    #   max_depth=10,
    #   max_features='log2',
    #   min_samples_leaf=8, random_state=3
    # )

    # print(grid_search_dt.score(x_test, y_test))
    # 0.8603351955307262
    # DecisionTreeClassifier(
    #   criterion='entropy',
    #   max_depth=10,
    #   max_features='sqrt',
    #   min_samples_split=9,
    #   random_state=5
    # )

    dt = DecisionTreeClassifier(
        max_depth=10,
        max_features="sqrt",
        random_state=5,
        criterion="entropy",
        min_samples_split=9,
    )

    Evaluate_Performance_Tools.evaluate_model(
        dt, "Decision Tree", x_train, x_test, y_train, y_test
    )

    ##################################################
    # LOGISTIC REGRESSION HYPERPARAMETER TUNING
    model = LogisticRegression(max_iter=400)
    solvers = ["newton-cg", "lbfgs", "liblinear"]
    penalty = ["l2"]
    c_values = [100, 50, 1.0, 0.1, 0.01]
    # define grid search
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        n_jobs=-1,
        cv=cv,
        scoring="accuracy",
        error_score=0,
    )
    grid_result = grid_search.fit(x_train, y_train)
    # summarize results
    print(
        "Best score: %f using %s" % (
            grid_result.best_score_,
            grid_result.best_params_
        )
    )

    # Best score
    lr = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="newton-cg",
        random_state=42
    )

    Evaluate_Performance_Tools.evaluate_model(
        lr, "Logistic Regression", x_train, x_test, y_train, y_test
    )

    ##################################################

    ##################################################
    # LR Feature importance

    importance = lr.coef_[0]
    # summarize feature importance
    for i, val in enumerate(importance):
        print("Feature: %0d, Score: %.5f" % (i, val))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], np.sort(importance))
    plt.show()

    lr_coefko = lr.coef_[0]
    x_train_columns = lr.feature_names_in_

    lr_coefko_df = pd.DataFrame(lr_coefko, columns=["feature_importance"])
    x_train_columns_df = pd.DataFrame(x_train_columns, columns=["features"])
    data2 = pd.concat([x_train_columns_df, lr_coefko_df], axis=1, join="inner")

    # feature_importance=pd.DataFrame({'feature':list(x_train.columns),
    # 'feature_importance':[abs(i) for i in lr.coef_[0]]})
    # feature_importance.sort_values('feature_importance',ascending=False)

    ####### barplot of sorted feature importance
    # WRONG ouTput DO not USE
    n_features = x_train.shape[1]
    plt.figure(figsize=(20, 20))
    plt.barh([x for x in x_train.columns],
        np.sort(lr.coef_[0]),
        align="center"
    )
    # plt.barh(range(n_features), lr.coef_[0], align='center')
    plt.yticks(np.arange(n_features), x_train.columns.values)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()

    ########
    # WoRKING
    plt.bar(range(len(lr.coef_[0])), lr.coef_[0])
    plt.xticks(range(len(lr.coef_[0])), x_train.columns)
    plt.show()

    ####################
    # WoRKING
    plt.barh([x for x in x_train.columns], lr.coef_[0])

    plt.barh(x_train.columns, lr.coef_[0])
    # lr.coef_[0], index=x_train.columns
    ####################
    # WORKING FEATURE IMPORTANCE PLOT - LOGISTIC REGRESSION
    # plot feature name and feature importance unsorted
    data2.plot(kind="barh", y="feature_importance", x="features")

    sorted_values = data2.sort_values(by="feature_importance", ascending=True)
    # plot ALL feature name and feature importance sorted via
    # feature importance largest to smallest; when displayed top
    # is the feature with largest feature importance
    sorted_values.plot(kind="barh", x="features")

    # plot top 10 smallest features importance. when displayed,
    # top is the last 10th smallest, followed by 9th smallest,...
    # last entry is the smallest
    sorted_values.head(10).plot(kind="barh", x="features")

    # plot top 15 largest feature importance. when displayed,
    # top is the largest feature importance, pababa
    sorted_values.tail(15).plot(
        kind="barh", x="features", title="LR Feature Importance"
    )

    # plot ALL feature name and feature importance sorted via
    # feature importance smallest to largest; when displayed top
    # is the feature with the smallest feature importance
    sorted_values.plot(kind="barh", x="features")

    # plot top 10 largest feature importance. when displayed top
    # is the last 10th largeest, followed by 9th largest,...
    # last entry is the largest
    sorted_values.head(10).plot(kind="barh", x="features")

    # plot top 10 smallest feature importance. when displayed top
    # is the smallest, followed by 2nd to the smallest,...
    # last entry is the 10th to the smallest
    sorted_values.tail(10).plot(kind="barh", x="features")

    # plot 10 largest features such that the feature with largest feature
    # importance is at the bottom
    data2.nlargest(10, "feature_importance").plot(kind="barh", x="features")

    # plot 10 smallest features such that the feature with smallest feature
    # importance at the bottom
    data2.nsmallest(10, "feature_importance").plot(kind="barh", x="features")

    ####################
    # working but displays unsorted feature importance
    plt.barh(range(len(lr.coef_[0])), lr.coef_[0])
    plt.yticks(range(len(lr.coef_[0])), x_train.columns)
    plt.show()

    ##################################################
    # RANDOM FOREST Hyperparamenter Tuning
    # param_grid = {'max_depth': np.arange(40, 50, 40),
    #               #'n_estimators': np.arange(50, 10, 500),
    #               'n_estimators': np.arange(400, 500),
    #               'random_state': [42]}
    param_grid2 = {
        "max_depth": np.arange(20, 81, 20),
        "n_estimators": np.arange(400, 551, 50),
        "random_state": [42],
    }

    rf_clf = RandomForestClassifier()
    rf_grid_search = GridSearchCV(
        rf_clf,
        param_grid2,
        cv=10,
        return_train_score=True,
        verbose=1,
        n_jobs=-1,
        scoring="accuracy",
    )
    rf_grid_search.fit(x_train, y_train)
    print(f"Best parameters: {rf_grid_search.best_params_}\n")
    print(f"Best estimator: {rf_grid_search.best_estimator_}\n")
    cvres_df = pd.DataFrame(rf_grid_search.cv_results_)
    cvres_df.sort_values(by="mean_test_score", ascending=False, inplace=True)
    print(cvres_df[["mean_test_score", "params"]])

    # Best estimator
    rf = RandomForestClassifier(
        max_depth=20,
        n_estimators=400,
        random_state=42
    )
    Evaluate_Performance_Tools.evaluate_model(
        rf, "Random Forest", x_train, x_test, y_train, y_test
    )


main()
