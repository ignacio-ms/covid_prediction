import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import model_selection

import misc


@misc.timed
def train_adaboost(X_train, X_test, y_train, y_test, S, verbose=False):

    params = {
        'base_estimator__criterion': ['gini', 'entropy'],
        'base_estimator__max_depth': [1, 3, 5, 6],
        'base_estimator__max_features': [None, 'auto', 'sqrt', 'log2'],
        'base_estimator__class_weight': [None, 'balanced'],
        'base_estimator__min_samples_split': [2, 10, 50],
        'base_estimator__min_samples_leaf': [1, 10, 50],
        'base_estimator__max_leaf_nodes': [7, 8, 9, 10],
        'base_estimator__ccp_alpha': [0.001, 0.05, 0.1],
        'n_estimators': [50, 100, 250],
        'learning_rate': [0.1, 1]
    }

    grid_search = model_selection.GridSearchCV(
        AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
        params,
        metrics.make_scorer(metrics.roc_auc_score, max_fpr=0.8, needs_threshold=True),
        cv=model_selection.StratifiedKFold(n_splits=10),
        return_train_score=True,
        n_jobs=-1
    ).fit(X_train[S == 1], y_train[S == 1])

    if verbose:
        print('AdaboostClasifier: ')
        print(f'Best params: {grid_search.best_params_}')

        print(f'Train: {metrics.recall_score(y_train, grid_search.predict(X_train), average="macro")}')
        print(f'Test: {metrics.recall_score(y_test, grid_search.predict(X_test), average="macro")}')

        cm_train = pd.DataFrame(
            metrics.confusion_matrix(y_train, grid_search.predict(X_train)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )

        cm_test = pd.DataFrame(
            metrics.confusion_matrix(y_test, grid_search.predict(X_test)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )

        print(f'Confusion matrix train: \n {cm_train}')
        print(f'Confusion matrix test: \n {cm_test}')
        print('\n')

    return grid_search.best_estimator_


@misc.timed
def train_tree(X_train, X_test, y_train, y_test, S, verbose=False):
    params = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'class_weight': [None, 'balanced'],
        'min_samples_split': [2, 10, 50],
        'min_samples_leaf': [1, 10, 50],
        'max_leaf_nodes': [5, 6, 7, 8, 9, 10],
        'ccp_alpha': [0.001, 0.05, 0.1, 0.3],
        'min_impurity_decrease': [0.001, 0.05, 0.1, 0.3]
    }

    grid_search = model_selection.GridSearchCV(
        DecisionTreeClassifier(),
        params,
        scoring=metrics.make_scorer(metrics.roc_auc_score, max_fpr=0.8, needs_threshold=True),
        cv=model_selection.StratifiedKFold(n_splits=10),
        n_jobs=-1,
        return_train_score=True
    ).fit(X_train[S == 1], y_train[S == 1])

    if verbose:
        print('DecisionTreeClassifier: ')
        print(f'Best params: {grid_search.best_params_}')

        print(f'Train: {metrics.recall_score(y_train, grid_search.predict(X_train), average="macro")}')
        print(f'Test: {metrics.recall_score(y_test, grid_search.predict(X_test), average="macro")}')

        cm_train = pd.DataFrame(
            metrics.confusion_matrix(y_train, grid_search.predict(X_train)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )

        cm_test = pd.DataFrame(
            metrics.confusion_matrix(y_test, grid_search.predict(X_test)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )

        print(f'Confusion matrix train: \n {cm_train}')
        print(f'Confusion matrix test: \n {cm_test}')

        fpr, tpr, thresholds = metrics.roc_curve(y_test, grid_search.predict_proba(X_test)[:, 1])

        plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
        plt.plot(fpr, tpr, marker='.', label='Tree')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()

        fig = plt.figure(figsize=(40, 20))
        tree.plot_tree(grid_search.best_estimator_)
        plt.show()

        print('\n')

    return grid_search.best_estimator_


@misc.timed
def train_forest(X_train, X_test, y_train, y_test, S, verbose=False):

    params = {
        'n_estimators': [50, 100, 150],
        'criterion': ['gini', 'entropy'],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'class_weight': [None, 'balanced'],
        'min_samples_split': [2, 10, 50],
        'min_samples_leaf': [1, 10, 50],
        'max_leaf_nodes': [7, 8, 9, 10],
        'ccp_alpha': [0.001, 0.05, 0.1],
    }

    grid_search = model_selection.GridSearchCV(
        RandomForestClassifier(),
        params,
        scoring=metrics.make_scorer(metrics.roc_auc_score, max_fpr=0.8, needs_threshold=True),
        cv=model_selection.StratifiedKFold(n_splits=10),
        n_jobs=-1,
        return_train_score=True
    ).fit(X_train[S == 1], y_train[S == 1])

    if verbose:
        print('RandomForestClassifier: ')
        print(f'Best params: {grid_search.best_params_}')

        print(f'Train: {metrics.recall_score(y_train, grid_search.predict(X_train), average="macro")}')
        print(f'Test: {metrics.recall_score(y_test, grid_search.predict(X_test), average="macro")}')

        cm_train = pd.DataFrame(
            metrics.confusion_matrix(y_train, grid_search.predict(X_train)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )

        cm_test = pd.DataFrame(
            metrics.confusion_matrix(y_test, grid_search.predict(X_test)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )

        print(f'Confusion matrix train: \n {cm_train}')
        print(f'Confusion matrix test: \n {cm_test}')
        print('\n')

    return grid_search.best_estimator_
