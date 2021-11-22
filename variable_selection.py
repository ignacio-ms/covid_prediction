import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import TransformerMixin
from sklearn import feature_selection as fs
from sklearn import model_selection
from sklearn import tree, metrics
from sklearn import pipeline
from sklearn.decomposition import PCA

import misc


class corr_matrix(TransformerMixin):

    def __init__(self, threshold=0.8, verbose=False):
        self.threshold = threshold
        self.verbose = verbose
        self.t = None

    def fit(self, X_train, y_train):
        c_matrix = np.abs(X_train.corr())

        upper = c_matrix.where(np.triu(np.ones(c_matrix.shape), k=1).astype(np.bool))
        index_var_to_drop = [i for i, column in enumerate(upper.columns) if any(upper[column] > self.threshold)]
        index_var_to_drop = np.arange(X_train.shape[1]) == index_var_to_drop

        self.t = np.logical_not(index_var_to_drop)
        if self.verbose:
            print("Correlation matrix variable selection: \n")
            print(f'Keeped variables: {X_train.columns[self.t].tolist()}')

        return self

    def transform(self, X):
        return X.iloc[:, self.t]

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def plot_correlation_matrix(X):
    """
    This function plots the correlation matrix between the variables.
    :param X: DataFrame containing The examples of the dataset
    :return:
    """
    correlaciones = X.corr()

    # ----- Correlarion Matrix ----- #
    fig = plt.figure(figsize=(50, 50))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlaciones, vmin=-1, vmax=1, cmap=plt.cm.rainbow)
    fig.colorbar(cax)
    ticks = np.arange(0, len(X.columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # ------- Variable Names ------- #
    names = X.columns
    ax.set_xticklabels(names, rotation='90')
    ax.set_yticklabels(names)
    plt.show()

    return


class t_test(TransformerMixin):

    def __init__(self, selection_type='Kbest', K=None, percentil=None):
        self.selection_type = selection_type
        self.K = K
        self.percentil = percentil
        self.t = None

    def fit(self, X_train, y_train):
        N0 = np.sum(np.array(y_train == 0))
        N1 = np.sum(np.array(y_train == 1))
        meansC0 = np.mean(X_train[np.array(y_train == 0)]).values
        meansC1 = np.mean(X_train[np.array(y_train == 1)]).values
        stdsC0 = np.std(X_train[np.array(y_train == 0)]).values
        stdsC1 = np.std(X_train[np.array(y_train == 1)]).values
        varsC0 = np.var(X_train[np.array(y_train == 0)]).values
        varsC1 = np.var(X_train[np.array(y_train == 1)]).values

        t = np.empty(X_train.shape[1])
        for i in range(len(t)):
            if varsC0[i] == varsC1[i]:
                t[i] = abs(meansC0[i] - meansC1[i]) / (stdsC0[i] * np.sqrt(abs(1 / N0 - 1 / N1)))
            else:
                t[i] = abs(meansC0[i] - meansC1[i]) / np.sqrt(abs(stdsC0[i] / N0 - stdsC1[i] / N1))

        self.t = t
        return self

    def transform(self, X):
        indices = np.argsort(self.t)[::-1]
        booleanos = np.zeros(X.shape[1], ).astype('bool')

        if self.selection_type == 'Kbest':
            booleanos[indices[:self.K]] = True
        elif self.selection_type == 'percentil':
            booleanos[indices[:int(X.shape[1] * (self.percentil / 100))]] = True
        else:
            print(f'{self.selection_type} is not a valid selection_type\n')
            return

        return X.iloc[:, booleanos]

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


@misc.timed
def test_selection(X_train, X_test, y_train, y_test, results=None, verbose=False):
    if results is None:
        results = {}

    params_selection = {
        corr_matrix: {
            'corr_matrix__threshold': [0.6, 0.7, 0.8]
        },
        t_test: {
            't_test__selection_type': ['percentil', 'Kbest'],
            't_test__k': np.arange(1, X_train.shape[1] + 1),
            't_test__percentil': [10, 20, 30]
        },
        fs.SelectKBest: {
            'SelectKBest__score_func': [fs.chi2, fs.f_classif, fs.mutual_info_classif],
            'SelectKBest__k': np.arange(1, X_train.shape[1] + 1)
        },
        fs.SelectPercentile: {
            'SelectPercentile__score_func': [fs.chi2, fs.f_classif, fs.mutual_info_classif],
            'SelectPercentile__percentile': [10, 20, 30]
        }
    }

    for selection_type in params_selection:
        print(f'Running {selection_type.__name__} variable selection: \n')
        pipe = pipeline.Pipeline([
            (selection_type.__name__, selection_type()),
            ('tree', tree.DecisionTreeClassifier())
        ])
        grid_search = model_selection.GridSearchCV(
            pipe,
            params_selection[selection_type],
            scoring=metrics.make_scorer(metrics.recall_score, average='macro'),
            cv=model_selection.StratifiedKFold(n_splits=10)
        ).fit(X_train, y_train)

        cm = pd.DataFrame(
            metrics.confusion_matrix(y_test, grid_search.predict(X_test)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )

        methodResults = {
            'acc_train': metrics.recall_score(y_train, grid_search.predict(X_train), average='macro'),
            'acc_test': metrics.recall_score(y_test, grid_search.predict(X_test), average='macro'),
            'params': grid_search.best_params_,
            'confusion_matrix': cm,
        }
        results[selection_type.__name__] = methodResults

        if verbose:
            print(f"Train accuracy: {methodResults['acc_train']} \n")
            print(f"Test accuracy: {methodResults['acc_test']} \n")
            print(f"Best parameters: {methodResults['params']} \n")
            print('Confusion matrix: ')
            print(cm)
            print('\n')

    return results


@misc.timed
def test_selection_wrapper(X_train, X_test, y_train, y_test, results=None, verbose=False):
    if results is None:
        results = {}

    lista_acc_train = []
    lista_acc_test = []

    classif_best = None
    n_best = None
    i_best = None
    X_test_sel = None

    for i, n in enumerate(range(1, X_train.shape[1])):

        rfe = fs.RFE(
            tree.DecisionTreeClassifier(),
            n_features_to_select=n,
            step=1
        ).fit(X_train, y_train)

        X_train_seleccion = rfe.transform(X_train)
        X_test_seleccion = rfe.transform(X_test)

        classif = tree.DecisionTreeClassifier().fit(X_train_seleccion, y_train)

        lista_acc_train.append(metrics.recall_score(y_train, classif.predict(X_train_seleccion), average='macro'))
        lista_acc_test.append(metrics.recall_score(y_test, classif.predict(X_test_seleccion), average='macro'))

        if np.argmax(lista_acc_test) == i:
            n_best = n
            i_best = i
            classif_best = classif
            X_test_sel = X_test_seleccion

    cm = pd.DataFrame(
        metrics.confusion_matrix(y_test, classif_best.predict(X_test_sel)),
        index=['Real negative', 'Real positive'],
        columns=['Predicted negative', 'Predicted positive']
    )

    methodResults = {
        'acc_train': lista_acc_train[i_best],
        'acc_test': lista_acc_test[i_best],
        'params': n_best,
        'confusion_matrix': cm
    }
    results['RFE'] = methodResults

    if verbose:
        print(f"Train accuracy: {methodResults['acc_train']} \n")
        print(f"Test accuracy: {methodResults['acc_test']} \n")
        print(f"Best parameters: {methodResults['params']} \n")
        print('Confusion matrix: ')
        print(cm)
        print('\n')

    return results


@misc.timed
def test_selection_PCA(X_train, X_test, y_train, y_test, results=None, verbose=False):
    if results is None:
        results = {}

    lista_acc_train = []
    lista_acc_test = []
    lista_por_inf = []

    classif_best = None
    n_best = None
    i_best = None
    X_test_sel = None

    for i, n in enumerate(range(1, X_train.shape[1])):
        pca = PCA(n_components=n, svd_solver='full').fit(X_train, y_train)

        X_train_seleccion = pca.transform(X_train)
        X_test_seleccion = pca.transform(X_test)

        classif = tree.DecisionTreeClassifier().fit(X_train_seleccion, y_train)

        lista_acc_train.append(metrics.recall_score(y_train, classif.predict(X_train_seleccion), average='macro') * 100)
        lista_acc_test.append(metrics.recall_score(y_test, classif.predict(X_test_seleccion), average='macro') * 100)
        lista_por_inf.append(np.sum(pca.explained_variance_ratio_) * 100)

        if np.argmax(lista_acc_test) == i:
            n_best = n
            i_best = i
            classif_best = classif
            X_test_sel = X_test_seleccion

    cm = pd.DataFrame(
        metrics.confusion_matrix(y_test, classif_best.predict(X_test_sel)),
        index=['Real negative', 'Real positive'],
        columns=['Predicted negative', 'Predicted positive']
    )

    methodResults = {
        'acc_train': lista_acc_train[i_best],
        'acc_test': lista_acc_test[i_best],
        'params': n_best,
        'confusion_matrix': cm
    }
    results['PCA'] = methodResults

    if verbose:
        print(f"Train accuracy: {methodResults['acc_train']} \n")
        print(f"Test accuracy: {methodResults['acc_test']} \n")
        print(f"Best parameters: {methodResults['params']} \n")
        print('Confusion matrix: ')
        print(cm)

        plt.figure(figsize=(7, 5))
        plt.plot(lista_acc_train, 'r', label='acc_train')
        plt.plot(lista_acc_test, 'b', label='acc_test')
        plt.plot(lista_por_inf, 'g', label='Info')

        plt.xlabel("n_components")
        plt.ylabel("%")
        plt.title("Comportamiento n_componentes PCA")

        plt.legend()
        plt.show()

        print('\n')

    return results
