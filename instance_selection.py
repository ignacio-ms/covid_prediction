import numpy as np
import pandas as pd

from sklearn import neighbors, tree, metrics
from scipy import stats

import misc


def leaveOneOut(clasificador, X, y):
    """
    :param clasificador: Instancia de un clasificador de Scikit-Learn entrenada (con fit hecho con los datos de train o el subconjunto seleccionado)
    :param X: Matriz con los ejemplos de entrenamiento completo (para hacer leave one out)
    :param y: Vector con la salida de los ejemplos de entrenamiento completo (correspondientes a X)
    :return: Vector con la salida obtenida para cada ejemplo de X (siguiendo el esquema leave-one-out)
    """
    if type(clasificador) != neighbors.KNeighborsClassifier:
        prediction = clasificador.predict(X)
        return prediction

    distancias, vecinos = clasificador.kneighbors(X, n_neighbors=clasificador.n_neighbors + 1, return_distance=True)
    vecinosClase = clasificador._y[vecinos]

    mascara = distancias[:, 0] == 0
    vecinosClase[mascara, 0] = vecinosClase[mascara, -1]
    prediction = stats.mode(vecinosClase[:, :clasificador.n_neighbors], axis=1)[0]

    return prediction


@misc.timed
def ejecutaMetodoIS(funcionMetodo, clasificador, X_train, y_train, X_test, y_test, verbose=True):
    """
    Esta función se encajar de ejecutar un método de selección de instancias y dar los resultados sobre train y test
    con el clasificador introducido como parámetro
    :param funcionMetodo: Función con el método de selección de instancias a ejecutar
    :param clasificador: Instancia del clasificador a utilizar para obtener la precisión en train y test
    :param train: Conjunto de datos de entrenamiento leído del formato keel con campos data y target
    :param test: Conjunto de datos de test leído del formato keel con campos data y target
    :return: tupla con la máscara de ejemplos seleccionados, la precisión en train, precisión en test y porcentaje de reducción obtenido
    """
    nombreMetodo = funcionMetodo.__name__
    if verbose:
        print("Ejecutando " + nombreMetodo + "...")
    S = funcionMetodo(X_train, y_train)

    clasificador.fit(X_train[S], y_train[S])

    predictionTrain = clasificador.predict(X_train)
    accTrain = metrics.recall_score(y_train, predictionTrain, average='macro')

    predictionTest = clasificador.predict(X_test)
    accTest = metrics.recall_score(y_test, predictionTest, average='macro')

    reduction = (np.size(S) - np.count_nonzero(S)) / np.size(S) * 100

    if verbose:
        print("Resultados " + nombreMetodo + " python")
        print("Precisión en train: {}".format(accTrain))
        print("Precisión en test: {}".format(accTest))
        print("Reducción " + nombreMetodo + ": {} de {}".format(S.sum(), S.size))
        print("Reducción: %2.2f%%" % reduction)
        # En caso de ser un árbol, imprimimos el número de reglas obtenidas
        if type(clasificador) == tree.DecisionTreeClassifier:
            print("Número de reglas: {}".format(clasificador.tree_.node_count))

        print("Confusion matrix train: ")
        cm_train = pd.DataFrame(
            metrics.confusion_matrix(y_train, clasificador.predict(X_train)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )
        print(cm_train)
        print("Confusion matrix test: ")
        cm_test = pd.DataFrame(
            metrics.confusion_matrix(y_test, clasificador.predict(X_test)),
            index=['Real negative', 'Real positive'],
            columns=['Predicted negative', 'Predicted positive']
        )
        print(cm_test)

    nReglas = -1
    if type(clasificador) == tree.DecisionTreeClassifier:
        nReglas = clasificador.tree_.node_count
    return S, accTrain, accTest, reduction, nReglas


def sin_seleccion(X, y):
    S = np.full((X.shape[0],), True, dtype=bool)
    return S


def CNN(X, y, k=1):
    """
    Algoritmo CNN para la selección de instancias. Se comienza con dos ejemplos aleatorios (uno de cada clase) y cada ejemplo
    que se falla al ser clasificado por los ya seleccionados se añade a la selección (se para cuando ya no se añaden más ejemplos)
    :param X: Matriz con los ejemplos de entrenamiento (se asume que los ejemplos están normalizados)
    :param y: Vector con la salida de los ejemplos en X
    :param k: Valor de k a utilizar en ENN
    :return: Vector con la máscara de instancias seleccionadas
            (La posición S[i]=True indica que la instancia i ha sido seleccionada y False lo contrario)
    """
    np.random.seed(12312)
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)

    S = np.full((X.shape[0],), False, dtype=bool)

    nClases = np.unique(y)
    # ----- Seeleccionamos todas las instancias de clase positiva y una de clase negativa ----- #
    for c in nClases:
        indicesClase = np.where(y == c)[0]
        if c == 1:
            S[indicesClase] = True
        else:
            instanciaAleatoria = indicesClase[np.random.randint(len(indicesClase))]
            S[instanciaAleatoria] = True

    notS = np.logical_not(S)

    knn.fit(X[S], y[S])
    fallados = -1
    while fallados != 0:
        fallados = 0
        indices = np.where(notS)[0]
        for i in np.random.permutation(indices):
            if knn.predict(X[i].reshape(1, -1)) != y[i]:
                S[i] = True
                knn.fit(X[S], y[S])
                fallados += 1

        notS = np.logical_not(S)
        print("CNN, fin de iteración, fallados: {}, ejemplos en S: {}".format(fallados, np.sum(S)))

    return S


def RNN(X, y, k=1):
    """
    Algoritmo RNN para la selección de instancias. Se parte de la selección obtenida con CNN y se eliminan aquellas instancias
    que no provoquen que se falle ninguna instancia no seleccionada (las seleccionadas se aciertan por definición al estar en el subconjunto)
    :param X: Matriz con los ejemplos de entrenamiento (se asume que los ejemplos están normalizados)
    :param y: Vector con la salida de los ejemplos en X
    :param k: Valor de k a utilizar en ENN
    :return: Vector con la máscara de instancias seleccionadas
            (La posición S[i]=True indica que la instancia i ha sido seleccionada y False lo contrario)
    """
    np.random.seed(12312)
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)

    S = CNN(X, y)

    knn.fit(X[S], y[S])
    notS = np.logical_not(S)
    indices = np.where(S)[0]
    for i in np.random.permutation(indices):
        # ----- No eliminamos instancias de clase positiva ----- #
        if y[i] == 0:
            S[i] = False
            notS[i] = True
            knn.fit(X[S], y[S])
            salidas = knn.predict(X[notS])
            if np.not_equal(salidas, y[notS]).any():
                S[i] = True
                notS[i] = False

    return S


def RMHC(X, y, iteraciones=1000, k=1):
    """
     Algoritmo RMHC (Random Mutation Hill Climbing) para la selección de instancias.
      Se comienza con una selección aleatoria de s * nEjemplos instancias. Para cada iteración, se elige una instancia
      seleccionada y una no seleccionada para ser intercambiadas. Si el intercambio mejora la precisión (leave-one-out) sobre train
      se mantiene el cambio, sino se deshace
    :param X: Matriz con los ejemplos de entrenamiento (se asume que los ejemplos están normalizados)
    :param y: Vector con la salida de los ejemplos en X
    :param iteraciones: Número de iteraciones (intercambios) a probar
    :param k: Valor de k a utilizar en ENN
    :return: Vector con la máscara de instancias seleccionadas
            (La posición S[i]=True indica que la instancia i ha sido seleccionada y False lo contrario)
    """
    np.random.seed(12312)
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)

    # ----- Seleccionamos todas las instancias positivas balancedas con las negativas ----- #
    seleccionadas_pos = np.where(y == 1)[0]
    n_sel_neg = len(seleccionadas_pos)
    neg = np.random.permutation(np.where(y == 0)[0])

    seleccionadas = np.random.permutation(np.append(seleccionadas_pos, neg[:n_sel_neg]))
    noSeleccionadas = neg[n_sel_neg:]

    knn.fit(X[seleccionadas], y[seleccionadas])
    salidas = leaveOneOut(knn, X, y)
    acc = metrics.recall_score(salidas, y, average='macro')

    for i in range(0, iteraciones):

        # ----- Intercambiamos solo negativas ----- #
        while True:
            quitar = np.random.randint(len(seleccionadas))
            poner = np.random.randint(len(noSeleccionadas))
            if y[quitar] == 0:
                break

        aux = seleccionadas[quitar]
        seleccionadas[quitar] = noSeleccionadas[poner]
        knn = knn.fit(X[seleccionadas], y[seleccionadas])

        salidas = leaveOneOut(knn, X, y)
        accNew = metrics.recall_score(salidas, y, average='macro')

        if accNew < acc:
            seleccionadas[quitar] = aux
        else:
            noSeleccionadas[poner] = aux
            acc = accNew

        if i % 100 == 0:
            print("precision en iteracion {}: {}".format(i, acc))

    S = np.full((X.shape[0],), False, dtype=bool)
    S[seleccionadas] = True

    return S
