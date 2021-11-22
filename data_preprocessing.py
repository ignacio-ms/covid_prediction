import numpy as np
import pandas as pd

import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import misc


def read_data(file_name='corona_data.csv', class_name='corona_result', verbose=False):
    """
    This function read the data of a .csv and split it into an Examples DataFrame and a Output DataFrame. If required it
    also print some info about the data, such as the number of instances and variables, variable types or Missing values
    :param file_name: Name of the file containing the data
    :param class_name: Name of the variable to predict
    :param verbose: Boolean for printing some dataset information
    :return: DataFrames containing the Examples and Output of the dataset
    """
    data = pd.read_csv(file_name, header=0, na_values=['None', ''])

    y = data[[class_name]]
    X = data.drop(columns=class_name)

    if verbose:
        print(f'{file_name} info:')
        print(f'Instances: {X.shape[0]}')
        print(f'Variables: {X.shape[1]}\n')

        for c in y.squeeze().unique():
            print(f'{c} instances: {len(np.where(y == c)[0])}')

        print('Dtypes: ')
        print(X.dtypes)
        print(y.dtypes)

        print('\nMissing values: ')
        print(np.sum(X.isnull()))
        print(np.sum(y.isnull()))

    return X, y.squeeze()


def clean_data_output(X, y):
    """
    This function eliminates the instances for which we don't have the output information and encodes the remaning ones
    :param X: DataFrame containing the Examples of the complete dataset
    :param y: DataFrame containing the output os the complete dataset examples
    :return: DataFrames containing the Input and Output cleaned data of the dataset
    """
    # ----- Cleaning missing output values ----- #
    print(f'\nUnique output variables: {y.unique()}')
    missing_index = np.where(y == y.unique()[2])[0]
    print(f'Cleaning {len(missing_index)} instances of "{y.unique()[2]}" corona results...\n')

    y.drop(missing_index, inplace=True)
    X.drop(missing_index, inplace=True)

    # ------------- Encode output -------------- #
    y = ce.OrdinalEncoder().fit_transform(y) - 1

    return X, y.squeeze()


@misc.timed
def impute_data_na(X, y):
    """
    This function imputes the missing values of the diferent variables of the input data.
    :param X: DataFrame containing the Examples of the complete dataset
    :param y: DataFrame containing the output os the dataset examples
    :return: DataFrames containing the Input and Output cleaned data of the dataset
    """

    # ----- Having count that all variables are categorical/booleans ---- #
    # --- we are going to apply most frequent imputer for all of them --- #
    var_names = X.columns
    print(f'Imputing missing values by most frequent...\n')
    preprocessing = ColumnTransformer(
        transformers=[
            ('most_frecuent', SimpleImputer(strategy='most_frequent'), var_names)
        ]
    ).fit(X)

    X = preprocessing.transform(X)
    X = pd.DataFrame(X, columns=var_names)

    return X, y


def encode_categorical(X, y):
    """
    This function encodes the categorical variables using an Ordinal Encoder and a Target Encoder
    :param X: DataFrame containing the Examples of the complete dataset
    :param y: DataFrame containing the output of the complete dataset
    :return: DataFrames containing the encoded variables
    """

    print('Encoding categorical variables...\n')
    categorical_vars = X.select_dtypes(include='object').columns.tolist()
    categorical_vars_2 = []
    categorical_vars_more_than_2 = []
    categorical_vars_between_3_6 = []

    for var in categorical_vars:
        n_values = len(X[var].unique())
        if n_values <= 2:
            categorical_vars_2.append(var)
        elif n_values <= 6:
            categorical_vars_between_3_6.append(var)
        else:
            categorical_vars_more_than_2.append(var)

    X = ce.OrdinalEncoder(cols=categorical_vars_2).fit_transform(X)
    X = ce.one_hot.OneHotEncoder(cols=categorical_vars_between_3_6).fit_transform(X)
    X = ce.target_encoder.TargetEncoder(cols=categorical_vars_more_than_2, smoothing=0.0000001).fit_transform(X, y)

    X[categorical_vars_2] = X[categorical_vars_2] - 1

    print()
    for var in X.columns:
        print(f'{var} values: {X[var].unique()}')

    return X


def reduce_dataset(file_name='corona_data.csv', size_to_drop=0.95):
    data = pd.read_csv(file_name, header=0, na_values=['None', ''])

    indexes = np.arange(data.shape[0])
    np.random.shuffle(indexes)
    data.drop(indexes[:int(data.shape[0]*size_to_drop)], inplace=True)

    data.to_csv('corona_data_reduced_95.csv', index=False)
