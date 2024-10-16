'''
Gradient Boosted Decision Tree (GBDT) implementation with Nested Cross-Validation (NCV)
and SHapely Additive explanations and figures (SHAP)

This code is a reused version with minor modifications by Danila Valko (d.v.valko[at]gmail.com) 
of Dr. David Steyrl's codebase (david.steyrl[at]gmail.com).
See source code and description can be accessed at https://github.com/univiemops/iml 
and one example here https://doi.org/10.31219/osf.io/um69t.
The SHAP project can be accessed here https://github.com/shap/shap.

You can cite this tool by mentioning one of Dr. Steryl's works, e.g. https://doi.org/10.1038/s41598-024-65088-z
or https://doi.org/10.1371/journal.pone.0304285

NCV involves repeated splits of the data into training and testing sets.
In the outer CV loop a 10 times 5-folds split scheme grouped by participants was applied. 
Model complexity tuning, carried out in a nested (inner) CV procedure using training data only 
(5-folds repeated until a minimum 1000 predictions), utilizes a random search scheme
to identify optimal complexity parameters (column sample per tree 0.1  to  1;  
using extra trees True/False; path smoothing 1 to 1000 log-scale).

The selected parameters are then employed in the main CV loop along 
with constant parameters (learning rate 0.01; number of leaves 100; 
number of boosting rounds 1000; max bin 100) for model training and testing.
Regression performance is measured using prediction coefficient of determination (prediction R²)
and mean absolute error, while classification performance is evaluated 
using balanced classification accuracy. Analysis of single predictors' predictive 
importance was performed using SHAP (SHapely Additive explanations).  

See:
    https://osf.io/m5uw7/?view_only=fd17613843f84c96a9c6a24f0afa152b
    Sklearn compatible Repeated Group KFold Cross-Validation
    Statistical testing using interpretable machine-learning, v697
    Plot results of statistical testing using interpretable machine-learning, v244
    @author: Dr. David Steyrl david.steyrl@gmail.com
    
    Software was initially developed for:
        Todorova, B., Steyrl, D., Hornsey, M., 
        Pearson, S., Brick, C., Lange, F., … Doell, K. C. (2024, September 10). 
        Machine learning identifies key individual and nation-level factors predicting 
        climate-relevant beliefs and behaviors. https://doi.org/10.31219/osf.io/um69t
        
'''

import numpy as np
import os
import math
import pandas as pd
import pickle
import warnings
from tqdm import tqdm 
from collections import Counter
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from scipy.stats import loguniform
from scipy.stats import randint
from scipy.stats import uniform
from shap.explainers import Tree as TreeExplainer
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import TargetEncoder
from sklearn.utils import shuffle
#from sklearn_repeated_group_k_fold import RepeatedGroupKFold
from time import time

from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_random_state
from sklearn.model_selection._split import _BaseKFold, _RepeatedSplits

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations
from scipy.stats import t
from shap import dependence_plot
from shap import Explanation
from shap.plots import beeswarm
from shap.plots import scatter
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore', 'Setting an item of incompatible dtype is')

################################################ KFold objects

'''
Sklearn compatible Repeated Group KFold Cross-Validation
v1 Source: https://github.com/BbChip0103/sklearn_repeated_group_k_fold/blob/
main/sklearn_repeated_group_k_fold.py
@author: Dr. David Steyrl david.steyrl@gmail.com
'''
class GroupKFold(_BaseKFold):
    '''
    K-fold iterator variant with non-overlapping groups.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    The folds are approximately balanced in the sense that the number of
    distinct groups is approximately the same in each fold.
    Read more in the :ref:`User Guide <group_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state : int, RandomState instance or None, default=None
        When `shuffle` is True, `random_state` affects the ordering of the
        indices, which controls the randomness of each fold. Otherwise, this
        parameter has no effect.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import GroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 2, 2])
    >>> groups = np.array([0, 1, 2, 3])
    >>> group_kfold = GroupKFold(n_splits=2, shuffle=True, random_state=12345)
    >>> group_kfold.get_n_splits(X, y, groups)
    2
    >>> print(group_kfold)
    GroupKFold(n_splits=2, random_state=12345, shuffle=True)
    >>> for train_index, test_index in group_kfold.split(X, y, groups):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    ...    print(X_train, X_test, y_train, y_test)
    ...
    TRAIN: [0 1] TEST: [2 3]
    [[1 2]
     [3 4]] [[5 6]
     [7 8]] [0 0] [2 2]
    TRAIN: [2 3] TEST: [0 1]
    [[5 6]
     [7 8]] [[1 2]
     [3 4]] [2 2] [0 0]

    See Also
    --------
    LeaveOneGroupOut : For splitting the data according to explicit
        domain-specific stratification of the dataset.
    '''

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle,
                         random_state=random_state)

    def _iter_test_indices(self, X, y, groups):
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        groups = check_array(groups, ensure_2d=False, dtype=None)

        unique_groups, groups = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)

        if self.n_splits > n_groups:
            raise ValueError("Cannot have number of splits n_splits=%d greater"
                             " than the number of groups: %d."
                             % (self.n_splits, n_groups))

        # Weight groups by their number of occurrences
        n_samples_per_group = np.bincount(groups)

        # Distribute the most frequent groups first
        indices = np.argsort(n_samples_per_group)[::-1]

        if self.shuffle:
            rng = check_random_state(self.random_state)
            for n_sample in np.unique(n_samples_per_group):
                same_n_indices_index = np.where(
                    n_samples_per_group == n_sample)[0]
                target_chunk = indices[same_n_indices_index]
                rng.shuffle(target_chunk)
                indices[same_n_indices_index] = target_chunk

        n_samples_per_group = n_samples_per_group[indices]

        # Total weight of each fold
        n_samples_per_fold = np.zeros(self.n_splits)

        # Mapping from group index to fold index
        group_to_fold = np.zeros(len(unique_groups))

        # Distribute samples by adding the largest weight to the lightest fold
        for group_index, weight in enumerate(n_samples_per_group):
            lightest_fold = np.argmin(n_samples_per_fold)
            n_samples_per_fold[lightest_fold] += weight
            group_to_fold[indices[group_index]] = lightest_fold

        indices = group_to_fold[groups]

        for f in range(self.n_splits):
            yield np.where(indices == f)[0]

    def split(self, X, y=None, groups=None):
        '''
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        '''

        return super().split(X, y, groups)


class RepeatedGroupKFold(_RepeatedSplits):
    '''
    Repeated Group K-Fold cross validator. Repeats Group K-Fold n times with
    different randomization in each repetition.
    Read more in the :ref:`User Guide <repeated_group_k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=10
        Number of times cross-validator needs to be repeated.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of each repeated cross-validation instance.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import RepeatedGroupKFold
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 0, 2, 2])
    >>> groups = np.array([0, 1, 2, 3])
    >>> rkf = RepeatedGroupKFold(n_splits=2, n_repeats=2, random_state=2652124)
    >>> for train_index, test_index in rkf.split(X, y, groups):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    ...
    TRAIN: [1 2] TEST: [0 3]
    TRAIN: [0 3] TEST: [1 2]
    TRAIN: [0 1] TEST: [2 3]
    TRAIN: [2 3] TEST: [0 1]

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    RepeatedStratifiedKFold : Repeats Stratified K-Fold n times.
    '''

    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            GroupKFold, n_repeats=n_repeats,
            random_state=random_state, n_splits=n_splits)

    def split(self, X, y=None, groups=None):
        '''
        Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        '''

        n_repeats = self.n_repeats
        rng = check_random_state(self.random_state)

        for idx in range(n_repeats):
            cv = self.cv(random_state=rng, shuffle=True, **self.cvargs)
            for train_index, test_index in cv.split(X, y, groups):
                yield train_index, test_index


################################################# analysis
#################################################
#################################################

def lfp(path_load):
    '''
    Returns pickle file at load path.

    Parameters
    ----------
    path_load : string
        Path to pickle file.

    Returns
    -------
    data : pickle
        Returns stored data.
    '''

    # Load from pickle file ---------------------------------------------------
    # Load
    with open(path_load, 'rb') as filehandle:
        # Load data from binary data stream
        data = pickle.load(filehandle)

    # Return data -------------------------------------------------------------
    return data

def create_dir(path):
    '''
    Create specified directory if not existing.

    Parameters
    ----------
    path : string
        Path to to check to be created.

    Returns
    -------
    None.
    '''

    # Create dir of not existing ----------------------------------------------
    # Check if dir exists
    os.makedirs(path, exist_ok=True)

    # Return None -------------------------------------------------------------
    return


def prepare(task):
    '''
    Prepare analysis pipeline, prepare seach_space.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.

    Returns
    -------
    pipe : scikit-learn compatible analysis pipeline
        Prepared pipe object.
    space : dict
        Space that should be searched for optimale parameters.
    '''

    # Make preprocessing pipe -------------------------------------------------
    # Instatiate target-encoder
    te = TargetEncoder(categories=task['te_categories'],
                       target_type='continuous',
                       smooth='auto',
                       cv=5,
                       shuffle=True,
                       random_state=None)
    # Get categorical predictors for target-encoder
    coltrans = ColumnTransformer(
        [('con_pred', 'passthrough', task['X_CON_NAMES']),
         ('bin_pred', 'passthrough', task['X_CAT_BIN_NAMES']),
         ('mult_pred', te, task['X_CAT_MULT_NAMES']),
         ],
        remainder='drop',
        sparse_threshold=0,
        n_jobs=1,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False)
    # Pipeline
    pre_pipe = Pipeline([('coltrans', coltrans),
                         ('std_scaler', StandardScaler())],
                        memory=None,
                        verbose=False)

    # Make predictor ----------------------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # Estimator
        estimator = LGBMRegressor(
            boosting_type='gbdt',
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=1000,
            subsample_for_bin=100000,
            objective=task['OBJECTIVE'],
            min_split_gain=0.000001,
            min_child_weight=0.000001,
            min_child_samples=2,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=None,
            n_jobs=1,
            importance_type='gain',
            **{'data_random_seed': None,
               'data_sample_strategy': 'goss',
               'extra_seed': None,
               'feature_fraction_seed': None,
               'feature_pre_filter': False,
               'force_col_wise': True,
               'min_data_in_bin': 1,
               'top_rate': 0.5,
               'verbosity': -1,
               })
        # Search space
        space = {
            'estimator__regressor__colsample_bytree': uniform(0.5, 0.5),
            'estimator__regressor__extra_trees': [True, False],
            'estimator__regressor__path_smooth': loguniform(1, 1000),
            }
        # Add scaler to the estimator
        estimator = TransformedTargetRegressor(
            regressor=estimator,
            transformer=StandardScaler(),
            func=None,
            inverse_func=None,
            check_inverse=True)
    # Classification
    elif task['OBJECTIVE'] == 'binary' or task['OBJECTIVE'] == 'multiclass':
        # Estimator
        estimator = LGBMClassifier(
            boosting_type='gbdt',
            num_leaves=100,
            max_depth=-1,
            learning_rate=0.01,
            n_estimators=1000,
            subsample_for_bin=100000,
            objective=task['OBJECTIVE'],
            class_weight=None,
            min_split_gain=0.000001,
            min_child_weight=0.000001,
            min_child_samples=2,
            subsample=1.0,
            subsample_freq=0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=None,
            n_jobs=1,
            importance_type='gain',
            **{'data_random_seed': None,
               'data_sample_strategy': 'goss',
               'extra_seed': None,
               'feature_fraction_seed': None,
               'feature_pre_filter': False,
               'force_col_wise': True,
               'min_data_in_bin': 1,
               'top_rate': 0.5,
               'verbosity': -1,
               })
        # Search space
        space = {
            'estimator__colsample_bytree': uniform(0.5, 0.5),
            'estimator__extra_trees': [True, False],
            'estimator__path_smooth': loguniform(1, 1000),
            }
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Make full pipeline ------------------------------------------------------
    # Analyis pipeline
    pipe = Pipeline(
        [('preprocessing', pre_pipe),
         ('estimator', estimator)],
        memory=None,
        verbose=False).set_output(transform='pandas')

    # Return pipe and space ---------------------------------------------------
    return pipe, space


def split_data(df, i_trn, i_tst):
    '''
    Split dataframe in training and testing dataframes.

    Parameters
    ----------
    df : dataframe
        Dataframe holding the data to split.
    i_trn : numpy array
        Array with indices of training data.
    i_tst : numpy array
        Array with indices of testing data.

    Returns
    -------
    df_trn : dataframe
        Dataframe holding the training data.
    df_tst : dataframe
         Dataframe holding the testing data.
    '''

    # Split dataframe via index -----------------------------------------------
    # Dataframe is not empty
    if not df.empty:
        # Make split
        df_trn = df.iloc[i_trn].reset_index(drop=True)
        # Make split
        df_tst = df.iloc[i_tst].reset_index(drop=True)
    # Dataframe is empty
    else:
        # Make empty dataframes
        df_trn, df_tst = pd.DataFrame(), pd.DataFrame()

    # Return train test dataframes --------------------------------------------
    return df_trn, df_tst


def get_class_w(y):
    '''
    Compute class weights over array by counting occurrences.

    Parameters
    ----------
    y : ndarray
        Array containing class labels.

    Returns
    -------
    class_weights : dictionary
        Dictionary of class weights with class labels as keys.
    '''

    # Get class weights -------------------------------------------------------
    # Count unique classes occurances
    counter = Counter(y.squeeze())
    # n_samples
    total_class = sum(counter.values())
    # Get weights
    w = {key: np.round(count/total_class, 4) for key, count in counter.items()}

    # Return class weights ----------------------------------------------------
    return w


def weighted_accuracy_score(y_true, y_pred, class_weights):
    '''
    Computes accuracy score weighted by the inverse if the frequency of a
    class.

    Parameters
    ----------
    y_true : ndarray
        True values.
    y_pred : ndarray
        Predicted values.
    class_weights : dictionary
        Class weights as inverse of frequency of class.

    Returns
    -------
    accuracy : float
        Prediction accuracy.
    '''

    # Get sample weights ------------------------------------------------------
    # Make sample weights dataframe
    w = y_true.squeeze().map(class_weights).to_numpy()

    # Return sample weighted accuracy -----------------------------------------
    return accuracy_score(y_true, y_pred, sample_weight=w)


def print_tune_summary(task, i_cv, n_splits, hp_params, hp_score):
    '''
    Print best paramters and related score to console.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    i_cv : int
        Current cv repetition.
    n_splits : int
        Number of splits in inner cv
    hp_params : dictionary
        Best hyper params found.
    hp_score : dictionary
        Score for best hyper params found.

    Returns
    -------
    None.
    '''

    # Print analysis name
    print('Analysis: '+task['ANALYSIS_NAME'])
    # Print data set
    # print('Dataset: '+task['PATH_TO_DATA'])
    # Cross-validation --------------------------------------------------------
    if task['TYPE'] == 'CV':
        # Regression
        if task['OBJECTIVE'] == 'regression':
            # Print general information
            print(str(task['i_y'])+'.'+str(i_cv)+' | ' +
                  'n rep outer cv: '+str(task['N_REP_OUTER_CV'])+' | ' +
                  'n rep inner cv: '+str(n_splits)+' | ' +
                  'best neg MSE: '+str(np.round(hp_score, decimals=4)))
        # Classification
        elif (task['OBJECTIVE'] == 'binary' or
              task['OBJECTIVE'] == 'multiclass'):
            # Print general information
            print(str(task['i_y'])+'.'+str(i_cv)+' | ' +
                  'n rep outer cv: '+str(task['N_REP_OUTER_CV'])+' | ' +
                  'n rep inner cv: '+str(n_splits)+' | ' +
                  'acc: '+str(np.round(hp_score, decimals=4)))
        # Other
        else:
            # Raise error
            raise ValueError('OBJECTIVE not found.')
    # Train-Test split --------------------------------------------------------
    elif task['TYPE'] == 'TT':
        # Regression
        if task['OBJECTIVE'] == 'regression':
            # Print general information
            print(str(task['i_y'])+'.'+str(i_cv)+' | ' +
                  'n rep inner cv: '+str(n_splits)+' | ' +
                  'best neg MSE: '+str(np.round(hp_score, decimals=4)))
        # Classification
        elif (task['OBJECTIVE'] == 'binary' or
              task['OBJECTIVE'] == 'multiclass'):
            # Print general information
            print(str(task['i_y'])+'.'+str(i_cv)+' | ' +
                  'n rep inner cv: '+str(n_splits)+' | ' +
                  'acc: '+str(np.round(hp_score, decimals=4)))
        # Other
        else:
            # Raise error
            raise ValueError('OBJECTIVE not found.')
    # Other -------------------------------------------------------------------
    else:
        # Raise error
        raise ValueError('TYPE not found.')
    # Print best hyperparameter and related score for regression task
    print(str(hp_params))

    # Return None -------------------------------------------------------------
    return


def tune_pipe(task, i_cv, pipe, space, g_trn, x_trn, y_trn):
    '''
    Inner loop of the nested cross-validation. Runs a search for optimal
    hyperparameter (random search).
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009.
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    i_cv : int
        Current iteration of outer cross-validation.
    pipe : pipeline object
        Analysis pipeline.
    space : dict
        Space that should be searched for optimale parameters.
    g_trn : ndarray (n_samples)
        Group data.
    x_trn : ndarray (n_features x n_samples)
        Predictor train data.
    y_trn : ndarray (n_samples)
        Target train data.

    Returns
    -------
    pipe : pipeline object
        Fitted pipeline object with tuned parameters.
    best parameters : dict
        Best hyperparameters of the pipe.
    '''

    # Get scorer --------------------------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # neg_mean_squared_error
        scorer = 'neg_mean_squared_error'
    # Classification
    elif task['OBJECTIVE'] == 'binary' or task['OBJECTIVE'] == 'multiclass':
        # Weighted accuracy for classification
        scorer = make_scorer(weighted_accuracy_score,
                             greater_is_better=True,
                             **{'class_weights': get_class_w(y_trn)})
        # Add current class weights to the pipe
        pipe.set_params(**{'estimator__class_weight': get_class_w(y_trn)})
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Tune analysis pipeline --------------------------------------------------
    # Choose n_repeats to approx N_SAMPLES_INNER_CV predictions, min 1, max 10
    n_repeats = min(10, max(1, int(task['N_SAMPLES_INNER_CV'] /
                                   g_trn.shape[0])))
    # Instatiate random parameter search
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=task['N_SAMPLES_RS'],
        scoring=scorer,
        n_jobs=task['N_JOBS'],
        refit=True,
        cv=RepeatedGroupKFold(n_splits=5,
                              n_repeats=n_repeats,
                              random_state=None),
        verbose=0,
        pre_dispatch='2*n_jobs',
        random_state=None,
        error_score=0,
        return_train_score=False)
    # Random search for best parameter
    search.fit(x_trn, y_trn.squeeze(), groups=g_trn)
    # Print tune summary
    print_tune_summary(task, i_cv, n_repeats, search.best_params_,
                       search.best_score_)

    # Return tuned analysis pipe ----------------------------------------------
    return search.best_estimator_, search.best_params_


def score_predictions(task, pipe, x_tst, y_tst, y):
    '''
    Compute scores for predictions based on task.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    pipe : pipeline object
        Analysis pipeline.
    x_tst : ndarray (n_features x n_samples)
        Predictor test data.
    y_tst : ndarray (n_samples)
        Target test data.
    y : ndarray
        All available target data to compute true class weights for scoring.

    Returns
    -------
    scores : dict
        Returns scoring results. MAE, MSE and R² if task is regression.
        ACC and true class weights if task is classification.
    '''

    # Predict -----------------------------------------------------------------
    # Predict test samples
    y_pred = pipe.predict(x_tst)

    # Score results -----------------------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # Score predictions in terms of mae
        mae = mean_absolute_error(y_tst, y_pred)
        # Score predictions in terms of mse
        mse = mean_squared_error(y_tst, y_pred)
        # Score predictions in terms of R²
        r2 = r2_score(y_tst, y_pred)
        # Results
        scores = {'y_true': y_tst.squeeze().to_numpy(),
                  'y_pred': y_pred,
                  'mae': mae,
                  'mse': mse,
                  'r2': r2}
    # Classification
    elif task['OBJECTIVE'] == 'binary' or task['OBJECTIVE'] == 'multiclass':
        # Get class weights
        class_weights = get_class_w(y)
        # Calculate model fit in terms of acc
        acc = weighted_accuracy_score(y_tst, y_pred, class_weights)
        # Results
        scores = {'y_true': y_tst.squeeze().to_numpy(),
                  'y_pred': y_pred,
                  'acc': acc,
                  'class_weights': class_weights}
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Return scores -----------------------------------------------------------
    return scores


def get_explainations(task, pipe, x_trn, x_tst):
    '''
    Get SHAP (SHapley Additive exPlainations) model explainations.
    Ref: Molnar, Christoph. 'Interpretable machine learning. A Guide for
    Making Black Box Models Explainable', 2019.
    https://christophm.github.io/interpretable-ml-book/.
    Ref: Lundberg, Scott M., and Su-In Lee. “A unified approach to
    interpreting model predictions.” Advances in Neural Information Processing
    Systems. 2017.
    Ref: Lundberg, Scott M., Gabriel G. Erion, and Su-In Lee. “Consistent
    individualized feature attribution for tree ensembles.” arXiv preprint
    arXiv:1802.03888 (2018).
    Ref: Sundararajan, Mukund, and Amir Najmi. “The many Shapley values for
    model explanation.” arXiv preprint arXiv:1908.08474 (2019).
    Ref: Janzing, Dominik, Lenon Minorics, and Patrick Blöbaum. “Feature
    relevance quantification in explainable AI: A causality problem.” arXiv
    preprint arXiv:1910.13413 (2019).
    Ref: Slack, Dylan, et al. “Fooling lime and shap: Adversarial attacks on
    post hoc explanation methods.” Proceedings of the AAAI/ACM Conference on
    AI, Ethics, and Society. 2020.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    pipe : pipeline object
        Fitted pipeline object with tuned parameters.
    x_trn : ndarray (n_features x n_samples)
        Background data.
    x_tst : ndarray (n_features x n_samples)
        Test data for shap computation.

    Returns
    -------
    imp : shap explainer object
        SHAP based predictor importance.
    '''

    # Get SHAP test data ------------------------------------------------------
    # Subsample test data
    x_tst_shap_orig = x_tst.sample(
        n=min(x_tst.shape[0], task['MAX_SAMPLES_SHAP']),
        random_state=3141592,
        ignore_index=True)
    # Transform shap test data
    x_tst_shap = pipe[0].transform(x_tst_shap_orig)

    # Explainer and Explainations ---------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # Get predictor
        predictor = pipe[1].regressor_
    # Classification
    elif task['OBJECTIVE'] == 'binary' or task['OBJECTIVE'] == 'multiclass':
        # Get predictor
        predictor = pipe[1]
    # Get explainer
    explainer = TreeExplainer(
        predictor,
        data=None,
        model_output='raw',
        feature_perturbation='tree_path_dependent',
        feature_names=None,
        approximate=False)
    # Get explainations with interactions
    if task['SHAP_INTERACTIONS']:
        # Get shap values
        shap_explainations = explainer(x_tst_shap,
                                       interactions=True,
                                       check_additivity=False)
    # Get explainations without interactions
    elif not task['SHAP_INTERACTIONS']:
        # Get shap values
        shap_explainations = explainer(x_tst_shap,
                                       interactions=False,
                                       check_additivity=False)
    # Other
    else:
        # Raise error
        raise ValueError('Invalid value for SHAP_INTERACTIONS.')

    # Prepare shap_explainations ----------------------------------------------
    # Replace scaled data in shap explainations with unscaled
    shap_explainations.data = x_tst_shap_orig
    # If regression
    if task['OBJECTIVE'] == 'regression':
        # Rescale shap values from scaled data to original space
        shap_explainations.values = (shap_explainations.values *
                                     pipe[1].transformer_.scale_[0])
        # Rescale shap base values from scaled data to original space
        shap_explainations.base_values = ((shap_explainations.base_values *
                                          pipe[1].transformer_.scale_[0]) +
                                          pipe[1].transformer_.mean_[0])

    # Return shap explainations -----------------------------------------------
    return shap_explainations


def s2p(path_save, variable):
    '''
    Save variable as pickle file at path.

    Parameters
    ----------
    path_save : string
        Path ro save variable.
    variable : string
        Variable to save.

    Returns
    -------
    None.
    '''

    # Save --------------------------------------------------------------------
    # Save variable as pickle file
    with open(path_save, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(variable, filehandle)

    # Return None -------------------------------------------------------------
    return


def print_current_results(task, t_start, scores, scores_sh):
    '''
    Print current results to console.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    t_start : time
        Start time of the current cross-validation loop.
    scores : dict
        Scores dict.
    scores_sh : dict
        Scores with shuffled data dict.

    Returns
    -------
    None.
    '''

    # Print results -----------------------------------------------------------
    # Regression
    if task['OBJECTIVE'] == 'regression':
        # Print current R2
        print('Current CV loop R2: '+str(np.round(
            scores[-1]['r2'], decimals=4)))
        # Print running mean R2
        print('Running mean R2: '+str(np.round(
            np.mean([i['r2'] for i in scores]), decimals=4)))
        # Print running mean shuffle R2
        print('Running shuffle mean R2: '+str(np.round(
            np.mean([i['r2'] for i in scores_sh]), decimals=4)))
        # Print elapsed time
        print('Elapsed time: '+str(np.round(
            time() - t_start, decimals=1)), end='\n\n')
    # Classification
    elif task['OBJECTIVE'] == 'binary' or task['OBJECTIVE'] == 'multiclass':
        # Print current acc
        print('Current CV loop acc: '+str(np.round(
            scores[-1]['acc'], decimals=4)))
        # Print running mean acc
        print('Running mean acc: '+str(np.round(
            np.mean([i['acc'] for i in scores]), decimals=4)))
        # Print running mean shuffle acc
        print('Running shuffle mean acc: '+str(np.round(
            np.mean([i['acc'] for i in scores_sh]), decimals=4)))
        # Print elapsed time
        print('Elapsed time: '+str(np.round(
            time() - t_start, decimals=1)), end='\n\n')
    # Other
    else:
        # Raise error
        raise ValueError('OBJECTIVE not found.')

    # Return None -------------------------------------------------------------
    return


def cross_validation(task, g, x, y):
    '''
    Performe cross-validation analysis. Saves results to pickle file in
    path_to_results directory.
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : dataframe
        Groups dataframe.
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Target dataframe.

    Returns
    -------
    None.
    '''

    # Initialize results lists ------------------------------------------------
    # Initialize best params list
    best_params = []
    # Initialize score list
    scores = []
    # Initialize SHAP based explainations list
    explainations = []
    # Initialize shuffle data score list
    scores_sh = []
    # Initialize shuffle data SHAP based explainations list
    explainations_sh = []
    # Get analysis pipeline and space
    pipe, space = prepare(task)

    # Main cross-validation loop ----------------------------------------------
    # Instatiate main cv splitter with fixed random state for comparison
    cv = RepeatedGroupKFold(n_splits=5,
                            n_repeats=task['N_REP_OUTER_CV'],
                            random_state=3141592)
    # Loop over main (outer) cross validation splits
    for i_cv, (i_trn, i_tst) in tqdm(enumerate(cv.split(g, groups=g))):
        # Save loop start time
        t_start = time()

        # Split data ----------------------------------------------------------
        # Split groups
        g_trn, g_tst = split_data(g, i_trn, i_tst)
        # Split targets
        y_trn, y_tst = split_data(y, i_trn, i_tst)
        # Split predictors
        x_trn, x_tst = split_data(x, i_trn, i_tst)

        # Tune and fit --------------------------------------------------------
        # Get optimized and fitted pipe
        pipe, params = tune_pipe(task, i_cv, pipe, space, g_trn, x_trn, y_trn)
        # Store best params
        best_params.append(params)

        # Analyze -------------------------------------------------------------
        # Score predictions
        scores.append(score_predictions(task, pipe, x_tst, y_tst, y))
        # SHAP explainations
        explainations.append(get_explainations(task, pipe, x_trn, x_tst))

        # Shuffle data analyze ------------------------------------------------
        # Clone pipe
        pipe_sh = clone(pipe)
        # Refit pipe with shuffled targets
        pipe_sh.fit(x_trn, shuffle(y_trn).squeeze())
        # Score predictions
        scores_sh.append(score_predictions(task, pipe_sh, x_tst, y_tst, y))
        # SHAP explainations
        explainations_sh.append(get_explainations(task, pipe_sh, x_trn, x_tst))

        # Compile and save intermediate results and task ----------------------
        # Create results
        results = {
            'best_params': best_params,
            'scores': scores,
            'explainations': explainations,
            'scores_sh': scores_sh,
            'explainations_sh': explainations_sh
            }
        # Make save path
        save_path = task['path_to_results']+'/'+task['y_name'][0]
        # Save results as pickle file
        s2p(save_path+'_results.pickle', results)
        # Save task as pickle file
        s2p(save_path+'_task.pickle', task)

        # Print current results -----------------------------------------------
        print_current_results(task, t_start, scores, scores_sh)

    # Return None -------------------------------------------------------------
    return


def train_test_split(task, g, x, y):
    '''
    Performe train-test split analysis. Saves results to pickle file in
    path_to_results directory.
    Ref: Hastie T, Tibshirani R, Friedman JH. The elements of statistical
    learning: data mining, inference, and prediction. 2nd ed. New York,
    NY: Springer; 2009
    Ref: Cawley GC, Talbot NLC. On Over-ﬁtting in Model Selection and
    Subsequent Selection Bias in Performance Evaluation. 2010;(11):2079–107.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    g : dataframe
        Groups dataframe.
    x : dataframe
        Predictors dataframe.
    y : dataframe
        Target dataframe.

    Returns
    -------
    None.
    '''

    # Initialize results lists ------------------------------------------------
    # Initialize best params list
    best_params = []
    # Initialize score list
    scores = []
    # Initialize SHAP based explainations list
    explainations = []
    # Initialize shuffle data score list
    scores_sh = []
    # Initialize shuffle data SHAP based explainations list
    explainations_sh = []
    # Get analysis pipeline and space
    pipe, space = prepare(task)
    # Save start time
    t_start = time()

    # Split data --------------------------------------------------------------
    # Get train data index
    i_trn = list(set(g.index).difference(set(task['TEST_SET_IND'])))
    # Get test data index
    i_tst = task['TEST_SET_IND']
    # Splitting groups
    g_trn, g_tst = split_data(g, i_trn, i_tst)
    # Splitting targets
    y_trn, y_tst = split_data(y, i_trn, i_tst)
    # Splitting predictors
    x_trn, x_tst = split_data(x, i_trn, i_tst)

    # Tune and fit ------------------------------------------------------------
    # Get optimized and fitted pipe
    pipe, params = tune_pipe(task, 0, pipe, space, g_trn, x_trn, y_trn)
    # Store best params
    best_params.append(params)

    # Analyze -----------------------------------------------------------------
    # Score predictions
    scores.append(score_predictions(task, pipe, x_tst, y_tst, y))
    # SHAP explainations
    explainations.append(get_explainations(task, pipe, x_trn, x_tst))

    # Shuffle data analyze ----------------------------------------------------
    # Clone pipe
    pipe_sh = clone(pipe)
    # Refit pipe with shuffled targets
    pipe_sh.fit(x_trn, shuffle(y_trn).squeeze())
    # Score predictions
    scores_sh.append(score_predictions(task, pipe_sh, x_tst, y_tst, y))
    # SHAP explainations
    explainations_sh.append(get_explainations(task, pipe_sh, x_trn, x_tst))

    # Compile and save intermediate results and task --------------------------
    # Create results
    results = {
        'best_params': best_params,
        'scores': scores,
        'explainations': explainations,
        'scores_sh': scores_sh,
        'explainations_sh': explainations_sh
        }
    # Make save path
    save_path = task['path_to_results']+'/'+task['y_name'][0]
    # Save results as pickle file
    s2p(save_path+'_results.pickle', results)
    # Save task as pickle file
    s2p(save_path+'_task.pickle', task)

    # Print current results ---------------------------------------------------
    print_current_results(task, t_start, scores, scores_sh)

    # Return None -------------------------------------------------------------
    return

def run_analysis(data_frame,
                 y_names,
                 x_con_names = [],
                 x_cat_bin_names = [],
                 x_cat_mult_names = [],
                 g_name = False,
                 objective = 'regression',
                 type = 'cv',
                 n_jobs = -2,
                 n_rep_outer_cv = 10,
                 n_samples_inner_cv = 10000,
                 n_samples_rs = 100,
                 max_samples_shap = 100,
                 shap_interactions = True,
                 analysis_name = 'analysis',
                 results_dir = './',
                 skip_rows = [],
                 test_set_ind = [],
                 def_test_size = 0.2,
                 ):
    '''
    Main function of the machine-learning based data analysis.

    ###########################################################################
    # Specify analysis
    ###########################################################################

    # 1. Specify task ---------------------------------------------------------

    # Type of analysis. string
    # Repeated Cross-validation: CV
    # Single Train-Test split: TT
    TYPE = 'CV'
    # Number parallel processing jobs. int (-1=all, -2=all-1)
    N_JOBS = -2
    # CV: Number of outer CV repetitions. int (default: 10)
    N_REP_OUTER_CV = 10
    # CV & TT: Total number of predictions in inner CV. int (default: 10000)
    N_SAMPLES_INNER_CV = 10000
    # Number of samples in random search. int (default: 100)
    N_SAMPLES_RS = 100
    # Limit number of samples for SHAP. int (default: 100).
    MAX_SAMPLES_SHAP = 100
    # Get SHAP interactions. Time consuming! bool (default: False)
    SHAP_INTERACTIONS = True

    # 2. Specify data ---------------------------------------------------------

    # Belief data - regression
    # Specifiy an analysis name
    ANALYSIS_NAME = 'belief'
    # Specify path to data. string
    PATH_TO_DATA = 'data/belief.xlsx'
    # Specify task OBJECTIVE. string (regression, binary, multiclass)
    OBJECTIVE = 'regression'
    # Specify grouping for CV split. list of string
    G_NAME = [
         'pcpid',
         ]
    # Specify continous predictor names. list of string or []
    X_CON_NAMES = [
        'govtrust',
        'humanid',
         ]
    # Specify binary categorical predictor names. list of string or []
    X_CAT_BIN_NAMES = [
         'gender',
         ]
    # Specify multi categorical predictor names. list of string or []
    X_CAT_MULT_NAMES = [
         # 'CountryCode',
         ]
    # Specify target name(s). list of strings or []
    Y_NAMES = [
         'ccbelief',
         ]
    # Rows to skip. list of int or []
    SKIP_ROWS = []
    # Specify index of rows for test set if TT. list of int or []
    TEST_SET_IND = list(randint.rvs(0, 4630, size=926))
    '''

    if not len(test_set_ind):
        test_set_ind = list(randint.rvs(0, len(data_frame) - 1, size=math.floor(len(data_frame) * def_test_size)))     
        
    # If shap with interactions
    if shap_interactions:
        # Update string
        analysis_name = analysis_name+'_'+type+'_inter'
    # If shap without interactions
    elif not shap_interactions:
        # Update string
        analysis_name = analysis_name+'_'+type
    # Other
    else:
        # Raise error
        raise ValueError('shap_interactions can be True or False only.')

    # Create results directory path -------------------------------------------
    #path_to_results = os.path.join(RESULTS_DIR, ANALYSIS_NAME)

    # Create task variable ----------------------------------------------------
    task = {
        'TYPE': type,
        'N_JOBS': n_jobs,
        'N_REP_OUTER_CV': n_rep_outer_cv,
        'N_SAMPLES_INNER_CV': n_samples_inner_cv,
        'N_SAMPLES_RS': n_samples_rs,
        'MAX_SAMPLES_SHAP': max_samples_shap,
        'SHAP_INTERACTIONS': shap_interactions,
        'ANALYSIS_NAME': analysis_name,
        #'PATH_TO_DATA': PATH_TO_DATA,
        #'SHEET_NAME': SHEET_NAME,
        'OBJECTIVE': objective,
        'G_NAME': g_name,
        'X_CON_NAMES': x_con_names,
        'X_CAT_BIN_NAMES': x_cat_bin_names,
        'X_CAT_MULT_NAMES': x_cat_mult_names,
        'Y_NAMES': y_names,
        'SKIP_ROWS': skip_rows,
        'TEST_SET_IND': test_set_ind,
        'path_to_results': results_dir,
        'x_names': x_con_names + x_cat_bin_names + x_cat_mult_names,
        }

    # Create results directory ------------------------------------------------
    create_dir(results_dir)

    # Load data ---------------------------------------------------------------
    x_cols = task['x_names']
    if not len(task['x_names']):
        if task['G_NAME']:
            x_cols = list(set(data_frame.columns) - set(task['Y_NAMES'] + [task['G_NAME']]))
        else:
            x_cols = list(set(data_frame.columns) - set(task['Y_NAMES']))
        task['X_CON_NAMES'] = x_cols
    task['x_names'] = x_cols
    
    cols = task['x_names'] + task['Y_NAMES']
    if task['G_NAME']:
        cols.append(task['G_NAME'])       
    d = data_frame[cols]
    print('Data frame size:', len(d))
    
    x = d[task['x_names']]

    if not task['G_NAME']:
        g = pd.DataFrame(pd.Series(x.index, name='idx'))
        task['G_NAME'] = 'idx'
    else:
        g = d[task['G_NAME']]
        
    # Reindex x to x_names
    x = x.reindex(task['x_names'], axis=1)
    
    y = d[task['Y_NAMES']]

    # Get target-encoding categories but don't do encoding --------------------
    if task['X_CAT_MULT_NAMES']:
        # Instatiate target-encoder
        te = TargetEncoder(categories='auto',
                           target_type='continuous',
                           smooth='auto',
                           cv=5,
                           shuffle=True,
                           random_state=13)
        # Fit target-encoder
        te.fit(x[task['X_CAT_MULT_NAMES']], y.squeeze())
        # Get target-encoder categories
        task['te_categories'] = te.categories_
    else:
        task['te_categories'] = []




    # Modelling and testing ---------------------------------------------------
    # Iterate over prediction targets (Y_NAMES)
    for i_y, y_name in tqdm(enumerate(y_names)):
        # Add prediction target index to task
        task['i_y'] = i_y
        # Add prediction target name to task
        task['y_name'] = [y_name]
        # Get current target
        yi = y[y_name].to_frame()
        # indexing full NOT-NA rows
        idx = np.hstack((g.notna(), 
                             yi.notna(),
                             np.reshape(x.notna().all(axis=1), 
                                        g.shape))).all(axis=1)
        print(f"\nDV={y_name}, N={int(idx.sum())}")
        # Cross-validation
        if type == 'CV':
            # Run cross-validation
            cross_validation(task, g[idx], x[idx], yi[idx])
        # Switch Type of analysis
        elif type == 'TT':
            # Run train-test split
            train_test_split(task, g[idx], x[idx], yi[idx])
        # Other
        else:
            # Raise error
            raise ValueError('Type not found.')

    # Return None -------------------------------------------------------------
    return

###############################################Figures
#################################################
################################################
##################################################
#################################################3
###############################################

def format_p(p, add_p=True, keep_space=False):
    if not np.isfinite(p) or p < 0.0:
        p = 'p = inf'
    elif p >= 0.999:
        p = 'p = 1.000'
    elif p < 0.001:
        p = 'p < .001'
    else:
        p = 'p = ' + f"{p:.3f}"[1:]
    p = p if add_p else p.replace('p ', '').replace('=', '')
    p = p if keep_space else p.replace(' ', '')
    return p

def format_r(r):
    if not np.isfinite(r):
        r = 'inf'
    elif abs(r) >= 0.999:
        r = '1.00'
    elif abs(r) < 0.005:
        r = '0.00'
    else:
        r = f"{'-' if r < 0.0 else ''}"+f"{abs(r):.2f}"[1:]
    return r

def get_stars(p, p001='***', p01='** ', p05='*  ', p10='⁺  ', p_='    '):
    if p < 0.001:
        return p001
    if p < 0.010:
        return p01
    if p < 0.050:
        return p05
    if p < 0.100:
        return p10
    return p_


def corrected_std(differences, n_tst_over_n_trn=0.25):
    '''
    Corrects standard deviation using Nadeau and Bengio's approach.
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    '''

    # Get corrected std -------------------------------------------------------
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    # Corrected variance
    corrected_var = np.var(differences, ddof=1) * (1/kr + n_tst_over_n_trn)
    # Corrected standard deviation
    corrected_std = np.sqrt(corrected_var)

    # Return corrected standard deviation -------------------------------------
    return corrected_std


def corrected_ttest(differences, n_tst_over_n_trn=0.25):
    '''
    Computes right-tailed paired t-test with corrected variance.
    Ref: Nadeau, C., Bengio, Y. Inference for the Generalization Error.
    Machine Learning 52, 239–281 (2003).
    https://doi.org/10.1023/A:1024068626366
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/
    plot_grid_search_stats.html

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_tst_over_n_trn : float
        Number of samples in the testing set over number of samples in the
        training set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    '''

    # Compute t statistics and p value ----------------------------------------
    # Get mean of differences
    mean = np.mean(differences)
    # Get corrected standard deviation
    std = corrected_std(differences, n_tst_over_n_trn)
    # Compute t statistics
    if std >= 0.0 and std <= 0.0000001: # for technical purposes
        t_stat = 0.0
        p_val = 1.0
    else:
        # Compute p value for right-tailed t-test
        t_stat = mean / std
        p_val = t.sf(np.abs(t_stat), df=len(differences)-1)
    # Return t statistics and p value -----------------------------------------
    return t_stat, p_val


def print_parameter_distributions(task, results, plots_path):
    '''
    Print model parameter distributions in histogram.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Get params
    params = pd.DataFrame(results['best_params'])

    # Make plot ---------------------------------------------------------------
    # Iterate over columns of params dataframe
    for (name, data) in params.items():
        # Make figure
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot hist of inlier score
        sns.histplot(data=data.astype('float'),
                     bins=30,
                     kde=True,
                     color='#777777',
                     ax=ax)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add x label
        ax.set_xlabel(name)
        # Add y label
        ax.set_ylabel('Number')
        # Set title
        ax.set_title(task['ANALYSIS_NAME']+' ' +
                     'parameter distribution for predicting'+' ' +
                     task['y_name'][0],
                     fontsize=10)

        # Save figure ---------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '0'+'_' +
                     task['y_name'][0]+'_' +
                     'hyperparameter'+'_' +
                     name)[:130]
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        #plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_regression_scatter(task, results, plots_path):
    '''
    Print model fit in a scatter plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # True values
    true_values = np.concatenate([i['y_true'] for i in results['scores']])
    # Predicted values
    pred_values = np.concatenate([i['y_pred'] for i in results['scores']])

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Print data
    ax.scatter(pred_values,
               true_values,
               zorder=2,
               alpha=0.1,
               color='#444444')
    # Add optimal fit line
    ax.plot([-10000, 10000], [-10000, 10000],
            color='#999999',
            zorder=3,
            linewidth=2,
            alpha=0.3)
    # Fix aspect
    ax.set_aspect(1)
    # Remove top, right and left frame elements
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Remove ticks
    ax.tick_params(axis='both', which='major', reset=True,
                   bottom=True, top=False, left=True, right=False)
    # Add grid
    ax.grid(visible=True, which='major', axis='both')
    # Modify grid
    ax.tick_params(grid_linestyle=':', grid_alpha=.5)
    # Get true values range
    true_values_range = max(true_values) - min(true_values)
    # Set x-axis limits
    ax.set_xlim(min(true_values) - true_values_range/20,
                max(true_values) + true_values_range/20)
    # Set y-axis limits
    ax.set_ylim(min(true_values) - true_values_range/20,
                max(true_values) + true_values_range/20)
    # Set title
    ax.set_title(task['ANALYSIS_NAME']+' ' +
                 'predicting'+' ' +
                 task['y_name'][0],
                 fontsize=10)
    # Set xlabel
    ax.set_xlabel('Predicted values', fontsize=10)
    # Set x ticks size
    plt.xticks(fontsize=10)
    # Set ylabel
    ax.set_ylabel('True values', fontsize=10)
    # Set y ticks size
    plt.yticks(fontsize=10)

    # Add MAE -----------------------------------------------------------------
    # Extract MAE
    mae = [i['mae'] for i in results['scores']]
    # Extract MAE shuffle
    mae_sh = [i['mae'] for i in results['scores_sh']]
    # Calculate p-value between MAE and shuffle MAE
    _, pval_mae = corrected_ttest(np.array(mae_sh)-np.array(mae))
    # Add MAE results to plot
    ax.text(.40, .055, ('MAE original mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}').format(
            np.mean(mae),
            np.std(mae),
            np.median(mae)),
            transform=ax.transAxes,
            fontsize=8)
    # Add MAE p val results to plot
    ax.text(.40, .02, ('MAE shuffle mean'+r'$\pm$'+'std:{:.2f}'+r'$\pm$' +
            '{:.2f}|med:{:.2f}|{}').format(
            np.mean(mae_sh),
            np.std(mae_sh),
            np.median(mae_sh),
            format_p(pval_mae)),
            transform=ax.transAxes,
            fontsize=8)

    # Add R² ------------------------------------------------------------------
    # Extract R²
    r2 = [i['r2'] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i['r2'] for i in results['scores_sh']]
    # Calculate p-value between R² and shuffle R²
    _, pval_r2 = corrected_ttest(np.array(r2)-np.array(r2_sh))
    # Add R² results to plot
    ax.text(.02, .96, ('R² original mean'+r'$\pm$'+'std:{:.3f}'+r'$\pm$' +
            '{:.3f}|med:{:.3f}').format(
            np.mean(r2),
            np.std(r2),
            np.median(r2)),
            transform=ax.transAxes,
            fontsize=8)
    # Add R² p val results to plot
    ax.text(.02, .925, ('R² shuffle mean'+r'$\pm$'+'std:{:.3f}'+r'$\pm$' +
            '{:.3f}|med:{:.3f}|{}').format(
            np.mean(r2_sh),
            np.std(r2_sh),
            np.median(r2_sh),
            format_p(pval_r2)),
            transform=ax.transAxes,
            fontsize=8)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 '1'+'_' +
                 task['y_name'][0])[:130]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg',  bbox_inches='tight')
    # Show figure
    #plt.show()
    plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_regression_violin(task, results, plots_path):
    '''
    Print model fit in a violin plot (regression).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Extract MAE
    mae = [i['mae'] for i in results['scores']]
    # Extract MAE shuffle
    mae_sh = [i['mae'] for i in results['scores_sh']]
    # Extract R²
    r2 = [i['r2'] for i in results['scores']]
    # Extract R² shuffle
    r2_sh = [i['r2'] for i in results['scores_sh']]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {'Mean Absolute Error': pd.Series(np.array(mae)),
         'R2': pd.Series(np.array(r2)),
         'Data': pd.Series(['original' for _ in mae]),
         'Dummy': pd.Series(np.ones(np.array(mae).shape).flatten())})
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'Mean Absolute Error': pd.Series(np.array(mae_sh)),
         'R2': pd.Series(np.array(r2_sh)),
         'Data': pd.Series(['shuffle' for _ in mae_sh]),
         'Dummy': pd.Series(np.ones(np.array(mae_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['Mean Absolute Error', 'R2']

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, len(metrics)*.75+1))
    # Set tight figure layout
    fig.tight_layout()
    # Make color palette
    mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(x=metric, y='Dummy', hue='Data', data=all_scores_df,
                       bw_method='scott', cut=2, density_norm='width', gridsize=100,
                       width=0.8, inner='box', orient='h', linewidth=1,
                       saturation=1, ax=ax[i], palette=mypal)
        # Remove top, right and left frame elements
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        # Remove ticks
        ax[i].tick_params(axis='both', which='major', reset=True,
                          bottom=True, top=False, left=False, right=False,
                          labelleft=False)
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel('', fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(axis='y', color='#bbbbbb', linestyle='dotted',
                         alpha=.3)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0])
    # set title
    fig.axes[0].set_title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 '1'+'_' +
                 task['y_name'][0]+'_' +
                 'distribution')[:130]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    #plt.show()
    plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_classification_confusion(task, results, plots_path):
    '''
    Print model fit as confusion matrix (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # True values
    true_values = [i['y_true'] for i in results['scores']]
    # Predicted values
    pred_values = [i['y_pred'] for i in results['scores']]
    # Sample weights  list
    sample_weights = [i['class_weights'] for i in results['scores']]
    # Accuracy
    acc = [i['acc'] for i in results['scores']]
    # Schuffle accuracy
    acc_sh = [i['acc'] for i in results['scores_sh']]
    # Get classes
    class_labels = np.unique(np.concatenate(true_values)).tolist()

    # Get confusion matrix ----------------------------------------------------
    # Loop over single results
    for true, pred, w in zip(true_values, pred_values, sample_weights):
        if 'con_mat' not in locals():
            # Compute confusion matrix
            con_mat = confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=np.array([w[i] for i in true]),
                normalize='all')
        else:
            # Add confusion matrix
            con_mat = np.add(con_mat, confusion_matrix(
                true,
                pred,
                labels=class_labels,
                sample_weight=np.array([w[i] for i in true]),
                normalize='all'))
    # Normalize confusion matrix
    con_mat_norm = con_mat / len(true_values)

    # Plot confusion matrix ---------------------------------------------------
    # Create figure
    fig, ax = plt.subplots(figsize=(con_mat.shape[0]*.5+3,
                                    con_mat.shape[0]*.5+3))
    # Plot confusion matrix
    sns.heatmap(con_mat_norm*100,
                vmin=None,
                vmax=None,
                cmap='Greys',
                center=None,
                robust=True,
                annot=True,
                fmt='.2f',
                annot_kws={'size': 10},
                linewidths=1,
                linecolor='#999999',
                cbar=False,
                cbar_kws=None,
                square=True,
                xticklabels=class_labels,
                yticklabels=class_labels,
                mask=None,
                ax=ax)
    # Add x label to plot
    plt.xlabel('Predicted class', fontsize=10)
    # Add y label to plot
    plt.ylabel('True class', fontsize=10)
    # Set y ticks size and sets the yticks 'upright' with 0
    plt.yticks(rotation=0, fontsize=10)
    # Calculate p-value of accuracy and shuffle accuracy
    _, pval_acc = corrected_ttest(np.array(acc)-np.array(acc_sh))
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0]+'\n' +
        'Orig. data accuracy mean'+r'$\pm$'+'std|median: {:.2f}'+r'$\pm$' +
        '{:.2f}|{:.2f}'+'\n' +
        'Shuffle data accuracy mean'+r'$\pm$'+'std|median: {:.2f}'+r'$\pm$' +
        '{:.2f}|{:.2f}'+'\n' +
        'p-value of orig. and shuffle: {}'+'\n').format(
        np.mean(acc)*100,
        np.std(acc)*100,
        np.median(acc)*100,
        np.mean(acc_sh)*100,
        np.std(acc_sh)*100,
        np.median(acc_sh)*100,
        format_p(pval_acc))
    # Set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 '1'+'_' +
                 task['y_name'][0])[:130]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show figure
    #plt.show()
    plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_classification_violin(task, results, plots_path):
    '''
    Print model fit in a violin plot (classification).

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Prepare results ---------------------------------------------------------
    # Extract accuracy
    acc = [i['acc'] for i in results['scores']]
    # Extract shuffle accuracy
    acc_sh = [i['acc'] for i in results['scores_sh']]
    # Compose scores dataframe
    scores_df = pd.DataFrame(
        {'Accuracy': pd.Series(np.array(acc)),
         'Data': pd.Series(['original' for _ in acc]),
         'Dummy': pd.Series(np.ones(np.array(acc).shape).flatten())})
    # Compose scores shuffle dataframe
    scores_sh_df = pd.DataFrame(
        {'Accuracy': pd.Series(np.array(acc_sh)),
         'Data': pd.Series(['shuffle' for _ in acc_sh]),
         'Dummy': pd.Series(np.ones(np.array(acc_sh).shape).flatten())})
    # Concatenate scores dataframes
    all_scores_df = pd.concat([scores_df, scores_sh_df], axis=0)
    # Make list of metrics
    metrics = ['Accuracy']

    # Make plot ---------------------------------------------------------------
    # Make figure
    fig, ax = plt.subplots(figsize=(8, len(metrics)*.75+1))
    # Make color palette
    mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
    # Put ax into list
    ax = [ax]
    # Loop over metrics
    for i, metric in enumerate(metrics):
        # Plot data
        sns.violinplot(x=metric, y='Dummy', hue='Data', data=all_scores_df,
                       bw_method='scott', cut=2, density_norm='width', gridsize=100,
                       width=0.8, inner='box', orient='h', linewidth=1,
                       saturation=1, ax=ax[i], palette=mypal)
        # Remove top, right and left frame elements
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        # Remove ticks
        ax[i].tick_params(axis='both', which='major', reset=True,
                          bottom=True, top=False, left=False, right=False,
                          labelleft=False)
        # Set x ticks and size
        ax[i].set_xlabel(metrics[i], fontsize=10)
        # Set y ticks and size
        ax[i].set_ylabel('', fontsize=10)
        # For other than first metric
        if i > 0:
            # Remove legend
            ax[i].legend().remove()
        # Add horizontal grid
        fig.axes[i].set_axisbelow(True)
        # Set grid style
        fig.axes[i].grid(axis='y', color='#bbbbbb', linestyle='dotted',
                         alpha=.3)
    # Make title string
    title_str = (
        task['ANALYSIS_NAME']+' ' +
        'predicting'+' ' +
        task['y_name'][0])
    # set title
    plt.title(title_str, fontsize=10)

    # Save figure -------------------------------------------------------------
    # Make save path
    save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                 '1'+'_' +
                 task['y_name'][0]+'_' +
                 'distribution')[:130]
    # Save figure
    plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
    # Check if save as svg is enabled
    if task['AS_SVG']:
        # Save figure
        plt.savefig(save_path+'.svg', bbox_inches='tight')
    # Show plot
    #plt.show()
    plt.close()

    # Return None -------------------------------------------------------------
    return fig


def get_shap_effects(task, explainations, c_class=-1):
    '''
    Get SHAP based global effects.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explainations : list of shap explaination objects
        SHAP explaination holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.

    Returns
    -------
    shap_effects : list
        SHAP effects.
    shap_base : float
        Base value corresponds to expected value of the predictor.
    '''

    # Get shap effects --------------------------------------------------------
    # Case 1: no interaction and regression
    if not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and regression
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Get SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: no interaction and binary
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values[:, :, c_class]), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 4: interaction and binary
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Get SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 5: no interaction and multiclass
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # SHAP effects
        shap_effects = [np.mean(np.abs(k.values[:, :, c_class]), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 6: interaction and multiclass
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Get SHAP effects
        shap_effects = [np.mean(np.abs(np.sum(k.values, axis=2)), axis=0)
                        for k in explainations]
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case else
    else:
        # Raise error
        raise ValueError('Unsupported task.')

    # Make SHAP effects dataframe
    shap_effects_df = pd.DataFrame(shap_effects, columns=task['x_names'])

    # Return shap effects -----------------------------------------------------
    return shap_effects_df, base


def print_shap_effects(task, results, plots_path):
    '''
    Print SHAP based global effects.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects -------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(task,
                                                 results['explainations'],
                                                 c_class)
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(task,
                                                 results['explainations_sh'],
                                                 c_class)

        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
            ascending=True)

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot ----------------------------------------------------------------
        # Make horizontal bar plot
        shap_effects_se_mean_sort.plot(
            kind='barh',
            figsize=(x_names_max_len*.1+7, x_names_count*.4+1),
            color='#777777',
            fontsize=10)
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('mean(|SHAP values|)', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis='y', color='#bbbbbb', linestyle='dotted', alpha=.3)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP effects for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = mean absolute deviation from expected' +
            ' value (' +
            str(np.round(base, decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Set title
        ax.set_title(title_str, fontsize=10)

        # Compute SHAP effect p values ----------------------------------------
        # Init p value list
        pval = []
        # Iterate over predictors
        for pred_name, pred_data in shap_effects_df.items():
            # Get current p value
            _, c_pval = corrected_ttest(
                pred_data.to_numpy()-shap_effects_sh_df[pred_name].to_numpy())
            # Add to pval list
            pval.append(np.around(c_pval, decimals=3))
        # Make pval series
        pval_se = pd.Series(data=pval, index=task['x_names'])
        # Multiple comparison correction
        if task['MCC']:
            # Multiply p value by number of tests
            pval_se = pval_se*x_names_count
            # Set p values > 1 to 1
            pval_se = pval_se.clip(upper=1)

        # Add SHAP effect values and p values as text -------------------------
        # Loop over values
        for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make test string
            txt_str = (str(np.around(c_val, decimals=2))+'|' +
                       #'p '+str(pval_se[c_pred]))
                       format_p(pval_se[c_pred]))
            # Add values to plot
            ax.text(c_val, i, txt_str, color='k',
                    va='center', fontsize=8)
        # Get x limits
        x_left, x_right = plt.xlim()
        plt.xlim(x_left, x_right + x_right*.1)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '2'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_effects')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        #plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_shap_effects_distribution(task, results, plots_path):
    '''
    Print SHAP values distribution.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects distribution ------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(task,
                                                 results['explainations'],
                                                 c_class)
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(task,
                                                 results['explainations_sh'],
                                                 c_class)

        # Process SHAP effects-------------------------------------------------
        # Sorting index by mean value of columns
        i_srt = shap_effects_df.mean().sort_values(ascending=False).index
        # Sort SHAP effects dataframe
        shap_effects_df_sort = shap_effects_df.reindex(i_srt, axis=1)
        # Sort shuffle SHAP effects dataframe
        shap_effects_sh_df_sort = shap_effects_sh_df.reindex(i_srt, axis=1)
        # Add data origin to SHAP effects dataframe
        shap_effects_df_sort['Data'] = pd.DataFrame(
            ['original' for _ in range(shap_effects_df_sort.shape[0])],
            columns=['Data'])
        # Add data origin to shuffle SHAP effects dataframe
        shap_effects_sh_df_sort['Data'] = pd.DataFrame(
            ['shuffle' for _ in range(shap_effects_sh_df_sort.shape[0])],
            columns=['Data'])
        # Get value name
        value_name = 'mean(|SHAP value|)'
        # Melt SHAP effects dataframe
        shap_effects_df_sort_melt = shap_effects_df_sort.melt(
            id_vars=['Data'], var_name='predictors',
            value_name=value_name)
        # Melt shuffle SHAP effects dataframe
        shap_effects_sh_df_sort_melt = shap_effects_sh_df_sort.melt(
            id_vars=['Data'], var_name='predictors',
            value_name=value_name)
        # Concatenate importances dataframes
        shap_effects_df_sort_melt_all = pd.concat([
            shap_effects_df_sort_melt,
            shap_effects_sh_df_sort_melt], axis=0)

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot ----------------------------------------------------------------
        # Make figure
        fig, ax = plt.subplots(figsize=(x_names_max_len*.1+7,
                                        x_names_count*.4+1))
        # Make color palette
        mypal = {'original': '#777777', 'shuffle': '#eeeeee'}
        # Plot data
        sns.violinplot(x=value_name, y='predictors', hue='Data',
                       data=shap_effects_df_sort_melt_all, bw_method='scott',
                       cut=2, density_norm='width', gridsize=100, width=0.8,
                       inner='box', orient='h', linewidth=.5,
                       saturation=1, ax=ax, palette=mypal)
        # Get the current figure and axes objects.
        _, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('mean(|SHAP values|)', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel('', fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Remove top, right and left frame elements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Add horizontal grid
        ax.set_axisbelow(True)
        # Set grid style
        ax.grid(axis='y', color='#bbbbbb', linestyle='dotted', alpha=.3)
        # Set legend position
        plt.legend(loc='lower right')
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP effects distribution for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = mean absolute deviation from expected' +
            ' value (' +
            str(np.round(base, decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Add title
        ax.set_title(title_str, fontsize=10)

        # Save plots and results ----------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '2'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_effects_distribution')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg', bbox_inches='tight')
        # Show figure
        # plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def get_shap_values(task, explainations, c_class=-1):
    '''
    Get SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    explainations : list of shap explaination objects
        SHAP explaination holding the results of the ml analyses.
    c_class : integer
        Current class for slicing.

    Returns
    -------
    shap_explainations : shap explaination object
        Explaination object with SHAP values.
    shap_base : float
        Base value corresponds to expected value of the predictor.
    '''

    # Get shap effects --------------------------------------------------------
    # Case 1: no interaction and regression
    if not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 2: interaction and regression
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'regression':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 3: no interaction and binary
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k[:, :, c_class].values for k in explainations]),
            base_values=np.hstack([k[:, :, c_class].base_values
                                   for k in explainations]),
            data=np.vstack([k[:, :, c_class].data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 4: interaction and binary
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'binary':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case 5: no interaction and multiclass
    elif not task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k[:, :, c_class].values for k in explainations]),
            base_values=np.hstack([k[:, :, c_class].base_values
                                   for k in explainations]),
            data=np.vstack([k[:, :, c_class].data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k[:, :, c_class].base_values
                                  for k in explainations]))
    # Case 6: interaction and multiclass
    elif task['SHAP_INTERACTIONS'] and task['OBJECTIVE'] == 'multiclass':
        # Explainer object
        shap_explainations = Explanation(
            np.vstack([k.values for k in explainations]),
            base_values=np.hstack([k.base_values for k in explainations]),
            data=np.vstack([k.data for k in explainations]),
            display_data=None,
            instance_names=None,
            feature_names=explainations[0].feature_names,
            output_names=None,
            output_indexes=None,
            lower_bounds=None,
            upper_bounds=None,
            error_std=None,
            main_effects=None,
            hierarchical_values=None,
            clustering=None,
            compute_time=np.sum([k.compute_time for k in explainations]))
        # Base value
        base = np.mean(np.hstack([k.base_values for k in explainations]))
    # Case else
    else:
        # Raise error
        raise ValueError('Unsupported task.')

    # Return shap effects -----------------------------------------------------
    return shap_explainations, base


def print_shap_values(task, results, plots_path):
    '''
    Plot SHAP values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results['explainations'],
                                            c_class)

        # If interactions
        if task['SHAP_INTERACTIONS']:
            # Sum over interaction to get full effects
            shap_explainations = shap_values.sum(axis=2)
            # Add base values
            shap_explainations.base_values = shap_values.base_values
            # Add data
            shap_explainations.data = shap_values.data
        # Other
        else:
            shap_explainations = shap_values

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Plot SHAP values beeswarm -------------------------------------------
        beeswarm(shap_explainations,
                 max_display=len(task['x_names']),
                 order=Explanation.abs.mean(0),
                 clustering=None,
                 cluster_threshold=0.5,
                 color=None,
                 axis_color='#333333',
                 alpha=.66,
                 show=False,
                 log_scale=False,
                 color_bar=True,
                 plot_size=(x_names_max_len*.1+7, x_names_count*.4+1),
                 color_bar_label='Predictor value')
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('SHAP values', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP values for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = mean absolute deviation from expected' +
            ' value (' +
            str(np.round(base, decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel('Predictor value', fontsize=10)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '3'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_values')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        # plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_shap_dependences(task, results, plots_path):
    '''
    Plot SHAP dependences.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results['explainations'],
                                            c_class)
        # If interactions
        if task['SHAP_INTERACTIONS']:
            # Sum over interaction to get full effects
            shap_explainations = shap_values.sum(axis=2)
            # Add base values
            shap_explainations.base_values = shap_values.base_values
            # Add data
            shap_explainations.data = shap_values.data
        # Other
        else:
            shap_explainations = shap_values

        # Print shap values dependencies --------------------------------------
        # Loop over predictors
        for i, c_pred in enumerate(shap_explainations.feature_names):
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plot SHAP Scatter plot
            scatter(shap_explainations[:, i],
                    color='#777777',
                    hist=True,
                    axis_color='#333333',
                    dot_size=16,
                    x_jitter='auto',
                    alpha=.5,
                    title=None,
                    xmin=None,
                    xmax=None,
                    ymin=None,
                    ymax=None,
                    overlay=None,
                    ax=ax,
                    ylabel='SHAP values',
                    show=False)
            # Get the current figure and axes objects.
            _, ax = plt.gcf(), plt.gca()
            # Set x label size
            plt.xlabel(ax.get_xlabel(), fontsize=10)
            # Set x ticks size
            plt.xticks(fontsize=10)
            # Set y label size
            plt.ylabel(ax.get_ylabel(), fontsize=10)
            # Set y ticks size
            plt.yticks(fontsize=10)
            # Make title string
            title_str = (
                task['ANALYSIS_NAME']+' ' +
                'SHAP values for'+' ' +
                task['y_name'][0]+'\n' +
                'SHAP values = deviation from expected value (' +
                str(np.round(base, decimals=2)) +
                ')'
                )
            # Add class if no interactions and binary or multiclass
            if not task['SHAP_INTERACTIONS'] and (
                    task['OBJECTIVE'] == 'binary' or
                    task['OBJECTIVE'] == 'multiclass'):
                # Make title string
                title_str = title_str+' class: '+str(c_class)
            # Add title
            plt.title(title_str, fontsize=10)

            # Save plot -------------------------------------------------------
            # Make save path
            save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                         '4'+'_' +
                         task['y_name'][0]+'_' +
                         'shap_values_dependency'+'_' +
                         str(c_pred))[:130]
            # Add class if no interactions and binary or multiclass
            if not task['SHAP_INTERACTIONS'] and (
                    task['OBJECTIVE'] == 'binary' or
                    task['OBJECTIVE'] == 'multiclass'):
                # Make title string
                save_path = save_path+'_class_'+str(c_class)
            # Save figure
            plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
            # Check if save as svg is enabled
            if task['AS_SVG']:
                # Save figure
                plt.savefig(save_path+'.svg',  bbox_inches='tight')
            # Show figure
            # plt.show()
            plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_shap_effects_interactions(task, results, plots_path):
    '''
    Plot SHAP effects inclusive interactions.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap effects -------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(task,
                                                 results['explainations'],
                                                 c_class)

        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
            ascending=False)

        # Get SHAP effects interactions ---------------------------------------
        # SHAP effects
        shap_effects_inter = np.array([np.mean(np.abs(k.values), axis=0)
                                       for k in results['explainations']])
        # Make dataframe
        shap_effects_inter_df = pd.DataFrame(
            np.mean(shap_effects_inter, axis=0),
            index=task['x_names'],
            columns=task['x_names'])
        # Reindex to sorted index
        shap_effects_inter_sort_df = \
            shap_effects_inter_df.reindex(shap_effects_se_mean_sort.index)
        # Reorder columns to sorted index
        shap_effects_inter_sort_df = \
            shap_effects_inter_sort_df.loc[:, shap_effects_se_mean_sort.index]
        # SHAP effects shuffle
        shap_effects_inter_sh = np.array(
            [np.mean(np.abs(k.values), axis=0) for k in
             results['explainations_sh']])

        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        x_names_count = len(task['x_names'])

        # Make labels with pvales ---------------------------------------------
        # Get p values
        pval = np.zeros((shap_effects_inter.shape[1],
                         shap_effects_inter.shape[2]))
        # Iterate over shap_effects
        for x, y in np.ndindex((shap_effects_inter.shape[1],
                                shap_effects_inter.shape[2])):
            # Get current SHAP effect
            c_effect = shap_effects_inter[:, x, y]
            # Get current SHAP effect shuffle
            c_effect_sh = shap_effects_inter_sh[:, x, y]
            # Calculate p-value
            _, pval[x, y] = corrected_ttest(c_effect-c_effect_sh)
        # Multiple comparison correction
        if task['MCC']:
            # Multiply p value by number of tests
            pval = pval*(x_names_count**2)
            # Set p values > 1 to 1
            pval = pval.clip(None, 1)
        # Initialize labels dataframe
        interaction_labels_df = pd.DataFrame(np.zeros([
            shap_effects_inter.shape[1],
            shap_effects_inter.shape[2]]))
        # Iterate labels
        for x, y in np.ndindex((shap_effects_inter.shape[1],
                                shap_effects_inter.shape[2])):
            # Make label
            interaction_labels_df.iloc[x, y] = (
                str(np.around(shap_effects_inter_df.iloc[x, y],
                              decimals=2)) +
                '\n'+'p'+' ' +
                str(np.around(pval[x, y], decimals=3)))
        # Index labels dataframe
        interaction_labels_df.index = shap_effects_inter_df.index
        # Column labels
        interaction_labels_df.columns = shap_effects_inter_df.columns
        # Reindex to sorted index
        interaction_labels_sort_df = \
            interaction_labels_df.reindex(shap_effects_se_mean_sort.index)
        # Reorder columns to sorted index
        interaction_labels_sort_df = \
            interaction_labels_sort_df.loc[:, shap_effects_se_mean_sort.index]

        # Plot interaction effects --------------------------------------------
        # Create figure
        fig, ax = plt.subplots(figsize=(x_names_max_len*.1+x_names_count*1+1,
                                        x_names_max_len*.1+x_names_count*1+1))
        # Make colorbar string
        clb_str = ('mean(|SHAP value|)')
        # Plot confusion matrix
        sns.heatmap(shap_effects_inter_sort_df,
                    vmin=None,
                    vmax=None,
                    cmap='Greys',
                    center=None,
                    robust=True,
                    annot=interaction_labels_sort_df,
                    fmt='',
                    annot_kws={'size': 10},
                    linewidths=1,
                    linecolor='#999999',
                    cbar=True,
                    cbar_kws={'label': clb_str, 'shrink': 0.6},
                    square=True,
                    xticklabels=True,
                    yticklabels=True,
                    mask=None,
                    ax=ax)
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel(ax.get_xlabel(), fontsize=10)
        # Set x ticks size
        plt.xticks(rotation=90, fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(rotation=0, fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP effects for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = deviation from expected value (' +
            str(np.round(np.mean(np.hstack(
                [k.base_values for k in results['explainations']])),
                decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Add title
        plt.title(title_str, fontsize=10)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel(clb_str, fontsize=10)
        cb_ax.set_box_aspect(50)

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '5'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_effects_interactions')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg',  bbox_inches='tight')
        # Show figure
        #plt.show()
        plt.close()

    # Return None -------------------------------------------------------------
    return fig


def print_shap_interaction_values(task, results, plots_path):
    '''
    Plot SHAP interaction values.

    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.

    Returns
    -------
    None.
    '''

    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1

    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results['explainations'],
                                            c_class)

        # Print shap values dependencies --------------------------------------
        # Make list of permutations
        permutations_list = (
            [(i, i) for i in shap_values.feature_names] +
            list(permutations(shap_values.feature_names, 2)))
        # Loop over predictor pairs
        for ind in permutations_list:
            # Make figure
            fig, ax = plt.subplots(figsize=(8, 5))
            # Plot SHAP dependence
            dependence_plot(ind,
                            shap_values=shap_values.values,
                            features=pd.DataFrame(
                                shap_values.data,
                                columns=shap_values.feature_names),
                            feature_names=shap_values.feature_names,
                            display_features=None,
                            interaction_index='auto',
                            color='#1E88E5',
                            axis_color='#333333',
                            cmap=None,
                            dot_size=16,
                            x_jitter=0,
                            alpha=.66,
                            title=None,
                            xmin=None,
                            xmax=None,
                            ax=ax,
                            show=False,
                            ymin=None,
                            ymax=None)
            # Get the current figure and axes objects.
            _, ax = plt.gcf(), plt.gca()
            # Set x label size
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)
            # Set x ticks size
            plt.xticks(fontsize=10)
            # Set y label size
            plt.ylabel(ax.get_ylabel(), fontsize=10)
            # Set y ticks size
            plt.yticks(fontsize=10)
            # Make title string
            title_str = (
                task['ANALYSIS_NAME']+' ' +
                'SHAP interaction values for'+' ' +
                task['y_name'][0]+'\n' +
                'SHAP values = deviation from expected value (' +
                str(np.round(np.mean(np.hstack(
                    [k.base_values for k in results['explainations']])),
                    decimals=2)) +
                ')'
                )
            # Add class if no interactions and binary or multiclass
            if not task['SHAP_INTERACTIONS'] and (
                    task['OBJECTIVE'] == 'binary' or
                    task['OBJECTIVE'] == 'multiclass'):
                # Make title string
                title_str = title_str+' class: '+str(c_class)
            # Add title
            ax.set_title(title_str, fontsize=10)
            # Check if mor than 1 axes are present
            if len(fig.axes) > 1:
                # Get colorbar
                cb_ax = fig.axes[1]
                # Modifying color bar tick size
                cb_ax.tick_params(labelsize=10)
                # Modifying color bar fontsize
                cb_ax.set_ylabel(cb_ax.get_ylabel(), fontsize=10)

            # Save plot -------------------------------------------------------
            # Make save path
            save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                         '6'+'_' +
                         task['y_name'][0]+'_' +
                         'shap_interaction_values'+'_' +
                         ind[0]+'_' +
                         ind[1])[:130]
            # Add class if no interactions and binary or multiclass
            if not task['SHAP_INTERACTIONS'] and (
                    task['OBJECTIVE'] == 'binary' or
                    task['OBJECTIVE'] == 'multiclass'):
                # Make title string
                save_path = save_path+'_class_'+str(c_class)
            # Save figure
            plt.savefig(save_path+'.png', dpi=150, bbox_inches='tight')
            # Check if save as svg is enabled
            if task['AS_SVG']:
                # Save figure
                plt.savefig(save_path+'.svg',  bbox_inches='tight')
            # Show figure
            #plt.show()
            plt.close()

    # Return None -------------------------------------------------------------
    return fig

def _print_shap_values(task, results, plots_path='./', title=None, exclude=[], rename_dict={}, dpi=600, stars=False):
    '''
    Plot SHAP values.
    Parameters
    ----------
    task : dictionary
        Dictionary holding the task describtion variables.
    results : dictionary
        Dictionary holding the results of the ml analyses.
    plots_path : string
        Path to the plots.
    Returns
    -------
    None.
    '''
    # Classes -----------------------------------------------------------------
    # If no interactions and binary or multiclass
    if not task['SHAP_INTERACTIONS'] and (
            task['OBJECTIVE'] == 'binary' or
            task['OBJECTIVE'] == 'multiclass'):
        # Get number of classes
        n_classes = results['explainations'][0].shape[2]
    # Other cases
    else:
        n_classes = 1
    # Plot shap values --------------------------------------------------------
    # Loop over classes
    for c_class in range(n_classes):
        # Get current shap values
        shap_values, base = get_shap_values(task, results['explainations'], c_class)
        # Get current shap effects
        shap_effects_df, base = get_shap_effects(task,
                                                 results['explainations'],
                                                 c_class)
        # Get current shuffle shap effects
        shap_effects_sh_df, _ = get_shap_effects(task,
                                                 results['explainations_sh'],
                                                 c_class)
        # Process SHAP effects-------------------------------------------------
        # Mean shap values
        shap_effects_se_mean = shap_effects_df.mean(axis=0)
        # Sort from highto low
        shap_effects_se_mean_sort = shap_effects_se_mean.sort_values(
            ascending=True)
        # If interactions
        if task['SHAP_INTERACTIONS']:
            # Sum over interaction to get full effects
            shap_explainations = shap_values.sum(axis=2)
            # Add base values
            shap_explainations.base_values = shap_values.base_values
            # Add data
            shap_explainations.data = shap_values.data
        # Other
        else:
            shap_explainations = shap_values
        # Additional info -----------------------------------------------------
        # x names lengths
        x_names_max_len = max([len(i) for i in task['x_names']])
        # x names count
        # Compute SHAP effect p values ----------------------------------------
        # Init p value list
        pval = {}
        # Iterate over predictors
        for pred_name, pred_data in shap_effects_df.items():
            # Get current p value
            _, c_pval = corrected_ttest(
                pred_data.to_numpy()-shap_effects_sh_df[pred_name].to_numpy())
            # Add to pval list
            pval[pred_name] = np.around(c_pval, decimals=3)
        excl = 0
        for e in exclude:
          idx = shap_explainations.feature_names.index(e)
          if idx:
            shap_explainations.feature_names.remove(e)
            shap_explainations.values = np.delete(shap_explainations.values, idx, axis=1)
            shap_explainations.data = np.delete(shap_explainations.data, idx, axis=1)
            excl += 1
            print(f"'{e}' has been removed.")
        # Make pval series
        x_names_count = len(task['x_names'])
        pval_se = pd.Series(data=pval, index=task['x_names'])
        # Multiple comparison correction
        if task['MCC']:
            # Multiply p value by number of tests
            pval_se = pval_se*(x_names_count - excl)
            # Set p values > 1 to 1
            pval_se = pval_se.clip(upper=1)
        extra_x_names = {}
        extra_x_names_max_len = []
        for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make test string
             txt_str = c_pred+'  '+f"{c_val:6<.2f}"+f"{', '+format_p(pval_se[c_pred]) if not stars else get_stars(pval_se[c_pred])}"
             extra_x_names[c_pred] = txt_str
        for i, f in enumerate(shap_explainations.feature_names):
            shap_explainations.feature_names[i] = extra_x_names[f].replace(f, rename_dict[f]) if f in rename_dict else extra_x_names[f]
            extra_x_names_max_len.append(len(shap_explainations.feature_names[i].split('\n')[0]))
        extra_x_names_max_len = max(extra_x_names_max_len)
        
        # Plot SHAP values beeswarm -------------------------------------------
        beeswarm(shap_explainations,
                 max_display=len(task['x_names']),
                 order=Explanation.abs.mean(0),
                 clustering=None,
                 cluster_threshold=0.5,
                 color=None,
                 axis_color='#333333',
                 alpha=.66,
                 show=False,
                 log_scale=False,
                 color_bar=True,
                 plot_size=(x_names_max_len*.05+7, x_names_count*.4+1),
                 color_bar_label='Predictor value')
        # Get the current figure and axes objects.
        fig, ax = plt.gcf(), plt.gca()
        # Set x label size
        plt.xlabel('Impact on model output\n(SHAP value of predictor)', fontsize=10)
        # Set x ticks size
        plt.xticks(fontsize=10)
        # Set y label size
        plt.ylabel(ax.get_ylabel(), fontsize=10)
        # Set y ticks size
        plt.yticks(fontsize=10)
        # Make title string
        title_str = (
            task['ANALYSIS_NAME']+' ' +
            'SHAP values for'+' ' +
            task['y_name'][0]+'\n' +
            'mean(|SHAP values|) = mean absolute deviation from expected' +
            ' value (' +
            str(np.round(base, decimals=2)) +
            ')'
            )
        # Add class if no interactions and binary or multiclass
        
        if not (task['OBJECTIVE'] == 'binary' or
                        task['OBJECTIVE'] == 'multiclass'):
            # Extract R²
            r2 = [i['r2'] for i in r['scores']]
              # Extract R² shuffle
            r2_sh = [i['r2'] for i in r['scores_sh']]
              # Calculate p-value between R² and shuffle R²
            _, pval_r2 = corrected_ttest(np.array(r2)-np.array(r2_sh))
                # Add R² results to plot
            metrics_str = f'R²pred={format_r(np.mean(r2))}'+r'±'+f'{format_p(np.std(r2), add_p=False)}, {format_p(pval_r2, keep_space=False)}'
        else:
            metrics_str = 'Acc.='

        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            title_str = title_str+' class: '+str(c_class)
        # Add title
        print(title_str)
        print(metrics_str)
        # Get colorbar
        cb_ax = fig.axes[1]
        # Modifying color bar tick size
        cb_ax.tick_params(labelsize=10)
        # Modifying color bar fontsize
        cb_ax.set_ylabel('Predictor value', fontsize=10)
        # Add SHAP effect values and p values as text -------------------------
        # Loop over values
        x_left, x_right = plt.xlim()
        y_bottom, y_top = plt.ylim()
        #for i, (c_pred, c_val) in enumerate(shap_effects_se_mean_sort.items()):
            # Make test string
        #    txt_str = (str(np.around(c_val, decimals=2))+', '+format_p(pval_se[c_pred])) #get_stars(pval_se[c_pred])
            # Add values to plot
        #    ax.text(x_left-.15, i-.05, txt_str, color='k', va='center', fontsize=9) #c_val
        # Get x limits
        s = 'Average impact   \nPredictor     (mean|SHAP values|)'
        ax.text(x_left  - x_left*0.05, y_top, s, color='k', va='center', ha='right', fontsize=10) # - (max([len(i) for i in s.split('\n')])*.02 / 2)
        plt.xlim(x_left - x_left*0.05, x_right + x_right*.1)

        if title!=None:
          plt.title(title, fontsize=10)
        else:
          f = task['y_name'][0]
          f = f.replace(f, rename_dict[f]) if f in rename_dict else f
          f += f'\n{metrics_str}'
          ax.text((x_left + x_right) / 2, y_top + len(f.split('\n'))*.4, f, fontsize=10.6, 
                  horizontalalignment='center', verticalalignment='top')

        # Save plot -----------------------------------------------------------
        # Make save path
        save_path = (plots_path+'/'+task['ANALYSIS_NAME']+'_' +
                     '3'+'_' +
                     task['y_name'][0]+'_' +
                     'shap_values')[:130]
        # Add class if no interactions and binary or multiclass
        if not task['SHAP_INTERACTIONS'] and (
                task['OBJECTIVE'] == 'binary' or
                task['OBJECTIVE'] == 'multiclass'):
            # Make title string
            save_path = save_path+'_class_'+str(c_class)
        # Save figure
        plt.savefig(save_path+'.png', dpi=dpi, bbox_inches='tight')
        # Check if save as svg is enabled
        if task['AS_SVG']:
            # Save figure
            plt.savefig(save_path+'.svg', dpi=dpi,  bbox_inches='tight')
        # Show figure
        # plt.show()
        plt.close()
    # Return None -------------------------------------------------------------
    return fig

def make_figures(results_dir = './',
                 figures_dir = './',
                 print_param_distrib = False,
                 print_shap_effects_inter = True,
                 add_multiple_comparison_correction = False,
                 def_shap_values = True,
                 as_svg = False):
    '''
    Main function of plot results of machine-learning based data analysis.
    ###########################################################################
    # Specify plot task
    ###########################################################################

    # Plot hyper parameter distributions
    PPD = True
    # Plot SHAP interactions
    PSI = True
    # Do multiple comparison correction
    MCC = False
    # Save plots additionally as svg
    AS_SVG = True

    ###########################################################################
    '''
    
    # Loop over result paths --------------------------------------------------
        # Load task paths
    task_paths = [f.name for f in os.scandir(results_dir)
                      if f.name.endswith('_task.pickle')]
        # Load result paths
    results_paths = [f.name for f in os.scandir(results_dir)
                         if f.name.endswith('_results.pickle')]

        # Loop over tasks -----------------------------------------------------
    for i_task, task_path in tqdm(enumerate(task_paths)):

            # Load task and results, create plots directory -------------------
            # Load task description
            task = lfp(os.path.join(results_dir, task_path))
            # Add multiple comparison correction to task
            task['MCC'] = add_multiple_comparison_correction
            # Add as svg to task
            task['AS_SVG'] = as_svg
            # Load results
            results = lfp(os.path.join(results_dir, results_paths[i_task]))
            # Plots path
            plots_path = os.path.join(figures_dir, task['y_name'][0]+'_plots')
            # Create plots dir
            create_dir(plots_path)

            # Plot parameter distributions ------------------------------------
            if print_param_distrib:
                print_parameter_distributions(task, results, plots_path)

            # Plot model fit --------------------------------------------------
            # Regressor
            if task['OBJECTIVE'] == 'regression':
                # Print model fit as scatter plot
                print_regression_scatter(task, results, plots_path)
                # Print model fit as violinplot of metrics
                print_regression_violin(task, results, plots_path)
            # Classification
            elif (task['OBJECTIVE'] == 'binary' or
                  task['OBJECTIVE'] == 'multiclass'):
                # Print model fit as confusion matrix
                print_classification_confusion(task, results, plots_path)
                # Print model fit as violinplot of metrics
                print_classification_violin(task, results, plots_path)
            # Other
            else:
                # Raise error
                raise ValueError('OBJECTIVE not found.')

            # Plot SHAP effects -----------------------------------------------
            print_shap_effects(task, results, plots_path)

            # Plot SHAP effects distribution ----------------------------------
            print_shap_effects_distribution(task, results, plots_path)

            # Plot SHAP values ------------------------------------------------
            if def_shap_values:
                print_shap_values(task, results, plots_path)
            else:
                _print_shap_values(task, results, plots_path)
        
            # Plot SHAP dependencies ------------------------------------------
            print_shap_dependences(task, results, plots_path)

            # Plot SHAP effects interactions ----------------------------------
            if task['SHAP_INTERACTIONS'] and print_shap_effects_inter:
                print_shap_effects_interactions(task, results, plots_path)

            # Plot SHAP interaction values ------------------------------------
            if task['SHAP_INTERACTIONS'] and print_shap_effects_inter:
                print_shap_interaction_values(task, results, plots_path)

    # Return None -------------------------------------------------------------
    return
