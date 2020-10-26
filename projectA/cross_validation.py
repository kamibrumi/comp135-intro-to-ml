import numpy as np
import sklearn.metrics
import sklearn
from performance_metrics_for_proba_predictions import calc_mean_binary_cross_entropy_from_probas


def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f

    Examples
    --------
    # Create simple dataset of N examples where y given x
    # is perfectly explained by a linear regression model
    >>> N = 101
    >>> n_folds = 7
    >>> x_N3 = np.random.RandomState(0).rand(N, 3)
    >>> y_N = np.dot(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
    >>> y_N.shape
    (101,)

    >>> import sklearn.linear_model
    >>> my_regr = sklearn.linear_model.LinearRegression()
    >>> tr_K, te_K = train_models_and_calc_scores_for_n_fold_cv(
    ...                 my_regr, x_N3, y_N, n_folds=n_folds, random_state=0)

    # Training error should be indistiguishable from zero
    >>> np.array2string(tr_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'

    # Testing error should be indistinguishable from zero
    >>> np.array2string(te_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'
    '''
    N, F = x_NF.shape
    N2, = y_N.shape
    assert N == N2
    tr_ids_per_fold, te_ids_per_fold = \
        make_train_and_test_row_ids_for_n_fold_cv(
            N, n_folds, random_state)

    train_error_rate_per_fold = np.zeros(n_folds, dtype=np.float64)
    test_error_rate_per_fold = np.zeros(n_folds, dtype=np.float64)
    
    train_bce_per_fold = np.zeros(n_folds, dtype=np.float64)
    test_bce_per_fold = np.zeros(n_folds, dtype=np.float64)
    
    for fold_id in range(n_folds):
        trai_ids = tr_ids_per_fold[fold_id]
        x_tr = x_NF[trai_ids].copy()
        y_tr = y_N[trai_ids].copy()

        test_ids = te_ids_per_fold[fold_id]
        x_te = x_NF[test_ids].copy()
        y_te = y_N[test_ids].copy()

        estimator.fit(x_tr, y_tr)

        # predict probabilities on training data and test
        yhat_tr = estimator.predict_proba(x_tr)[:,1]
        yhat_te = estimator.predict_proba(x_te)[:,1]
        
        
        train_error_rate_per_fold[fold_id] = sklearn.metrics.zero_one_loss(y_tr, yhat_tr >= 0.5)
        test_error_rate_per_fold[fold_id] = sklearn.metrics.zero_one_loss(y_te, yhat_te >= 0.5)
        
        train_bce_per_fold[fold_id] = calc_mean_binary_cross_entropy_from_probas(y_tr, yhat_tr)
        test_bce_per_fold[fold_id] = calc_mean_binary_cross_entropy_from_probas(y_te, yhat_te)
        
    return train_error_rate_per_fold, test_error_rate_per_fold, train_bce_per_fold, test_bce_per_fold


def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    Guarantees for Return Values
    ----------------------------
    Across all folds, guarantee that no two folds put same object in test set.
    For each fold f, we need to guarantee:
    * The *union* of train_ids_per_fold[f] and test_ids_per_fold[f]
    is equal to [0, 1, ... N-1]
    * The *intersection* of the two is the empty set
    * The total size of train and test ids for any fold is equal to N

    Examples
    --------
    >>> N = 11
    >>> n_folds = 3
    >>> tr_ids_per_fold, te_ids_per_fold = (
    ...     make_train_and_test_row_ids_for_n_fold_cv(N, n_folds))
    >>> len(tr_ids_per_fold)
    3

    # Count of items in training sets
    >>> np.sort([len(tr) for tr in tr_ids_per_fold])
    array([7, 7, 8])

    # Count of items in the test sets
    >>> np.sort([len(te) for te in te_ids_per_fold])
    array([3, 4, 4])

    # Test ids should uniquely cover the interval [0, N)
    >>> np.sort(np.hstack([te_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    # Train ids should cover the interval [0, N) TWICE
    >>> np.sort(np.hstack([tr_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
            8,  9,  9, 10, 10])
    '''
    K = int(n_folds)
    N = int(n_examples)
    if hasattr(random_state, 'rand'):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    shuffled_row_ids = random_state.permutation(N)

    train_ids_per_fold = list()
    test_ids_per_fold = list()

    for f in range(K):
        cur_test_ids = shuffled_row_ids[f::K].copy()
        cur_train_ids = np.hstack([
            shuffled_row_ids[g::K].copy() for g in range(K) if g != f])

        train_ids_per_fold.append(cur_train_ids)
        test_ids_per_fold.append(cur_test_ids)

    sizes_K = np.asarray([len(test_ids) for test_ids in test_ids_per_fold])
    assert sizes_K.min() >= 0
    assert np.max(sizes_K) - np.min(sizes_K) <= 1

    return train_ids_per_fold, test_ids_per_fold
