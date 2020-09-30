import numpy as np

from performance_metrics import calc_mean_squared_error # was this here??


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
    x_NF.astype(np.float64)
    y_N.astype(np.float64)

    train_error_per_fold = list() 
    test_error_per_fold = list() 
    
    

    # TODO define the folds here by calling your function
    # e.g. ... = make_train_and_test_row_ids_for_n_fold_cv(...)
    
    
    y_N = y_N.reshape((y_N.shape[0], 1))
    
    # get the indices for training and testing per fold
    tr_ids_per_fold, te_ids_per_fold = make_train_and_test_row_ids_for_n_fold_cv(x_NF.shape[0], n_folds, random_state)
    
    for i in range(n_folds):
        indexes_tr = np.array(tr_ids_per_fold[i])

        ith_training_xs = x_NF[indexes_tr]
        ith_training_ys = y_N[indexes_tr]

        # Now we have all the training points, we can fit our model
        ith_training_xs = np.asarray(ith_training_xs)
        ith_training_ys = np.asarray(ith_training_ys)

        #ith_training_xs = ith_training_xs.reshape((ith_training_xs.shape[0], 1))
        #ith_training_ys = ith_training_ys.reshape((ith_training_ys.shape[0], 1))
        
        #ith_training_xs = ith_training_xs.reshape((ith_training_xs.shape[0], 1))
        #ith_training_ys = ith_training_ys.reshape((ith_training_ys.shape[0], 1))
        #print("ith_training_xs.shape", ith_training_xs.shape)
        #print("ith_training_ys.shape", ith_training_ys.shape)


        estimator.fit(ith_training_xs, ith_training_ys)
        # we obtain the predictions for the training points
        ith_tr_y_hat = estimator.predict(ith_training_xs)
        # we compute the training error and add it to the train_error_per_fold numpy array
        #print("calc_mean_squared_error(ith_training_ys, ith_tr_y_hat)", calc_mean_squared_error(ith_training_ys, ith_tr_y_hat))
        train_error_per_fold.extend(calc_mean_squared_error(ith_training_ys, ith_tr_y_hat))
        
        # now we compute the testing/cross validation error
        indexes_te = np.array(te_ids_per_fold[i])
        ith_testing_xs = x_NF[indexes_te]
        ith_testing_ys = y_N[indexes_te]
        
        ith_te_y_hat = estimator.predict(ith_testing_xs)
        #print("calc_mean_squared_error(ith_testing_ys, ith_te_y_hat)", calc_mean_squared_error(ith_testing_ys, ith_te_y_hat))
        test_error_per_fold.extend(calc_mean_squared_error(ith_testing_ys, ith_te_y_hat))
        
    # TODO loop over folds and compute the train and test error
    # for the provided estimator

    return np.asarray(train_error_per_fold).astype(np.float64), np.asarray(test_error_per_fold).astype(np.float64)


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
    if hasattr(random_state, 'rand'):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state # just remind us we use the passed-in value
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    # TODO obtain a shuffled order of the n_examples
    number_of_items_per_fold = int(np.ceil(n_examples/n_folds)) # TODO: not sure if I should use ceil or the lower bound, see if I pass tests like this

    # I could have used np.split(indices, n_folds) instead. That would have avoided the above line. USE https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
    #print("number_of_items_per_fold", number_of_items_per_fold)
    #print("number_of_items_per_fold CEIL", int(np.ceil(n_examples/n_folds)))
    indices = np.asarray([n for n in range(n_examples)])
    random_state.shuffle(indices)

    # split the indices array into folds
    indices_split = np.array_split(indices, n_folds)
    #print(indices_split)
    train_ids_per_fold = list()
    test_ids_per_fold = list()
    
    for k in range(n_folds):
        kth_training_folds = list()
        for i in range(n_folds):
            if (i == k): # there will be only one testing fold
                test_ids_per_fold.append(indices_split[i])
                #test_ids_per_fold.append(np.asarray([j for j in indices[i*number_of_items_per_fold:number_of_items_per_fold*(i+1)]]))
            else: # there will be n_folds - 1 training folds
                kth_training_folds.extend(indices_split[i])
                #kth_training_folds.append(np.asarray([j for j in indices[i*number_of_items_per_fold:number_of_items_per_fold*(i+1)]]))


        train_ids_per_fold.append(kth_training_folds)

        
    # TODO establish the row ids that belong to each fold's 
    # train subset and test subset - DONE

    return train_ids_per_fold, test_ids_per_fold


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # N = 11
    # n_folds = 3
    # tr_ids_per_fold, te_ids_per_fold = (make_train_and_test_row_ids_for_n_fold_cv(N, n_folds))
    # print(len(tr_ids_per_fold))
    # #3
    #
    # # Count of items in training sets
    # print("sort", np.sort([len(tr) for tr in tr_ids_per_fold]))
    # print("tr_ids_per_fold", tr_ids_per_fold)
    #array([7, 7, 8])

    # N = 11
    # n_folds = 3
    # x_N3 = np.random.RandomState(0).rand(N, 3)
    # y_N = np.dot(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
    # y_N.shape
    # #(101,)
    #
    # import sklearn.linear_model
    # my_regr = sklearn.linear_model.LinearRegression()
    # tr_K, te_K = train_models_and_calc_scores_for_n_fold_cv(
    #
    # my_regr, x_N3, y_N, n_folds = n_folds, random_state = 0)
    #
    # # Training error should be indistiguishable from zero
    # print(np.array2string(tr_K, precision=8, suppress_small=True))
    # #'[0. 0. 0. 0. 0. 0. 0.]'
    #
    # # Testing error should be indistinguishable from zero
    # print(np.array2string(te_K, precision=8, suppress_small=True))
    # #'[0. 0. 0. 0. 0. 0. 0.]'
