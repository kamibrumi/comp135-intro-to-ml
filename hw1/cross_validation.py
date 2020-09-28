import numpy as np
from LeastSquaresLinearRegression import LeastSquaresLinearRegressor # TODO remove this line before submitting and the main function

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
    train_error_per_fold = np.zeros(2, dtype=np.int32)
    test_error_per_fold = np.zeros(2, dtype=np.int32)
    
    #for i in range(n_folds):

    # TODO define the folds here by calling your function
    # e.g. ... = make_train_and_test_row_ids_for_n_fold_cv(...)
    
    N = x_NF.shape[0]
    # get the indices for training and testing per fold
    tr_ids_per_fold, te_ids_per_fold = make_train_and_test_row_ids_for_n_fold_cv(N, n_folds, random_state)
    
    for i in range(n_folds):
        # concatenate in one array all the training points
        ith_training_xs = list()
        ith_training_ys = list()
        for j in range (n_folds - 1):
            ith_training_xs = np.append(ith_training_points, x_NF[np.array(tr_ids_per_fold[i][j])]) # TODO change the final 0)
            ith_training_ys = np.append(ith_training_ys, y_N[np.array(tr_ids_per_fold[i][j])])

        # Now we have all the training points, we can fit our model
        estimator.fit(ith_training_xs, ith_training_ys)
        # we obtain the predictions for the training points
        ith_tr_y_hat = estimator.predict(ith_training_xs)
        # we compute the training error and add it to the train_error_per_fold numpy array
        train_error_per_fold[i] = calc_mean_squared_error(ith_training_ys, ith_y_hat)
        
        # now we compute the testing/cross validation error
        ith_testing_xs = x_NF[np.array(te_ids_per_fold[i])]
        ith_testing_ys = y_N[np.array(te_ids_per_fold[i])]
        
        ith_te_y_hat = estimator.predict(ith_testing_xs)
        test_error_per_fold[i] = calc_mean_squared_error(ith_testing_ys, ith_te_y_hat)
        
    # TODO loop over folds and compute the train and test error
    # for the provided estimator

    return train_error_per_fold, test_error_per_fold


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
    number_of_items_per_fold = int(N/n_folds) # TODO: not sure if I should use ceil or the lower bound, see if I pass tests like this
    #print("~~~~~~~~~~~~~~~~~~")
    #print(int(number_of_items_per_fold))
    indices = [n for n in range(N)]
    random_state.shuffle(indices)
    train_ids_per_fold = list()
    test_ids_per_fold = list()
    
    for k in range(n_folds):
        #print("k = ", k)
        kth_training_folds = list()
        for i in range(n_folds):
            #print("i = ", i)
            if (i == k): # there will be only one testing fold
                test_ids_per_fold.append(np.asarray([j for j in indices[i*number_of_items_per_fold:number_of_items_per_fold*(i+1)]]))
                print("i == k is true / add testing")
            else: # there will be n_folds - 1 training folds
                kth_training_folds.append(np.asarray([j for j in indices[i*number_of_items_per_fold:number_of_items_per_fold*(i+1)]]))
                print("i == k is false / add training")
        #print("adding set of training folds to the big list")
        train_ids_per_fold.append(kth_training_folds)
        #print("now the train_ids_per_fold has size ", len(train_ids_per_fold))
        
    # TODO establish the row ids that belong to each fold's 
    # train subset and test subset - DONE

    return train_ids_per_fold, test_ids_per_fold



if __name__ == '__main__': # TODO eliminate the main function
    # Simple example use case
    # With toy dataset with N=100 examples
    # created via a known linear regression model plus small noise

    #prng = np.random.RandomState(0)
    #N = 100

    #true_w_F = np.asarray([1.1, -2.2, 3.3])
    #true_b = 0.0
    #x_NF = prng.randn(N, 3)
    #y_N = true_b + np.dot(x_NF, true_w_F) + 0.03 * prng.randn(N)

    #linear_regr = LeastSquaresLinearRegressor()
    #linear_regr.fit(x_NF, y_N)
    #indices = [n for n in range(N)]
    #print(indices)

    #print(x_NF)
    #print(x_NF.shape)
    #prng.shuffle(indices)
    #print(indices)
    #print(x_NF)
    #print(x_NF.shape)
    

    #yhat_N = linear_regr.predict(x_NF)
    #print(yhat_N)
    #print("shape")
    #print(yhat_N.shape)
    
    #tr_ids_per_fold, te_ids_per_fold = (make_train_and_test_row_ids_for_n_fold_cv(11, 3))
    #print("the size of tr_ids_per_fold is", len(tr_ids_per_fold), "where each element has size ", len(tr_ids_per_fold[0]))
    #print(tr_ids_per_fold)
    #print("<---------------------------->")
    #print("the size of te_ids_per_fold is", len(te_ids_per_fold), "where each element has size ", len(te_ids_per_fold[0]))
    #print(te_ids_per_fold)
    
    # Create simple dataset of N examples where y given x
    # is perfectly explained by a linear regression model
    N = 101
    n_folds = 7
    x_N3 = np.random.RandomState(0).rand(N, 3)
    y_N = np.dot(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
    
    arr1 = [[np.asarray([1, 2]), np.asarray([3, 4])], 
            [np.asarray([5, 6]), np.asarray([7, 8])]]
    
    
    arr2 = np.append(arr1[1][0], arr1[1][1])
    print(arr2)
    print(arr1[1][0])
                                        
    
    print(arr2[np.array([3, 3, 1, 2])])
    train_error_per_fold = np.zeros(2, dtype=np.int32)
    print(train_error_per_fold)
    #print(print(train_error_per_fold).shape)
    train_error_per_fold[0] = 1
    print(train_error_per_fold)
    
    


