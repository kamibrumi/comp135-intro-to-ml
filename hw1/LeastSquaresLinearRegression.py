import numpy as np
# No other imports allowed!

class LeastSquaresLinearRegressor(object):
    ''' A linear regression model with sklearn-like API

    Fit by solving the "least squares" optimization problem.

    Attributes
    ----------
    * self.w_F : 1D numpy array, size n_features (= F)
        vector of weights, one value for each feature
    * self.b : float
        scalar real-valued bias or "intercept"
    '''

    def __init__(self):
        ''' Constructor of an sklearn-like regressor

        Should do nothing. Attributes are only set after calling 'fit'.
        '''
        # Leave this alone
        pass

    def fit(self, x_NF, y_N):
        ''' Compute and store weights that solve least-squares problem.

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
            Input measurements ("features") for all examples in train set.
            Each row is a feature vector for one example.
        y_N : 1D numpy array, shape (n_examples,) = (N,)
            Response measurements for all examples in train set.
            Each row is a feature vector for one example.

        Returns
        -------
        Nothing. 

        Post-Condition
        --------------
        Internal attributes updated:
        * self.w_F (vector of weights for each feature)
        * self.b (scalar real bias, if desired)

        Notes
        -----
        The least-squares optimization problem is:
        
        .. math:
            \min_{w \in \mathbb{R}^F, b \in \mathbb{R}}
                \sum_{n=1}^N (y_n - b - \sum_f x_{nf} w_f)^2
        '''
        N, F = x_NF.shape
        
        xtilde_NG = np.hstack([x_NF, np.ones((N, 1))])
        xTx_GG = np.dot(xtilde_NG.T, xtilde_NG)
        
        theta_G1 = np.linalg.solve(xTx_GG, np.dot(xtilde_NG.T, y_N))
        #print(theta_G1.shape)
        self.w_F = theta_G1[:-1]
        self.b = theta_G1[-1]
        
        pass # TODO - DONE - should I leave the pass here?



    def predict(self, x_MF):
        ''' Make predictions given input features for M examples

        Args
        ----
        x_NF : 2D numpy array, shape (n_examples, n_features) (M, F)
            Input measurements ("features") for all examples of interest.
            Each row is a feature vector for one example.

        Returns
        -------
        yhat_N : 1D array, size M
            Each value is the predicted scalar for one example
        '''
        # TODO FIX ME
        return np.dot(x_MF, self.w_F) + self.b





if __name__ == '__main__':
    import doctest
    doctest.testmod()

