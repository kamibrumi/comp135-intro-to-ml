# Homework # 1
Camelia D. Brumar

## Problem 1: Polynomial Regression - Model Selection on a Fixed Validation Set
Line plot of mean-squared error on y-axis vs. polynomial degree on x-axis. It shows two lines, one for error on training set (dotted blue line) and another line for error on validation set (solid red line with square markers).
![](figure1-err_vs_degree-fv.png)
*Figure 1*

### Short Answer 1a in Report
> If your goal is to select a model that will generalize well to new data from the same distribution, which polynomial degree do you recommend based on this assessment? Are there other degrees that seem to give nearly the same performance?

**Answer:** I would recommend choosing the degree 2 polynomial since it
minimizes the testing MSE.This means higher degree polynomials
will tend to over-fit the training data, which will decrease the
training MSE, but it will drastically increase the testing
MSE. So, if the goal for the chosen model is to generalize to points that
were not used for the training, then the degree 2 polynomial will
be the one to chose in this case. Perhaps the degree 1 polynomial
has similar performance as the degree 2 polynomial, but both
its training and testing MSE are higher, so the degree 2 polynomial
remains the best choice.

### Short Answer 1b in Report
> At some point, the mean squared error on the training set should become very close to zero (say, within 0.5 or so). At what degree value do you observe this happening for this particular training dataset? What technical argument can you make to justify this (e.g. for this dataset, why is training error zero at X degrees but not X-1 degrees?)


