# Homework # 1
Camelia D. Brumar

## Problem 1: Polynomial Regression - Model Selection on a Fixed Validation Set

![](figure1-err_vs_degree-fv.png)
*Figure 1: Line plot of mean-squared error on y-axis vs. 
polynomial degree on x-axis. On one hand, the error on training set 
(dotted blue line) is strictly decreasing as the degree
 of the polynomial increases. On the other hand, the error 
 on validation set (solid red line with square 
markers) is decreasing until the polynomial reaches degree 2,
then it increases as the degree also increases.*

### Short Answer 1a
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

### Short Answer 1b
> At some point, the mean squared error on the training set should become very close to zero (say, within 0.5 or so). At what degree value do you observe this happening for this particular training dataset? What technical argument can you make to justify this (e.g. for this dataset, why is training error zero at X degrees but not X-1 degrees?)

Answer: To answer this, I created a table that shows the degrees of the
polynomials with their corresponding errors:

![](figure1_table1_3decimals.png)

The training error of the degree 6 polynomial is approx. 0.2999, while the training error for
the degree 7 polynomial is approx. 0.0138. Thus, I can say that at degree 6 the training
MSE becomes very close to 0. The degree 7 polynomial has a lower training error since it
is a more flexible model that overfits the training more closely than the degree 6 one.

### Short Answer 1c
> You'll notice that our pipelines include a preprocessing step that rescales each feature column to be in the unit interval from 0 to 1. Why is this necessary for this particular dataset? What happens (in terms of both training error and test error) if this step is omitted?

Answer: I removed the scaling step from the pipeline to empirically see what happens, and here is a table with
the degrees vs the errors

![](figure1_table2_3decimals.png)

Apparently, the training error does not decrease so dramatically as it did in the case when 
the pipeline was doing the scaling step. This happens because the data measured for the 
different features have different magnitudes. For example, the cylinders feature can be 4 or 8, while
the weight is a number above a thousand (lb). This difference in scales it makes it more difficult
for the polynomials to overfit the training data, and thus more difficult to minimize the training MSE.
Interestingly the testing error seems to improve in this case, and this is for the same reason
as before, the polynomials are not overfitting the training data, hence the model generalizes
better to unknown data points. 

It is necessary to scale the data since Linear Regression uses
Euclidean distance in its computations in order to fit the data. In this particular case, the
weight feature will weight in a lot more in distance computations than features with low
magnitudes such as the cylinders feature. To supress this effect, we need to bring all features to the same level of magnitudes by scaling.

### Short Answer 1d
> Consider the model with degree 6. Print out its intercept coeficient value, as well as the minimum and maximum weight coeficient value (out of all the features). What do you notice about these values? How might they be connected to the training and validation set performance you observe in Figure 1?

- The intercept of the 6th degree polynomial is 1,2901,120,738.129
- Min weight of the 6th degree polynomial is -9,693,939,910,344.908, which corresponds to the cylinders^4 feature
- Max weight of the 6th degree polynomial is 5,546,402,397,351.329, and it corresponds to the cylinders^3 feature

These values may occur because there is collinearity between the variables.

## Problem 2: Polynomial Regression - Model Selection with Cross-Validation

![](figure2-err_vs_degree-cv-seed=12345.jpg)
*Figure 2: Line plot of mean-squared error on y-axis vs. 
polynomial degree on x-axis. On one hand, the error on training set 
(dotted blue line) is strictly decreasing as the degree
 of the polynomial increases. On the other hand, the error 
 on validation set (solid red line with square 
markers) is decreasing until the polynomial reaches degree 2,
then it increases as the degree also increases.*



### Short Answer 2a
> If your goal is to select hyperparameters for your pipeline that will generalize well to new data from the same distribution, which polynomial degree do you recommend based on this assessment? Are there other degrees that seem to give nearly the same performance? What (if anything) changed from 1a?

I would recommend degree 2 again, just like in 1a. The degree 1 polynomial seems to give
nearly the same performance, but it's worse than in the quadratic case. The difference 
with Figure 1 is that in Figure 2 the testing error grows even faster. Past the 
degree 3, the testing MSE already tends to infinite.

### Short Answer 2b
> What are two benefits of using cross validation when compared to a fixed validation set (as in Problem 1)?

First, cross validation reduces the variability of the MSE compared with the fixed validation
set, i.e. if the training set is slightly altered, the MSE will be barely changed, which is not the
case of the fixed validation set (this is explained in the ISL book, page 178).

Second, cross validation tends to better estimate the error, while the fixed validation set
approach tends to overestimate it since in this approach the
training set used to fit the statistical learning method contains only half
the observations of the entire data set.

### Short Answer 2c
> What are two drawbacks to using cross validation when compared to a fixed validation set (as in Problem 1)?

First, cross-validation is computationally more expensive than the fixed validation approach.
This is because in cross validation we have to train k models and asses their training and testing errors, 
while in the fixed validation set we only train one model.

Second, in some cases we have clustered/hierarchical/grouped data, in which it is hard to 
split them into k folds without them overlapping. If overlapping the data in two different
 folds happens, this could cause underestimates of the MSE since there will be datapoints
 in one of the training folds that also falls in the validation fold. (I found this interesting
 article about why we are still using Hold-out validation nowadays: 
 https://stats.stackexchange.com/questions/104713/hold-out-validation-vs-cross-validation/104750#104750).
 
### Short Answer 2d
> Remember, your task is to develop models that will accurately predict miles per gallon given some basic features of car engines. Suppose your available data is augmented so that each example is associated with a specific manufacturer (e.g. each row of your development set could be labeled 'Toyota' or 'Ford' or 'Hyundai'). You have data labeled with 10 different manufacturers. You'd like your prediction to be accurate for new manufacturers, that you do not have available in your training set. (Thus, your regression model should not use manufacturer label as a feature). How would you suggest we change our cross validation procedure to do better at this task?

I would suggest making sure that in each fold of the cross validation data from all
10 manufacturers is included, in this way when, when the model is being trained, it is not
specific to only one or two manufacturers, but it will accurately predict for all the manufacturers,
and hopefully also for the manufacturers that the model hasn't seen.

## Problem 3: Polynomial Regression with L2 Regularization - Model Selection with Cross-Validation
