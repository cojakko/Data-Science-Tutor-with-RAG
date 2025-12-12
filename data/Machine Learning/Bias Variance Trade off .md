The bias-variance tradeoff is a fundamental and widely discussed concept in the area of Data Science. Understanding the bias-variance tradeoff is essential for developing accurate and reliable machine learning models, as it can help us optimize model performance and avoid common pitfalls such as underfitting and overfitting.

Before defining it, it is necessary to define what is bias and variance separately.
Bias
Bias refers to the error that is introduced by approximating a real-life problem with a simplified model. A model with high bias is not able to capture the true complexity of the data and tends to underfit, leading to poor performance on both the training and test data. The bias is represented by the difference between the expected or true value of the target variable and the predicted value of the model.

Variance
Variance refers to the error introduced by the model’s sensitivity to small fluctuations in the training data. A model with high variance tends to overfit the training data, leading to poor performance on new, unseen data. Variance is represented by the degree of variability or spreads in the model’s predictions for different training sets.

Understanding the bias-variance tradeoff is essential for developing accurate and reliable Machine Learning models. It can help to optimize the model performance and avoid common pitfalls such as underfitting and overfitting. One of the best ways to visualize the bias and variance concepts is through a dartboard like the one shown below.

Source: V. Gudivada, A. Apon & J. Ding, 2017
Source: V. Gudivada, A. Apon & J. Ding, 2017
The figure shows how variance and bias are related:

A model with high bias and high variance is a model that makes a lot of mistakes and is very inconsistent.
A model with high variance and low bias tends to be more accurate, but the results suffer a lot of variations.
A model with high bias and low variance is a model that makes a lot of bad predictions but is very consistent in its results.
Lastly, a model with low bias and variance makes good predictions and is consistent with its results.
Looking at the figure, it’s intuitive that all the models should have a low bias and low variance, since this combination generates the best results. However, this is where the bias-variance tradeoff appears.
The bias-variance tradeoff arises because increasing the model’s complexity can reduce the bias but increase the variance. On the other hand, decreasing the complexity can reduce the variance but increase the bias. The goal is to find the optimal balance between bias and variance, which results in the best generalization performance on new, unseen data.

This is directly related to the complexity of the model used, as shown in the figure below.

Bias-variance tradeoff and error relationship (image by author)
Bias-variance tradeoff and error relationship (image by author)
The graph shows how the complexity of the model is related to the values of bias and variance. Models that have a low complexity can be too simple to understand the patterns of the data used in the training, a phenomenon called underfitting. **** Consequently, it won’t be able to make good predictions on the test data, resulting in a high bias.

On the other hand, a model with too much degree of liberty can result in what is called overfitting, **** which is when the model has an excellent performance in the training data, but has a significant decrease in performance when evaluating the test data. This happens when the model becomes extremely accustomed to the training data, thus losing its generalization capability and, when it needs to interpret a data sample never seen before, it cannot get a good result.

As the model’s complexity increases, the bias decreases (the model fits the training data better) but the variance increases (the model becomes more sensitive to the training data). The optimal tradeoff occurs at the point where the error is minimized, which in this case is at a moderate level of complexity.

To help understanding, let’s look at a practical example that illustrates the concept of bias-variance tradeoff.
To illustrate the impacts of the bias-variance tradeoff in Machine Learning models, let’s see how models with different levels of complexity will perform when trained and tested on the same datasets.

For this example, a random dataset with a quadratic relationship between the input X and the output y will be generated. We then split the data into training and test sets and fit three polynomial regression models of different degrees (1, 2, and 20). We plot the resulting models along with the training and test data and calculate the mean squared error for both the training and test sets.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate random data
np.random.seed(0)
X = np.linspace(-5, 5, 100)
y = X**2 + np.random.normal(0, 4, size=100)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit polynomial regression models with different degrees
degrees = [1, 2, 20]
plt.figure(figsize=(15, 5))
for i, degree in enumerate(degrees):

    # Generating the data
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train.reshape(-1, 1))
    X_poly_test = poly.transform(X_test.reshape(-1, 1))
    
    # Creating the model
    model = LinearRegression()
    
    # Training the model
    model.fit(X_poly_train, y_train)
    
    # Evaluating the model
    y_pred_train = model.predict(X_poly_train)
    y_pred_test = model.predict(X_poly_test)
    train_error = mean_squared_error(y_train, y_pred_train)
    test_error = mean_squared_error(y_test, y_pred_test)
    
    # Plotting the results  
    plt.subplot(1, len(degrees), i+1)
    plt.scatter(X_train, y_train, color='blue')
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X, model.predict(poly.transform(X.reshape(-1, 1))), color='green')
    plt.title('Degree={}, Train Error={:.2f}, Test Error={:.2f}'.format(degree, train_error, test_error))
plt.show()
view rawbias_variance_tradeoff.py hosted with ❤ by GitHub
The resulting plot shows the bias-variance tradeoff for the different polynomial regression models:

Results obtained by the model for different degrees (image by author).
Results obtained by the model for different degrees (image by author).
The model with degree = 1 is way too simplistic and has high bias and low variance, resulting in underfitting and high errors on both the training and test data. The model with degree = 20 is too complex and has low bias and high variance, resulting in overfitting and low error on the training data but a high error on the test data. The model with degree = 2 has a good balance between bias and variance and results in the lowest test error.

This example demonstrates the importance of finding the right level of complexity for a machine learning model to balance bias and variance and achieve good generalization performance on new, unseen data.

Hopefully, this article was able to help you understand the bias-variance tradeoff and how to consider it when developing Machine Learning models.

Any comments and suggestions are more than welcome.