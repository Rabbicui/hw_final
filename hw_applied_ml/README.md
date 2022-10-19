# HW_final
This is a code repository of daily homework, which is filled with rabbi's naive thought; The world here involves only choosing topics, writing reports, and running code. More details can be introducted as follow.

### env
python--3.8.0

pytorch

pip install statsmodel

pip install sklearn

## Prediction and Improvement of Housing Prices by Classical Linear Regression Model
### house price prediction.ipynb    (dataset\Real estate valuation data set.xlsx)
The data comes from a set of housing price data in Taipei City, Taiwan Province and its surrounding areas from 2012 to 2013 on https://archive.ics.uci.edu/ml/datasets.php. The data set contains 414 housing price information, each piece of information It is composed of 7 factors: "transaction date", "room age", "distance to MRT station (subway station)", "number of convenience stores", "latitude", "longitude", and "unit square price". The data set is roughly in line with the basic assumptions of the linear regression model, and the data set after cleaning and normalization is input into the classical regression model: multiple linear regression, ridge regression, robust regression, polynomial with degrees 2 and 3 regression model. By comparing the coefficient of determination, the standard deviation of prediction, and the comprehensive evaluation index Evalue to test the pros and cons of the model, it is found that as the complexity of the model increases, the fit on the training set is better, and the error on the prediction set is greater. big. In order to seek a balance between goodness of fit and prediction accuracy, this paper calls the PolynomialFeature function to construct a complete set of explanatory variable combinations with degree=3, and selects the optimal combination of independent variables through stepwise regression based on the goodness-of-fit test coefficient. Compared with the general linear model, the accuracy of this composite model is improved by 18.5%, and the comprehensive index is also the best among all models, which can be used for the prediction of housing price datasets whose sample size is much larger than the number of independent variables. Considered from different perspectives, this model can provide decision-making reference for developers to select and set prices, and for the public to buy and live in or invest.


## Visualization of International Trade Data in the Era of COVID-19
### International trade visualization.ipynb  (dataset\Effects.csv)

Based on the trade data between countries from 2015 to 2020, this article takes "the impact of the new crown pneumonia epidemic on import and export trade" as the analysis purpose, takes the data in the two columns of Year and Value as the main line, and focuses on exploring the transaction value of each feature of the data set. The performance in different years, the main analysis tool is data visualization. According to the data type and size, the appropriate chart is called to visualize the data of each variable in turn. Combined with the visualization results, it is given how to maintain the normality between countries under the trend of normalization of the epidemic. trade.

### customer-segmentation.ipynb   (dataset\Train.csv & Test.csv)
The customer-segmentation dataset is selected from Kaggle, which aims to locate and classify by the behavioral characteristics of customers. Based on the selected data set, the category feature data is encoded and normalized under the pandas framework, and the processed data set is input into the classic logistic regression, decision tree, and other classification models, and the GridSearchCV function in sklearn is called to perform Xgboost Optimization with lightgbm hyperparameters. Based on the optimized model, create a customer class with the attributes of the customer, and check the parameters of the exception constructor. If it is unreasonable, an exception will be triggered to refuse to generate an object, and the customer category group can be predicted by defining a classification method. This class encapsulates the optimized classification model and implements the function of customer category prediction.
more details：https://github.com/Rabbicui/customer-segmentation

### pytorch-regression-crossvadilation-earlystop.ipynb
1. Use the model to randomly generate 100 data points (x, y), where x to be uniformly distributed within [-1, 1]. Then all samples are randomly divided into training set and test set.
Please train a polynomial regression model of degree n to predict y given x, for n=1, 2, 3, and 8, respectively. For different n, plot the training and test errors as a function of the number of training iterations. The training algorithm uses the gradient descent based Adam algorithm with default parameters and iterates 5000 steps.

2. Observe whether there is overfitting in the above question, and use the 5-fold cross-validation method to find the most suitable model, that is, the optimal n. And finally report the test error on the test set.

3. Generate another 50 points as a validation set. For n=8, use the earlystopping method to obtain a polynomial regression model to solve the overfitting problem. And finally report the test error on the test set. The Earlystopping method is to find the iteration step when the validation set error reaches the minimum point, and use the model at this moment as the final model.

## training minist by Keras
### Dnn_minist.ipynb
Learn about the handwritten character dataset, minist from the website https://www.kaggle.com/c/digit-recognizer/overview, and use the training data train.csv to train a multi-layer fully connected feedforward neural network + Softmax classifier model. Predict on the data test.csv, refer to the format of sample_submission.csv, submit your prediction results on the website, and record the score. Note that the data features represent pixel values from 0-255, and the data can be normalized to [0,1] first. If computing resources are limited, PCA dimensionality reduction can be considered first (optional)


## xgboost to predict income
### income_ml.ipynb
Download the census data set from the website https://archive.ics.uci.edu/ml/datasets/Adult.
1) Understand this dataset;
2) Prepare the data and preprocess the features;
3) Train any model you have learned in this lesson on the adult.data training set and find a way to choose appropriate hyperparameters;
4) Calculate the classification accuracy and F1 score on both the adult.data training set and the adult.test test set, and try to improve the performance of the model on the test set.


