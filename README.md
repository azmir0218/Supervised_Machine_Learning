# Supervised Machine Learning - Predicting Credit Risk

The goal of this project is to build a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not. 

## Background

LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.

We will be using this data to create machine learning models to classify the risk level of given loans. Specifically, we will be comparing the Logistic Regression model and Random Forest Classifier.


### Data Used

* `2019loans.csv`
* `2020Q1loans.csv`

We will be using an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).

Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. 

## Preprocessing: Convert categorical data to numeric

We will create a training set from the 2019 loans using `pd.get_dummies()` to convert the categorical data to numeric columns. Similarly, we will create a testing set from the 2020 loans, also using `pd.get_dummies()`. Note! There are categories in the 2019 loans that do not exist in the testing set. If we fit a model to the training set and try to score it on the testing set as is, we will get an error. We need to use code to fill in the missing categories in the testing set. 

## Scale the data

The data going into these models was never scaled, an important step in preprocessing. Use `StandardScaler` to scale the training and testing sets. Fit and score the LogisticRegression and RandomForestClassifier models on the scaled data. 


### References

LendingClub (2019-2020) _Loan Stats_. Retrieved from: [https://resources.lendingclub.com/]

