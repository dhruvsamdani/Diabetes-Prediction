# Diabetes-Prediction

### Logistic Regression is a work in progress

Diabetes Prediction uses a Random Forest algorithim and a Logistic Regression to predict if someone has diabetes based on a set of data.

## How it works

### Random Forest:

A random forest algorithim is one that complies **n** number of decision trees and takes the majority classification of the results of them. A decision tree classifies something with a class by splitting it based on some of the features of the data. Random Forests provide more value; however, as they are not as prone to overfitting in comparision to a decision tree.

### Logistic Regression:

Logistic Regressions work by putting an equation through a sigmoid function. To find the equation there is a loss function that needs to be minimized. This can be done through gradient descent.

**Sigmoid Function:**

<img src="https://latex.codecogs.com/svg.image?\large&space;\frac{\large&space;1}{\large&space;1&space;&plus;&space;e^{-z}}" title="\large \frac{\large 1}{\large 1 + e^{-z}}" />

where z is the equation based on the data that takes the form

<img src="https://latex.codecogs.com/svg.image?\large&space;a&space;&plus;&space;b_1x&space;&plus;&space;b_2x&space;&plus;&space;...&space;&plus;&space;b_nx" title="\large a + b_1x + b_2x + ... + b_nx" />

**Loss Function:**

<img src="https://latex.codecogs.com/svg.image?\large&space;-\frac{\large&space;1}{\large&space;n}\sum_{i=1}^{n}&space;{y_i\log{P(x_i)}&space;&plus;&space;(1-y_i)\log{(1-P(x_i))}}" title="\large -\frac{\large 1}{\large n}\sum_{i=1}^{n} {y_i\log{P(x_i)} + (1-y_i)\log{(1-P(x_i))}}" />

## Data:

The data is from the [**Pima Indian Diabetes Dataset**](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

The dataset has **768** members and **8** features, such as, Pregnancies, Glucose, Insulin, and more

## Dependencies:

Python 3.0+

```
pip install numpy
pip install scipy
```

## Usage:

Import a dataset, strip the title row, and seperate the X (feature) values from the y (outcomes) values
Run:

```Python
from RandomForest import RandomForest as RTC
randomForest = RTC(X_train,Y_train)

# 100 is the number of trees and 20 is the max depth for each tree
randomForest.randomForest(100,20)
outcome = randomForest.predict(testing_value)
```

Where `X_train` and `Y_train` are the training data and `testing_value` is the data you want to test the outcome will be stored in `outcome`
