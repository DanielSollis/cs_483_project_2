#!/usr/bin/env python3

#CPSC 483 - Project 2

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

file_path = 'Wage.csv'
df = pd.read_csv(file_path)
df = df.sort_values(by=["age"])

X = pd.DataFrame(df.age)
y = pd.DataFrame(df.wage)

# Experiment 1
def exp_1(df):
    plt.scatter(df.age, df.wage)
    plt.xlabel('Age')
    plt.ylabel('Wage')
    

# Experiment 2
X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=0.2, shuffle=True)

# convert pandas series to DataFrames
# alternative to reshape(-1, 1) - think this works?
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

x_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

# Experiment 3
lm = LinearRegression().fit(X_train, y_train)

plt.figure(0)
exp_1(df)

plt.plot(X, lm.predict(X), color='red', label='First Order')

# Experiment 4
print('First order coefficients', lm.coef_)
print('R^2 first order: ', lm.score(X_test, y_test))

# Experiment 5
poly = PolynomialFeatures(4)
X_train = poly.fit_transform(X_train)
lm.fit(X_train, y_train)

print('Fourth order coefficients', lm.coef_)
print('R^2 fourth order: ', lm.score(X_train, y_train))

X_4 = poly.fit_transform(X)

fourth_order_predicted = lm.predict(X_4)

plt.figure(1)
exp_1(df)

plt.plot(X, fourth_order_predicted, color='green', label='Fourth Order')
plt.legend()
plt.show()