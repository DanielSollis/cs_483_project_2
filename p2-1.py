#!/usr/bin/env python3
# CPSC 483 - Project 2

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV


file_path = 'Wage.csv'
df = pd.read_csv(file_path)
df = df.sort_values(by=["age"])

X = pd.DataFrame(df.age)
y = pd.DataFrame(df.wage)



# Experiment 1
def show_scatter(df):
    plt.scatter(df.age, df.wage, label='Data')
    plt.xlabel('Age')
    plt.ylabel('Wage')


# Experiment 2
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, shuffle=True)


# Experiment 3
lm = LinearRegression().fit(X_train, y_train)
plt.figure(1)
show_scatter(df)
plt.plot(X, lm.predict(X), color='red', label='First Order')
plt.legend()


# Experiment 4
print('First order coefficients:', lm.coef_)
print('R^2 first order: ', lm.score(X_test, y_test))


# Experiment 5
poly = PolynomialFeatures(4)
X_train_4 = poly.fit_transform(X_train)
lm = lm.fit(X_train_4, y_train)

X_test_4 = poly.fit_transform(X_test)
print('Fourth order coefficients:', lm.coef_)
print('R^2 fourth order: ', lm.score(X_test_4, y_test))


X_4 = poly.fit_transform(X)
fourth_order_pred = lm.predict(X_4)
plt.figure(2)
show_scatter(df)
plt.plot(X, fourth_order_pred, color='red', label='Fourth Order')
plt.legend()


# Experiment 6 - added normalization, not sure if needed but removes
# ill-conditioned matrix warning and everything still works?
ridge = Ridge(alpha=0.1, normalize=True).fit(X_train_4, y_train)
ridge_4_pred = ridge.predict(X_4)
print(ridge_4_pred)
print('Ridge fourth order coefficients:', ridge.coef_)
print('R^2 Ridge fourth order: ', ridge.score(X_test_4, y_test))
plt.figure(3)
show_scatter(df)
plt.plot(X, ridge_4_pred, color='red', label='Ridge-Fourth')
plt.legend()


# Experiment 7 - added alpha values below .001 due to normalization
ridge_cv = RidgeCV(alphas=[1e-07, 1e-06, 1e-05, 1e-04, 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0],
                        normalize=True).fit(X_train_4, y_train)

ridge_4_cv_pred = ridge_cv.predict(X_4)
print('Fourth order RidgeCV coefficients', ridge_cv.coef_)
print('R^2 fourth order RidgeCV: ', ridge_cv.score(X_test_4, y_test))
# Gives you the alpha value w/ the lowest MSE - 1e-06 is best for our normalized data
print('Ridge_cv alpha: ', ridge_cv.alpha_)

plt.figure(4)
show_scatter(df)
plt.plot(X, ridge_4_cv_pred, color='red', label='RidgeCV-Fourth')
plt.legend()
#plt.show()


# Experiment 8
X = df.drop(columns=['Unnamed: 0', 'jobclass', 'logwage', 'wage'])

y = df.logwage

def to_categorical(df):
    '''returns a new pd DataFrame with categorical data
    separated into columns of binary values'''
    dfc = pd.DataFrame(df)
    for column in dfc.columns:
        if dfc[column].dtype == object:
            dummyCols = pd.get_dummies(dfc[column])
            dfc = dfc.join(dummyCols)
            del dfc[column]
    return dfc

# one-hot encode categorical data
X = to_categorical(X)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, shuffle=True)

lm = LinearRegression().fit(X_train, y_train)
coefs = pd.DataFrame(list(zip(X.columns, lm.coef_)),
                     columns=["Feature", "Coefficient"])
print(coefs)

# Extra experiment to find the best order poly
NUM_TRIALS = 50
for degree in [1, 2]:
    r2 = 0
    for trial in range(NUM_TRIALS):
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.fit_transform(X_test)

        # Experiment 7 - added alpha values below .001 due to normalization
        ridge_cv = RidgeCV(alphas=[1e-07, 1e-06, 1e-05, 1e-04, 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1.0, 2.0, 3.0], 
                            normalize=True).fit(X_train_poly, y_train)

        r2 += ridge_cv.score(X_test_poly, y_test)

    print('R^2 degree ' + str(degree) +  ' RidgeCV: ', str((r2/NUM_TRIALS)))
    print('Alpha for degree ' + str(degree) + ' RidgeCV: ', ridge_cv.alpha_)

