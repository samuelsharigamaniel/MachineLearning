from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


 
## 

#df = pd.read_csv("data_1.txt")

data = np.loadtxt("data_1.txt")
l = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)
print(f'Original data shape: l = {l.shape} and y = {y.shape}')

#Start by splitting the training and testing data in a 70-30 split and shuffling with the same seed
l_train, l_test, y_train, y_test = train_test_split(l, y, test_size = 0.3, random_state = 42)
print(f'After train_test_split: l_train = {l_train.shape}, l_test = {l_test.shape} and y_train = {y_train.shape} and y_test = {y_test.shape}')

#######################################################################################################################################################################
#Set up the linear regression function from Scikit-learn:
#######################################################################################################################################################################
reg1 = linear_model.LinearRegression()

reg1.fit(l_train, y_train)

a = reg1.coef_
b = reg1.intercept_

print(f'coefficients a: {a[0]} and b: {b}')

x = np.linspace(0,20,100).reshape(-1,1)

yy = reg1.predict(x)
#yy = np.array([(a[0]*x[i] + b) for i in range(len(x))])

# Prediction of test sets using LR model 
yLR = reg1.predict(l_test)
#yLR = np.array([(a[0]*l_test[i] + b) for i in range(len(l_test))])


#######################################################################################################################################################################
#Set up the linear regression function for a polynomial function from Scikit-learn:
#######################################################################################################################################################################

reg2 = linear_model.LinearRegression()      # Creates a new LR model for the Polynomial Regression

poly = PolynomialFeatures(2)                # Creates a Polynomial Feature transformer of degree 2. This will generate [1, x, x^2] feature columns from a single input x

X_train = poly.fit_transform(l_train)       # Fits the polynomial feature transformer to the training input data and transforms it into its polynomial feature representation
X_test = poly.transform(l_test)

reg2.fit(X_train, y_train)                  # Fits the LR model using the polynomial features (X_train) and target variable (y_train). Model learns the coef. for the polynomial ax^2 + bx + c 

c = reg2.intercept_                         # Intercept term
b = reg2.coef_[0, 1]                        
a = reg2.coef_[0, 2]

#y2 = reg2.predict(X_train)
#x2 = poly.transform(x)
y2 = np.array([(a*(x[i]**2) + b*x[i] + c) for i in range(len(x))])  # plots the function to predict performances (y) for a given new set of length data (x)
#y2 = reg2.predict(x2)

print(y2.shape, x.shape)

# Prediction of test sets using Polynomial Regression
yPR = reg2.predict(X_test)
#yPR = np.array([(a*(l_test[i]**2) + b*l_test[i] + c) for i in range(len(l_test))])

# Computing the root mean squared error

total_LR = 0
total_PR = 0

for i in range(len(y_test)):
    total_LR += (yLR[i] - y_test[i])**2
    total_PR += (yPR[i] - y_test[i])**2

meanSquaredError_LR = total_LR/len(y_test)
meanSquaredError_PR = total_PR/len(y_test)


meanSquaredError_LR = mean_squared_error(y_test,yLR)
meanSquaredError_PR = mean_squared_error(y_test,yPR)


RMSE_LR = meanSquaredError_LR**0.5
RMSE_PR = meanSquaredError_PR**0.5

R2_LR = metrics.r2_score(y_test, yLR)
R2_PR = metrics.r2_score(y_test, yPR)

print(f"The RMSE for LR is {RMSE_LR:.4f} and the RMSE for the PR is {RMSE_PR:.4f}.")
print(f"The R2_score for LR is {R2_LR} and the R2_score for the PR is {R2_PR}.")

#######################################################################################################################################################################
### Ridge regression problem
#######################################################################################################################################################################

reg3 = linear_model.LinearRegression()      # Creates a new LR model for the Polynomial Regression

poly_3 = PolynomialFeatures(5)                # Creates a Polynomial Feature transformer of degree 2. This will generate [1, x, x^2] feature columns from a single input x
X_train = poly_3.fit_transform(l_train)       # Fits the polynomial feature transformer to the training input data and transforms it into its polynomial feature representation
X_test = poly_3.transform(l_test)

plt.figure(figsize=(12,8))

plt.scatter(l_train, y_train, label = 'Train data', color = 'blue')
plt.scatter(l_test, y_test, label = 'Test data', color = 'cyan')
#plt.scatter(l_test, yLR, label = 'Linear Regression predicted Test', color = 'cyan', marker = '*')
#plt.scatter(l_test, yPR, label = 'Polynomial Regression predicted Test', color = 'cyan', marker = 'D')
plt.plot(x,yy, color = 'red', label = 'Linear regression fit')
plt.plot(x, y2, color = 'yellow', label = 'Polynomial fit')

alp = [0.001, 0.1, 1, 10, 100, 1000, 10000, 1000000]
colr = ['green', 'green', 'green', 'green', 'purple', 'purple', 'purple', 'purple']
ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']
for i in range(len(alp)):
    reg3 = linear_model.Ridge(alpha = alp[i])
    reg3.fit(X_train, y_train)                  # Fits the LR model using the polynomial features (X_train) and target variable (y_train). Model learns the coef. for the polynomial ax^2 + bx + c 

    coefs = reg3.coef_
    intercept = reg3.intercept_

    yRR = np.array([(coefs[5]*(x[i]**5) + coefs[4]*(x[i]**4) + coefs[3]*(x[i]**3) + coefs[2]*(x[i]**2) + coefs[1]*x[i] + intercept) for i in range(len(x))])
    plt.plot(x, yRR, color = colr[i], linestyle = ls[i], label = f'Ridge Regression - α = {alp[i]}')

print(f'The Rigde Regression coefficients: {coefs} and intercept: {intercept}.')


## Plotter 


#plt.plot(x, yRR, color = 'brown', label = 'Rigde Regression fit')
plt.xlabel('lengths ($l_m$)', fontsize = 20)
plt.ylabel('performances ($y_m$)', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend()
plt.show()

"""

## 

data = np.loadtxt("data_2.txt")
X = data[:, 0:8]
y = data[:, 8:10]
print(X.shape, y.shape)

#Start by splitting the training and testing data in a 70-30 split and shuffling with the same seed
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3, random_state=12)

print(X_test.shape, Y_test.shape)

lr = linear_model.LinearRegression() #create an instance of the Linear Regression class

lr.fit(X_train, Y_train)            # Fit the training data

yy = lr.predict(X_train)            # Use the fitted model to predict the s,v value for the training data

x = np.linspace(0,1,1000)

s_ = Y_train[:,0]                   # this gives the s values
sm = yy[:,0]

v_ = Y_train[:,1]
vm = yy[:,1]

R2_s = metrics.r2_score(s_, sm)
R2_v = metrics.r2_score(v_, vm)

print(R2_s, R2_v)

plt.figure(figsize = (12,8))
plt.scatter(s_, sm)
plt.plot(sm,sm)
plt.plot(s_,s_, color = 'red', linestyle = ':')

#plt.scatter(v_, vm, color = 'red')
#plt.plot(v_,v_, color = 'red')

plt.show()

"""