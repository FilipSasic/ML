import math
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0,2.0,2.5,3.0, 4.0,7])
y_train = np.array([300.0, 500.0, 520.0, 600.0, 900.0 , 1000.0])

def linear_regression_model(x,w,b):
    return w*x + b

#divide by 2 for easier derivation later
def squared_error_loss(y,y_pred):
    diff = (y - y_pred)**2
    return diff.mean()/2

def compute_gradient(x, y,w,b):
    dw_i = (x*w + b - y)*x
    db_i = (x*w + b - y)
    return dw_i.mean(), db_i.mean() 
    
def train():
    w = 0
    b = 0
    epoch = 10000
    lr = 0.01
    
    for i in range(epoch):
        
        y_pred = linear_regression_model(x_train, w, b)
        print(squared_error_loss( y_train,y_pred))
    
        grad_w,grad_b = compute_gradient(x_train,y_train,w,b)
        w = w - lr*grad_w
        b = b - lr*grad_b
    
    return w,b
        
        
w,b = train()

x_test = np.array([0.0,10.0])
y_test = w*x_test + b
    
# Create a figure containing a single axes.   
fig, ax = plt.subplots() 
# Plot some data on the axes, color red, marker x
ax.scatter(x_train, y_train,c = 'r',marker = 'x', label = 'real data')
ax.plot(x_test,y_test, label = 'predicted')
ax.legend()