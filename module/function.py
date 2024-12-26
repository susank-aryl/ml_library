import numpy as np
import math, copy

def linear_regression(x, w, b)-> np.ndarray:
    f_wb = np.dot(w,x) + b
    return f_wb

def compute_cost(x, y, w, b):
    total_cost = 0
    m = x.shape[0]

    for i in range(m):
        f_wb_i = np.dot(w, x[i]) + b    
        total_cost += total_cost + (f_wb_i - y[i])**2
    total_cost = total_cost / (2 * m)
    return total_cost


def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha , num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history.append(cost_function(X,y,w,b))

        if i%math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i : 4d}: cost {J_history[-1]: 8.2f}")

    return w,b,J_history

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

#Parameters
initial_w = np.zeros_like(w_init)
initial_b = 0
iteration = 1000
alpha = 5.0e-7

#running gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w,initial_b,compute_cost,compute_gradient,alpha,iteration)

print(f"The best value for w and b are: {w_final}, {b_final:.2f}")
# print(linear_regression(x,w,b))
# print(compute_cost(x,y,w,b))exit
# print(compute_gradient(X_train,y_train,w_init,b_init))

