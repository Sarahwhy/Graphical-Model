import numpy as np
import math

def convert_index(i,j):
    return (i-1)*7+j


def neighbour(i,j):
    neigh = [[i-1,j],[i+1,j],[i,j-1],[i,j+1]]
    if i == 1:
        neigh[0] = [7,j]
    if i == 7:
        neigh[1] = [1,j]
    if j == 1:
        neigh[2] = [i,7]
    if j == 7:
        neigh[3] = [i,1]
    return neigh

# This sampling is for Ising model with binary variables for 
#  a 2-D grid with toroidal (donut-like) boundary conditions

def generate_toroidal(k):
    Mat = np.zeros((k**2,k**2))
    for i in range(1,k+1,1):
        for j in range(1,k+1,1):
            index = convert_index(i,j)
            neigh = neighbour(i,j)
            for p in range(len(neigh)):
                neigh_index = convert_index(neigh[p][0],neigh[p][1])
                Mat[index-1,neigh_index-1] = 1
    return Mat


# Gibbs Sampling

def Gibbs(Mat, theta_s, theta_st, burnIn, num):
    n = Mat.shape[0]
    samples = []
    X = np.ones((n,1))
    # Randomize the X_t
    X[np.random.random((n,1))>0.5] = -1.0
    iteration = burnIn+num
    for i in range(iteration):
        # Update for each node based on randomized sequence
        permutation = np.random.permutation(n)
        for s in permutation:
            X_N = X[Mat[s]!=0]
            alpha = theta_s[s] + theta_st * np.sum(X_N)
            # Check the probability
            if np.random.uniform(0,1,1)[0] < math.exp(alpha)/(math.exp(alpha)+math.exp(-alpha)):
                X[s] = 1.0
            else:
                X[s] = -1.0
        if i >= burnIn:
            samples.append(np.concatenate(X))
    return samples


# Naive mean field method
def Naive_Mean(Mat, theta_s, theta_st):
    n = Mat.shape[0]
    X = np.ones((n,1))
    X[np.random.random((n, 1)) > 0.5] = -1
    X = np.concatenate(X)
    delta = 1
    i = 0
    while i<1000:
        permutation = np.random.permutation(n)
        for s in permutation:
            X_N = X[Mat[s] != 0]
            beta = theta_s[s] + theta_st * np.sum(X_N)
            X[s] = (math.exp(2*beta)-1)/(1+math.exp(2*beta))
        i +=1
    return X

# Generate the example when k = 7
Mat = generate_toroidal(7)
theta_s = [(-1)**k for k in range(1,50,1)]
theta_st = 0.25
print (Naive_Mean(Mat, theta_s, theta_st))
