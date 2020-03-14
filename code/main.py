import pandas as pd
import numpy as np
from scipy.linalg import block_diag

# Data
data = pd.read_csv(r'C:/Users/jcdav/Documents/GitHub/ODS-lab-05/code/The data for #5.csv')
data = np.array(data)
d = np.array(data[1:48,2:11],dtype=float)
f = np.array(data[:,0],dtype=str)
t = np.array(data[:,1],dtype=str)


# Misclosure Matrix and Covariance
w=[]
count = 0
cov = np.array([])

for i in range (47):
    base_i = d[i,2:5]
    cov_i = np.array([[d[i,3],d[i,4],d[i,5]],[d[i,4],d[i,6],d[i,7]],[d[i,5],d[i,7],d[i,8]]])
    for j in range (i+1,47):
        base_j = d[j,2:5]
        cov_j = np.array([[d[j,3],d[j,4],d[j,5]],[d[j,4],d[j,6],d[j,7]],[d[j,5],d[j,7],d[j,8]]])
        f_val = f[j]
        t_val = t[j]
        if f_val == f[i] and t_val == t[i]:
            w.append(base_i + base_j)  
            cov_val = cov_i + cov_j
            cov = block_diag(cov,cov_val)
        elif f_val == t[i] and t_val == f[i]:
            w.append(base_i - base_j)
            cov_val = cov_i + cov_j
            cov = block_diag(cov,cov_val)
            
cov = np.delete(cov, 0, 0)
w = np.ravel(w)

# A priori estimation
apr = (w.T@cov@w)/w.size
n_cov = apr*cov


# 95% Confidence interval misclosure test:
variance = np.diag(n_cov)
compare = np.less_equal(w,1.96*variance)
compare = np.reshape(compare,[21,3])
print(compare)


# Chi squared test
for i in range(0,62,3):
    w_i = np.array([w[i], w[i+1], w[i+2]])
    cov_i = cov[i:i+3,i:i+3]
    val = w_i.T@cov_i@w_i
    if val > 1.82:
        val = 1



