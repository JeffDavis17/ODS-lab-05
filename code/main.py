import pandas as pd
import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import inv


# Data
data = pd.read_csv(r'C:/Users/jcdav/Documents/GitHub/ODS-lab-05/code/The data for #5.csv')
data = np.array(data)
d = np.array(data[1:48,2:11],dtype=float)
f = np.array(data[1:48,0],dtype=str)
t = np.array(data[1:48,1],dtype=str)


# Misclosure Matrix and Covariance
w=[]
cov = np.array([])
lines = []

for i in range (47):
    base_i = d[i,0:3]
    cov_i = np.array([[d[i,3],d[i,4],d[i,5]],[d[i,4],d[i,6],d[i,7]],[d[i,5],d[i,7],d[i,8]]])
    for j in range (i+1,47):
        base_j = d[j,0:3]
        cov_j = np.array([[d[j,3],d[j,4],d[j,5]],[d[j,4],d[j,6],d[j,7]],[d[j,5],d[j,7],d[j,8]]])
        f_val = f[j]
        t_val = t[j]
        if f_val == f[i] and t_val == t[i]:
            w.append(base_i - base_j)  
            cov_val = cov_i + cov_j
            cov = block_diag(cov,cov_val)
            lines.append([f_val,t_val,'repeated forward'])
        elif f_val == t[i] and t_val == f[i]:
            w.append(base_i + base_j)
            cov_val = cov_i + cov_j
            cov = block_diag(cov,cov_val)
            lines.append([f_val,t_val,'forward/reverse'])       

cov = np.delete(cov, 0, 0)
w = np.ravel(w)
lines = np.array(lines)

# A priori estimation
apr = (w.T@inv(cov)@w)/w.size
n_cov = apr*cov


# 95% Confidence interval misclosure test:
variance = np.diag(n_cov)
compare = np.less_equal(w,1.96*variance**(1/2)) # Test if values are less than 196σ
compare = np.reshape(compare,[22,3])


val = [];
# Chi squared test
for i in range(0,66,3):
    w_i = np.array([w[i], w[i+1], w[i+2]])
    cov_i = n_cov[i:i+3,i:i+3]
    val.append(w_i.T@inv(cov_i)@w_i)
val = np.ravel(val)
compare1 = np.less_equal(val,7.82)


# Possible Outliers 
compare2 = compare[:,0]*compare[:,1]*compare[:,2]
count = np.arange(22)
pos1 = count[compare1<1]
pos2 = count[compare2<1]
pos = []
pos.append(lines[pos1[:],0:2])
pos.append(lines[pos2[:],0:2])
pos = np.array(pos)
print(pos)
