import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Data=pd.read_csv('ex1data1.txt',names=['DataX','DataY'])
#Number training samples
m=len(Data.index)
#Adding X0
Data.insert(0,'X0',np.ones(m))
X=Data.iloc[:,[0,1]]#mX2
y=Data.iloc[:,2]

#1x2
#Theta=np.ones(len(X.columns))
Theta=np.array([-1,2])
#plt.plot(X['DataX'],y,'r+')
iterations=5000
alpha=0.01
J=np.zeros(iterations)
for i in range(iterations):
    Theta=Theta-alpha*np.dot((np.dot(X,Theta)-y),X)/m
    h=(np.dot(X,Theta)-y)
    J[i]=0.5/m*np.dot(h,h)
print(Theta)

plt.plot(J)
