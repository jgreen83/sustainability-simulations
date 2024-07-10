import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

PROBS = [[1.  , 0.36, 0.44, 0.12, 0.04],
       [0.76, 0.4 , 0.32, 0.2 , 0.12],
       [0.56, 0.28, 0.32, 0.  , 0.  ],
       [0.24, 0.16, 0.08, 0.12, 0.12],
       [0.16, 0.  , 0.04, 0.  , 0.  ]]

trialX = np.linspace(0,2,5)
trialR = np.linspace(0,2,5)
TX, TR = np.meshgrid(trialX, trialR)

def P(x,r):
    return np.exp(-1)*np.exp(-(r+1)*(np.log(r+1)-1)) * np.exp(-x)

errs = []

x = np.linspace(0, 2, 100)
r = np.linspace(0, 2, 200)

X, R = np.meshgrid(x, r)
Z = np.empty((200,100))
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i][j] = P(x[j],r[i])

    
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, R, Z, 50, cmap=cm.hsv)
ax.contour3D(TX,TR,PROBS,50,cmap=cm.viridis)
ax.set_xlabel('x')
ax.set_ylabel('r')
ax.set_zlabel('Pi')
#ax.set_title('L(l,x) 2D case, a = 5, b = 20, r = 5')
plt.show()
    
errorSum = 0
for i in range(len(trialR)):
    for j in range(len(trialX)):
        errorSum += abs(PROBS[i][j] - P(trialX[j],trialR[i]))
print(errorSum)
print(errorSum/(len(trialR)*len(trialX)))
