import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

trialX1 = np.linspace(0,2,5)
trialR1 = np.linspace(0,2,5)
trialX2 = np.linspace(0,2,5)
trialR2 = np.linspace(0.01,2,5)

X1, R1 = np.meshgrid(trialX1, trialR1)
X2, R2 = np.meshgrid(trialX2, trialR2)

PROBS1 = [[1.  , 0.36, 0.28, 0.16, 0.04],
       [0.76, 0.32, 0.4 , 0.12, 0.04],
       [0.4 , 0.36, 0.36, 0.2 , 0.04],
       [0.4 , 0.32, 0.12, 0.12, 0.08],
       [0.28, 0.12, 0.08, 0.04, 0.  ]]
PROBS2 = [[1.  , 0.48, 0.4 , 0.28, 0.04],
       [0.96, 0.6 , 0.28, 0.16, 0.08],
       [0.72, 0.36, 0.12, 0.12, 0.08],
       [0.32, 0.16, 0.08, 0.04, 0.  ],
       [0.24, 0.16, 0.04, 0.08, 0.08]]

errs = []

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X1, R1, PROBS1, 50, cmap=cm.hsv)
ax.contour3D(X2,R2,PROBS2,50,cmap=cm.viridis)
ax.set_xlabel('x')
ax.set_ylabel('r')
ax.set_zlabel('Pi')
#ax.set_title('L(l,x) 2D case, a = 5, b = 20, r = 5')
plt.show()
