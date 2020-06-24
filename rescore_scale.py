import numpy as np
import matplotlib.pyplot as plt

std = np.array([0.189987493, 0.189136073, 0.190970921, 0.197133962, 0.204727957, 0.22133, 0.240853688, 0.292728071])

x = np.array([600, 540, 480, 420, 360, 300, 240, 180]) #scales
y = np.array([0.924859161, 0.926361658, 0.925119198, 0.920566263, 0.91405962, 0.900039856, 0.878185468, 0.811734458]) # Mean confidence of TP

degree = 3
z = np.polyfit(x, y, degree)
print(z)

plt.errorbar(x, y, fmt='o', yerr=std, label='Original data', markersize=10)
x_ = np.arange(128, 601)

yhat = np.zeros(601-128)
for i in range(degree+1):
    yhat += z[i]*(x_**(degree-i))
print(yhat[-1])
plt.plot(x_, yhat, 'r', label='Fitted line')
plt.axis([0, 650, 0, 1.2])
plt.legend()
plt.show()