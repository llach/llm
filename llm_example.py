from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# add 3D plot subfigure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# generate N random samples
n = 1000

xs = np.random.rand(n)     
ys = np.random.rand(n)
zs = np.random.rand(n)

# init network

# plot data
ax.scatter(xs, ys, zs)

# make plot nicer
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
