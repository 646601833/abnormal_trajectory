import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#三维曲面显示
X=np.linspace(-2,2,500)
Y=np.linspace(-2,2,500)
XX, YY = np.meshgrid(X, Y)
Z=YY*np.sin(2*math.pi*XX)+XX*np.cos(2*math.pi*YY)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(XX, YY, Z,rstride=1, cstride=1, cmap='rainbow')
plt.show()

