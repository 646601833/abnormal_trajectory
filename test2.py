from sklearn.neighbors import NearestNeighbors
import numpy as np


neigh = NearestNeighbors(radius=1.6)
samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
neigh.fit(samples)
rng = neigh.radius_neighbors([[1., 1., 1.]], return_distance=False)
print(rng)