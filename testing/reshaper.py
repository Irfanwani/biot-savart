import numpy as np

## Preamble
x = np.linspace(0, 3, 3+1)
y = np.linspace(0, 2, 2+1)
z = np.linspace(0, 1, 1+1)

xx, yy, zz, = np.meshgrid(x, y, z, indexing="ij")

positions = np.transpose(np.array([xx, yy, zz]), axes=(1,2,3,0))


coil = np.array([[1, 0, 0], [2, 1, 1], [3, 2, 2], [4, 3, 3], [5, 4, 4]])
current = np.array([1, 2, 3, 4, 5])

## Start Integrating
midpoints = (coil[1:] + coil[:-1])/2
dl = np.diff(coil, axis=0) # dl row vectors for each segment

R_Rprime = midpoints[np.newaxis, np.newaxis, np.newaxis, :] - positions[:, :, :, np.newaxis, :]
mags = np.linalg.norm(R_Rprime, axis=-1)

elementals = current[np.newaxis, np.newaxis, np.newaxis, :-1, np.newaxis] * np.cross(dl, R_Rprime) / mags[:, :, :, :, np.newaxis]**3

B = np.sum(elementals, axis=-2)