import numpy as np
from layers import DilatedConvFlatten

dc = DilatedConvFlatten((3,3), 2, 1)
X = np.ones((128,28,28))
Y = dc.forward(X)
print(Y.shape)

