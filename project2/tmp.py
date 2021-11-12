'''Test file '''

import numpy as np
x = np.arange(-10,10,4)
print(x)
a = 0.5
y1 = ((x > 0) * x)
y2 = ((x <= 0) * x * a)
y = y1 + y2
print(y)

dy1 =((x > 0) * 1)
dy2 = ((x <= 0) * a)
dy = dy1 + dy2
print(dy)
