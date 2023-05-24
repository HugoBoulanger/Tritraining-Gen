import numpy as np
import matplotlib.pyplot as plt


x = [[+8.9, +4.4, +3.1, +2.8, +1.7,    +4.8, +1.4, -0.1, +0.6, -0.0,    +6.3, +4.5, +4.2, +3.6, +2.3,    +0.8, +0.2, -0.7, -0.2, -0.8],
     [+7.8, +4.7, +2.7, +2.9, +1.7,    +3.9, +0.8, -0.4, +0.6, +0.1,    +6.3, +5.6, +4.9, +4.3, +2.4,    -0.5, +0.3, -1.4, -0.9, -0.7],
     [+9.0, +4.1, +2.8, +2.6, +1.2,    +6.0, +2.5, +0.7, +0.8, +0.2,   +10.4, +7.4, +7.1, +4.6, +2.6,    +4.5, +2.9, +3.0, +2.0, -0.1],
     [+7.3, +2.9, +1.2, +1.7, +1.3,    +4.3, -0.5, -1.1, -0.2, -1.0,    +5.6, +7.7, +6.0, +4.2, +2.4,    +3.6, +3.6, +3.9, +2.2, +0.3],
     [+9.3, +4.6, +3.4, +2.9, +1.6,    +4.9, +1.4, -0.4, +0.9, +0.1,    +7.7, +6.4, +5.7, +4.4, +2.9,    +1.6, +0.7, -0.1, +0.6, -0.5]]

print(np.corrcoef(x))

plt.scatter(x[4], x[0], label='follow')
plt.scatter(x[4], x[1], label='complete')
plt.scatter(x[4], x[2], label='mention')
plt.scatter(x[4], x[3], label='context')
plt.legend(loc='best')
plt.show()