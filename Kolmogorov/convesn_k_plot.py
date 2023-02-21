import numpy as np
import matplotlib.pyplot as plt

k = np.arange(1, 11, dtype=np.float32)

def f(Cres, input_dim, k):
    output = k*k*Cres - 1
    output /= Cres * (input_dim+k-1)**2
    return output

Cres = np.arange(40, 401, 40)

for i in range(Cres.shape[0]):
    plt.plot(k, f(Cres[i], 4, k), 'o', linestyle='--', label='C : {}'.format(Cres[i]))
plt.legend()
plt.grid(True)
plt.show()
