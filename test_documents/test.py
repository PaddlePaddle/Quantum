from paddle_quantum.utils import img_to_density_matrix
import paddle
import matplotlib.image
import numpy as np


img_file = '/home/aistudio/f1.jpeg'
rho = (img_to_density_matrix(img_file))

#半正定
w,_=np.linalg.eig(rho)
print(all(w>=0))
#迹为1
print(np.trace(rho))
#shape为[2**n,2**n]
print(rho.shape)
