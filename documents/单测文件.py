from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import Hamiltonian,NKron, gate_fidelity,SpinOps,dagger
from paddle_quantum.trotter import construct_trotter_circuit, get_1d_heisenberg_hamiltonian
from paddle import matmul, transpose, trace
import paddle
import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt

def get_evolve_op(t,h): return scipy.linalg.expm(-1j * t * h.construct_h_matrix())

def test(h,n):
    t = 2
    r = 1
    cir = UAnsatz(n)
    construct_trotter_circuit(cir, h, tau=t/r, steps=r) 
    print('系统的哈密顿量为：')
    print(h)
    print('电路的酉矩阵与正确的演化算符之间的保真度为：%.2f' % gate_fidelity(cir.U.numpy(), get_evolve_op(t,h)))
    print(cir)


print('--------------test1------------')
h1 = get_1d_heisenberg_hamiltonian(length=2, j_x=1, j_y=1, j_z=2,h_z=2 * np.random.rand(2) - 1,periodic_boundary_condition=False)# 
test(h1,2)
print('--------------test2------------')
h2 = Hamiltonian([[1., 'X0, X1'], [1., 'Z2, Z3'], [1., 'Y0, Y1'], [1., 'X1, X2'], [1., 'Y2, Y3'], [1., 'Z0, Z1']])
test(h2,4)
print('--------------test3------------')
h3 = Hamiltonian([ [1, 'Y0, Y1'], [1, 'X1, X0'],[1, 'X0, Y1'],[1, 'Z0, Z1']])
test(h3,2)

"""

运行耗时: 130毫秒
--------------test1------------
([1.0, 1.0, 2.0, 0.8812020131972615, 0.7453483155128535], ['XX', 'YY', 'ZZ', 'Z', 'Z'], [[0, 1], [0, 1], [0, 1], [0], [1]])
系统的哈密顿量为：
1.0 X0, X1
1.0 Y0, Y1
2.0 Z0, Z1
0.8812020131972615 Z0
0.7453483155128535 Z1
电路的酉矩阵与正确的演化算符之间的保真度为：0.99
---------------x----Rz(6.429)----*-----------------x----Rz(-1.57)----Rz(3.525)--
               |                 |                 |                            
--Rz(1.571)----*----Ry(-3.85)----x----Ry(3.854)----*----Rz(2.981)---------------
                                                                                
--------------test2------------
([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], ['XX', 'YY', 'ZZ', 'ZZ', 'XX', 'YY'], [[0, 1], [0, 1], [0, 1], [2, 3], [1, 2], [2, 3]])
系统的哈密顿量为：
1.0 X0, X1
1.0 Z2, Z3
1.0 Y0, Y1
1.0 X1, X2
1.0 Y2, Y3
1.0 Z0, Z1
电路的酉矩阵与正确的演化算符之间的保真度为：0.51
-------------------x--------Rz(2.429)--------*---------------------x----Rz(-1.57)-------------------------------------------------------------------------------
                   |                         |                     |                                                                                            
--Rz(1.571)--------*--------Ry(-3.85)--------x--------Ry(3.854)----*--------H--------*-----------------*----H---------------------------------------------------
                                                                                     |                 |                                                        
------*-------------------------*------------H---------------------------------------x----Rz(4.000)----x----H----Rx(1.571)----*-----------------*----Rx(-1.57)--
      |                         |                                                                                             |                 |               
------x--------Rz(4.000)--------x--------Rx(1.571)----------------------------------------------------------------------------x----Rz(4.000)----x----Rx(-1.57)--
                                                                                                                                                                
--------------test3------------
([1.0, 1.0, 1.0, 1.0], ['YY', 'XX', 'ZZ', 'XY'], [[0, 1], [1, 0], [0, 1], [0, 1]])
系统的哈密顿量为：
1.0 Y0, Y1
1.0 X1, X0
1.0 X0, Y1
1.0 Z0, Z1
电路的酉矩阵与正确的演化算符之间的保真度为：0.46
---------------x----Rz(2.429)----*-----------------x----Rz(-1.57)----H----*-----------------*--------H------
               |                 |                 |                      |                 |               
--Rz(1.571)----*----Ry(-3.85)----x----Ry(3.854)----*----Rx(1.571)---------x----Rz(4.000)----x----Rx(-1.57)--
"""

from paddle_quantum.utils import partial_trace,plot_state_in_bloch_sphere,partial_trace_discontiguous,NKron,plot_n_qubit_state_in_bloch_sphere
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import paddle
cir1 = UAnsatz(1)
cir2 = UAnsatz(1)
phi, theta, omega = 2 * np.pi * np.random.uniform(size=3)
phi = paddle.to_tensor(phi, dtype='float64')
theta = paddle.to_tensor(theta, dtype='float64')
omega = paddle.to_tensor(omega, dtype='float64')
cir1.rx(phi,0)
cir1.rz(omega,0)
cir2.ry(theta,0)

mat1,mat2 = np.array(cir1.run_density_matrix()),np.array(cir2.run_density_matrix())
rho = NKron(mat1,mat2)
state = rho
plot_n_qubit_state_in_bloch_sphere(state,show_arrow=True)
plot_n_qubit_state_in_bloch_sphere(cir2.run_density_matrix(),show_arrow=True)
plot_n_qubit_state_in_bloch_sphere(cir1.run_state_vector(),show_arrow=True)

