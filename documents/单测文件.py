from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import Hamiltonian,NKron, gate_fidelity,SpinOps,dagger
from paddle_quantum.trotter import construct_trotter_circuit, get_1d_heisenberg_hamiltonian,__group_hamiltonian_xyz,optimal_circuit,__sort_pauli_word
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

h1 = get_1d_heisenberg_hamiltonian(length=2, j_x=1, j_y=1, j_z=2,h_z=2 * np.random.rand(2) - 1,periodic_boundary_condition=False)# 
h2 = Hamiltonian([[1., 'X0, X1'], [1., 'Z2, Z3'], [1., 'Y0, Y1'], [1., 'X1, X2'], [1., 'Y2, Y3'], [1., 'Z0, Z1']])
test(h1,2)
test(h2,4)

"""
系统的哈密顿量为：
1.0 X0, X1
1.0 Y0, Y1
2.0 Z0, Z1
-0.08627686700375259 Z0
-0.7839872913953019 Z1
电路的酉矩阵与正确的演化算符之间的保真度为：0.68
---------------x----Rz(6.429)----*-----------------x----Rz(-1.57)----Rz(-0.34)--
               |                 |                 |                            
--Rz(1.571)----*----Ry(-3.85)----x----Ry(3.854)----*----Rz(-3.13)---------------
                                                                                
系统的哈密顿量为：
1.0 X0, X1
1.0 Z2, Z3
1.0 Y0, Y1
1.0 X1, X2
1.0 Y2, Y3
1.0 Z0, Z1
电路的酉矩阵与正确的演化算符之间的保真度为：0.19
--H--------*-------------------------*--------H----Rx(1.571)----*-----------------*----Rx(-1.57)-----------------------------------------*-------------------------*------------------------
           |                         |                          |                 |                                                      |                         |                        
--H--------x--------Rz(4.000)--------x--------H----Rx(1.571)----x----Rz(4.000)----x----Rx(-1.57)----H----*-----------------*----H--------x--------Rz(4.000)--------x------------------------
                                                                                                         |                 |                                                                
--*---------------------*------------H-------------------------------------------------------------------x----Rz(4.000)----x----H----Rx(1.571)--------*---------------------*----Rx(-1.57)--
  |                     |                                                                                                                             |                     |               
--x----Rz(4.000)--------x--------Rx(1.571)------------------------------------------------------------------------------------------------------------x--------Rz(4.000)----x----Rx(-1.57)--
"""
