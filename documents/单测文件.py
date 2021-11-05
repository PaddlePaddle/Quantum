from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import Hamiltonian,NKron, gate_fidelity,SpinOps,dagger
from paddle_quantum.trotter import construct_trotter_circuit, get_1d_heisenberg_hamiltonian,__group_hamiltonian_xyz,optimal_circuit,__sort_pauli_word
from paddle import matmul, transpose, trace
import paddle
import numpy as np
import scipy
from scipy import linalg
import matplotlib.pyplot as plt

def get_evolve_op(t): return scipy.linalg.expm(-1j * t * h.construct_h_matrix())
n_qubits =2
h = get_1d_heisenberg_hamiltonian(length=n_qubits, j_x=1, j_y=1, j_z=2,h_z=2 * np.random.rand(2) - 1,
periodic_boundary_condition=False)# 

t = 2
r = 1
cir = UAnsatz(n_qubits)
construct_trotter_circuit(cir, h, tau=t/r, steps=r) 
print('系统的哈密顿量为：')
print(h)
print('电路的酉矩阵与正确的演化算符之间的保真度为：%.2f' % gate_fidelity(cir.U.numpy(), get_evolve_op(t)))

optimal_cir = UAnsatz(n_qubits)
construct_trotter_circuit(optimal_cir,h,tau=t,steps=r,grouping='optimal')
print('优化电路的酉矩阵与正确的演化算符之间的保真度为：%.2f' % gate_fidelity(optimal_cir.U.numpy(), get_evolve_op(t)))

print(cir)
print(optimal_cir)

"""
系统的哈密顿量为：
1.0 X0, X1
1.0 Y0, Y1
2.0 Z0, Z1
0.8437864330659737 Z0
0.13446464627645072 Z1
电路的酉矩阵与正确的演化算符之间的保真度为：0.67
优化电路的酉矩阵与正确的演化算符之间的保真度为：0.67
--H----*-----------------*----H----Rx(1.571)----*-----------------*----Rx(-1.57)----*-----------------*----Rz(3.375)--
       |                 |                      |                 |                 |                 |               
--H----x----Rz(4.000)----x----H----Rx(1.571)----x----Rz(4.000)----x----Rx(-1.57)----x----Rz(8.000)----x----Rz(0.538)--
                                                                                                                      
---------------x----Rz(6.429)----*-----------------x----Rz(-1.57)----Rz(3.375)--
               |                 |                 |                            
--Rz(1.571)----*----Ry(-3.85)----x----Ry(3.854)----*----Rz(0.538)---------------
"""
