from paddle_quantum.state import density_op_random
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import partial_trace,plot_state_in_bloch_sphere,partial_trace_discontiguous,NKron,plot_n_qubit_state_in_bloch_sphere
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
state = [cir1.run_state_vector(),cir2.run_state_vector(),rho]
plot_n_qubit_state_in_bloch_sphere(state,show_arrow=True)


n = 2
rho = density_op_random(n)
#print(rho)
plot_n_qubit_state_in_bloch_sphere(rho,show_arrow=True)
plot_n_qubit_state_in_bloch_sphere(rho,show_qubits=[[0]],show_arrow=True)
