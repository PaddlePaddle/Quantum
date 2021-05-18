# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Paddle_GIBBS
"""

from numpy import pi as PI
import paddle
from paddle import matmul, trace
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.state import density_op
from paddle_quantum.utils import state_fidelity, partial_trace
from paddle_quantum.GIBBS.HGenerator import H_generator

SEED = 14  # Choose the seed for random generator

__all__ = [
    "U_theta",
    "Net",
    "Paddle_GIBBS",
]


def U_theta(initial_state, theta, N, D):
    """
    Quantum Neural Network
    """

    # Initialize the quantum neural network by the number of qubits (width of the network)
    cir = UAnsatz(N)

    # Use in-built template (R_y + CNOT)
    cir.real_entangled_layer(theta[:D], D)

    # Add a layer of R_y rotation gate
    for i in range(N):
        cir.ry(theta=theta[D][i][0], which_qubit=i)

    # Act quantum neural network on initialized state
    final_state = cir.run_density_matrix(initial_state)

    return final_state


class Net(paddle.nn.Layer):
    """
    Construct the model net
    """

    def __init__(self, N, shape, param_attr=paddle.nn.initializer.Uniform(low=0.0, high=2 * PI),
                 dtype='float64'):
        super(Net, self).__init__()
        # Initialize theta by sampling from a uniform distribution [0, 2*pi]
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        # Set the initial state as rho = |0..0><0..0|
        self.initial_state = paddle.to_tensor(density_op(N))

    # Define the loss function and forward propagation mechanism
    def forward(self, H, N, N_SYS_B, beta, D):
        # Apply quantum neural network onto the initial state
        rho_AB = U_theta(self.initial_state, self.theta, N, D)
        
        # Calculate the partial trace to get the state rho_B of subsystem B
        rho_B = partial_trace(rho_AB, 2 ** (N - N_SYS_B), 2 ** N_SYS_B, 1)

        # Calculate the three components of the loss function
        rho_B_squre = matmul(rho_B, rho_B)
        loss1 = paddle.real(trace(matmul(rho_B, H)))
        loss2 = paddle.real(trace(rho_B_squre)) * 2 / beta
        loss3 = - (paddle.real(trace(matmul(rho_B_squre, rho_B))) + 3) / (2 * beta)
        
        # Get the final loss function
        loss = loss1 + loss2 + loss3

        return loss, rho_B


def Paddle_GIBBS(hamiltonian, rho_G, N=4, N_SYS_B=3, beta=1.5, D=1, ITR=50, LR=0.2):
    r"""
    Paddle_GIBBS
    :param hamiltonian: Hamiltonian
    :param rho_G: Target Gibbs state rho
    :param N: Width of QNN
    :param N_SYS_B: Number of qubits in subsystem B used to generate Gibbs state
    :param D: Depth of QNN
    :param ITR: Number of iterations
    :param LR: Learning rate
    :return: State prepared by optimized QNN
    """
    # We need to convert Numpy array to variable supported in PaddlePaddle
    H = paddle.to_tensor(hamiltonian)

    # Fix the dimensions of network
    net = Net(N, shape=[D + 1, N, 1])

    # Usually, we recommend Adam optimizer for better results. If you wish, you could use SGD or RMS prop.
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    # Optimization iterations
    for itr in range(1, ITR + 1):
        # Run forward propagation to calculate loss function and obtain state rho_B
        loss, rho_B = net(H, N, N_SYS_B, beta, D)

        # In dynamic graph, run backward propagation to minimize loss function
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        # Convert variable to Numpy array to calculate fidelity F(rho_B, rho_G)
        rho_B = rho_B.numpy()
        fid = state_fidelity(rho_B, rho_G)
        # Print results
        if itr % 5 == 0:
            print('iter:', itr, 'loss:', '%.4f' % loss.numpy(), 'fid:', '%.4f' % fid)
    return rho_B


def main():
    paddle.seed(SEED)
    # Generate gibbs Hamiltonian
    hamiltonian, rho_G = H_generator()
    rho_B = Paddle_GIBBS(hamiltonian, rho_G)
    print(rho_B)


if __name__ == '__main__':
    main()
