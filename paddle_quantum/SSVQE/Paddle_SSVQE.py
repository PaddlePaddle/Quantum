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
Paddle_SSVQE: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import numpy

from paddle.complex import matmul
from paddle import fluid
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import dagger
from paddle_quantum.SSVQE.HGenerator import H_generator

SEED = 14  # Choose the seed for random generator

__all__ = [
    "U_theta",
    "Net",
    "Paddle_SSVQE",
]


def U_theta(theta, N):
    """
    Quantum Neural Network
    """

    # Initialize the quantum neural network by the number of qubits (width of the network)
    cir = UAnsatz(N)

    # Use a built-in QNN template
    cir.universal_2_qubit_gate(theta)

    # Return the Unitary matrix simulated by QNN
    return cir.U


class Net(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self, shape, param_attr=fluid.initializer.Uniform(low=0.0, high=2 * numpy.pi, seed=SEED),
                 dtype='float64'):
        super(Net, self).__init__()

        # Initialize theta by sampling from a uniform distribution [0, 2*pi]
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)
        
    # Define the loss function and forward propagation mechanism
    def forward(self, H, N):
        # Apply QNN onto the initial state
        U = U_theta(self.theta, N)

        # Calculate loss function
        loss_struct = matmul(matmul(dagger(U), H), U).real

        # Use computational basis to calculate each expectation value, which is the same
        # as a diagonal element in U^dagger*H*U
        loss_components = [
            loss_struct[0][0],
            loss_struct[1][1],
            loss_struct[2][2],
            loss_struct[3][3]
        ]

        # Calculate the weighted loss function
        loss = 4 * loss_components[0] + 3 * loss_components[1] + 2 * loss_components[2] + 1 * loss_components[3]

        return loss, loss_components


def Paddle_SSVQE(H, N=2, THETA_SIZE=15, ITR=50, LR=0.3):
    r"""
    Paddle_SSVQE
    :param H: Hamiltonian
    :param N: Number of qubits/Width of QNN
    :param THETA_SIZE: Number of paramaters in QNN
    :param ITR: Number of iterations
    :param LR: Learning rate
    :return: First several smallest eigenvalues of the Hamiltonian
    """
    
    # Initialize PaddlePaddle dynamic graph machanism
    with fluid.dygraph.guard():
        # We need to convert Numpy array to variable supported in PaddlePaddle
        hamiltonian = fluid.dygraph.to_variable(H)

        # Fix the dimensions of network
        net = Net(shape=[THETA_SIZE])
        
        # Use Adagrad optimizer
        opt = fluid.optimizer.AdagradOptimizer(learning_rate=LR, parameter_list=net.parameters())

        # Optimization iterations
        for itr in range(1, ITR + 1):

            # Run forward propagation to calculate loss function and obtain energy spectrum
            loss, loss_components = net(hamiltonian, N)

            # In dynamic graph, run backward propogation to minimize loss function
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()
            
            # Print results
            if itr % 10 == 0:
                print('iter:', itr, 'loss:', '%.4f' % loss.numpy()[0])
    return loss_components


def main():
    N = 2
    H = H_generator(N)

    loss_components = Paddle_SSVQE(H)

    print('The estimated ground state energy is: ', loss_components[0].numpy())
    print('The theoretical ground state energy: ', numpy.linalg.eigh(H)[0][0])

    print('The estimated 1st excited state energy is: ', loss_components[1].numpy())
    print('The theoretical 1st excited state energy: ', numpy.linalg.eigh(H)[0][1])

    print('The estimated 2nd excited state energy is: ', loss_components[2].numpy())
    print('The theoretical 2nd excited state energy: ', numpy.linalg.eigh(H)[0][2])

    print('The estimated 3rd excited state energy is: ', loss_components[3].numpy())
    print('The theoretical 3rd excited state energy: ', numpy.linalg.eigh(H)[0][3])


if __name__ == '__main__':
    main()
