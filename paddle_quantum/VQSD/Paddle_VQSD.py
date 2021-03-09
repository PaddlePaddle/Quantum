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
Paddle_VQSD: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import numpy
import paddle
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import dagger
from paddle import matmul, trace
from paddle_quantum.VQSD.HGenerator import generate_rho_sigma

SEED = 14

__all__ = [
    "U_theta",
    "Net",
    "Paddle_VQSD",
]


def U_theta(theta, N):
    """
    Quantum Neural Network
    """
    
    # Initialize the quantum neural network by the number of qubits (width of the network)
    cir = UAnsatz(N)

    # Use built-in template
    cir.universal_2_qubit_gate(theta, [0, 1])

    # Return the Unitary matrix simulated by QNN
    return cir.U


class Net(paddle.nn.Layer):
    """
    Construct the model net
    """

    def __init__(self, shape, rho, sigma, param_attr=paddle.nn.initializer.Uniform(low=0.0, high=2 * numpy.pi),
                 dtype='float64'):
        super(Net, self).__init__()
        # Convert Numpy array to variable supported in PaddlePaddle
        self.rho = paddle.to_tensor(rho)
        self.sigma = paddle.to_tensor(sigma)

        # Initialize theta by sampling from a uniform distribution [0, 2*pi]
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    # Define the loss function and forward propagation mechanism
    def forward(self, N):
        # Apply quantum neural network onto the initial state
        U = U_theta(self.theta, N)

        # rho_tilda is the quantum state obtained by acting U on rho, which is U*rho*U^dagger
        rho_tilde = matmul(matmul(U, self.rho), dagger(U))
        
        # Calculate loss function
        loss = trace(matmul(self.sigma, rho_tilde))

        return paddle.real(loss), rho_tilde


def Paddle_VQSD(rho, sigma, N=2, THETA_SIZE=15, ITR=50, LR=0.2):
    r"""
    Paddle_VQSD
    :param rho: Qauntum state to be diagonalized
    :param sigma: Quantum state sigma
    :param N: Width of QNN
    :param THETA_SIZE: Number of parameters in QNN
    :param ITR: Number of iterations
    :param LR: Learning rate
    :return: Diagonalized quantum state after optimization 
    """

    # Fix the dimensions of network
    net = Net(shape=[THETA_SIZE], rho=rho, sigma=sigma)

    # Use Adagrad optimizer
    opt = paddle.optimizer.Adagrad(learning_rate=LR, parameters=net.parameters())

    # Optimization iterations
    for itr in range(ITR):

        # Run forward propagation to calculate loss function and obtain energy spectrum
        loss, rho_tilde = net(N)
        rho_tilde_np = rho_tilde.numpy()

        # In dynamic graph, run backward propagation to minimize loss function
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()

        # Print results
        if itr % 10 == 0:
            print('iter:', itr, 'loss:', '%.4f' % loss.numpy()[0])

    return rho_tilde_np


def main():

    D = [0.5, 0.3, 0.1, 0.1]

    rho, sigma = generate_rho_sigma()

    rho_tilde_np = Paddle_VQSD(rho, sigma)

    print("The estimated spectrum is:", numpy.real(numpy.diag(rho_tilde_np)))
    print('The target spectrum is:', D)


if __name__ == '__main__':
    main()
