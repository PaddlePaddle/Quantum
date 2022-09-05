# !/usr/bin/env python3
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

r"""
Paddle_SSVQE: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import numpy

import paddle
from paddle import matmul

import paddle_quantum
from paddle_quantum.ansatz import Circuit
from paddle_quantum.linalg import dagger
from paddle_quantum.SSVQE.HGenerator import H_generator

SEED = 14  # Choose the seed for random generator

__all__ = [
    "loss_func",
    "Paddle_SSVQE",
]


def loss_func(U, H):
    r"""Compute the loss function of SSVQE
    
    Args:
        H: Hamiltonian
        U: unitary of the circuit
    
    Returns: Tutle: inlcuding following elements
        - loss function
        - loss components
    """
    # Calculate loss function
    loss_struct = paddle.real(matmul(matmul(dagger(U), H), U))
    # Use computational basis to calculate each expectation value, which is the same
    # as a diagonal element in U^dagger*H*U
    loss_components = []
    for i in range(len(loss_struct)):
        loss_components.append(loss_struct[i][i])

    # Calculate the weighted loss function
    loss = 0
    for i in range(len(loss_components)):
        weight = 4 - i
        loss += weight * loss_components[i]

    return loss, loss_components


def Paddle_SSVQE(H, N=2, ITR=50, LR=0.3):
    r"""Paddle_SSVQE
    
    Args:
        H: Hamiltonian
        N: Number of qubits/Width of QNN
        ITR: Number of iterations
        LR: Learning rate
    
    Returns: 
        First several smallest eigenvalues of the Hamiltonian
    """

    # We need to convert Numpy array to variable supported in PaddlePaddle
    hamiltonian = H
    net = Circuit(N)
    net.universal_two_qubits([0, 1])

    # Use Adagrad optimizer
    opt = paddle.optimizer.Adagrad(learning_rate=LR, parameters=net.parameters())

    # Optimization iterations
    for itr in range(1, ITR + 1):

        # Run forward propagation to calculate loss function and obtain energy spectrum
        U = net.unitary_matrix()
        loss, loss_components = loss_func(U, hamiltonian)
        # In dynamic graph, run backward propagation to minimize loss function
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()

        # Print results
        if itr % 10 == 0:
            print('iter:', itr, 'loss:', '%.4f' % loss.numpy()[0])
    return loss_components


if __name__ == '__main__':
    paddle.seed(SEED)
    N = 2
    H = H_generator(N)

    loss_components = Paddle_SSVQE(H)

    def output_ordinalvalue(num):
        r"""
        Convert to ordinal value

        Args:
            num (int): input number

        Return:
            (str): output ordinal value
        """
        if num == 1:
            return str(num) + "st"
        elif num == 2:
            return str(num) + "nd"
        elif num == 3:
            return str(num) + "rd"
        else:
            return str(num) + 'th'

    for i in range(len(loss_components)):
        if i == 0:
            print('The estimated ground state energy is: ', loss_components[i].numpy())
            print('The theoretical ground state energy is: ', numpy.linalg.eigh(H)[0][i])
        else:
            print('The estimated {} excited state energy is: {}'.format(
                output_ordinalvalue(i), loss_components[i].numpy())
            )
            print('The theoretical {} excited state energy is: {}'.format(
                output_ordinalvalue(i), numpy.linalg.eigh(H)[0][i])
            )

