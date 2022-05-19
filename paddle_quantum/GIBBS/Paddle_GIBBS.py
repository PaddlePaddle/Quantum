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
Paddle_GIBBS
"""

from numpy import pi as PI
import paddle
from paddle import matmul, trace
import paddle_quantum
from paddle_quantum.ansatz import Circuit
from paddle_quantum.state import zero_state
from paddle_quantum.qinfo import state_fidelity, partial_trace
from paddle_quantum.GIBBS.HGenerator import H_generator

SEED = 14  # Choose the seed for random generator

__all__ = [
    "loss_func",
    "Paddle_GIBBS",
]


def loss_func(rho_AB, H, N, N_SYS_B, beta):
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
    paddle_quantum.set_backend(paddle_quantum.Backend.DensityMatrix)
    # We need to convert Numpy array to variable supported in PaddlePaddle
    H = paddle.to_tensor(hamiltonian, dtype = paddle_quantum.get_dtype())

    # Fix the dimensions of network
    net = Circuit(N)
    net.real_entangled_layer(depth=D)
    net.ry('full')
    # Usually, we recommend Adam optimizer for better results. If you wish, you could use SGD or RMS prop.
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    # Optimization iterations
    for itr in range(1, ITR + 1):
        # Run forward propagation to calculate loss function and obtain state rho_B
        rho_AB = net()
        loss, rho_B = loss_func(rho_AB, H, N, N_SYS_B, beta)
        # In dynamic graph, run backward propagation to minimize loss function
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()
        # Convert variable to Numpy array to calculate fidelity F(rho_B, rho_G)
        fid = state_fidelity(paddle_quantum.state.to_state(rho_B), paddle_quantum.state.to_state(rho_G))
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
