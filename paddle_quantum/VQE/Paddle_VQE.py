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
VQE: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import os
import platform

import paddle
from numpy import savez

import paddle_quantum
from paddle_quantum.ansatz import Circuit
from paddle_quantum.loss import ExpecVal
from paddle_quantum.VQE.benchmark import benchmark_result
from paddle_quantum.VQE.chemistrysub import H2_generator


__all__ = [
    "Paddle_VQE",
]


def Paddle_VQE(Hamiltonian, N, D=2, ITR=80, LR=0.2):
    r"""Main Learning network using dynamic graph

    Args:
        Hamiltonian: Hamiltonian
        N: Width of QNN
        D: Depth of QNN. Defaults to 2.
        ITR: Number of iterations. Defaults to 80.
        LR: Learning rate. Defaults to 0.2.
    """

    # Determine the dimensions of network
    net = Circuit(N)
    net.real_entangled_layer(depth=D)
    net.ry(qubits_idx='full')

    loss_func = ExpecVal(paddle_quantum.Hamiltonian(Hamiltonian))

    # Usually, we recommend Adam optimizer for better result. If you wish, you could use SGD or RMS prop.
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    # Record optimization results
    summary_iter, summary_loss = [], []

    # Optimization iterations
    for itr in range(1, ITR + 1):

        # Run forward propagation to calculate loss function
        state = net()
        loss = loss_func(state)

        # In dynamic graph, run backward propagation to minimize loss function
        loss.backward()
        opt.minimize(loss)
        opt.clear_grad()

        # Update optimized results
        summary_loss.append(loss.numpy())
        summary_iter.append(itr)

        # Print results
        if itr % 20 == 0:
            print("iter:", itr, "loss:", "%.4f" % loss.numpy())
            print("iter:", itr, "Ground state energy:", "%.4f Ha" % loss.numpy())

    # Save results in the 'output' directory
    os.makedirs("output", exist_ok=True)
    savez("./output/summary_data", iter=summary_iter, energy=summary_loss)


if __name__ == '__main__':
    # Read data from built-in function or xyz file depending on OS
    sysStr = platform.system()

    if sysStr == 'Windows':
        #  Windows does not support SCF, using H2_generator instead
        print('Molecule data will be read from built-in function')
        hamiltonian, N = H2_generator()
        print('Read Process Finished')

    elif sysStr in ('Linux', 'Darwin'):
        # for linux only
        from paddle_quantum.VQE.chemistrygen import read_calc_H
        # Hamiltonian and cnot module preparing, must be executed under Linux
        # Read the H2 molecule data
        print('Molecule data will be read from h2.xyz')
        hamiltonian, N = read_calc_H(geo_fn='h2.xyz')
        print('Read Process Finished')

    else:
        print("Don't support this OS.")

    Paddle_VQE(hamiltonian, N)
    benchmark_result()
