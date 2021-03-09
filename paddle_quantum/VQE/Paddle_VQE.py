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
VQE: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import os
import platform

import paddle
from numpy import pi as PI
from numpy import savez
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.VQE.benchmark import benchmark_result
from paddle_quantum.VQE.chemistrysub import H2_generator


__all__ = [
    "U_theta",
    "StateNet",
    "Paddle_VQE",
]


def U_theta(theta, Hamiltonian, N, D):
    """
    Quantum Neural Network
    """
    # Initialize the quantum neural network by the number of qubits (width of the network)
    cir = UAnsatz(N)

    # Use built-in template (R_y + CNOT)
    cir.real_entangled_layer(theta[:D], D)

    # Add a layer of R_y rotation gates
    for i in range(N):
        cir.ry(theta=theta[D][i][0], which_qubit=i)

    # Act QNN on the default initial state |0000>
    cir.run_state_vector()

    # Calculate the expectation value of the given Hamiltonian
    expectation_val = cir.expecval(Hamiltonian)

    return expectation_val


class StateNet(paddle.nn.Layer):
    """
    Construct the model net
    """

    def __init__(self, shape, param_attr=paddle.nn.initializer.Uniform(low=0.0, high=2 * PI), dtype="float64"):
        super(StateNet, self).__init__()

        # Initialize theta by sampling from a uniform distribution [0, 2*pi]
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    # Define the loss function and forward propagation mechanism
    def forward(self, Hamiltonian, N, D):
        # Calculate loss function (expectation value)
        loss = U_theta(self.theta, Hamiltonian, N, D)

        return loss


def Paddle_VQE(Hamiltonian, N, D=2, ITR=80, LR=0.2):
    r"""
    Main Learning network using dynamic graph
    :param Hamiltonian: Hamiltonian
    :param N: Width of QNN
    :param D: Depth of QNN
    :param ITR: Number of iterations
    :param LR: Learning rate
    :return: No return
    """

    # Determine the dimensions of network
    net = StateNet(shape=[D + 1, N, 1])

    # Usually, we recommend Adam optimizer for better result. If you wish, you could use SGD or RMS prop.
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    # Record optimization results
    summary_iter, summary_loss = [], []

    # Optimization iterations
    for itr in range(1, ITR + 1):

        # Run forward propagation to calculate loss function
        loss = net(Hamiltonian, N, D)

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


def main():
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


if __name__ == '__main__':
    main()
