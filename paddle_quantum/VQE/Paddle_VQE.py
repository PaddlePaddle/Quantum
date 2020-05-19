# Copyright (c) 2020 Paddle Quantum Authors. All Rights Reserved.
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

from numpy import concatenate
from numpy import pi as PI
from numpy import savez, zeros
from paddle import fluid
from paddle.complex import matmul, transpose

from paddle_quantum.circuit import UAnsatz

__all__ = [
    "U_theta",
    "StateNet",
    "Paddle_VQE",
]


def U_theta(theta, input_state, N, D):
    """
    Circuit
    """

    cir = UAnsatz(N, input_state=input_state)
    for i in range(N):
        cir.rz(theta=theta[0][0][i], which_qubit=i + 1)
        cir.ry(theta=theta[0][1][i], which_qubit=i + 1)
        cir.rz(theta=theta[0][2][i], which_qubit=i + 1)

    for repeat in range(D):
        for i in range(1, N):
            cir.cnot(control=[i, i + 1])

        for i in range(N):
            cir.ry(theta=theta[repeat][0][i], which_qubit=i + 1)
            cir.ry(theta=theta[repeat][1][i], which_qubit=i + 1)
            cir.rz(theta=theta[repeat][2][i], which_qubit=i + 1)

    return cir.state


class StateNet(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(
            self,
            shape,
            param_attr=fluid.initializer.Uniform(
                low=0.0, high=2 * PI),
            dtype="float32", ):
        super(StateNet, self).__init__()
        self.theta = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    def forward(self, input_state, H, N, D):
        """
        :param input_state: The initial state with default |0..>, 'mat'
        :param H: The target Hamiltonian, 'mat'
        :return: The loss, 'float'
        """

        out_state = U_theta(self.theta, input_state, N, D)
        loss = matmul(
            matmul(out_state, H),
            transpose(
                fluid.framework.ComplexVariable(out_state.real,
                                                -out_state.imag),
                perm=[1, 0], ), )

        return loss.real


def Paddle_VQE(Hamiltonian, N, D=1, ITR=120, LR=0.15):
    """
        Main Learning network using dynamic graph
        :return: Plot or No return
    """
    with fluid.dygraph.guard():
        # initial state preparing
        _initial_state_np = concatenate(
            ([[1.0]], zeros([1, 2**N - 1])), axis=1).astype("complex64")
        initial_state = fluid.dygraph.to_variable(_initial_state_np)

        # Store H
        H = fluid.dygraph.to_variable(Hamiltonian)

        # net
        net = StateNet(shape=[D + 1, 3, N])

        # optimizer
        opt = fluid.optimizer.AdamOptimizer(
            learning_rate=LR, parameter_list=net.parameters())

        # gradient descent loop
        summary_iter, summary_loss = [], []
        for itr in range(1, ITR + 1):
            # forward calc, loss
            loss = net(initial_state, H, N, D)

            # backward calculation for gradient value
            loss.backward()
            # using gradients to update the variable theta
            opt.minimize(loss)
            # clear gradients
            net.clear_gradients()

            summary_loss.append(loss[0][0].numpy())
            summary_iter.append(itr)

            print("iter:", itr, "loss:", "%.4f" % loss.numpy())
            print("iter:", itr, "Ground state energy:",
                  "%.4f Ha" % loss.numpy())
            # print('theta:', net.parameters()[0].numpy())

        os.makedirs("output", exist_ok=True)
        savez("./output/summary_data", iter=summary_iter, energy=summary_loss)


def main():
    """
    :return:
    """


if __name__ == "__main__":
    main()
