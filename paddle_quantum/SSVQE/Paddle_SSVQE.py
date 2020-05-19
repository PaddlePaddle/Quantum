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
Paddle_SSVQE: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""
import numpy
from paddle.complex import matmul, transpose
from paddle import fluid
from paddle_quantum.circuit import UAnsatz

SEED = 1

__all__ = [
    "U_theta",
    "Net",
    "Paddle_SSVQE",
]


# definition of U_theta
def U_theta(theta, N):
    """
    U_theta
    """

    cir = UAnsatz(N)
    # ============== D1=2 ==============
    cir.ry(theta[0], 2)
    cir.rz(theta[1], 2)
    cir.cnot([2, 1])
    cir.ry(theta[2], 2)
    cir.rz(theta[3], 2)
    cir.cnot([2, 1])

    # ============== D2=2 ==============
    cir.ry(theta[4], 1)
    cir.ry(theta[5], 2)
    cir.rz(theta[6], 1)
    cir.rz(theta[7], 2)
    cir.cnot([1, 2])

    cir.ry(theta[8], 1)
    cir.ry(theta[9], 2)
    cir.rz(theta[10], 1)
    cir.rz(theta[11], 2)
    cir.cnot([1, 2])

    return cir.state


class Net(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self,
                 shape,
                 param_attr=fluid.initializer.Uniform(
                     low=0.0, high=2 * numpy.pi, seed=SEED),
                 dtype='float32'):
        super(Net, self).__init__()

        self.theta = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    def forward(self, H, N):
        """
        Args:
            input_state: The initial state with default |0..>
            H: The target Hamiltonian
        Returns:
            The loss.
        """
        out_state = U_theta(self.theta, N)

        loss_struct = matmul(
            matmul(
                transpose(
                    fluid.framework.ComplexVariable(out_state.real,
                                                    -out_state.imag),
                    perm=[1, 0]),
                H),
            out_state).real

        loss_components = [
            loss_struct[0][0], loss_struct[1][1], loss_struct[2][2],
            loss_struct[3][3]
        ]

        loss = 4 * loss_components[0] + 3 * loss_components[
            1] + 2 * loss_components[2] + 1 * loss_components[3]
        return loss, loss_components


def Paddle_SSVQE(H, N=2, THETA_SIZE=12, ITR=60, LR=0.2):
    """
       main
       """
    with fluid.dygraph.guard():
        # Harmiltonian preparing
        H = fluid.dygraph.to_variable(H)

        # net
        net = Net(shape=[THETA_SIZE])

        # optimizer
        opt = fluid.optimizer.AdagradOptimizer(
            learning_rate=LR, parameter_list=net.parameters())

        # gradient descent loop
        for itr in range(1, ITR + 1):
            loss, loss_components = net(H, N)

            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()

            print('iter:', itr, 'loss:', '%.4f' % loss.numpy()[0])

        return loss_components
