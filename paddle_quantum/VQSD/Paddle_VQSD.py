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
Paddle_VQSD: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import numpy

from paddle import fluid
from paddle_quantum.circuit import UAnsatz
from paddle.complex import matmul, trace, transpose

SEED = 1

__all__ = [
    "U_theta",
    "Net",
    "Paddle_VQSD",
]


# definition of U_theta
def U_theta(theta, N):
    """
    U_theta
    """

    cir = UAnsatz(N)
    cir.rz(theta[0], 1)
    cir.ry(theta[1], 1)
    cir.rz(theta[2], 1)

    cir.rz(theta[3], 2)
    cir.ry(theta[4], 2)
    cir.rz(theta[5], 2)

    cir.cnot([2, 1])

    cir.rz(theta[6], 1)
    cir.ry(theta[7], 2)

    cir.cnot([1, 2])

    cir.rz(theta[8], 1)
    cir.ry(theta[9], 1)
    cir.rz(theta[10], 1)

    cir.rz(theta[11], 2)
    cir.ry(theta[12], 2)
    cir.rz(theta[13], 2)

    return cir.state


class Net(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self,
                 shape,
                 rho,
                 sigma,
                 param_attr=fluid.initializer.Uniform(
                     low=0.0, high=2 * numpy.pi, seed=SEED),
                 dtype='float32'):
        super(Net, self).__init__()

        self.rho = fluid.dygraph.to_variable(rho)
        self.sigma = fluid.dygraph.to_variable(sigma)

        self.theta = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    def forward(self, N):
        """
        Args:
        Returns:
            The loss.
        """

        out_state = U_theta(self.theta, N)

        # rho_tilde is what you get after you put self.rho through the circuit
        rho_tilde = matmul(
            matmul(out_state, self.rho),
            transpose(
                fluid.framework.ComplexVariable(out_state.real,
                                                -out_state.imag),
                perm=[1, 0]))

        # record the new loss
        loss = trace(matmul(self.sigma, rho_tilde))

        return loss.real, rho_tilde


def Paddle_VQSD(rho, sigma, N=2, THETA_SIZE=14, ITR=50, LR=0.1):
    """
    Paddle_VQSD
    """

    with fluid.dygraph.guard():
        # net
        net = Net(shape=[THETA_SIZE], rho=rho, sigma=sigma)

        # optimizer
        opt = fluid.optimizer.AdagradOptimizer(
            learning_rate=LR, parameter_list=net.parameters())
        # gradient descent loop
        for itr in range(ITR):
            loss, rho_tilde = net(N)

            rho_tilde_np = rho_tilde.numpy()
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()

            print('iter:', itr, 'loss:', '%.4f' % loss.numpy()[0])

    return rho_tilde_np
