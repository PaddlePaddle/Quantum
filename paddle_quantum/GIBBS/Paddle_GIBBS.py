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
Paddle_GIBBS
"""

from numpy import concatenate, zeros
from numpy import pi as PI

from paddle import fluid
from paddle.complex import matmul, transpose, trace
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import compute_fid, partial_trace

from paddle_quantum.GIBBS.HGenerator import H_generator

SEED = 1

__all__ = [
    "U_theta",
    "Net",
    "Paddle_GIBBS",
]


def U_theta(theta, input_state, N, D):  # definition of U_theta
    """
    :param theta:
    :param input_state:
    :return:
    """

    cir = UAnsatz(N, input_state=input_state)
    for i in range(N):
        cir.rx(theta=theta[0][0][i], which_qubit=i + 1)
        cir.ry(theta=theta[0][1][i], which_qubit=i + 1)
        cir.rx(theta=theta[0][2][i], which_qubit=i + 1)

    for repeat in range(D):
        for i in range(1, N):
            cir.cnot(control=[i, i + 1])

        for i in range(N):
            cir.ry(theta=theta[repeat][0][i], which_qubit=i + 1)
            # cir.ry(theta=theta[repeat][1][i], which_qubit=i + 1)
            # cir.ry(theta=theta[repeat][2][i], which_qubit=i + 1)

    return cir.state


class Net(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self,
                 shape,
                 param_attr=fluid.initializer.Uniform(
                     low=0.0, high=PI, seed=SEED),
                 dtype='float32'):
        super(Net, self).__init__()

        self.theta = self.create_parameter(
            shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    def forward(self, input_state, H, N, N_SYS_B, D):
        """
        Args:
            input_state: The initial state with default |0..>
            H: The target Hamiltonian
        Returns:
            The loss.
        """

        out_state = U_theta(self.theta, input_state, N, D)

        # rho_AB = utils.matmul(utils.matrix_conjugate_transpose(out_state), out_state)
        rho_AB = matmul(
            transpose(
                fluid.framework.ComplexVariable(out_state.real,
                                                -out_state.imag),
                perm=[1, 0]),
            out_state)

        # compute the partial trace and three losses
        rho_B = partial_trace(rho_AB, 2**(N - N_SYS_B), 2**(N_SYS_B), 1)
        rho_B_squre = matmul(rho_B, rho_B)
        loss1 = (trace(matmul(rho_B, H))).real
        loss2 = (trace(rho_B_squre)).real * 2
        loss3 = -(trace(matmul(rho_B_squre, rho_B))).real / 2

        loss = loss1 + loss2 + loss3  # 损失函数

        # option: if you want to check whether the imaginary part is 0, uncomment the following
        # print('loss_iminary_part: ', loss.numpy()[1])
        return loss - 3 / 2, rho_B


def Paddle_GIBBS(hamiltonian, rho=None, N=5, N_SYS_B=3, D=1, ITR=100, LR=0.5):
    """
    Paddle_GIBBS
    """

    with fluid.dygraph.guard():
        # initial state preparing
        _initial_state_np = concatenate(
            ([[1.]], zeros([1, 2**N - 1])), axis=1).astype('complex64')
        initial_state = fluid.dygraph.to_variable(_initial_state_np)

        # gibbs Hamiltonian preparing
        H = fluid.dygraph.to_variable(hamiltonian)

        # net
        net = Net(shape=[D + 1, 3, N])

        # optimizer
        opt = fluid.optimizer.AdamOptimizer(
            learning_rate=LR, parameter_list=net.parameters())

        # gradient descent loop
        for itr in range(1, ITR + 1):
            loss, rho_B = net(initial_state, H, N, N_SYS_B, D)

            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()

            rho_B = rho_B.numpy()

            if rho is not None:
                fid = compute_fid(rho_B, rho)
                print('iter:', itr, 'loss:', '%.4f' % loss.numpy(), 'fid:',
                      '%.4f' % fid)

    return rho_B


def main():
    """
    main
    """

    # gibbs Hamiltonian preparing
    hamiltonian, rho = H_generator()
    rho_B = Paddle_GIBBS(hamiltonian, rho)
    print(rho_B)


if __name__ == '__main__':
    main()
