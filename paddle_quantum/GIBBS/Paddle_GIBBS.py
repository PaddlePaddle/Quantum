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

"""
Paddle_GIBBS
"""

import scipy

from numpy import pi as PI

from paddle import fluid
from paddle.complex import matmul, trace
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.state import density_op
from paddle_quantum.utils import state_fidelity, partial_trace
from paddle_quantum.GIBBS.HGenerator import H_generator

SEED = 14  # 固定随机种子

__all__ = [
    "U_theta",
    "Net",
    "Paddle_GIBBS",
]


def U_theta(initial_state, theta, N, D):
    """
    Quantum Neural Network
    """

    # 按照量子比特数量/网络宽度初始化量子神经网络
    cir = UAnsatz(N)

    # 内置的 {R_y + CNOT} 电路模板
    cir.real_entangled_layer(theta[:D], D)

    # 铺上最后一列 R_y 旋转门
    for i in range(N):
        cir.ry(theta=theta[D][i][0], which_qubit=i)

    # 量子神经网络作用在给定的初始态上
    final_state = cir.run_density_matrix(initial_state)

    return final_state


class Net(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self, N, shape, param_attr=fluid.initializer.Uniform(low=0.0, high=2 * PI, seed=SEED),
                 dtype='float64'):
        super(Net, self).__init__()

        # 初始化 theta 参数列表，并用 [0, 2*pi] 的均匀分布来填充初始值
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

        # 初始化 rho = |0..0><0..0| 的密度矩阵
        self.initial_state = fluid.dygraph.to_variable(density_op(N))

    # 定义损失函数和前向传播机制
    def forward(self, H, N, N_SYS_B, beta, D):
        # 施加量子神经网络
        rho_AB = U_theta(self.initial_state, self.theta, N, D)

        # 计算偏迹 partial trace 来获得子系统B所处的量子态 rho_B
        rho_B = partial_trace(rho_AB, 2 ** (N - N_SYS_B), 2 ** (N_SYS_B), 1)

        # 计算三个子损失函数
        rho_B_squre = matmul(rho_B, rho_B)
        loss1 = (trace(matmul(rho_B, H))).real
        loss2 = (trace(rho_B_squre)).real * 2 / beta
        loss3 = - ((trace(matmul(rho_B_squre, rho_B))).real + 3) / (2 * beta)

        # 最终的损失函数
        loss = loss1 + loss2 + loss3

        return loss, rho_B


def Paddle_GIBBS(hamiltonian, rho_G, N=4, N_SYS_B=3, beta=1.5, D=1, ITR=50, LR=0.5):
    r"""
    Paddle_GIBBS
    :param hamiltonian: 哈密顿量
    :param rho_G: 目标吉布斯态 rho
    :param N: 量子神经网络的宽度
    :param N_SYS_B: 用于生成吉布斯态的子系统B的量子比特数
    :param D: 设置量子神经网络中重复计算模块的深度 Depth
    :param ITR: 设置训练的总迭代次数
    :param LR: 设置学习速率
    :return: todo
    """
    # 初始化paddle动态图机制
    with fluid.dygraph.guard():
        # 我们需要将 Numpy array 转换成 Paddle 动态图模式中支持的 variable
        H = fluid.dygraph.to_variable(hamiltonian)

        # 确定网络的参数维度
        net = Net(N, shape=[D + 1, N, 1])

        # 一般来说，我们利用Adam优化器来获得相对好的收敛，当然你可以改成SGD或者是RMS prop.
        opt = fluid.optimizer.AdamOptimizer(learning_rate=LR, parameter_list=net.parameters())

        # 优化循环
        for itr in range(1, ITR + 1):
            # 前向传播计算损失函数并返回生成的量子态 rho_B
            loss, rho_B = net(H, N, N_SYS_B, beta, D)

            # 在动态图机制下，反向传播极小化损失函数
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()

            # 转换成 Numpy array 用以计算量子态的保真度 F(rho_B, rho_G)
            rho_B = rho_B.numpy()
            fid = state_fidelity(rho_B, rho_G)

            # 打印训练结果
            if itr % 5 == 0:
                print('iter:', itr, 'loss:', '%.4f' % loss.numpy(), 'fid:', '%.4f' % fid)
    return rho_B


def main():
    # gibbs Hamiltonian preparing
    hamiltonian, rho_G = H_generator()
    rho_B = Paddle_GIBBS(hamiltonian, rho_G)
    print(rho_B)


if __name__ == '__main__':
    main()
