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
Paddle_SSVQE: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import numpy

from paddle.complex import matmul
from paddle import fluid
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import hermitian
from paddle_quantum.SSVQE.HGenerator import H_generator

SEED = 14  # 固定随机种子

__all__ = [
    "U_theta",
    "Net",
    "Paddle_SSVQE",
]


def U_theta(theta, N):
    """
    Quantum Neural Network
    """

    # 按照量子比特数量/网络宽度初始化量子神经网络
    cir = UAnsatz(N)

    # 调用内置的量子神经网络模板
    cir.universal_2_qubit_gate(theta)

    # 返回量子神经网络所模拟的酉矩阵 U
    return cir.U


class Net(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self, shape, param_attr=fluid.initializer.Uniform(low=0.0, high=2 * numpy.pi, seed=SEED),
                 dtype='float64'):
        super(Net, self).__init__()

        # 初始化 theta 参数列表，并用 [0, 2*pi] 的均匀分布来填充初始值
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    # 定义损失函数和前向传播机制
    def forward(self, H, N):
        # 施加量子神经网络
        U = U_theta(self.theta, N)

        # 计算损失函数
        loss_struct = matmul(matmul(hermitian(U), H), U).real

        # 输入计算基去计算每个子期望值，相当于取 U^dagger*H*U 的对角元
        loss_components = [
            loss_struct[0][0],
            loss_struct[1][1],
            loss_struct[2][2],
            loss_struct[3][3]
        ]

        # 最终加权求和后的损失函数
        loss = 4 * loss_components[0] + 3 * loss_components[1] + 2 * loss_components[2] + 1 * loss_components[3]

        return loss, loss_components


def Paddle_SSVQE(H, N=2, THETA_SIZE=15, ITR=50, LR=0.3):
    r"""
    Paddle_SSVQE
    :param H: 哈密顿量
    :param N: 量子比特数/量子神经网络的宽度
    :param THETA_SIZE: 量子神经网络中参数的数量
    :param ITR: 设置训练的总迭代次数
    :param LR: 设置学习速率
    :return: 哈密顿量的前几个最小特征值
    """
    # 初始化paddle动态图机制
    with fluid.dygraph.guard():
        # 我们需要将 Numpy array 转换成 Paddle 动态图模式中支持的 variable
        hamiltonian = fluid.dygraph.to_variable(H)

        # 确定网络的参数维度
        net = Net(shape=[THETA_SIZE])

        # 一般来说，我们利用Adam优化器来获得相对好的收敛，当然你可以改成SGD或者是RMS prop.
        opt = fluid.optimizer.AdagradOptimizer(learning_rate=LR, parameter_list=net.parameters())

        # 优化循环
        for itr in range(1, ITR + 1):

            # 前向传播计算损失函数并返回估计的能谱
            loss, loss_components = net(hamiltonian, N)

            # 在动态图机制下，反向传播极小化损失函数
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()

            # 打印训练结果
            if itr % 10 == 0:
                print('iter:', itr, 'loss:', '%.4f' % loss.numpy()[0])
    return loss_components


def main():
    N = 2
    H = H_generator(N)

    loss_components = Paddle_SSVQE(H)

    print('The estimated ground state energy is: ', loss_components[0].numpy())
    print('The theoretical ground state energy: ', numpy.linalg.eigh(H)[0][0])

    print('The estimated 1st excited state energy is: ', loss_components[1].numpy())
    print('The theoretical 1st excited state energy: ', numpy.linalg.eigh(H)[0][1])

    print('The estimated 2nd excited state energy is: ', loss_components[2].numpy())
    print('The theoretical 2nd excited state energy: ', numpy.linalg.eigh(H)[0][2])

    print('The estimated 3rd excited state energy is: ', loss_components[3].numpy())
    print('The theoretical 3rd excited state energy: ', numpy.linalg.eigh(H)[0][3])


if __name__ == '__main__':
    main()
