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
Paddle_VQSD: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import numpy
from paddle import fluid
from paddle_quantum.circuit import UAnsatz
from paddle_quantum.utils import dagger
from paddle.complex import matmul, trace
from paddle_quantum.VQSD.HGenerator import generate_rho_sigma

SEED = 14

__all__ = [
    "U_theta",
    "Net",
    "Paddle_VQSD",
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

    def __init__(self, shape, rho, sigma, param_attr=fluid.initializer.Uniform(low=0.0, high=2 * numpy.pi, seed=SEED),
                 dtype='float64'):
        super(Net, self).__init__()

        # 将 Numpy array 转换成 Paddle 动态图模式中支持的 variable
        self.rho = fluid.dygraph.to_variable(rho)
        self.sigma = fluid.dygraph.to_variable(sigma)

        # 初始化 theta 参数列表，并用 [0, 2*pi] 的均匀分布来填充初始值
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    # 定义损失函数和前向传播机制
    def forward(self, N):
        # 施加量子神经网络
        U = U_theta(self.theta, N)

        # rho_tilde 是将 U 作用在 rho 后得到的量子态 U*rho*U^dagger
        rho_tilde = matmul(matmul(U, self.rho), dagger(U))

        # 计算损失函数
        loss = trace(matmul(self.sigma, rho_tilde))

        return loss.real, rho_tilde


def Paddle_VQSD(rho, sigma, N=2, THETA_SIZE=15, ITR=50, LR=0.1):
    r"""
    Paddle_VQSD
    :param rho: 待对角化的量子态
    :param sigma: 输入用来标记的量子态sigma
    :param N: 量子神经网络的宽度
    :param THETA_SIZE: 量子神经网络中参数的数量
    :param ITR: 设置训练的总的迭代次数
    :param LR: 设置学习速率
    :return: 优化之后量子态rho接近对角态的numpy形式
    """
    # 初始化paddle动态图机制
    with fluid.dygraph.guard():
        # 确定网络的参数维度
        net = Net(shape=[THETA_SIZE], rho=rho, sigma=sigma)

        # 一般来说，我们利用Adam优化器来获得相对好的收敛，当然你可以改成SGD或者是RMS prop.
        opt = fluid.optimizer.AdagradOptimizer(learning_rate=LR, parameter_list=net.parameters())

        # 优化循环
        for itr in range(ITR):

            # 前向传播计算损失函数并返回估计的能谱
            loss, rho_tilde = net(N)
            rho_tilde_np = rho_tilde.numpy()

            # 在动态图机制下，反向传播极小化损失函数
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()

            # 打印训练结果
            if itr % 10 == 0:
                print('iter:', itr, 'loss:', '%.4f' % loss.numpy()[0])
    return rho_tilde_np


def main():

    D = [0.5, 0.3, 0.1, 0.1]

    rho, sigma = generate_rho_sigma()

    rho_tilde_np = Paddle_VQSD(rho, sigma)

    print("The estimated spectrum is:", numpy.real(numpy.diag(rho_tilde_np)))
    print('The target spectrum is:', D)


if __name__ == '__main__':
    main()
