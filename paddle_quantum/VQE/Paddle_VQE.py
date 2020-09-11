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
VQE: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import os
import platform

from numpy import pi as PI
from numpy import savez
from paddle import fluid
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

    # 按照量子比特数量/网络宽度初始化量子神经网络
    cir = UAnsatz(N)

    # 内置的 {R_y + CNOT} 电路模板
    cir.real_entangled_layer(theta[:D], D)

    # 铺上最后一列 R_y 旋转门
    for i in range(N):
        cir.ry(theta=theta[D][i][0], which_qubit=i)

    # 量子神经网络作用在默认的初始态 |0000>上
    cir.run_state_vector()

    # 计算给定哈密顿量的期望值
    expectation_val = cir.expecval(Hamiltonian)

    return expectation_val


class StateNet(fluid.dygraph.Layer):
    """
    Construct the model net
    """

    def __init__(self, shape, param_attr=fluid.initializer.Uniform(low=0.0, high=2 * PI), dtype="float64"):
        super(StateNet, self).__init__()

        # 初始化 theta 参数列表，并用 [0, 2*pi] 的均匀分布来填充初始值
        self.theta = self.create_parameter(shape=shape, attr=param_attr, dtype=dtype, is_bias=False)

    # 定义损失函数和前向传播机制
    def forward(self, Hamiltonian, N, D):
        # 计算损失函数/期望值
        loss = U_theta(self.theta, Hamiltonian, N, D)

        return loss


def Paddle_VQE(Hamiltonian, N, D=2, ITR=80, LR=0.2):
    r"""
    Main Learning network using dynamic graph
    :param Hamiltonian:
    :param N:
    :param D: 设置量子神经网络中重复计算模块的深度 Depth
    :param ITR: 设置训练的总迭代次数
    :param LR: 设置学习速率
    :return: return: Plot or No return
    """

    # 初始化paddle动态图机制
    with fluid.dygraph.guard():
        # 确定网络的参数维度
        net = StateNet(shape=[D + 1, N, 1])

        # 一般来说，我们利用Adam优化器来获得相对好的收敛，当然你可以改成SGD或者是RMS prop.
        opt = fluid.optimizer.AdamOptimizer(learning_rate=LR, parameter_list=net.parameters())

        # 记录优化结果
        summary_iter, summary_loss = [], []

        # 优化循环
        for itr in range(1, ITR + 1):

            # 前向传播计算损失函数
            loss = net(Hamiltonian, N, D)

            # 在动态图机制下，反向传播极小化损失函数
            loss.backward()
            opt.minimize(loss)
            net.clear_gradients()

            # 更新优化结果
            summary_loss.append(loss.numpy())
            summary_iter.append(itr)

            # 打印结果
            if itr % 20 == 0:
                print("iter:", itr, "loss:", "%.4f" % loss.numpy())
                print("iter:", itr, "Ground state energy:", "%.4f Ha" % loss.numpy())

        # 储存训练结果到 output 文件夹
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
