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
梯度分析工具模块
"""


import numpy as np
import paddle
from paddle import reshape
from random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt

__all__ = [
    "StateNet",
    "show_gradient",
    "random_sample",
    "random_sample_supervised",
    "plot_loss_grad",
    "plot_supervised_loss_grad",
    "plot_distribution"
]


class StateNet(paddle.nn.Layer):
    r"""定义用于量子机器学习的量子神经网络模型

    用户可以通过实例化该类定义自己的量子神经网络模型。
    """

    def __init__(self, shape, dtype='float64'):
        r"""构造函数，用于实例化一个 ``StateNet`` 对象

        Args:
            shape (paddle.Tensor): 表示传入的量子电路中的需要被优化的参数个数
        """
        super(StateNet, self).__init__()
        self.theta = self.create_parameter(shape=shape,
                                           default_initializer=paddle.nn.initializer.Uniform(low=0.0, high=2*np.pi),
                                           dtype=dtype, is_bias=False)

    def forward(self, circuit, loss_func, *args):
        r"""用于更新电路参数并计算该量子神经网络的损失值。

        Args:
            circuit (UAnsatz): 表示传入的参数化量子电路，即要训练的量子神经网络
            loss_func (function): 表示计算该量子神经网络损失值的函数
            *args (list): 表示用于损失函数计算的额外参数列表

        Note:
            这里的 ``loss_func`` 是一个用户自定义的计算损失值的函数，参数为电路和一个可变参数列表。

        Returns:
            tuple: 包含如下两个元素:
                - loss (paddle.Tensor): 表示该量子神经网络损失值
                - circuit (UAnsatz): 更新参数后的量子电路
        """
        circuit.update_param(self.theta)
        circuit.run_state_vector()
        loss = loss_func(circuit, *args)
        return loss, circuit


def show_gradient(circuit, loss_func, ITR, LR, *args):
    r"""计算量子神经网络中各可变参数的梯度值和损失函数值

    Args:
        circuit (UAnsatz): 表示传入的参数化量子电路，即要训练的量子神经网络
        loss_func (function): 表示计算该量子神经网络损失值的函数
        ITR (int): 表示训练的次数
        LR (float): 表示学习训练的速率
        *args (list): 表示用于损失函数计算的额外参数列表

    Returns:
        tuple: 包含如下两个元素:
            - loss_list (list): 表示损失函数值随训练次数变化的列表
            - grad_list(list): 表示各参数梯度随训练次变化的列表
    """

    grad_list = []
    loss_list = []
    shape = paddle.shape(circuit.get_param())
    net = StateNet(shape=shape)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    pbar = tqdm(
        desc="Training: ", total=ITR, ncols=100, ascii=True
    )

    for itr in range(ITR):
        pbar.update(1)
        loss, cir = net(circuit, loss_func, *args)
        loss.backward()
        grad = net.theta.grad.numpy()
        grad_list.append(grad)
        loss_list.append(loss.numpy()[0])
        opt.minimize(loss)
        opt.clear_grad()
    pbar.close()

    return loss_list, grad_list


def plot_distribution(grad):
    r"""根据输入的梯度的列表，画出梯度的分布图

    Args:
        grad (np.array): 表示量子神经网络某参数的梯度列表
    """

    grad = np.abs(grad)
    grad_list = [0, 0, 0, 0, 0]
    x = ['<0.0001', ' (0.0001,0.001)', '(0.001,0.01)', '(0.01,0.1)', '>0.1']
    for g in grad:
        if g > 0.1:
            grad_list[4] += 1
        elif g > 0.01:
            grad_list[3] += 1
        elif g > 0.001:
            grad_list[2] += 1
        elif g > 0.0001:
            grad_list[1] += 1
        else:
            grad_list[0] += 1
    grad_list = np.array(grad_list) / len(grad)

    plt.figure()
    plt.bar(x, grad_list, width=0.5)
    plt.title('The gradient distribution of variables')
    plt.ylabel('ratio')
    plt.show()


def random_sample(circuit, loss_func, sample_num, *args, mode='single', if_plot=True, param=0):
    r"""表示对模型进行随机采样，根据不同的计算模式，获得对应的平均值和方差

    Args:
        circuit (UAnsatz): 表示传入的参数化量子电路，即要训练的量子神经网络
        loss_func (function): 表示计算该量子神经网络损失值的函数
        sample_num (int): 表示随机采样的次数
        mode (string): 表示随机采样后的计算模式，默认为 'single'
        if_plot(boolean): 表示是否对梯度进行画图表示
        param (int): 表示 ``Single`` 模式中对第几个参数进行画图，默认为第一个参数
        *args (list): 表示用于损失函数计算的额外参数列表

    Note:
        在本函数中提供了三种计算模式，``mode`` 分别可以选择 ``'single'``, ``'max'``, 以及 ``'random'``
            - mode='single': 表示计算电路中的每个可变参数梯度的平均值和方差
            - mode='max': 表示对电路中每轮采样的所有参数梯度的最大值求平均值和方差
            - mode='random': 表示对电路中每轮采样的所有参数随机取一个梯度，求平均值和方差

    Returns:
        tuple: 包含如下两个元素:
            - loss_list (list): 表示多次采样后损失函数值的列表
            - grad_list(list): 表示多次采样后各参数梯度的列表
    """

    loss_list, grad_list = [], []
    pbar = tqdm(
        desc="Sampling: ", total=sample_num, ncols=100, ascii=True
    )
    for itr in range(sample_num):
        pbar.update(1)
        shape = paddle.shape(circuit.get_param())
        net = StateNet(shape=shape)
        loss, cir = net(circuit, loss_func, *args)
        loss.backward()
        grad = net.theta.grad.numpy()
        loss_list.append(loss.numpy()[0])
        grad_list.append(grad)

    pbar.close()

    if mode == 'single':
        grad_list = np.array(grad_list)
        grad_list = grad_list.transpose()
        grad_variance_list = []
        grad_mean_list = []
        for idx in range(len(grad_list)):
            grad_variance_list.append(np.var(grad_list[idx]))
            grad_mean_list.append(np.mean(grad_list[idx]))

        print("Mean of gradient for all parameters: ")
        for i in range(len(grad_mean_list)):
            print("theta", i+1, ": ", grad_mean_list[i])
        print("Variance of gradient for all parameters: ")
        for i in range(len(grad_variance_list)):
            print("theta", i+1, ": ", grad_variance_list[i])

        if if_plot:
            plot_distribution(grad_list[param])

        return grad_mean_list, grad_variance_list

    if mode == 'max':
        max_grad_list = []
        for idx in range(len(grad_list)):
            max_grad_list.append(np.max(np.abs(grad_list[idx])))

        print("Mean of max gradient")
        print(np.mean(max_grad_list))
        print("Variance of max gradient")
        print(np.var(max_grad_list))

        if if_plot:
            plot_distribution(max_grad_list)

        return np.mean(max_grad_list), np.var(max_grad_list)

    if mode == 'random':
        random_grad_list = []
        for idx in range(len(grad_list)):
            random_grad = choice(grad_list[idx])
            random_grad_list.append(random_grad)
        print("Mean of random gradient")
        print(np.mean(random_grad_list))
        print("Variance of random gradient")
        print(np.var(random_grad_list))

        if if_plot:
            plot_distribution(random_grad_list)

        return np.mean(random_grad_list), np.var(random_grad_list)

    return loss_list, grad_list


def plot_loss_grad(circuit, loss_func, ITR, LR, *args):
    r"""绘制损失值和梯度随训练次数变化的图

    Args:
        circuit (UAnsatz): 表示传入的参数化量子电路，即要训练的量子神经网络
        loss_func (function): 表示计算该量子神经网络损失值的函数
        ITR (int): 表示训练的次数
        LR (float): 表示学习训练的速率
        *args (list): 表示用于损失函数计算的额外参数列表

    """
    loss, grad = show_gradient(circuit, loss_func, ITR, LR, *args)
    plt.xlabel(r"Iteration")
    plt.ylabel(r"Loss")
    plt.plot(range(1, ITR+1), loss, 'r', label='loss')
    plt.legend()
    plt.show()

    max_grad = [np.max(np.abs(i)) for i in grad]
    plt.xlabel(r"Iteration")
    plt.ylabel(r"Gradient")
    plt.plot(range(1, ITR+1), max_grad, 'b', label='gradient')
    plt.legend()
    plt.show()


def plot_supervised_loss_grad(circuit, loss_func, N, EPOCH, LR, BATCH, TRAIN_X, TRAIN_Y, *args):
    r"""绘制监督学习中损失值和梯度随训练次数变化的图

    Args:
        circuit (UAnsatz): 表示传入的参数化量子电路，即要训练的量子神经网络
        loss_func (function): 表示计算该量子神经网络损失值的函数
        N (int): 表示量子比特的数量
        EPOCH (int): 表示训练的轮数
        LR (float): 表示学习训练的速率
        BATCH (int): 表示训练时 batch 的大小
        TRAIN_X (paddle.Tensor): 表示训练数据集
        TRAIN_Y (list): 表示训练数据集的标签
        *args (list): 表示用于损失函数计算的额外参数列表

    Returns:
        tuple: 包含如下两个元素:
             - loss_list (list): 表示多次训练的损失函数值列表
             - grad_list(list): 表示多次训练后各参数梯度的列表
    """
    grad_list = []
    loss_list = []

    if type(TRAIN_X) != paddle.Tensor:
        raise Exception("Training data should be paddle.Tensor type")

    shape = paddle.shape(circuit.get_param())
    net = StateNet(shape=shape)
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=net.parameters())

    for ep in range(EPOCH):
        for itr in range(len(TRAIN_X)//BATCH):
            input_state = TRAIN_X[itr*BATCH:(itr+1)*BATCH]
            input_state = reshape(input_state, [-1, 1, 2**N])
            label = TRAIN_Y[itr * BATCH:(itr + 1) * BATCH]
            loss, circuit = net(circuit, loss_func, input_state, label)
            loss.backward()
            grad = net.theta.grad.numpy()
            grad_list.append(grad)
            loss_list.append(loss.numpy()[0])
            opt.minimize(loss)
            opt.clear_grad()

    max_grad = [np.max(np.abs(i)) for i in grad_list]
    plt.xlabel(r"Iteration")
    plt.ylabel(r"Loss")
    plt.plot(range(1, EPOCH*len(TRAIN_X)//BATCH+1), loss_list, 'r', label='loss')
    plt.legend()
    plt.show()

    plt.xlabel(r"Iteration")
    plt.ylabel(r"Gradient")
    plt.plot(range(1, EPOCH*len(TRAIN_X)//BATCH+1), max_grad, 'b', label='gradient')
    plt.legend()
    plt.show()

    return loss_list, grad_list


def random_sample_supervised(circuit, loss_func, N, sample_num, BATCH, TRAIN_X, TRAIN_Y, *args, mode='single', if_plot=True, param=0):
    r"""表示对监督学习模型进行随机采样，根据不同的计算模式，获得对应的平均值和方差

    Args:
        circuit (UAnsatz): 表示传入的参数化量子电路，即要训练的量子神经网络
        loss_func (function): 表示计算该量子神经网络损失值的函数
        N (int): 表示量子比特的数量
        sample_num (int): 表示随机采样的次数
        BATCH (int): 表示训练时 batch 的大小
        TRAIN_X (paddle.Tensor): 表示训练数据集
        TRAIN_Y (list): 表示训练数据集的标签
        mode (string): 表示随机采样后的计算模式，默认为 'single'
        if_plot(boolean): 表示是否对梯度进行画图表示
        param (int): 表示 ``Single`` 模式中对第几个参数进行画图，默认为第一个参数
        *args (list): 表示用于损失函数计算的额外参数列表

    Note:
        在本函数中提供了三种计算模式，``mode`` 分别可以选择 ``'single'``, ``'max'``, 以及 ``'random'``
            - mode='single': 表示计算电路中的每个可变参数梯度的平均值和方差
            - mode='max': 表示对电路中所有参数梯度的最大值求平均值和方差
            - mode='random': 表示随机对电路中采样的所有参数随机取一个梯度，求平均值和方差

    Returns:
        tuple: 包含如下两个元素:
            - loss_list (list): 表示多次采样后损失函数值的列表
            - grad_list(list): 表示多次采样后各参数梯度的列表
    """
    grad_list = []
    loss_list = []
    input_state = TRAIN_X[0:BATCH]
    input_state = reshape(input_state, [-1, 1, 2**N])
    label = TRAIN_Y[0: BATCH]

    if type(TRAIN_X) != paddle.Tensor:
        raise Exception("Training data should be paddle.Tensor type")

    label = TRAIN_Y[0: BATCH]

    pbar = tqdm(
        desc="Sampling: ", total=sample_num, ncols=100, ascii=True
    )
    for idx in range(sample_num):
        pbar.update(1)
        shape = paddle.shape(circuit.get_param())
        net = StateNet(shape=shape)

        loss, circuit = net(circuit, loss_func, input_state, label)
        loss.backward()
        grad = net.theta.grad.numpy()
        grad_list.append(grad)
        loss_list.append(loss.numpy()[0])
    pbar.close()

    if mode == 'single':
        grad_list = np.array(grad_list)
        grad_list = grad_list.transpose()
        grad_variance_list = []
        grad_mean_list = []
        for idx in range(len(grad_list)):
            grad_variance_list.append(np.var(grad_list[idx]))
            grad_mean_list.append(np.mean(grad_list[idx]))

        print("Mean of gradient for all parameters: ")
        for i in range(len(grad_mean_list)):
            print("theta", i+1, ": ", grad_mean_list[i])
        print("Variance of gradient for all parameters: ")
        for i in range(len(grad_variance_list)):
            print("theta", i+1, ": ", grad_variance_list[i])

        if if_plot:
            plot_distribution(grad_list[param])

        return grad_mean_list, grad_variance_list

    if mode == 'max':
        max_grad_list = []
        for idx in range(len(grad_list)):
            max_grad_list.append(np.max(np.abs(grad_list[idx])))

        print("Mean of max gradient")
        print(np.mean(max_grad_list))
        print("Variance of max gradient")
        print(np.var(max_grad_list))

        if if_plot:
            plot_distribution(max_grad_list)

        return np.mean(max_grad_list), np.var(max_grad_list)

    if mode == 'random':
        random_grad_list = []
        for idx in range(len(grad_list)):
            random_grad = choice(grad_list[idx])
            random_grad_list.append(random_grad)
        print("Mean of random gradient")
        print(np.mean(random_grad_list))
        print("Variance of random gradient")
        print(np.var(random_grad_list))

        if if_plot:
            plot_distribution(random_grad_list)

        return np.mean(random_grad_list), np.var(random_grad_list)

    return loss_list, grad_list
