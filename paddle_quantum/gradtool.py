# !/usr/bin/env python3
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

r"""
The module of the gradient tool.
"""


from typing import Any, Callable, Tuple, List
import numpy as np
import paddle
import paddle_quantum
from paddle_quantum.ansatz import Circuit
from math import pi
from random import choice
from tqdm import tqdm
import matplotlib.pyplot as plt

__all__ = [
    "show_gradient",
    "plot_distribution",
    "random_sample",
    "plot_loss_grad",
    "plot_supervised_loss_grad",
    "random_sample_supervised"
]


def show_gradient(circuit: Circuit, loss_func: Callable[[Circuit, Any], paddle.Tensor], 
                  ITR: int, LR: float, *args: Any) -> Tuple[List[float], List[float]]:
    r"""Calculate the gradient and loss function for every parameter in QNN.

    Args:
        circuit: QNN to be trained.
        loss_func: Loss function that evaluates the QNN.
        ITR: Number of iterations.
        LR: Learning rate.
        *args: Parameters for ``loss_func`` other than ``circuit``.

    Returns:
        Contains following two elements.
            - loss_list: A list of losses for each iteration.
            - grad_list: A list of gradients for each iteration.
    """
    grad_list = []
    loss_list = []
    pbar = tqdm(
        desc="Training: ", total=ITR, ncols=100, ascii=True
    )
    
    # randomize initial parameters
    circuit.randomize_param()
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=circuit.parameters())

    for _ in range(ITR):
        pbar.update(1)
        loss = loss_func(circuit, *args)
        loss.backward()
        grad_list.append(circuit.grad)
        loss_list.append(loss.numpy()[0])
        opt.minimize(loss)
        opt.clear_grad()
    pbar.close()

    return loss_list, grad_list


def plot_distribution(grad: np.ndarray) -> None:
    r"""Plot the distribution map according to the input gradients.

    Args:
        grad: List of gradients with respect to a parameter.
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


def random_sample(circuit: Circuit, loss_func: Callable[[Circuit, Any], paddle.Tensor], sample_num: int, *args: Any, 
                  mode: str = 'single', if_plot: bool = True, param: int = 0) -> Tuple[List[float], List[float]]:
    r"""Randomly sample the model. Obtain mean and variance of gradients according to different calculation modes.

    Args:
        circuit: QNN to be trained.
        loss_func: Loss function that evaluates the QNN.
        sample_num: Number of samplings.
        mode: Mode for calculation. Defaults to ``'single'``.
        if_plot: Whether plot the calculation. Defaults to ``True``.
        param: Which parameter to be plotted in single mode, Defaults to ``0``, which means the first one.
        *args: Parameters for ``loss_func`` other than ``circuit``.

    Note:
        This function provides three calculation modes: single, max and random.
            - In single mode, we calculate the mean and variance of gradients of every trainable parameter.
            - In max mode, we calculate the mean and variance of maximum gradients of for every trainable parameter.
            - In random mode, we calculate the mean and variance of data randomly extracted from gradients of every trainable parameter.

    Returns:
        Contains the following two elements.
            - loss_list: A list of losses for each iteration.
            - grad_list: A list of gradients for each iteration.
    """
    loss_list, grad_list = [], []
    pbar = tqdm(
        desc="Sampling: ", total=sample_num, ncols=100, ascii=True
    )
    for _ in range(sample_num):
        pbar.update(1)
        circuit.randomize_param()
        loss = loss_func(circuit, *args)
        loss.backward()
        loss_list.append(loss.numpy()[0])
        grad_list.append(circuit.grad)

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


def plot_loss_grad(circuit: Circuit, loss_func: Callable[[Circuit, Any], paddle.Tensor], ITR: int, LR: float, *args: Any) -> None:
    r"""Plot the distribution maps between loss values & gradients and number of iterations.
    
    Args:
        circuit: QNN to be trained.
        loss_func: Loss function that evaluate QNN.
        ITR: Number of iterations.
        LR: Learning rate.
        *args: Parameters for ``loss_func`` other than ``circuit``.
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


def plot_supervised_loss_grad(circuit: Circuit, loss_func: Callable[[Circuit, Any], paddle.Tensor], N: int, EPOCH: int, LR: float, 
                              BATCH: int, TRAIN_X: paddle.Tensor, TRAIN_Y: list, *args: Any) -> Tuple[List[float], List[float]]:
    r""" plot the distribution maps between loss values & gradients and number of iterations in supervised training

    Args:
        circuit: QNN ready to be trained.
        loss_func: Loss function that evaluates the QNN.
        N: Number of qubits.
        EPOCH: Number of training iterations.
        LR: Learning rate.
        BATCH: Size of batches.
        TRAIN_X: Data set .
        TRAIN_Y: Label set.
        *args: Parameters for ``loss_func`` other than ``circuit``.

    Raises:
        Exception: Training data should be paddle.Tensor type

    Returns:
        Contains the following two elements.
            - loss_list: A list of losses for each iteration.
            - grad_list: A list of gradients for each iteration.
    """
    grad_list = []
    loss_list = []

    if type(TRAIN_X) != paddle.Tensor:
        raise Exception("Training data should be paddle.Tensor type")

    circuit.randomize_param()
    opt = paddle.optimizer.Adam(learning_rate=LR, parameters=circuit.parameters())

    for _ in range(EPOCH):
        for itr in range(len(TRAIN_X)//BATCH):
            input_state = TRAIN_X[itr*BATCH:(itr+1)*BATCH]
            input_state = input_state.reshape([-1, 1, 2**N])
            label = TRAIN_Y[itr * BATCH:(itr + 1) * BATCH]
            
            loss = loss_func(circuit, input_state, label)
            loss.backward()
            grad_list.append(circuit.grad)
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


def random_sample_supervised(circuit: Circuit, loss_func: Callable[[Circuit, Any], paddle.Tensor], 
                             N: int, sample_num: int, BATCH: int, TRAIN_X: paddle.Tensor, TRAIN_Y: paddle.Tensor, 
                             *args: Any, mode: str = 'single', if_plot: bool = True, param: int = 0) -> Tuple[List[float], List[float]]:
    r"""Random sample the supervised model. Obtain mean and variance of gradients according to different calculation modes.

    Args:
        circuit: QNN to be trained.
        loss_func: Loss function that evaluates the QNN.
        N: Number of qubits.
        sample_num: Number of samplings.
        BATCH: Size of batches.
        TRAIN_X: Data set.
        TRAIN_Y: Label set.
        mode: Mode for calculation. Defaults to ``'single'``.
        if_plot: Whether plot the calculation. Defaults to ``True``.
        param: Which parameter to be plotted in single mode. Defaults to ``0``, which means the first one.
        *args: Parameters for ``loss_func`` other than ``circuit``.

    Note:
        This function provides three calculation modes: single, max and random.
            - In single mode, we calculate the mean and variance of gradients of every trainable parameters.
            - In max mode, we calculate the mean and variance of maximum gradients of for every trainable parameters.
            - In random mode, we calculate the mean and variance of data randomly extracted from gradients of every trainable parameters.
    
    Raises:
        Exception: Training data should be paddle.Tensor type

    Returns:
        Contains the following two elements.
            - loss_list: A list of losses for each iteration.
            - grad_list: A list of gradients for each iteration.
    """
    grad_list = []
    loss_list = []
    input_state = TRAIN_X[0:BATCH]
    input_state = input_state.reshape([-1, 1, 2**N])
    label = TRAIN_Y[0: BATCH]

    if type(TRAIN_X) != paddle.Tensor:
        raise Exception("Training data should be paddle.Tensor type")

    pbar = tqdm(
        desc="Sampling: ", total=sample_num, ncols=100, ascii=True
    )
    for idx in range(sample_num):
        pbar.update(1)
        circuit.randomize_param()
        loss = loss_func(circuit, input_state, label)
        loss.backward()
        grad_list.append(circuit.grad)
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
