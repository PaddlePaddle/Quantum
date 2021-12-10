# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
dataset: To learn more about the functions and properties of this application,
you could check the corresponding Jupyter notebook under the Tutorial folder.
"""

import math
import random
from math import sqrt
import time
import paddle.vision.transforms as transform
import numpy as np
import paddle
from paddle_quantum.circuit import UAnsatz
from paddle.fluid.layers import reshape

from sklearn.model_selection import train_test_split
from sklearn import datasets

__all__ = [
    "VisionDataset",
    "SimpleDataset",
    "MNIST",
    "FashionMNIST",
    "Iris",
    "BreastCancer"
]

# data modes
DATAMODE_TRAIN = "train"
DATAMODE_TEST = "test"

# encoding methods
ANGLE_ENCODING = "angle_encoding"
AMPLITUDE_ENCODING = "amplitude_encoding"
PAULI_ROTATION_ENCODING = "pauli_rotation_encoding"

LINEAR_ENTANGLED_ENCODING = "linear_entangled_encoding"
REAL_ENTANGLED_ENCODING = "real_entangled_encoding"
COMPLEX_ENTANGLED_ENCODING = "complex_entangled_encoding"
IQP_ENCODING = "IQP_encoding"

# downscaling method
DOWNSCALINGMETHOD_PCA = "PCA"
DOWNSCALINGMETHOD_RESIZE = "resize"


def _normalize(x):
    r"""normalize vector ``x`` and the maximum will be pi. This is an internal function.

    Args:
        x (ndarray): 需要归一化的向量

    Returns:
        ndarray: 归一化之后的向量
    """
    xx = np.abs(x)
    if xx.max() > 0:
        return x * np.pi / xx.max()
    else:
        return x


def _normalize_image(x):
    r"""normalize image vector ``x`` and the maximum will be pi. This is an internal function.

    Args:
        x (ndarray): 需要归一化的图片向量

    Returns:
        ndarray: 归一化之后的向量
    """
    return x * np.pi / 256


def _crop(images, border):
    r"""crop ``images`` according to ``border``. This is an internal function.

    Args:
        images (list/ndarray): 每一个元素是拉成一维的图片向量
        border(list): 裁剪的边界，从第一个元素切到第二个元素，比如说[4,24]就是从图片的第四行第四列到第24行第24列

    Returns:
        new_images(list): 裁剪之后被拉成一维的图片向量组成的list
    """
    new_images = []
    for i in range(len(images)):
        size = int(sqrt(len(images[i])))
        temp_image = images[i].reshape((size, size))
        temp_image = temp_image[border[0]:border[1], border[0]:border[1]]
        new_images.append(temp_image.flatten())
    return new_images


class Dataset(object):
    r"""所有数据集的基类，集成了多种量子编码方法。
    """
    def __init__(self):
        return

    def data2circuit(self, classical_data, encoding, num_qubits, can_describe_dimension, split_circuit,
                     return_state, is_image=False):
        r"""将输入的经典数据 ``classical_data`` 用编码方式 ``encoding`` 编码成量子态，这里的经典数据经过了截断或者补零，因而可以正好被编码。

        Args:
            classical_data (list): 待编码的向量，ndarray 组成的 list，经过了截断或者补零，刚好可以被编码
            encoding (str): 编码方式，参见 MNIST 编码注释
            num_qubits (int): 量子比特数目
            can_describe_dimension (int): 以 ``encoding`` 编码方式可以编码的数目，比如说振幅编码为 ``2 ** n`` ，其他编码因为可以进行层层堆叠需要计算
            split_circuit (bool): 是否切分电路
            return_state (bool): 是否返回量子态
            is_image (bool): 是否是图片，如果是图片，归一化方法不太一样

        Returns:
            List: 如果 ``return_state == True`` ，返回编码后的量子态，否则返回编码的电路
        """
        quantum_states = classical_data.copy()
        quantum_circuits = classical_data.copy()
        if encoding == AMPLITUDE_ENCODING:
            # Not support to return circuit in amplitude encoding
            if return_state is False or split_circuit is True:
                raise Exception("Not support to return circuit in amplitude encoding")
            for i in range(len(classical_data)):
                built_in_amplitude_enc = UAnsatz(num_qubits)
                x = paddle.to_tensor(_normalize(classical_data[i]))
                if is_image:
                    x = paddle.to_tensor(_normalize_image(classical_data[i]))
                state = built_in_amplitude_enc.amplitude_encoding(x, 'state_vector')
                quantum_states[i] = state.numpy()

        elif encoding == ANGLE_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 1 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]))
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]))
                param = reshape(param, (depth, num_qubits, 1))
                which_qubits = [k for k in range(num_qubits)]
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = UAnsatz(num_qubits)
                        for k, q in enumerate(which_qubits):
                            circuit.ry(param[repeat][k][0], q)

                        quantum_circuits[i].append(circuit)
                else:
                    circuit = UAnsatz(num_qubits)
                    for repeat in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.ry(param[repeat][k][0], q)
                    state_out = circuit.run_state_vector()
                    quantum_states[i] = state_out.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == IQP_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 1 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]))
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]))
                param = reshape(param, (depth, num_qubits))
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = UAnsatz(num_qubits)
                        S = []
                        for k in range(num_qubits - 1):
                            S.append([k, k + 1])
                        # r 是 U 重复的次数
                        r = 1
                        circuit.iqp_encoding(param[repeat], r, S)
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = UAnsatz(num_qubits)
                    for repeat in range(depth):
                        temp_circuit = UAnsatz(num_qubits)
                        S = []
                        for k in range(num_qubits - 1):
                            S.append([k, k + 1])
                        # r 是 U 重复的次数
                        r = 1
                        temp_circuit.iqp_encoding(param[repeat], r, S)
                        circuit = circuit + temp_circuit
                    state_out = circuit.run_state_vector()
                    quantum_states[i] = state_out.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == PAULI_ROTATION_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 3 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]))
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]))
                param = reshape(param, (depth, num_qubits, 3))
                which_qubits = [k for k in range(num_qubits)]
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = UAnsatz(num_qubits)
                        for k, q in enumerate(which_qubits):
                            circuit.ry(param[repeat][k][0], q)
                            circuit.rz(param[repeat][k][1], q)
                            circuit.ry(param[repeat][k][2], q)
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = UAnsatz(num_qubits)
                    for repeat in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.ry(param[repeat][k][0], q)
                            circuit.rz(param[repeat][k][1], q)
                            circuit.ry(param[repeat][k][2], q)
                    state_out = circuit.run_state_vector()
                    quantum_states[i] = state_out.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == LINEAR_ENTANGLED_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 2 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]))
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]))
                param = reshape(param, (depth, num_qubits, 2))
                which_qubits = [k for k in range(num_qubits)]
                if split_circuit:
                    quantum_circuits[i] = []
                    for j in range(depth):
                        circuit = UAnsatz(num_qubits)
                        for k, q in enumerate(which_qubits):
                            circuit.ry(param[j][k][0], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.cnot([which_qubits[k], which_qubits[k + 1]])
                        for k, q in enumerate(which_qubits):
                            circuit.rz(param[j][k][1], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.cnot([which_qubits[k + 1], which_qubits[k]])
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = UAnsatz(num_qubits)
                    for j in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.ry(param[j][k][0], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.cnot([which_qubits[k], which_qubits[k + 1]])
                        for k, q in enumerate(which_qubits):
                            circuit.rz(param[j][k][1], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.cnot([which_qubits[k + 1], which_qubits[k]])
                    state_out = circuit.run_state_vector()
                    quantum_states[i] = state_out.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == REAL_ENTANGLED_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 1 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]))
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]))
                param = reshape(param, (depth, num_qubits, 1))
                which_qubits = [k for k in range(num_qubits)]
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = UAnsatz(num_qubits)
                        for k, q in enumerate(which_qubits):
                            circuit.ry(param[repeat][k][0], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.cnot([which_qubits[k], which_qubits[k + 1]])
                        circuit.cnot([which_qubits[-1], which_qubits[0]])
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = UAnsatz(num_qubits)
                    for repeat in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.ry(param[repeat][k][0], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.cnot([which_qubits[k], which_qubits[k + 1]])
                        circuit.cnot([which_qubits[-1], which_qubits[0]])
                    state_out = circuit.run_state_vector()
                    quantum_states[i] = state_out.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == COMPLEX_ENTANGLED_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 3 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]))
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]))
                param = reshape(param, (depth, num_qubits, 3))
                which_qubits = [k for k in range(num_qubits)]
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = UAnsatz(num_qubits)
                        for k, q in enumerate(which_qubits):
                            circuit.u3(param[repeat][k][0], param[repeat][k][1], param[repeat][k][2], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.cnot([which_qubits[k], which_qubits[k + 1]])
                        circuit.cnot([which_qubits[-1], which_qubits[0]])
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = UAnsatz(num_qubits)
                    for repeat in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.u3(param[repeat][k][0], param[repeat][k][1], param[repeat][k][2], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.cnot([which_qubits[k], which_qubits[k + 1]])
                        circuit.cnot([which_qubits[-1], which_qubits[0]])
                    state_out = circuit.run_state_vector()
                    quantum_states[i] = state_out.numpy()
                    quantum_circuits[i] = [circuit]
        return quantum_states, quantum_circuits

    def filter_class(self, x, y, classes, data_num, need_relabel, seed=0):
        r"""将输入的 ``x`` , ``y`` 按照 ``classes`` 给出的类别进行筛选，数目为 ``data_num`` 。

        Args:
            x (ndarray/list): 样本的特征
            y (ndarray/list): 样本标签， ``classes`` 是其中某几个标签的取值
            classes (list): 需要筛选的类别
            data_num (int): 筛选出来的样本数目
            need_relabel (bool): 将原有类别按照顺序重新标记为 0、1、2 等新的名字，比如传入 ``[1,2]`` ， 重新标记之后变为 ``[0,1]`` 主要用于二分类
            seed (int): 随机种子，默认为 ``0``

        Returns:
            tuple: 包含如下元素:
                - new_x (list): 筛选出的特征
                - new_y (list): 对应于 ``new_x`` 的标签

        """
        new_x = []
        new_y = []
        if need_relabel:
            for i in range(len(x)):
                if y[i] in classes:
                    new_x.append(x[i])
                    new_y.append(classes.index(y[i]))
        else:
            for i in range(len(x)):
                if y[i] in classes:
                    new_x.append(x[i])
                    new_y.append(y[i])

        # sample to data_num randomly
        if data_num > 0 and data_num < len(new_x):
            random_index = [k for k in range(len(new_x))]
            random.seed(seed)
            random.shuffle(random_index)
            random_index = random_index[:data_num]
            filter_x = []
            filter_y = []
            for index in random_index:
                filter_x.append(new_x[index])
                filter_y.append(new_y[index])
            return filter_x, filter_y
        return new_x, new_y


class VisionDataset(Dataset):
    r"""图片数据集类，通过继承 VisionDataset 类，用户可以快速生成自己的图片量子数据。

    Attributes:
        original_images (ndarray): 图片经过类别过滤，但是还没有降维、补零的特征，是一个一维向量（可调用 ``reshape()`` 函数转成图片）
        classical_image_vectors (ndarray): 经过类别过滤和降维、补零等操作之后的特征，并未编码为量子态
        quantum_image_states (paddle.tensor): 经过类别过滤之后的所有特征经编码形成的量子态
        quantum_image_circuits (list): 所有特征编码的电路
    """

    def __init__(self, figure_size):
        r"""构造函数

        Args:
            figure_size (int): 图片大小，也就是长和高的数值
        """
        Dataset.__init__(self)
        self.figure_size = figure_size
        return

    # The encode function only needs to import images to form one-dimensional vector features.
    # The pre-processing of images (except dimensionality reduction) is completed before the import of features
    def encode(self, feature, encoding, num_qubits, split_circuit=False,
               downscaling_method=DOWNSCALINGMETHOD_RESIZE, target_dimension=-1, return_state=True, full_return=False):
        r"""根据降尺度方式、目标尺度、编码方式、量子比特数目进行编码。只需要输入一维图片向量就行。

        Args:
            feature (list/ndarray): 一维图片向量组成的list/ndarray
            encoding (str): ``"angle_encoding"`` 表示角度编码，一个量子比特编码一个旋转门； ``"amplitude_encoding"`` 表示振幅编码；
                           ``"pauli_rotation_encoding"`` 表示SU(3)的角度编码; 还有 ``"linear_entangled_encoding"`` ,
                           ``"real_entangled_encoding"`` , ``"complex_entangled_encoding"`` 三种纠缠编码和 ``"IQP_encoding"`` 编码
            num_qubits (int): 编码后的量子比特数目
            split_circuit (bool): 是否需要切分电路。除了振幅之外的所有电路都会存在堆叠的情况，如果选择 ``True`` 就将块与块分开
            downscaling_method (str): 包括 ``"PCA"`` 和 ``"resize"``
            target_dimension (int): 降维之后的尺度大小，如果是 ``"PCA"`` ，不能超过图片大小；如果是 ``"resize"`` ，不能超过原图大小
            return_state (bool): 是否返回量子态，如果是 ``False`` 返回量子电路

        Returns:
            tuple: 包含如下元素:
                - quantum_image_states (paddle.tensor): 量子态，只有 ``full_return==True`` 或者 ``return_state==True`` 的时候会返回
                - quantum_image_circuits (list): 所有特征编码的电路， 只有 ``full_return==False`` 或者 ``return_state==True`` 的时候会返回
                - original_images (ndarray): 图片经过类别过滤，但是还没有降维、补零的特征，是一个一维向量（可以 reshape 成图片），只有 ``return_state==True`` 的时候会返回
                - classical_image_vectors (ndarray): 经过类别过滤和降维、补零等操作之后的特征，并未编码为量子态，只有 ``return_state==True`` 的时候会返回

        """
        assert num_qubits > 0
        if encoding in [IQP_ENCODING, COMPLEX_ENTANGLED_ENCODING, REAL_ENTANGLED_ENCODING,
                        LINEAR_ENTANGLED_ENCODING]:
            assert num_qubits > 1

        if type(feature) == np.ndarray:
            feature = list(feature)

        # The first step: judge whether `target_dimension` is reasonable
        if target_dimension > -1:
            if downscaling_method == DOWNSCALINGMETHOD_PCA:
                if target_dimension > self.figure_size:
                    raise Exception("PCA dimension should be less than {}.".format(self.figure_size))
            elif downscaling_method == DOWNSCALINGMETHOD_RESIZE:
                if int(sqrt(target_dimension)) ** 2 != target_dimension:  # not a square
                    raise Exception("Resize dimension should be a square.")
            else:
                raise Exception("Downscaling methods can only be resize and PCA.")
        else:
            if downscaling_method == DOWNSCALINGMETHOD_PCA:
                target_dimension = self.figure_size
            elif downscaling_method == DOWNSCALINGMETHOD_RESIZE:
                target_dimension = self.figure_size ** 2

        # The second step: calculate `can_describe_dimension`
        if encoding == AMPLITUDE_ENCODING:  # amplitude encoding, encoding 2^N-dimension feature
            self.can_describe_dimension = 2 ** num_qubits

        elif encoding == LINEAR_ENTANGLED_ENCODING:
            one_block_param = 2 * num_qubits
            self.can_describe_dimension = math.ceil(target_dimension / one_block_param) * one_block_param

        elif encoding in [REAL_ENTANGLED_ENCODING, ANGLE_ENCODING, IQP_ENCODING]:
            one_block_param = 1 * num_qubits
            self.can_describe_dimension = math.ceil(target_dimension / one_block_param) * one_block_param

        elif encoding in [COMPLEX_ENTANGLED_ENCODING, PAULI_ROTATION_ENCODING]:
            one_block_param = 3 * num_qubits
            self.can_describe_dimension = math.ceil(target_dimension / one_block_param) * one_block_param

        else:
            raise Exception("Invalid encoding methods!")
        self.dimension = target_dimension

        # The third step: download MNIST data from paddlepaddle and crop or fill the vector to ``can_describe_vector``
        self.original_images = np.array(feature)
        self.classical_image_vectors = feature.copy()

        # What need to mention if ``Resize`` needs uint8, but MNIST in paddle is float32, so we should change its type.
        if downscaling_method == DOWNSCALINGMETHOD_RESIZE:
            # iterating all items
            for i in range(len(self.classical_image_vectors)):
                cur_image = self.classical_image_vectors[i].astype(np.uint8)
                new_size = int(sqrt(self.dimension))
                cur_image = transform.resize(cur_image.reshape((self.figure_size, self.figure_size)),
                                             (new_size, new_size))
                self.classical_image_vectors[i] = cur_image.reshape(-1).astype(np.float64)  # now it is one-dimension

                if self.can_describe_dimension < len(self.classical_image_vectors[i]):
                    self.classical_image_vectors[i] = self.classical_image_vectors[i][:self.can_describe_dimension]
                else:
                    self.classical_image_vectors[i] = np.append(self.classical_image_vectors[i], np.array(
                        [0.0] * (self.can_describe_dimension - len(self.classical_image_vectors[i]))))

        elif downscaling_method == DOWNSCALINGMETHOD_PCA:
            for i in range(len(self.classical_image_vectors)):
                U, s, V = np.linalg.svd(self.classical_image_vectors[i].reshape((self.figure_size, self.figure_size)))
                s = s[:self.dimension].astype(np.float64)
                if self.can_describe_dimension > self.dimension:
                    self.classical_image_vectors[i] = np.append(s, np.array(
                        [0.0] * (self.can_describe_dimension - self.dimension)))
                else:
                    self.classical_image_vectors[i] = s[:self.can_describe_dimension]

        # Step 4: Encode the data, which must be of float64 type(needed in paddle quantum)
        self.quantum_image_states, self.quantum_image_circuits = self.data2circuit(
            self.classical_image_vectors, encoding, num_qubits, self.can_describe_dimension, split_circuit,
            return_state, is_image=True)
        self.classical_image_vectors = np.array(self.classical_image_vectors)
        if return_state:
            self.quantum_image_states = paddle.to_tensor(np.array(self.quantum_image_states))  # transfer to tensor

        if full_return:
            return self.quantum_image_states, self.quantum_image_circuits, self.original_images, \
                   self.classical_image_vectors
        else:
            if return_state:
                return self.quantum_image_states
            else:
                return self.quantum_image_circuits


class MNIST(VisionDataset):
    r"""MNIST 数据集，它继承了 VisionDataset 图片数据集类。

    Attributes:
        original_images(ndarray): 图片经过类别过滤，但是还没有降维、补零的特征，是一个一维向量（可调用 ``reshape()`` 方法转成图片）
        classical_image_vectors(ndarray): 经过类别过滤和降维、补零等操作之后的特征，并未编码为量子态
        quantum_image_states(paddle.tensor): 经过类别过滤之后的所有特征经编码形成的量子态
        quantum_image_circuits(list): 所有特征编码的电路
        labels(ndarray): 经过类别过滤之后的所有标签

    代码示例:

    .. code-block:: python

        from paddle_quantum.dataset import MNIST

        # main parameters
        training_data_num = 80
        testing_data_num = 20
        qubit_num = 4

        # acquiring training dataset
        train_dataset = MNIST(mode='train', encoding='pauli_rotation_encoding', num_qubits=qubit_num, classes=[3,6],
                              data_num=training_data_num,need_cropping=True,
                              downscaling_method='resize', target_dimension=16, return_state=True)

        # acquiring testing dataset
        val_dataset = MNIST(mode='test', encoding='pauli_rotation_encoding', num_qubits=qubit_num, classes=[3,6],
                            data_num=testing_data_num,need_cropping=True,
                            downscaling_method='resize', target_dimension=16,return_state=True)

        # acquiring features and labels
        train_x, train_y = train_dataset.quantum_image_states, train_dataset.labels # paddle.tensor, ndarray
        test_x, test_y = val_dataset.quantum_image_states, val_dataset.labels

        print(train_x[0])
        print(train_y[0])

    """

    def __init__(self, mode, encoding, num_qubits, classes, data_num=-1, split_circuit=False,
                 downscaling_method=DOWNSCALINGMETHOD_RESIZE, target_dimension=-1, need_cropping=True,
                 need_relabel=True, return_state=True, seed=0):
        r"""构造函数

        Args:
            mode (str): 数据模式，包括 ``"train"`` 和 ``"test"``
            encoding (str): ``"angle_encoding"`` 表示角度编码，一个量子比特编码一个旋转门； ``"amplitude_encoding"`` 表示振幅编码；
                           ``"pauli_rotation_encoding"`` 表示SU(3)的角度编码; 还有 ``"linear_entangled_encoding"`` ,
                           ``"real_entangled_encoding"`` , ``"complex_entangled_encoding"`` 三种纠缠编码和 ``"IQP_encoding"`` 编码
            num_qubits (int): 编码后的量子比特数目
            classes (list): 用列表给出需要的类别，类别用数字标签表示，不支持传入名字
            data_num (int): 使用的数据量大小，这样可以不用所有数据都进行编码，这样会很慢
            split_circuit (bool): 是否需要切分电路。除了振幅之外的所有电路都会存在堆叠的情况，如果选择 ``True`` 就将块与块分开
            need_cropping (bool): 是否需要裁边，如果为 ``True`` ，则从 ``image[0:27][0:27]`` 裁剪为 ``image[4:24][4:24]``
            need_relabel (bool): 将原有类别按照顺序重新标记为 0，1，2 等新的名字，比如传入 ``[1,2]`` ，重新标记之后变为 ``[0,1]`` ，主要用于二分类
            downscaling_method (str): 包括 ``"PCA"`` 和 ``"resize"``
            target_dimension (int): 降维之后的尺度大小，如果是 ``"PCA"`` ，不能超过图片大小；如果是 ``"resize"`` ，不能超过原图大小
            return_state (bool): 是否返回量子态，如果是 ``False`` 返回量子电路
            seed (int): 筛选样本的随机种子，默认为 ``0``
        """
        VisionDataset.__init__(self, 28)

        if need_cropping:
            self.figure_size = 20

        # Download data from paddlepaddle
        if mode == DATAMODE_TRAIN:
            train_dataset = paddle.vision.datasets.MNIST(mode='train')
            feature, self.labels = self.filter_class(train_dataset.images, train_dataset.labels,
                                                     classes=classes,
                                                     data_num=data_num, need_relabel=need_relabel, seed=seed)
            if need_cropping:
                feature = _crop(feature, [4, 24])

        elif mode == DATAMODE_TEST:
            test_dataset = paddle.vision.datasets.MNIST(mode='test')
            # test_dataset.images is now a list of (784,1) shape
            feature, self.labels = self.filter_class(test_dataset.images, test_dataset.labels,
                                                     classes=classes,
                                                     data_num=data_num, need_relabel=need_relabel, seed=seed)
            if need_cropping:
                feature = _crop(feature, [4, 24])

        else:
            raise Exception("data mode can only be train and test.")

        # Start to encode
        self.quantum_image_states, self.quantum_image_circuits, self.original_images, self.classical_image_vectors = \
            self.encode(feature, encoding, num_qubits, split_circuit, downscaling_method, target_dimension,
                        return_state, True)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.quantum_image_states)


class FashionMNIST(VisionDataset):
    r""" FashionMNIST 数据集，它继承了 VisionDataset 图片数据集类

    Attributes:
        original_images (ndarray): 图片经过类别过滤，但是还没有降维、补零的特征，是一个一维向量（可调用 ``reshape()`` 方法转成图片）
        classical_image_vectors (ndarray): 经过类别过滤和降维、补零等操作之后的特征，并未编码为量子态
        quantum_image_states (paddle.tensor): 经过类别过滤之后的所有特征经编码形成的量子态
        quantum_image_circuits (list): 所有特征编码的电路
        labels (ndarray): 经过类别过滤之后的所有标签

    代码示例:

    .. code-block:: python

        from paddle_quantum.dataset import FashionMNIST

        training_data_num=80
        testing_data_num=20
        qubit_num=4

        # acquiring training dataset
        train_dataset = FashionMNIST(mode='train', encoding='pauli_rotation_encoding', num_qubits=qubit_num, classes=[3,6],
                              data_num=training_data_num,downscaling_method='resize', target_dimension=16, return_state=True)

        # 验acquiring testing dataset
        val_dataset = FashionMNIST(mode='test', encoding='pauli_rotation_encoding', num_qubits=qubit_num, classes=[3,6],
                            data_num=testing_data_num,downscaling_method='resize', target_dimension=16,return_state=True)

        # acquiring features and labels
        train_x, train_y = train_dataset.quantum_image_states, train_dataset.labels # paddle.tensor, ndarray
        test_x, test_y = val_dataset.quantum_image_states, val_dataset.labels

        print(train_x[0])
        print(train_y[0])
    """

    def __init__(self, mode, encoding, num_qubits, classes, data_num=-1, split_circuit=False,
                 downscaling_method=DOWNSCALINGMETHOD_RESIZE, target_dimension=-1,
                 need_relabel=True, return_state=True, seed=0):
        r""" 构造函数

        Args:
            mode (str): 数据模式，包括 ``"train"`` 和 ``"test"``
            encoding (str): ``"angle_encoding"`` 表示角度编码，一个量子比特编码一个旋转门； ``"amplitude_encoding"`` 表示振幅编码；
                           ``"pauli_rotation_encoding"`` 表示SU(3)的角度编码；还有 ``"linear_entangled_encoding"`` 、
                           ``"real_entangled_encoding"`` 、 ``"complex_entangled_encoding"`` 三种纠缠编码和 ``"IQP_encoding"`` 编码
            num_qubits (int): 编码后的量子比特数目
            classes (list): 用列表给出需要的类别，类别用数字标签表示，不支持传入名字
            data_num (int): 使用的数据量大小，这样可以不用所有数据都进行编码，这样会很慢
            split_circuit (bool): 是否需要切分电路。除了振幅之外的所有电路都会存在堆叠的情况，如果选择true就将块与块分开
            need_relabel (bool): 将原有类别按照顺序重新标记为0，1，2等新的名字，比如传入 ``[1,2]`` ，重新标记之后变为 ``[0,1]`` ，主要用于二分类
            downscaling_method (str): 包括 ``"PCA"`` 和 ``"resize"``
            target_dimension (int): 降维之后的尺度大小，如果是 ``"PCA"`` ，不能超过图片大小；如果是 ``"resize"`` ，不能超过原图大小
            return_state (bool): 是否返回量子态，如果是 ``False`` 返回量子电路
            seed (int): 随机种子，默认为 ``0``
        """
        VisionDataset.__init__(self, 28)

        # Download data from paddlepaddle
        if mode == DATAMODE_TRAIN:
            train_dataset = paddle.vision.datasets.FashionMNIST(mode='train')
            feature, self.labels = self.filter_class(train_dataset.images, train_dataset.labels,
                                                     classes=classes,
                                                     data_num=data_num, need_relabel=need_relabel, seed=seed)

        elif mode == DATAMODE_TEST:
            test_dataset = paddle.vision.datasets.FashionMNIST(mode='test')
            # test_dataset.images is now a list of (784,1) shape
            feature, self.labels = self.filter_class(test_dataset.images, test_dataset.labels,
                                                     classes=classes,
                                                     data_num=data_num, need_relabel=need_relabel, seed=seed)

        else:
            raise Exception("data mode can only be train and test.")

        # Start to encode
        self.quantum_image_states, self.quantum_image_circuits, self.original_images, self.classical_image_vectors = \
            self.encode(feature, encoding, num_qubits, split_circuit, downscaling_method, target_dimension,
                        return_state, True)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.quantum_image_states)


class SimpleDataset(Dataset):
    r""" 用于不需要降维的简单分类数据。用户可以通过继承 ``SimpleDataset`` ，将自己的分类数据变为量子态。下面的几个属性也会被继承。

    Attributes:
        quantum_states (ndarray): 经过类别过滤之后的所有特征经编码形成的量子态
        quantum_circuits (list): 所有特征编码的电路
        origin_feature (ndarray): 经过类别过滤之后的所有特征，并未编码为量子态
        feature (ndarray): ``origin_feature`` 经过了补零之后的特征， ``quantum_states`` 就是将 ``feature`` 编码之后的结果

    代码示例:

    .. code-block:: python

        from paddle_quantum.dataset import SimpleDataset

        def circle_data_point_generator(Ntrain, Ntest, boundary_gap, seed_data):
            # generate binary classification data with circle boundaries
            # The former Ntrain samples are training data, the latter Ntest samples are testing data.

            train_x, train_y = [], []
            num_samples, seed_para = 0, 0
            while num_samples < Ntrain + Ntest:
                np.random.seed((seed_data + 10) * 1000 + seed_para + num_samples)
                data_point = np.random.rand(2) * 2 - 1  # generator two-dimension vectors in [-1, 1]

                # If the norm is below (0.7 - gap), the data point is mark as zero
                if np.linalg.norm(data_point) < 0.7 - boundary_gap / 2:
                    train_x.append(data_point)
                    train_y.append(0.)
                    num_samples += 1

                # If the norm is over (0.7 + gap), the data point is mark as one
                elif np.linalg.norm(data_point) > 0.7 + boundary_gap / 2:
                    train_x.append(data_point)
                    train_y.append(1.)
                    num_samples += 1
                else:
                    seed_para += 1

            train_x = np.array(train_x).astype("float64")
            train_y = np.array([train_y]).astype("float64").T

            print("The dimension of the training data: x {} 和 y {}".format(np.shape(train_x[0:Ntrain]), np.shape(train_y[0:Ntrain])))
            print("The dimension of the testing data: x {} 和 y {}".format(np.shape(train_x[Ntrain:]), np.shape(train_y[Ntrain:])), "\n")

            return train_x[0:Ntrain], train_y[0:Ntrain], train_x[Ntrain:], train_y[Ntrain:]

        Ntrain = 200        # the size of the training dataset
        Ntest = 100         # the size of the testing dataset
        boundary_gap = 0.5  # the gap between two classes
        seed_data = 2       # fixed seed

        # generate a new dataset
        train_x, train_y, test_x, test_y = circle_data_point_generator(Ntrain, Ntest, boundary_gap, seed_data)
        encoding="angle_encoding"
        num_qubit=2
        dimension=2
        train_x=SimpleDataset(dimension).encode(train_x,encoding,num_qubit)
        test_x=SimpleDataset(dimension).encode(test_x,encoding,num_qubit)

        print(train_x[0])
        print(test_x[0])

    ::
    """

    def __init__(self, dimension):
        r""" 构造函数

        Args:
            dimension (int): 编码数据的维度。
        """
        Dataset.__init__(self)
        self.dimension = dimension
        return

    def encode(self, feature, encoding, num_qubits, return_state=True, full_return=False):
        r""" 进行编码

        Args:
            feature (list/ndarray): 编码的特征，每一个分量都是一个 ndarray 的特征向量
            encoding (str): 编码方法
            num_qubits (int): 编码的量子比特数目
            return_state (bool): 是否返回量子态

        Returns:
            tuple: 包含如下元素:
                - quantum_states (ndarray): 量子态，只有 ``full_return==True`` 或者 ``return_state==True`` 的时候会返回
                - quantum_circuits (list): 所有特征编码的电路，只有 ``full_return==False`` 或者 ``return_state==True`` 的时候会返回
                - origin_feature (ndarray): 经过类别过滤之后的所有特征，并未编码为量子态，只有 ``return_state==True`` 的时候会返回
                - feature (ndarray): ``origin_feature`` 经过了补零之后的特征， ``quantum_states`` 就是将 ``feature`` 编码之后的结果。 只有 ``return_state==True`` 的时候会返回
        """

        assert num_qubits > 0
        if encoding in [IQP_ENCODING, COMPLEX_ENTANGLED_ENCODING, REAL_ENTANGLED_ENCODING,
                        LINEAR_ENTANGLED_ENCODING]:
            assert num_qubits > 1

        if type(feature) == np.ndarray:
            self.feature = list(feature)
        elif type(feature) is list:
            self.feature = feature
        else:
            raise Exception("invalid type of feature")

        self.origin_feature = np.array(feature)

        # The first step, calculate ``self.can_describe_dimension``, and judge whether the qubit number is small
        if encoding == AMPLITUDE_ENCODING:  # amplitude encoding, encoding 2^N-dimension feature
            self.can_describe_dimension = 2 ** num_qubits
        # For these three kinds of entanglement encoding: lay these parameters block by block.
        elif encoding == LINEAR_ENTANGLED_ENCODING:
            one_block_param = 2 * num_qubits
            self.can_describe_dimension = math.ceil(self.dimension / one_block_param) * one_block_param

        elif encoding in [REAL_ENTANGLED_ENCODING, IQP_ENCODING, ANGLE_ENCODING]:
            one_block_param = 1 * num_qubits
            self.can_describe_dimension = math.ceil(self.dimension / one_block_param) * one_block_param

        elif encoding in [COMPLEX_ENTANGLED_ENCODING, PAULI_ROTATION_ENCODING]:
            one_block_param = 3 * num_qubits
            self.can_describe_dimension = math.ceil(self.dimension / one_block_param) * one_block_param

        else:
            raise Exception("Invalid encoding methods!")

        if self.can_describe_dimension < self.dimension:
            raise Exception("The qubit number is not enough to encode the features.")

        # The second step: fill the vector to ``can_describe_dimension`` using zero
        for i in range(len(self.feature)):
            self.feature[i] = self.feature[i].reshape(-1).astype(
                np.float64)  # now self.images[i] is a numpy with (new_size*new_size,1) shape
            self.feature[i] = np.append(self.feature[i],
                                        np.array([0.0] * (
                                                self.can_describe_dimension - self.dimension)))  # now self.images[i] is filled to ``self.can_describe_dimension``

        # Step 3: Encode the data, which must be of float64 type(needed in paddle quantum)
        self.quantum_states, self.quantum_circuits = self.data2circuit(
            self.feature, encoding, num_qubits, self.can_describe_dimension, False,  # split_circuit=False
            return_state)

        self.feature = np.array(self.feature)
        self.quantum_states = np.array(self.quantum_states)

        if full_return:
            return self.quantum_states, self.quantum_circuits, self.origin_feature, self.feature
        else:
            if return_state:
                return self.quantum_states
            else:
                return self.quantum_circuits


class Iris(SimpleDataset):
    r""" Iris 数据集

    Attributes:
        quantum_states (ndarray): 经过类别过滤之后的所有特征经编码形成的量子态
        quantum_circuits (list): 所有特征编码的电路
        origin_feature (ndarray): 经过类别过滤之后的所有特征，并未编码为量子态
        feature (ndarray): ``origin_feature`` 经过了补零之后的特征， ``quantum_states`` 就是将 ``feature`` 编码之后的结果
        target (ndarray): 经过类别过滤之后的所有标签
        train_x (paddle.tensor): 从 ``quantum_states`` 中选出的训练集
        test_x (paddle.tensor): 从 ``quantum_states`` 中选出的测试集
        train_circuits (list): 对应于 ``train_x`` 的编码电路
        test_circuits (list): 对应于 ``test_x`` 的编码电路
        origin_train_x (ndarray): 和 ``train_x`` 对应的经典数据
        origin_test_x (ndarray): 和 ``test_x`` 对应的经典数据
        train_y (ndarray): 对应于 ``train_x`` 的标签
        test_y (ndarray): 对应于 ``test_x`` 的标签

    代码示例:

    .. code-block:: python

        from paddle_quantum.dataset import Iris

        test_rate=0.2
        qubit_num=4

        # Get Iris data, select two classes 0,1, and encode the data into four qubits using angle coding and return the quantum state.
        # The proportion of the test set is 0.2.
        iris =Iris (encoding='angle_encoding', num_qubits=qubit_num, test_rate=test_rate,classes=[0,1], return_state=True)

        # Get the features and labels of the classical Iris dataset
        origin_feature=iris.origin_feature # ndarray
        origin_target=iris.target

        # Gets the quantum states and labels of the training and test datasets
        train_x, train_y = iris.train_x, iris.train_y # paddle.tensor, ndarray
        test_x, test_y = iris.test_x, iris.test_y

        testing_data_num=len(test_y)
        training_data_num=len(train_y)
        print(training_data_num)
        print(testing_data_num)
    """

    def __init__(self, encoding, num_qubits, classes, test_rate=0.2, need_relabel=True, return_state=True):
        r""" 构造函数

        Args:
            encoding (str): ``"angle_encoding"`` 表示角度编码，一个量子比特编码一个旋转门； ``"amplitude_encoding"`` 表示振幅编码；
                           ``"pauli_rotation_encoding"`` 表示SU(3)的角度编码；还有 ``"linear_entangled_encoding"`` 、
                           ``"real_entangled_encoding"`` 、 ``"complex_entangled_encoding"`` 三种纠缠编码和 ``"IQP_encoding"`` 编码
            num_qubits (int): 量子比特数目
            classes (list): 用列表给出需要的类别，类别用数字标签表示，不支持传入名字
            test_rate (float): 测试集的占比
            need_relabel (bool): 将原有类别按照顺序重标记为 0、1、2 等新的名字，比如传入 ``[1,2]`` ，重标记之后变为 ``[0,1]`` ，主要用于二分类
            return_state (bool): 是否返回量子态，如果是 ``False`` 返回量子电路
        """

        SimpleDataset.__init__(self, dimension=4)

        # Download data from scikit-learn
        iris = datasets.load_iris()
        self.dimension = 4  # dimension of Iris dataset
        feature, self.target = self.filter_class(iris.data, iris.target, classes, -1,
                                                 need_relabel)  # here -1 means all data
        self.target = np.array(self.target)

        # Start to encode
        self.quantum_states, self.quantum_circuits, self.origin_feature, self.feature = \
            self.encode(feature, encoding, num_qubits, return_state, True)

        # Divide training and testing dataset
        seed = int(time.time())
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.quantum_states, self.target, test_size=test_rate,
                             random_state=seed)

        self.train_circuits, self.test_circuits, temp1, temp2 = \
            train_test_split(self.quantum_circuits, self.target, test_size=test_rate,
                             random_state=seed)

        self.origin_train_x, self.origin_test_x, temp1, temp2 = \
            train_test_split(self.origin_feature, self.target, test_size=test_rate,
                             random_state=seed)
        if return_state:
            self.train_x = paddle.to_tensor(self.train_x)
            self.test_x = paddle.to_tensor(self.test_x)


class BreastCancer(SimpleDataset):
    r"""BreastCancer 数据集，569 组数据 30 维，只有两类。

    Attributes:
        quantum_states (ndarray): 经过类别过滤之后的所有特征经编码形成的量子态
        quantum_circuits (list): 所有特征编码的电路
        origin_feature (ndarray): 经过类别过滤之后的所有特征，并未编码为量子态
        feature (ndarray): ``origin_feature`` 经过了补零之后的特征， ``quantum_states`` 就是将 ``feature`` 编码之后的结果
        target (ndarray): 经过类别过滤之后的所有标签
        train_x (paddle.tensor): 从 ``quantum_states`` 中选出的训练集
        test_x (paddle.tensor): 从 ``quantum_states`` 中选出的测试集
        train_circuits (list): 对应于 ``train_x`` 的编码电路
        test_circuits (list): 对应于 ``test_x`` 的编码电路
        origin_train_x (ndarray): 和 ``train_x`` 对应的经典数据
        origin_test_x (ndarray): 和 ``test_x`` 对应的经典数据
        train_y (ndarray): 对应于 ``train_x`` 的标签
        test_y (ndarray): 对应于 ``test_x`` 的标签

    代码示例:

    .. code-block:: python

        from paddle_quantum.dataset import BreastCancer

        test_rate = 0.2
        qubit_num = 4

        # Get BreastCancer data, select two classes 0,1, and encode the data into four qubits using angle coding and return the quantum state.
        # The proportion of the test set is 0.2.
        breast_cancer =BreastCancer(encoding='angle_encoding', num_qubits=qubit_num, test_rate=test_rate, return_state=True)

        # Get the features and labels of the classical BreastCancer dataset
        origin_feature=breast_cancer.origin_feature # ndarray
        origin_target=breast_cancer.target

        # Gets the quantum states and labels of the training and test datasets
        train_x, train_y = breast_cancer.train_x, breast_cancer.train_y # paddle.tensor, ndarray
        test_x, test_y = breast_cancer.test_x, breast_cancer.test_y

        testing_data_num=len(test_y)
        training_data_num=len(train_y)
        print(training_data_num)
        print(testing_data_num)
    """

    def __init__(self, encoding, num_qubits, test_rate=0.2, return_state=True):
        r"""构造函数

        Args:
            encoding (str): ``"angle_encoding"`` 表示角度编码，一个量子比特编码一个旋转门； ``"amplitude_encoding"`` 表示振幅编码；
                           ``"pauli_rotation_encoding"`` 表示SU(3)的角度编码；还有 ``"linear_entangled_encoding"`` 、
                           ``"real_entangled_encoding"`` 、 ``"complex_entangled_encoding"`` 三种纠缠编码和 ``"IQP_encoding"`` 编码
            num_qubits (int): 量子比特数目
            test_rate (float): 测试集的占比
            return_state (bool): 是否返回量子态，如果是 ``False`` 返回量子电路
        """
        SimpleDataset.__init__(self, dimension=30)  # The dimension is 30
        self.dimension = 30

        # Download data from scikit-learn
        breast_cancer = datasets.load_breast_cancer()
        feature = breast_cancer["data"]
        self.target = breast_cancer["target"]

        self.target = np.array(self.target)

        # Start to encode
        self.quantum_states, self.quantum_circuits, self.origin_feature, self.feature = \
            self.encode(feature, encoding, num_qubits, return_state, True)

        # Divide training and testing dataset
        seed = int(time.time())
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.quantum_states, self.target, test_size=test_rate,
                             random_state=seed)

        self.train_circuits, self.test_circuits, temp1, temp2 = \
            train_test_split(self.quantum_circuits, self.target, test_size=test_rate,
                             random_state=seed)

        self.origin_train_x, self.origin_test_x, temp1, temp2 = \
            train_test_split(self.origin_feature, self.target, test_size=test_rate,
                             random_state=seed)
        if return_state:
            self.train_x = paddle.to_tensor(self.train_x)
            self.test_x = paddle.to_tensor(self.test_x)
