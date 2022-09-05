# !/usr/bin/env python3
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


r"""
The source file of the dataset.
"""

import random
import math
from typing import Tuple, Union, Optional
import numpy as np
import paddle
import paddle.vision.transforms as transform
from sklearn.model_selection import train_test_split
from sklearn import datasets
from paddle_quantum.gate import RY, RZ, U3, CNOT, IQPEncoding, AmplitudeEncoding
from .base import get_dtype
from .intrinsic import _get_float_dtype

__all__ = [
    "Dataset",
    "VisionDataset",
    "SimpleDataset",
    "MNIST",
    "FashionMNIST",
    "Iris",
    "BreastCancer"
]

# data modes
import paddle_quantum.gate

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
    """
    xx = np.abs(x)
    if xx.max() > 0:
        return x * np.pi / xx.max()
    return x


def _normalize_image(x):
    r"""normalize image vector ``x`` and the maximum will be pi. This is an internal function.
    """
    return x * np.pi / 256


def _crop(images, border):
    r"""crop ``images`` according to ``border``. This is an internal function.
    """
    new_images = []
    for i in range(len(images)):
        size = int(np.sqrt(len(images[i])))
        temp_image = images[i].reshape((size, size))
        temp_image = temp_image[border[0]:border[1], border[0]:border[1]]
        new_images.append(temp_image.flatten())
    return new_images


class Dataset(object):
    r"""Base class for all datasets, integrating multiple quantum encoding methods.
    """

    def __init__(self):
        return

    def data2circuit(
            self, classical_data: list, encoding: str, num_qubits: int, can_describe_dimension: int, split_circuit: bool,
            return_state: bool, is_image: Optional[bool] = False
    ) -> list:
        r"""Encode the input ``classical data`` into quantum states using ``encoding``, where the classical data is truncated or filled with zero.

        Args:
            classical_data: vectors needed to encode, which have been already truncated or filled with zero to the length ``can_describe_dimension``
                For example, amplitude encoding can describe ``2 ** n`` dimension vectors.
            encoding: The encoding method.
            num_qubits: The number of qubits.
            can_describe_dimension: The dimension which the circuit can describe by ``encoding``. 
            split_circuit: Whether to split the circuit.
            return_state: Whether to return quantum state.
            is_image:Whether it is a picture, if it is a picture, the normalization method is not quite the same. Defaults to ``False``.
        
        Raises:
            Exception: Not support to return circuit in amplitude encoding.

        Returns:
            If ``return_state == True``, return encoded quantum state, otherwise return encoding circuits.
        """
        quantum_states = classical_data.copy()
        quantum_circuits = classical_data.copy()
        float_dtype = _get_float_dtype(get_dtype())
        if encoding == AMPLITUDE_ENCODING:
            # Not support to return circuit in amplitude encoding
            if return_state is False or split_circuit is True:
                raise Exception("Not support to return circuit in amplitude encoding")
            for i in range(len(classical_data)):
                x = paddle.to_tensor(_normalize(classical_data[i]), dtype=float_dtype)
                if is_image:
                    x = paddle.to_tensor(_normalize_image(classical_data[i]), dtype=float_dtype)
                circuit = AmplitudeEncoding(qubits_idx='full', num_qubits=num_qubits)
                state = circuit(x)
                quantum_states[i] = state.data.numpy()

        elif encoding == ANGLE_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 1 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]), dtype=float_dtype)
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]), dtype=float_dtype)
                param = paddle.reshape(param, (depth, num_qubits, 1))
                which_qubits = list(range(num_qubits))
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = paddle_quantum.ansatz.Sequential()
                        for k, q in enumerate(which_qubits):
                            circuit.append(RY(qubits_idx=q, param=param[repeat][k][0]))
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = paddle_quantum.ansatz.Sequential()
                    for repeat in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.append(RY(qubits_idx=q, param=param[repeat][k][0]))
                    state_out = circuit(paddle_quantum.state.zero_state(num_qubits))
                    quantum_states[i] = state_out.data.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == IQP_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 1 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]), dtype=float_dtype)
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]), dtype=float_dtype)
                param = paddle.reshape(param, (depth, num_qubits))
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = paddle_quantum.ansatz.Sequential()
                        s = []
                        for k in range(num_qubits - 1):
                            s.append([k, k + 1])
                        # r 是 U 重复的次数
                        r = 1
                        circuit.append(IQPEncoding(feature=param[repeat], num_repeat=r, qubits_idx=s))
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = paddle_quantum.ansatz.Sequential()
                    for repeat in range(depth):
                        s = []
                        for k in range(num_qubits - 1):
                            s.append([k, k + 1])
                        # r 是 U 重复的次数
                        r = 1
                        circuit.append(IQPEncoding(feature=param[repeat], num_repeat=r, qubits_idx=s))
                    state_out = circuit(paddle_quantum.state.zero_state(num_qubits))
                    quantum_states[i] = state_out.data.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == PAULI_ROTATION_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 3 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]), dtype=float_dtype)
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]), dtype=float_dtype)
                param = paddle.reshape(param, (depth, num_qubits, 3))
                which_qubits = list(range(num_qubits))
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = paddle_quantum.ansatz.Sequential()
                        for k, q in enumerate(which_qubits):
                            circuit.append(RY(q, param=param[repeat][k][0]))
                            circuit.append(RZ(q, param=param[repeat][k][1]))
                            circuit.append(RY(q, param=param[repeat][k][2]))
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = paddle_quantum.ansatz.Sequential()
                    for repeat in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.append(RY(q, param=param[repeat][k][0]))
                            circuit.append(RZ(q, param=param[repeat][k][1]))
                            circuit.append(RY(q, param=param[repeat][k][2]))
                    state_out = circuit(paddle_quantum.state.zero_state(num_qubits))
                    quantum_states[i] = state_out.data.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == LINEAR_ENTANGLED_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 2 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]), dtype=float_dtype)
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]), dtype=float_dtype)
                param = paddle.reshape(param, (depth, num_qubits, 2))
                which_qubits = [k for k in range(num_qubits)]
                if split_circuit:
                    quantum_circuits[i] = []
                    for j in range(depth):
                        circuit = paddle_quantum.ansatz.Sequential()
                        for k, q in enumerate(which_qubits):
                            circuit.append(RY(q, param=param[j][k][0]))
                        for k in range(len(which_qubits) - 1):
                            circuit.append(CNOT(qubits_idx=[which_qubits[k], which_qubits[k + 1]]))
                        for k, q in enumerate(which_qubits):
                            circuit.append(RZ(q, param=param[j][k][1]))
                        for k in range(len(which_qubits) - 1):
                            circuit.append(CNOT(qubits_idx=[which_qubits[k + 1], which_qubits[k]]))
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = paddle_quantum.ansatz.Sequential()
                    for j in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.append(RY(q, param=param[j][k][0]))
                        for k in range(len(which_qubits) - 1):
                            circuit.append(CNOT(qubits_idx=[which_qubits[k], which_qubits[k + 1]]))
                        for k, q in enumerate(which_qubits):
                            circuit.append(RZ(q, param=param[j][k][1]))
                        for k in range(len(which_qubits) - 1):
                            circuit.append(CNOT(qubits_idx=[which_qubits[k + 1], which_qubits[k]]))
                    state_out = circuit(paddle_quantum.state.zero_state(num_qubits))
                    quantum_states[i] = state_out.data.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == REAL_ENTANGLED_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 1 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]), dtype=float_dtype)
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]), dtype=float_dtype)
                param = paddle.reshape(param, (depth, num_qubits, 1))
                which_qubits = [k for k in range(num_qubits)]
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = paddle_quantum.ansatz.Sequential()
                        for k, q in enumerate(which_qubits):
                            circuit.append(RY(q, param=param[repeat][k][0]))
                        for k in range(len(which_qubits) - 1):
                            circuit.append(CNOT(qubits_idx=[which_qubits[k], which_qubits[k + 1]]))
                        circuit.append(CNOT(qubits_idx=[which_qubits[-1], which_qubits[0]]))
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = paddle_quantum.ansatz.Sequential()
                    for repeat in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.append(RY(q, param=param[repeat][k][0]))
                        for k in range(len(which_qubits) - 1):
                            circuit.append(CNOT(qubits_idx=[which_qubits[k], which_qubits[k + 1]]))
                        circuit.append(CNOT(qubits_idx=[which_qubits[-1], which_qubits[0]]))
                    state_out = circuit(paddle_quantum.state.zero_state(num_qubits))
                    quantum_states[i] = state_out.data.numpy()
                    quantum_circuits[i] = [circuit]

        elif encoding == COMPLEX_ENTANGLED_ENCODING:
            for i in range(len(classical_data)):
                one_block_param = 3 * num_qubits
                depth = int(can_describe_dimension / one_block_param)
                param = paddle.to_tensor(_normalize(classical_data[i]), dtype=float_dtype)
                if is_image:
                    param = paddle.to_tensor(_normalize_image(classical_data[i]), dtype=float_dtype)
                param = paddle.reshape(param, (depth, num_qubits, 3))
                which_qubits = [k for k in range(num_qubits)]
                if split_circuit:
                    quantum_circuits[i] = []
                    for repeat in range(depth):
                        circuit = paddle_quantum.ansatz.Sequential()
                        for k, q in enumerate(which_qubits):
                            circuit.append(U3(q, param=param[repeat][k]))
                            circuit.u3(param[repeat][k][0], param[repeat][k][1], param[repeat][k][2], q)
                        for k in range(len(which_qubits) - 1):
                            circuit.append(CNOT(qubits_idx=[which_qubits[k], which_qubits[k + 1]]))
                        circuit.append(CNOT(qubits_idx=[which_qubits[-1], which_qubits[0]]))
                        quantum_circuits[i].append(circuit)
                else:
                    circuit = paddle_quantum.ansatz.Sequential()
                    for repeat in range(depth):
                        for k, q in enumerate(which_qubits):
                            circuit.append(U3(q, param=param[repeat][k]))
                        for k in range(len(which_qubits) - 1):
                            circuit.append(CNOT(qubits_idx=[which_qubits[k], which_qubits[k + 1]]))
                        circuit.append(CNOT(qubits_idx=[which_qubits[-1], which_qubits[0]]))
                    state_out = circuit(paddle_quantum.state.zero_state(num_qubits))
                    quantum_states[i] = state_out.data.numpy()
                    quantum_circuits[i] = [circuit]
        return quantum_states, quantum_circuits

    def filter_class(self, x: Union[list, np.ndarray], y: Union[list, np.ndarray], classes: list,
                     data_num: int, need_relabel: bool, seed: Optional[int] = 0) -> Tuple[list]:
        r"""Select ``data_num`` samples from ``x`` , ``y``, whose label is in ``classes``.

        Args:
            x: Training features.
            y: Training labels.
            classes: Classes needed to select.
            data_num: The number of data needed to select.
            need_relabel: Whether we need to relabel the labels to 0,1,2 for binary classification. For example ``[1,2]`` will be relabeled to ``[0,1]``. 
            seed: Random seed. Defaults to ``0``.

        Returns:
            contains elements

            - new_x: selected features.
            - new_y: selected labels corresponded to ``new_x``.
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
    r""" ``VisionDataset`` is the base class of all image datasets. By inheriting ``VisionDataset``, users can easily generate their own quantum data.

    Args:
        figure_size: The size of the figure.
    """

    def __init__(self, figure_size: int):
        Dataset.__init__(self)
        self.figure_size = figure_size
        return

    # The encode function only needs to import images to form one-dimensional vector features.
    # The pre-processing of images (except dimensionality reduction) is completed before the import of features
    def encode(self, feature: Union[list, np.ndarray], encoding: str, num_qubits: int, split_circuit: Optional[bool] = False,
               downscaling_method: Optional[str] = DOWNSCALINGMETHOD_RESIZE, target_dimension: Optional[int] = -1, 
               return_state: Optional[bool] = True, full_return: Optional[bool] = False) -> Tuple[paddle.Tensor, list, np.ndarray, np.ndarray]:
        r"""Encode ``feature`` into ``num_qubits`` qubits using ``encoding`` after downscaling to ``target_dimension``. ``feature`` is one-dimension image vectors.

        Args:
            feature: One-dimension image vectors which can be list or ndarray.
            encoding: ``angle_encoding`` denotes angle encoding, and one qubit encodes one number with a Ry gate. ``amplitude_encoding`` denotes amplitude encoding;
                      ``pauli_rotation_encoding`` denotes using SU(3) rotation gate. ``linear_entanglement_encoding``, ``real_entanglement_encoding`` , ``complex_entanglement_encoding`` 
                      and ``IQP_encoding`` encoding methods.
            num_qubits: Qubit number.
            split_circuit: Whether to split the circuits. If true, every layer of the encoding circuit will be split into a list. Defaults to ``False``.
            downscaling_method: Including ``PCA`` and ``resize``. Defaults to ``resize``.
            target_dimension: The dimension after downscaling. ``target_dimension`` is not allowed to surpass the figure size. Defaults to ``-1``.
            return_state: Whether to return quantum states. If it is ``False``, return quantum circuits. Defaults to ``True``.
            full_return: Whether to return ``quantum_image_states``, ``quantum_image_circuits``, ``original_images`` and ``classical_image_vectors``. Defaults to ``False``.

        Raises:
            Exception: PCA dimension should be less than figure size.
            Exception: Resize dimension should be a square.
            Exception: Downscaling methods can only be resize and PCA.
            Exception: Invalid encoding methods

        Returns:
            contain these elements

            - quantum_image_states: Quantum states, only ``full_return==True`` or ``return_state==True`` will return.
            - quantum_image_circuits: A list of circuits generating quantum states, only ``full_return==True`` or ``return_state==True`` will return.
            - original_images: One-dimension original vectors without any processing, only ``return_state==True`` will return.
            - classical_image_vectors: One-dimension original vectors after filling with zero, which are encoded to quantum states. only ``return_state==True`` will return.
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
                if int(np.sqrt(target_dimension)) ** 2 != target_dimension:  # not a square
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
                new_size = int(np.sqrt(self.dimension))
                cur_image = transform.resize(cur_image.reshape((self.figure_size, self.figure_size)),
                                             (new_size, new_size))
                self.classical_image_vectors[i] = cur_image.reshape(-1)  # now it is one-dimension

                if self.can_describe_dimension < len(self.classical_image_vectors[i]):
                    self.classical_image_vectors[i] = self.classical_image_vectors[i][:self.can_describe_dimension]
                else:
                    self.classical_image_vectors[i] = np.append(
                        self.classical_image_vectors[i],
                        np.array([0.0] * (self.can_describe_dimension - len(self.classical_image_vectors[i])))
                    )

        elif downscaling_method == DOWNSCALINGMETHOD_PCA:
            for i in range(len(self.classical_image_vectors)):
                _, s, _ = np.linalg.svd(self.classical_image_vectors[i].reshape((self.figure_size, self.figure_size)))
                s = s[:self.dimension]
                if self.can_describe_dimension > self.dimension:
                    self.classical_image_vectors[i] = np.append(s, np.array(
                        [0.0] * (self.can_describe_dimension - self.dimension)))
                else:
                    self.classical_image_vectors[i] = s[:self.can_describe_dimension]

        # Step 4: Encode the data
        self.quantum_image_states, self.quantum_image_circuits = self.data2circuit(
            self.classical_image_vectors, encoding, num_qubits, self.can_describe_dimension, split_circuit,
            return_state, is_image=True)
        self.classical_image_vectors = np.array(self.classical_image_vectors)
        if return_state:
            self.quantum_image_states = paddle.to_tensor(np.array(self.quantum_image_states))  # transfer to tensor

        if full_return:
            return (
                self.quantum_image_states, self.quantum_image_circuits,
                self.original_images, self.classical_image_vectors
            )
        if return_state:
            return self.quantum_image_states
        return self.quantum_image_circuits


class MNIST(VisionDataset):
    r"""MNIST quantum dataset. It inherits ``VisionDataset``.

    Args:
        mode: Data mode including ``train`` and ``test``.
        encoding: ``angle_encoding`` denotes angle encoding, and one qubit encodes one number with a Ry gate. ``amplitude_encoding`` denotes amplitude encoding;
                    ``pauli_rotation_encoding`` denotes using SU(3) rotation gate. ``linear_entanglement_encoding``, ``real_entanglement_encoding`` , ``complex_entanglement_encoding`` 
                    and ``IQP_encoding`` encoding methods.
        num_qubits: Qubit number.
        classes: Classes needed to classify, categories are indicated by numeric labels.
        data_num: Data number returned. Defaults to ``-1``.
        split_circuit: Whether to split the circuits. If True, every layer of the encoding circuit will be split into a list. Defaults to ``False``.
        downscaling_method: Including ``PCA`` and ``resize``. Defaults to ``resize``.
        target_dimension:  The dimension after downscaling, which is not allowed to surpass the figure size. Defaults to ``-1``.
        need_cropping: Whether needed to crop, If ``True``, ``image[0:27][0:27]`` will be cropped to ``image[4:24][4:24]``. Defaults to ``True``.
        need_relabel: Whether we need to relabel the labels to 0,1,2… for binary classification.For example [1,2] will be relabeled to [0,1] Defaults to ``True``.
        return_state: Whether to return quantum states. Defaults to ``True``.
        seed: Select random seed. Defaults to ``0``.

    Raises:
        Exception: Data mode can only be train and test.

    """

    def __init__(
            self, mode: str, encoding: str, num_qubits: int, classes : list, data_num: Optional[int]=-1, 
            split_circuit: Optional[bool]=False, downscaling_method: Optional[str] =DOWNSCALINGMETHOD_RESIZE, 
            target_dimension: Optional[int] = -1, need_cropping: Optional[bool] = True,
            need_relabel: Optional[bool] = True, return_state: Optional[bool] =True, seed: Optional[int]=0
    ) -> None:
        VisionDataset.__init__(self, 28)

        if need_cropping:
            self.figure_size = 20

        # Download data from paddlepaddle
        if mode == DATAMODE_TRAIN:
            train_dataset = paddle.vision.datasets.MNIST(mode='train')
            feature, self.labels = self.filter_class(
                train_dataset.images, train_dataset.labels,
                classes=classes, data_num=data_num, need_relabel=need_relabel, seed=seed
            )
            if need_cropping:
                feature = _crop(feature, [4, 24])

        elif mode == DATAMODE_TEST:
            test_dataset = paddle.vision.datasets.MNIST(mode='test')
            # test_dataset.images is now a list of (784,1) shape
            feature, self.labels = self.filter_class(
                test_dataset.images, test_dataset.labels,
                classes=classes, data_num=data_num, need_relabel=need_relabel, seed=seed
            )
            if need_cropping:
                feature = _crop(feature, [4, 24])

        else:
            raise Exception("data mode can only be train and test.")

        # Start to encode
        self.quantum_image_states, self.quantum_image_circuits, self.original_images, self.classical_image_vectors = \
            self.encode(
                feature, encoding, num_qubits, split_circuit,
                downscaling_method, target_dimension, return_state, True
            )
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.quantum_image_states)


class FashionMNIST(VisionDataset):
    r"""FashionMNIST quantum dataset. It inherits ``VisionDataset``.

    Args:
        mode: Data mode including ``train`` and ``test``.
        encoding: ``angle_encoding`` denotes angle encoding, and one qubit encodes one number with a Ry gate. ``amplitude_encoding`` denotes amplitude encoding;
                    ``pauli_rotation_encoding`` denotes using SU(3) rotation gate. ``linear_entanglement_encoding``, ``real_entanglement_encoding`` , ``complex_entanglement_encoding`` 
                    and ``IQP_encoding`` encoding methods.
        num_qubits: Qubit number.
        classes: Classes needed to classify, categories are indicated by numeric labels.
        data_num: Data number returned. Defaults to ``-1``.
        split_circuit: Whether to split the circuits. If True, every layer of the encoding circuit will be split into a list. Defaults to ``False``.
        downscaling_method: Including ``PCA`` and ``resize``. Defaults to ``resize``.
        target_dimension:  The dimension after downscaling, which is not allowed to surpass the figure size. Defaults to ``-1``.
        need_cropping: Whether needed to crop, If ``True``, ``image[0:27][0:27]`` will be cropped to ``image[4:24][4:24]``. Defaults to ``True``.
        need_relabel: Whether we need to relabel the labels to 0,1,2… for binary classification.For example [1,2] will be relabeled to [0,1] Defaults to ``True``.
        return_state: Whether to return quantum states. Defaults to ``True``.
        seed: Select random seed. Defaults to ``0``.

    Raises:
        Exception: Data mode can only be train and test.
    """

    def __init__(
            self, mode: str, encoding: str, num_qubits: int, classes: list, data_num: Optional[int] = -1, 
            split_circuit: Optional[bool] = False, downscaling_method: Optional[str] = DOWNSCALINGMETHOD_RESIZE, 
            target_dimension: Optional[int] = -1, need_relabel: Optional[bool] = True,
            return_state: Optional[bool] = True, seed: Optional[int] = 0) -> None:

        r"""Constructor

        """
        VisionDataset.__init__(self, 28)

        # Download data from paddlepaddle
        if mode == DATAMODE_TRAIN:
            train_dataset = paddle.vision.datasets.FashionMNIST(mode='train')
            feature, self.labels = self.filter_class(
                train_dataset.images, train_dataset.labels,
                classes=classes,data_num=data_num, need_relabel=need_relabel, seed=seed
            )

        elif mode == DATAMODE_TEST:
            test_dataset = paddle.vision.datasets.FashionMNIST(mode='test')
            # test_dataset.images is now a list of (784,1) shape
            feature, self.labels = self.filter_class(
                test_dataset.images, test_dataset.labels,
                classes=classes,data_num=data_num, need_relabel=need_relabel, seed=seed
            )

        else:
            raise Exception("data mode can only be train and test.")

        # Start to encode
        self.quantum_image_states, self.quantum_image_circuits, self.original_images, self.classical_image_vectors = \
            self.encode(
                feature, encoding, num_qubits, split_circuit, downscaling_method, target_dimension,
                return_state, True
            )
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.quantum_image_states)


class SimpleDataset(Dataset):
    r"""For simple dataset that does not require dimension reduction. You can inherit ``SimpleDataset`` to generate quantum states from your classical datasets.

    Args:
        dimension: Dimension of encoding data.

    """

    def __init__(self, dimension: int):
        Dataset.__init__(self)
        self.dimension = dimension
        return

    def encode(self, feature: Union[list, np.ndarray], encoding: str, num_qubits: int, 
               return_state: Optional[bool] = True, full_return: Optional[bool] = False) -> Tuple[np.ndarray, list, np.ndarray, np.ndarray]:
        r"""Encode ``feature`` with ``num_qubits`` qubits by ``encoding``.

        Args:
            feature: Features needed to encode.
            encoding: Encoding methods.
            num_qubits: Qubit number.
            return_state: Whether to return quantum states. Defaults to ``True``.
            full_return: Whether to return quantum_states, quantum_circuits, origin_feature and feature. Defaults to ``False``.

        Raises:
            Exception: Invalid type of feature.
            Exception: Invalid encoding methods.
            Exception: The qubit number is not enough to encode the features.

        Returns:
            contain these element

            - quantum_states: Quantum states, only ``full_return==True`` or ``return_state==True`` will return;
            - quantum_circuits: A list of circuits generating quantum states, only ``full_return==True`` or ``return_state==True`` will return;
            - origin_feature: One-dimension original vectors without any processing, only ``return_state==True`` will return
            - feature: One-dimension original vectors after filling with zero, which are encoded to quantum states. only ``return_state==True`` will return.
        """
        assert num_qubits > 0
        encoding_list = [
            IQP_ENCODING, COMPLEX_ENTANGLED_ENCODING,
            REAL_ENTANGLED_ENCODING, LINEAR_ENTANGLED_ENCODING
        ]
        if encoding in encoding_list:
            assert num_qubits > 1

        if isinstance(feature, np.ndarray):
            self.feature = list(feature)
        elif isinstance(feature, list):
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
            self.feature[i] = self.feature[i].reshape(-1) # now self.images[i] is a numpy with (new_size*new_size,1) shape
            self.feature[i] = np.append(
                self.feature[i],
                np.array([0.0] * (self.can_describe_dimension - self.dimension))
            )  # now self.images[i] is filled to ``self.can_describe_dimension``

        # Step 3: Encode the data
        self.quantum_states, self.quantum_circuits = self.data2circuit(
            self.feature, encoding, num_qubits, self.can_describe_dimension, False,  # split_circuit=False
            return_state
        )

        self.feature = np.array(self.feature)
        self.quantum_states = np.array(self.quantum_states)

        if full_return:
            return self.quantum_states, self.quantum_circuits, self.origin_feature, self.feature
        if return_state:
            return self.quantum_states
        return self.quantum_circuits


class Iris(SimpleDataset):
    r"""Iris dataset

    Args:
        encoding: ``angle_encoding`` denotes angle encoding, and one qubit encodes one number with a Ry gate. ``amplitude_encoding`` denotes amplitude encoding;
                    ``pauli_rotation_encoding`` denotes using SU(3) rotation gate. ``linear_entanglement_encoding``, ``real_entanglement_encoding`` , ``complex_entanglement_encoding`` 
                    and ``IQP_encoding`` encoding methods.
        num_qubits: Qubit number.
        classes: Classes needed to classify, categories are indicated by numeric labels.
        test_rate: The proportion of the testing dataset. Defaults to ``0.2``.
        need_relabel: Whether we need to relabel the labels to 0,1,2… for binary classification.For example [1,2] will be relabeled to [0,1]. Defaults to ``True``.
        return_state: Whether to return quantum states. Defaults to ``True``.
        seed: Select random seed. Defaults to ``0``.
    
    """

    def __init__(self, encoding: str, num_qubits: int, classes: list, test_rate: Optional[float] = 0.2, 
                 need_relabel: Optional[bool] = True, return_state: Optional[bool] = True, seed: Optional[int] = 0) -> None:
        SimpleDataset.__init__(self, dimension=4)

        # Download data from scikit-learn
        iris = datasets.load_iris()
        self.dimension = 4  # dimension of Iris dataset
        feature, self.target = self.filter_class(
            iris.data, iris.target, classes, -1,need_relabel
        )  # here -1 means all data
        self.target = np.array(self.target)

        # Start to encode
        self.quantum_states, self.quantum_circuits, self.origin_feature, self.feature = \
            self.encode(feature, encoding, num_qubits, return_state, True)

        # Divide training and testing dataset
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
    r"""BreastCancer quantum dataset.

    Args:
        encoding: ``angle_encoding`` denotes angle encoding, and one qubit encodes one number with a Ry gate. ``amplitude_encoding`` denotes amplitude encoding;
                    ``pauli_rotation_encoding`` denotes using SU(3) rotation gate. ``linear_entanglement_encoding``, ``real_entanglement_encoding`` , ``complex_entanglement_encoding`` 
                    and ``IQP_encoding`` encoding methods.
        num_qubits: Qubit number.
        test_rate:The proportion of the testing dataset. Defaults to ``0.2``.
        return_state: Whether to return quantum states. Defaults to ``True``.
        seed: Select random seed. Defaults to ``0``.

    """

    def __init__(self, encoding: str, num_qubits: int, test_rate: Optional[float] = 0.2, 
                 return_state: Optional[bool] = True, seed: Optional[int] = 0) -> None:
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
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(self.quantum_states, self.target, test_size=test_rate, random_state=seed)

        self.train_circuits, self.test_circuits, temp1, temp2 = \
            train_test_split(self.quantum_circuits, self.target, test_size=test_rate, random_state=seed)

        self.origin_train_x, self.origin_test_x, temp1, temp2 = \
            train_test_split(self.origin_feature, self.target, test_size=test_rate, random_state=seed)
        if return_state:
            self.train_x = paddle.to_tensor(self.train_x)
            self.test_x = paddle.to_tensor(self.test_x)
