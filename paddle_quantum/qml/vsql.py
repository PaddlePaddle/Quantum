# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
The VSQL model.
"""

import random
from typing import Optional, List, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.vision.datasets import MNIST

import paddle_quantum as pq
from paddle_quantum.ansatz import Circuit


def norm_image(images: List[np.ndarray], num_qubits: int) -> List[paddle.Tensor]:
    r"""
    Normalize the input images. Flatten them and make them to normalized vectors.

    Args:
        images: The input images.
        num_qubits: The number of qubits, which decides the dimension of the vector.

    Returns:
        Return the normalized vectors, which is the list of paddle's tensors.
    """
    # pad and normalize the image
    _images = []
    for image in images:
        image = image.flatten()
        if image.size < 2 ** num_qubits:
            _images.append(np.pad(image, pad_width=(0, 2 ** num_qubits - image.size)))
        else:
            _images.append(image[:2 ** num_qubits])
    return [paddle.to_tensor(image / np.linalg.norm(image), dtype=pq.get_dtype()) for image in _images]


def data_loading(
        num_qubits: int, mode: str, classes: list, num_data: Optional[int] = None
) -> Tuple[List[np.ndarray], List[int]]:
    r"""
    Loading the MNIST dataset, which only contains the specified data.

    Args:
        num_qubits: The number of qubits, which determines the dimension of the normalized vector.
        mode: Specifies the loaded dataset: ``'train'`` | ``'test'`` .

            - ``'train'`` : Load the training dataset.
            - ``'test'`` : Load the test dataset.

        classes: The labels of the data which will be loaded.
        num_data: The number of data to be loaded. Defaults to ``None``, which means loading all data.

    Returns:
       Return the loaded dataset, which is ``(images, labels)`` .
    """
    data = MNIST(mode=mode, backend='cv2')
    filtered_data = [item for item in data if item[1].item() in classes]
    random.shuffle(filtered_data)
    if num_data is None:
        num_data = len(filtered_data)
    images = [filtered_data[idx][0] for idx in range(min(len(filtered_data), num_data))]
    labels = [filtered_data[idx][1] for idx in range(min(len(filtered_data), num_data))]
    images = norm_image(images, num_qubits=num_qubits)
    labels = [label.item() for label in labels]
    return images, labels


def _slide_circuit(cir: pq.ansatz.Circuit, distance: int):
    # slide to get the local feature
    for sublayer in cir.sublayers():
        qubits_idx = np.array(sublayer.qubits_idx)
        qubits_idx = qubits_idx + distance
        sublayer.qubits_idx = qubits_idx.tolist()


def observable(start_idx: int, num_shadow: int) -> pq.Hamiltonian:
    r"""
    Generate the observable to measure the quantum states.

    Args:
        start_idx: The start index of the qubits.
        num_shadow: The number of qubits which the shadow circuit contains.

    Returns:
        Return the generated observable.
    """
    # construct the observable to get te output of the circuit
    pauli_str = ','.join(f'x{str(i)}' for i in range(start_idx, start_idx + num_shadow))
    return pq.Hamiltonian([[1.0, pauli_str]])


class VSQL(paddle.nn.Layer):
    r"""
    The class of the variational shadow quantum learning (VSQL).

    The details can be referred to https://ojs.aaai.org/index.php/AAAI/article/view/17016 .

    Args:
        num_qubits: The number of qubits which the quantum circuit contains.
        num_shadow: The number of qubits which the shadow circuit contains.
        num_classes: The number of class which the modell will classify.
        depth: The depth of the quantum circuit. Defaults to ``1`` .
    """
    def __init__(self, num_qubits: int, num_shadow: int, num_classes: int, depth: int = 1):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_shadow = num_shadow
        self.depth = depth
        cir = Circuit(num_qubits)
        for idx in range(num_shadow):
            cir.rx(qubits_idx=idx)
            cir.ry(qubits_idx=idx)
            cir.rx(qubits_idx=idx)
        for _ in range(depth):
            for idx in range(num_shadow - 1):
                cir.cnot([idx, idx + 1])
            cir.cnot([num_shadow - 1, 0])
            for idx in range(num_shadow):
                cir.ry(qubits_idx=idx)
        self.cir = cir
        self.fc = paddle.nn.Linear(
            num_qubits - num_shadow + 1, num_classes,
            weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal()),
            bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal())
        )

    def forward(self, batch_input: List[paddle.Tensor]) -> paddle.Tensor:
        r"""
        The forward function.

        Args:
            batch_input: The input of the model. It's shape is :math:`(\text{batch_size}, 2^{\text{num_qubits}})` .

        Returns:
            Return the output of the model. It's shape is :math:`(\text{batch_size}, \text{num_classes})` .
        """
        batch_feature = []
        for input in batch_input:
            _state = pq.State(input)
            feature = []
            for idx_start in range(self.num_qubits - self.num_shadow + 1):
                ob = observable(idx_start, num_shadow=self.num_shadow)
                _slide_circuit(cir=self.cir, distance=1 if idx_start != 0 else 0)
                expec_val_func = pq.loss.ExpecVal(ob)
                out_state = self.cir(_state)
                expec_val = expec_val_func(out_state)
                feature.append(expec_val)
            # slide the circuit to the initial position
            _slide_circuit(self.cir, -idx_start)
            feature = paddle.concat(feature)
            batch_feature.append(feature)
        batch_feature = paddle.stack(batch_feature)
        return self.fc(batch_feature)


def train(
        num_qubits: int, num_shadow: int, depth: int = 1,
        batch_size: int = 16, epoch: int = 10, learning_rate: float = 0.01,
        classes: Optional[list] = None, num_train: Optional[int] = None, num_test: Optional[int] = None
) -> None:
    """
    The function of training the VSQL model.

    Args:
        num_qubits: The number of qubits which the quantum circuit contains.
        num_shadow: The number of qubits which the shadow circuit contains.
        depth: The depth of the quantum circuit. Defaults to ``1`` .
        batch_size: The size of the batch samplers. Defaults to ``16`` .
        epoch: The number of epochs to train the model. Defaults to ``10`` .
        learning_rate: The learning rate used to update the parameters. Defaults to ``0.01`` .
        classes: The classes of handwrite digits to be predicted.
            Defaults to ``None`` , which means predict all the classes.
        num_train: The number of the data in the training dataset.
            Defaults to ``None`` , which will use all training data.
        num_test: The number of the data in the test dataset. Defaults to ``None`` , which will use all test data.
    """
    if classes is None:
        classes = list(range(10))
    train_input, train_label = data_loading(num_qubits=num_qubits, mode='train', classes=classes, num_data=num_train)
    test_input, test_label = data_loading(num_qubits=num_qubits, mode='test', classes=classes, num_data=num_test)
    net = VSQL(num_qubits, num_shadow=num_shadow, num_classes=len(classes), depth=depth)
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=net.parameters())
    num_train = len(train_label) if num_train is None else num_train
    num_test = len(test_label) if num_test is None else num_test
    for idx_epoch in range(epoch):
        for itr in range(num_train // batch_size):
            output = net(train_input[itr * batch_size:(itr + 1) * batch_size])
            labels = paddle.to_tensor(train_label[itr * batch_size:(itr + 1) * batch_size])
            loss = F.cross_entropy(output, labels)
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()
            if itr % 10 == 0:
                predictions = paddle.argmax(output, axis=-1).tolist()
                labels = labels.tolist()
                train_acc = sum(labels[idx] == predictions[idx] for idx in range(len(labels))) / len(labels)
                output = net(test_input[:num_test])
                labels = test_label[:num_test]
                predictions = paddle.argmax(output, axis=-1).tolist()
                test_acc = sum(labels[idx] == predictions[idx] for idx in range(len(labels))) / num_test
                print(
                    f"Epoch: {idx_epoch: 3d}, iter: {itr: 3d}, loss: {loss.item(): .4f}, "
                    f"batch_acc: {train_acc: .2%}, test_acc: {test_acc: .2%}."
                )
                state_dict = net.state_dict()
                paddle.save(state_dict, 'vsql.pdparams')


if __name__ == '__main__':
    train(
        num_qubits=10,
        num_shadow=2,
        depth=1,
        batch_size=20,
        epoch=10,
        learning_rate=0.01,
        classes=[0, 1],
        num_train=1000,
        num_test=100
    )
