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
The Variational Shadow Quantum Learning (VSQL) model.
"""

import logging
import os
from functools import partial
from tqdm import tqdm
from typing import Optional, List, Tuple, Callable

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.vision.datasets import MNIST
from paddle.io import Dataset, DataLoader

import paddle_quantum as pq
from paddle_quantum.ansatz import Circuit


def image_preprocess(image: np.ndarray, num_qubits: int) -> np.ndarray:
    r"""
    Normalize the input image. Flatten them and make them to normalized vectors.

    Args:
        image: The input image.
        num_qubits: The number of the qubits, which decides the dimension of the vector.

    Returns:
        Return the preprocessed image which is a normalized vector.
    """
    image = np.pad(image.flatten(), pad_width=(0, 2 ** num_qubits - image.size))
    return image / np.linalg.norm(image)


class ImageDataset(Dataset):
    r"""
    The class to implement the image dataset.

    Args:
        file_path: The path of the dataset file, each line of which represents a piece of data.
            Each line contains the file path and label of the image, separated by tabs.
        num_samples: The number of the samples. It is the number of the data that the dataset contains.
            Defaults to ``0`` , which means contains all data in the dataset file.
        transform: The function to transform the image. Defaults to ``None`` , which means do nothing on the image.
    """
    def __init__(self, file_path: str, num_samples: int = 0, transform: Optional[Callable] = None):
        super().__init__()
        self.transform = transform
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data_list = [line.strip().split('\t') for line in lines]
        if num_samples > 0:
            self.data_list = self.data_list[:self.data_list]
        self.num_samples = len(self.data_list)

    def __getitem__(self, idx):
        image_path, label = self.data_list[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(paddle.get_default_dtype())
        if self.transform is not None:
            image = self.transform(image)
        return image, np.array(int(label))

    def __len__(self):
        return self.num_samples


def _slide_circuit(cir: pq.ansatz.Circuit, distance: int):
    r"""
    Slide the shadow circuit to obtain the different shadow feature.

    Args:
        cir: The quantum circuit which only contains shadow circuit.
        distance: The distance to move the shadow circuit.
    """
    # slide to get the local feature
    for sublayer in cir.sublayers():
        qubits_idx = np.array(sublayer.qubits_idx)
        qubits_idx = qubits_idx + distance
        sublayer.qubits_idx = qubits_idx.tolist()


def generate_observable(start_idx: int, num_shadow: int) -> pq.Hamiltonian:
    r"""
    Generate the observable to measure the quantum states.

    Args:
        start_idx: The start index of the qubits.
        num_shadow: The number of the qubits which the shadow circuit contains.

    Returns:
        Return the generated observable.
    """
    # construct the observable to get te output of the circuit
    pauli_str = ','.join(f'x{str(i)}' for i in range(start_idx, start_idx + num_shadow))
    return pq.Hamiltonian([[1.0, pauli_str]])


class VSQL(paddle.nn.Layer):
    r"""
    The class of the variational shadow quantum learning (VSQL) model.

    The details can be referred to https://ojs.aaai.org/index.php/AAAI/article/view/17016 .

    Args:
        num_qubits: The number of the qubits which the quantum circuit contains.
        num_shadow: The number of the qubits which the shadow circuit contains.
        num_classes: The number of the classes which the model will classify.
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
                ob = generate_observable(idx_start, num_shadow=self.num_shadow)
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
        model_name: str, num_qubits: int, num_shadow: int, classes: list,
        batch_size: int, num_epochs: int, depth: int = 1, dataset: str = 'MNIST', saved_dir: str = '',
        learning_rate: float = 0.01, using_validation: bool = False, num_workers: int = 0,
        early_stopping: int = 1000, num_train: int = 0, num_dev: int = 0, num_test: int = 0,
) -> None:
    """
    The function of training the VSQL model.

    Args:
        model_name: The name of the model. It is the filename of the saved model.
        num_qubits: The number of the qubits which the quantum circuit contains.
        num_shadow: The number of the qubits which the shadow circuit contains.
        classes: The classes of handwrite digits to be predicted.
            Defaults to ``None`` , which means predict all the classes.
        batch_size: The size of the batch samplers.
        num_epochs: The number of the epochs to train the model.
        depth: The depth of the quantum circuit. Defaults to ``1`` .
        dataset: The dataset used to train the model, which should be a directory.
            Defaults to ``'MNIST'`` , which means using the built-in MNIST dataset.
        saved_dir: The directory to saved the trained model and the training log. Defaults to use the current path.
        learning_rate: The learning rate used to update the parameters. Defaults to ``0.01`` .
        using_validation: If the datasets contains the validation dataset.
            Defaults to ``False`` , which means the validation dataset is not included.
        num_workers: The number of the subprocess to load data, 0 for no subprocess used and loading data in main process.
            Defaults to ``0`` .
        early_stopping: Number of epochs with no improvement after which training will be stopped.
            Defaults to ``1000`` .
        num_train: The number of the data in the training dataset.
            Defaults to ``0`` , which will use all training data.
        num_dev: The number of the data in the test dataset. Defaults to ``0`` , which will use all validation data.
        num_test: The number of the data in the test dataset. Defaults to ``0`` , which will use all test data.
    """
    if not saved_dir:
        saved_dir = './'
    elif saved_dir[-1] != '/':
        saved_dir += '/'
    if dataset != 'MNIST' and dataset[-1] != '/':
        dataset += '/'
    logging.basicConfig(
        filename=f'{saved_dir}{model_name}.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
    )
    if dataset == 'MNIST':
        train_dataset = MNIST(
            mode='train', backend='cv2',
            transform=partial(image_preprocess, num_qubits=num_qubits)
        )
        test_dataset = MNIST(
            mode='test', backend='cv2',
            transform=partial(image_preprocess, num_qubits=num_qubits)
        )

        def filter_data(dataset: MNIST, classes: list):
            _images = dataset.images
            _labels = dataset.labels
            _len = len(dataset)
            dataset.images = [_images[idx] for idx in range(_len) if _labels[idx] in classes]
            dataset.labels = [_labels[idx] for idx in range(_len) if _labels[idx] in classes]
        if classes != list(range(10)):
            filter_data(train_dataset, classes)
            filter_data(test_dataset, classes)
        if num_train > 0:
            train_dataset.images = train_dataset.images[:num_train]
            train_dataset.labels = train_dataset.labels[:num_train]
        if num_test > 0:
            test_dataset.images = test_dataset.images[:num_test]
            test_dataset.labels = test_dataset.labels[:num_test]
    else:
        train_dataset = ImageDataset(
            file_path=f'{dataset}train.txt', num_samples=num_train,
            transform=partial(image_preprocess, num_qubits=num_qubits)
        )
        if using_validation:
            dev_dataset = ImageDataset(
                file_path=f'{dataset}dev.txt', num_samples=num_dev,
                transform=partial(image_preprocess, num_qubits=num_qubits)
            )
        test_dataset = ImageDataset(
            file_path=f'{dataset}test.txt', num_samples=num_test,
            transform=partial(image_preprocess, num_qubits=num_qubits)
        )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if using_validation:
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = VSQL(num_qubits, num_shadow=num_shadow, num_classes=len(classes), depth=depth)
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    stopping_flag = False
    for epoch in range(num_epochs):
        p_bar = tqdm(
            total=len(train_loader),
            desc=f'Epoch[{epoch: 3d}]',
            ascii=True,
            dynamic_ncols=True,
        )
        for images, labels in train_loader:
            p_bar.update(1)
            model.clear_gradients()
            output = model(images)
            loss = F.cross_entropy(output, labels)
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()
            if total_batch % 10 == 0:
                predicts = paddle.argmax(output, axis=1).tolist()
                labels = labels.flatten().tolist()
                train_acc = sum(labels[idx] == predicts[idx] for idx in range(len(labels))) / len(labels)
                if using_validation:
                    with paddle.no_grad():
                        dev_loss, dev_acc = evaluate(model, dev_loader)
                        if dev_loss < dev_best_loss:
                            paddle.save(model.state_dict(), f'{saved_dir}/{model_name}.pdparams')
                            improve = '*'
                            last_improve = total_batch
                            dev_best_loss = dev_loss
                        else:
                            improve = ' '
                    msg = (
                        f"Iter:{total_batch: 5d}, Train loss:{loss.item(): 3.5f}, acc:{train_acc: 3.2%}; "
                        f"Val loss:{dev_loss: 3.5f}, acc:{dev_acc: 3.2%}{improve}"
                    )
                else:
                    with paddle.no_grad():
                        test_loss, test_acc = evaluate(model, test_loader)
                        paddle.save(model.state_dict(), f'{saved_dir}{model_name}.pdparams')
                    msg = (
                        f"Iter:{total_batch: 5d}, Train loss:{loss.item(): 3.5f}, acc:{train_acc: 3.2%}; "
                        f"Test loss:{test_loss: 3.5f}, acc:{test_acc: 3.2%}"
                    )
                model.train()
                p_bar.set_postfix_str(msg)
                logging.info(msg)
            total_batch += 1
            if using_validation and total_batch - last_improve >= early_stopping:
                stopping_flag = True
                break
        p_bar.close()
        if stopping_flag:
            break
    if stopping_flag:
        msg = "No optimization for a long time, auto-stopping..."
    else:
        msg = "The training of the model has been finished."
    logging.info(msg)
    print(msg)
    if using_validation:
        test(model, f'{saved_dir}/{model_name}.pdparams', test_loader)
    else:
        paddle.save(model.state_dict(), f'{saved_dir}/{model_name}.pdparams')
        with paddle.no_grad():
            test_loss, test_acc = evaluate(model, test_loader)
        msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
        logging.info(msg)
        print(msg)


def evaluate(model: paddle.nn.Layer, data_loader: paddle.io.DataLoader) -> Tuple[float, float]:
    r"""
    Evaluate the model.

    Args:
        model: The trained model to be evaluated.
        data_loader: The dataloader of the data used to evaluate the model.
    Returns:
        Return the average loss and accuracy in the data of the input dataloader.
    """
    dev_loss = 0
    model.eval()
    labels_all = []
    predicts_all = []
    with paddle.no_grad():
        for images, labels in data_loader:
            prob = model(images)
            loss = paddle.nn.functional.cross_entropy(prob, labels)
            labels = labels.flatten().tolist()
            dev_loss += loss.item() * len(labels)
            labels_all.extend(labels)
            predict = paddle.argmax(prob, axis=1)
            predicts_all.extend(predict.tolist())
    dev_acc = sum(labels_all[idx] == predicts_all[idx] for idx in range(len(labels_all)))
    return dev_loss / len(labels_all), dev_acc / len(labels_all)


def test(model: paddle.nn.Layer, model_path: str, test_loader: paddle.io.DataLoader) -> None:
    r"""
    Use the test dataset to test the model.

    Args:
        model: The model to be tested.
        model_path: The file path of the models' file.
        test_loader: The dataloader of the test dataset.
    """
    model.set_state_dict(paddle.load(f'{model_path}'))
    with paddle.no_grad():
        test_loss, test_acc = evaluate(model, test_loader)
    msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
    logging.info(msg)
    print(msg)


def inference(
        image_path: str, is_dir: bool, model_path: str,
        num_qubits: int, num_shadow: int, classes: list, depth: int = 1,
) -> Tuple[int, list]:
    r"""
    The inference function. Using the trained model to predict new data.

    Args:
        image_path: The path of the image to be predicted.
        is_dir: Whether the input ``image_path`` is a directory.
            If it is a directory, the model will predict all images under it.
        model_path: The path of the model file.
        num_qubits: The number of the qubits which the quantum circuit contains.
        num_shadow: The number of the qubits which the shadow circuit contains.
        classes: The classes which the input image belongs to.
        depth: The depth of the quantum circuit. Defaults to ``1`` .

    Returns:
        Return the class the model predicted and the probability of each class.
    """
    num_classes = len(classes)
    model = VSQL(
        num_qubits=num_qubits, num_shadow=num_shadow,
        num_classes=num_classes, depth=depth
    )
    model.set_state_dict(paddle.load(model_path))
    if is_dir:
        filter_ext = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
        path_list = [
            image_path + filename
            for filename in os.listdir(image_path)
            if os.path.splitext(filename)[-1] in filter_ext
        ]
    else:
        path_list = [image_path]
    images = []
    for path in path_list:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(paddle.get_default_dtype())
        images.append(image_preprocess(image, num_qubits))
    images = paddle.to_tensor(np.stack(images))
    model.eval()
    prob = model(images)
    prediction = paddle.argmax(prob, axis=1).tolist()
    return prediction, paddle.nn.functional.softmax(prob).tolist()


if __name__ == '__main__':
    exit(0)
