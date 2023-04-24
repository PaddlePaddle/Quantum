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
The quality detection model based on quantum neural networks.
"""

import logging
import os
import warnings
from tqdm import tqdm
from typing import Optional, List, Tuple, Union

from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
import paddle_quantum as pq
from paddle_quantum.ansatz import Circuit

warnings.filterwarnings("ignore", category=Warning)
pq.set_dtype('complex128')


class ImageDataset(Dataset):
    r"""
    The class used for loading classical datasets.

    Args:
        file_path: The path of the input image.
        num_samples: The number of the data in the test dataset.
        pca: Whether use principal component analysis. Default to None.
        scaler: Whether scale the data. Default to None.
        centering: Whether remove the mean. Default to None.

    """
    def __init__(
            self, file_path: str, num_samples: int, pca: PCA=None,
            scaler: StandardScaler=None, centering: MinMaxScaler=None
    ):
        super().__init__()

        # load data
        data, labels = [], []
        pos_files = os.listdir(file_path + 'Positive/')
        neg_files = os.listdir(file_path + 'Negative/')

        for fd in pos_files:
            pos_img = Image.open(file_path + 'Positive/' +  fd)
            data.append(np.reshape(pos_img, 784))
            labels.append([1.0, 0.0])

        for fd in neg_files:
            pos_img = Image.open(file_path + 'Negative/' +  fd)
            data.append(np.reshape(pos_img, 784))
            labels.append([0.0, 1.0])

        data = np.array(data)
        labels = np.array(labels)
        idx = np.arange(len(data))
        np.random.seed(0)
        np.random.shuffle(idx)
        data, labels = data[idx], labels[idx]

        if pca is None: # training task
            # preprocess
            # remove the mean
            self.centering = StandardScaler(with_std=False)
            self.centering.fit(data)
            data = self.centering.transform(data)
            # PCA
            self.pca = PCA(n_components=16)
            self.pca.fit(data)
            data = self.pca.transform(data)
            # scale the data
            self.scaler = MinMaxScaler((0, np.pi))
            self.scaler.fit(data)
            data = self.scaler.transform(data)
        else: # test or validation task
            data = centering.transform(data)
            data = pca.transform(data)
            data = scaler.transform(data)

        if num_samples == -1:
            self.num_samples = len(data)
        elif num_samples >= len(data):
            self.num_samples = len(data)
        else:
            self.num_samples = num_samples
        self.image_list = data[:self.num_samples]
        self.label_list = labels[:self.num_samples]

    def __getitem__(self, idx):
        return self.image_list[idx], self.label_list[idx]

    def __len__(self):
        return self.num_samples


def _filter_circuit(num_qubits: int, depth:int) -> pq.ansatz.Circuit:
    r"""
    The function that generates a filter circuit for extracting features.
    """
    cir = Circuit(num_qubits)

    cir.complex_entangled_layer(depth=depth)

    return cir


def _encoding_circuit(num_qubits: int, data: paddle.Tensor) -> pq.ansatz.Circuit:
    r"""
    The function that encodes the classical data into quantum states.
    """
    cir = Circuit(num_qubits)
    depth = int(np.ceil(len(data)/num_qubits))
    t = 0

    for d in range(depth):
        if d%2==0:
            for q in range(num_qubits):
                cir.ry(qubits_idx=q, param=data[t%len(data)])
                t += 1
        else:
            for q in range(num_qubits):
                cir.rx(qubits_idx=q, param=data[t%len(data)])
                t += 1

    return cir


class QNNQD(paddle.nn.Layer):
    r"""
    The class of the quality detection using quantum neural networks (QNN).

    Args:
        num_qubits: The number of qubits of the quantum circuit in each layer.
        num_depths: The depth of quantum circuit in each layer.
        observables: The observables of the quantum circuit in each layer. 
    """
    def __init__(self, num_qubits: List[int], num_depths: List[int], observables: List) -> None:
        super(QNNQD, self).__init__()
        self.num_qubits = num_qubits
        self.num_depths = num_depths
        self.observables = observables
        self.cirs = paddle.nn.Sequential(*[_filter_circuit(num_qubits[j], num_depths[j]) for j in range(len(self.num_qubits))])

        self.fc = paddle.nn.Linear(len(self.observables[-1]), 2,
                                   weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal()),
                                   bias_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal()))


    def forward(self, batch_input: List[paddle.Tensor]) -> paddle.Tensor:
        r"""
        The forward function.

        Args:
            batch_input: The input of the model. It's shape is :math:`(\text{batch_size}, -1)` .

        Returns:
            Return the output of the model. It's shape is :math:`(\text{batch_size}, \text{num_classes})` .
        """
        features = []

        # quantum part
        for data in batch_input:
            # from layer one to layer N
            for j in range(len(self.num_qubits)):
                if j==0: # initialization
                    f_i = data
                enc_cir = _encoding_circuit(self.num_qubits[j], f_i)
                init_state = pq.state.zero_state(self.num_qubits[j])
                enc_state = enc_cir(init_state)
                layer_state = self.cirs[j](enc_state)
                
                f_j = []
                for ob_str in self.observables[j]:
                    ob = pq.Hamiltonian([[1.0, ob_str]])
                    expecval = pq.loss.ExpecVal(ob)
                    f_ij = expecval(layer_state)
                    f_j.append(f_ij)
                f_i = paddle.concat(f_j)
    
            features.append(f_i)

        features = paddle.stack(features)
    
        # classical part
        outputs = self.fc(features)

        return F.softmax(outputs)


def train(
        model_name: str, num_qubits: List, num_depths: List[int], observables: List,
        batch_size: int=20, num_epochs: int=4, learning_rate: float = 0.1,
        dataset: str = 'SurfaceCrack', saved_dir: str = './',
        using_validation: bool = False,
        num_train: int = -1, num_val: int = -1, num_test: int = -1,
        early_stopping: Optional[int] = 1000, num_workers: Optional[int] = 0
) -> None:
    """
    The function of training the QNNQD model.

    Args:
        model_name: The name of the model, which is used to save the model.
        num_qubits: The number of qubits of the quantum circuit in each layer.
        num_depths: The depth of quantum circuit in each layer.
        observables: The observables of the quantum circuit in each layer.
        batch_size: The size of the batch samplers. Default to ``20`` .
        num_epochs: The number of epochs to train the model. Default to ``4`` .
        learning_rate: The learning rate used to update the parameters. Default to ``0.1``.
        dataset: The path of the dataset. It defaults to SurfaceCrack.
        saved_dir: The path used to save logs. Default to ``./``.
        using_validation:  Whether use the validation. It is false means the dataset only contains training datasets and test datasets. 
        num_train: The number of the data in the training dataset. Default to ``-1`` , which will use all training data.
        num_val: The number of the data in the validation dataset. Default to ``-1`` , which will use all validation data.
        num_test: The number of the data in the test dataset. Default to ``-1`` , which will use all test data.
        early_stopping: Number of the iterations with no improvement after which training will be stopped. Defulat to ``1000``.
        num_workers: The number of subprocess to load data, 0 for no subprocess used and loading data in main process. Default to 0.
    """
    if saved_dir[-1] != '/':
        saved_dir += '/'
    if dataset[-1] != '/':
        dataset += '/'

    logging.basicConfig(
        filename=f'{saved_dir}{model_name}.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
    )

    train_dataset = ImageDataset(file_path=dataset+'training_data/', num_samples=num_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = ImageDataset(file_path=dataset+'test_data/', num_samples=num_test, 
                                pca=train_dataset.pca, scaler=train_dataset.scaler, centering=train_dataset.centering)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if using_validation:
        val_dataset = ImageDataset(file_path=dataset+'validation_data/', num_samples=num_val,
                                pca=train_dataset.pca, scaler=train_dataset.scaler, centering=train_dataset.centering)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    paddle.seed(0)
    model = QNNQD(num_qubits, num_depths, observables)
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())

    total_batch = 0
    val_best_loss = float('inf')
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
            prob = model(images)
            loss = - paddle.mean(paddle.log(prob) * labels)
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()

            if total_batch % 10 == 0:
                predicts = paddle.argmax(prob, axis=1).tolist()
                labels = paddle.argmax(labels, axis=1).tolist()
                train_acc = sum(labels[idx] == predicts[idx] for idx in range(len(labels))) / len(labels)
                if using_validation:
                    with paddle.no_grad():
                        val_loss, val_acc = evaluate(model, val_loader)
                        if val_loss < val_best_loss:
                            paddle.save(model.state_dict(), f'{saved_dir}/{model_name}.pdparams')
                            improve = '*'
                            last_improve = total_batch
                            val_best_loss = val_acc
                        else:
                            improve = ' '
                    msg = (
                        f"Iter:{total_batch: 5d}, Train loss:{loss.item(): 3.5f}, acc:{train_acc: 3.2%}; "
                        f"Val loss:{val_loss: 3.5f}, acc:{val_acc: 3.2%}{improve}"
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
        test(model, saved_dir, model_name, test_loader)
    else:
        paddle.save(model.state_dict(), f'{saved_dir}/{model_name}.pdparams')
        with paddle.no_grad():
            test_loss, test_acc = evaluate(model, test_loader)
        msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
        logging.info(msg)
        print(msg)


def evaluate(
    model:paddle.nn.Layer, 
    data_loader: paddle.io.DataLoader
) -> Tuple[float, float]:
    r"""
    Evaluating the performance of the model on the dataset.

    Args:
        model: The QNN model.
        data_loader: The data loader for the data.

    Returns:
        Return the accuracy of the model on the given datasets.
    """
    val_loss = 0
    model.eval()
    labels_all = []
    predicts_all = []

    with paddle.no_grad():
        for images, labels in data_loader:
            prob = model(images)
            loss = - paddle.mean(paddle.log(prob) * labels)
            labels = paddle.argmax(labels, axis=1).tolist()
            val_loss += loss.item() * len(labels)
            labels_all.extend(labels)
            predict = paddle.argmax(prob, axis=1).tolist()
            predicts_all.extend(predict)
    val_acc = sum(labels_all[idx] == predicts_all[idx] for idx in range(len(labels_all)))

    return val_loss / len(labels_all), val_acc / len(labels_all)


def test(
        model: paddle.nn.Layer, 
        saved_dir: str, 
        model_name: str, 
        test_loader: paddle.io.DataLoader
) -> None:
    r"""
    Evaluating the performance of the model on the test dataset.

    Args:
        model: QNN model.
        saved_dir: The path of the saved model.
        model_name: The name of the model.
        test_loader: The data loader for testing datasets.
    """
    model.set_state_dict(paddle.load(f'{saved_dir}{model_name}.pdparams'))
    test_loss, test_acc = evaluate(model, test_loader)
    msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
    logging.info(msg)
    print(msg)


def inference(
        image_path: str, num_samples: int, model_path: str,
        num_qubits: List, num_depths: List, observables: List,
) -> Union[float, list]:
    r"""
    The prediction function for the provided test dataset.

    Args:
        image_path: The path of the input image.
        num_samples: The number of the data need to be classified.
        model_path: The path of the trained model, which will be loaded.
        num_qubits: The number of qubits of the quantum circuit in each layer.
        num_depths: The depth of quantum circuit in each layer.
        observables: The observables of the quantum circuit in each layer.

    Returns:
        Return the prediction of the given datasets together with the associated level of certainty.
    """
    logging.basicConfig(
        filename=f'./{"qnnqd"}.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
    )
    msg = f"Start Inferencing"
    logging.info(msg)
    
    model = QNNQD(num_qubits, num_depths, observables)
    model.set_state_dict(paddle.load(model_path))
    train_dataset = ImageDataset(file_path='SurfaceCrack/training_data/', num_samples=500)
    dataset = ImageDataset(file_path=image_path, num_samples=num_samples, pca=train_dataset.pca, scaler=train_dataset.scaler, centering=train_dataset.centering)

    model.eval()
    prob = model(paddle.to_tensor(dataset.image_list))
    prediction = paddle.argmin(prob, axis=1)
    label = np.argmin(dataset.label_list[:num_samples], axis=1)

    msg = f"Finish Inferencing"
    logging.info(msg)

    return prediction.tolist(), prob.tolist(), label.tolist()


if __name__ == '__main__':
    exit(0)