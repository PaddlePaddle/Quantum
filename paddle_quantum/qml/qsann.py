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
The Quantum Self-Attention Neural Network (QSANN) model.
"""

import logging
import random
from tqdm import tqdm
from typing import List, Tuple, Dict

import numpy as np
import paddle
import paddle_quantum as pq
from paddle.io import Dataset
from paddle_quantum.loss import ExpecVal
from paddle_quantum.gate import functional


def generate_observable(num_qubits: int, num_terms: int) -> List[list]:
    """
    Generate the observables to observe the quantum state.

    Args:
        num_qubits: The number of the qubits.
        num_terms: The number of the generated observables.

    Returns:
        Return the generated observables.
    """
    ob = [[[1.0, f'z{idx:d}']] for idx in range(num_qubits)]
    ob.extend([[1.0, f'y{idx:d}']] for idx in range(num_qubits))
    ob.extend([[1.0, f'x{idx:d}']] for idx in range(num_qubits))
    if len(ob) >= num_terms:
        ob = ob[:num_terms]
    else:
        ob.extend(ob * (num_terms // len(ob) - 1))
        ob.extend(ob[:num_terms % len(ob)])
    return ob


class QSANN(paddle.nn.Layer):
    r"""
    The class of the quantum self-attention neural network (QSANN) model.

    Args:
        num_qubits: The number of the qubits which the quantum circuit contains.
        len_vocab: The length of the vocabulary.
        num_layers: The number of the self-attention layers.
        depth_ebd: The depth of the embedding circuit.
        depth_query: The depth of the query circuit.
        depth_key: The depth of the key circuit.
        depth_value: The depth of the value circuit.
    """
    def __init__(
            self, num_qubits: int, len_vocab: int, num_layers: int,
            depth_ebd: int, depth_query: int, depth_key: int, depth_value: int,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.len_vocab = len_vocab
        self.num_layers = num_layers
        self.depth_ebd = depth_ebd
        self.depth_query = depth_query
        self.depth_key = depth_key
        self.depth_value = depth_value
        self.embedding_param = self.create_parameter(
            shape=[len_vocab, num_qubits * (depth_ebd * 2 + 1), 2],
            default_initializer=paddle.nn.initializer.Uniform(low=-np.pi, high=np.pi),
            dtype=paddle.get_default_dtype(),
            is_bias=False,
        )
        self.weight = self.create_parameter(
            shape=[num_qubits * (depth_ebd * 2 + 1) * 2],
            default_initializer=paddle.nn.initializer.Normal(std=0.001),
            dtype=paddle.get_default_dtype(),
            is_bias=False)
        self.bias = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Normal(std=0.001),
            dtype=paddle.get_default_dtype(),
            is_bias=False)
        query_circuits = self.__circuit_list(num_layers, num_qubits, depth_query)
        self.query_circuits = paddle.nn.LayerList(query_circuits)
        key_circuits = self.__circuit_list(num_layers, num_qubits, depth_key)
        self.key_circuits = paddle.nn.LayerList(key_circuits)
        value_circuits = self.__circuit_list(num_layers, num_qubits, depth_value)
        self.value_circuits = paddle.nn.LayerList(value_circuits)
        observables = generate_observable(self.num_qubits, self.embedding_param[0].size)
        self.ob_query = pq.Hamiltonian(observables[0])
        self.ob_key = pq.Hamiltonian(observables[0])
        self.ob_value = [pq.Hamiltonian(ob_item) for ob_item in observables]

    def __embedding_circuit(self, num_qubits, params, depth=1) -> pq.State:
        r"""
        The circuit to implement the embedding.

        Args:
            num_qubits: The number of the qubits.
            params: The parameters in the quantum circuit.
            depth: The depth of the quantum circuit. Defaults to ``1``.

        Returns:
            The quantum state which embeds the word.
        """
        embedding_state = pq.state.zero_state(num_qubits)
        for d in range(depth):
            for idx in range(num_qubits):
                qubits_idx = [idx, (idx + 1) % num_qubits]
                param_idx = 2 * num_qubits * d + 2 * idx
                
                cir = pq.Circuit(embedding_state.num_qubits)
                cir.rx(qubits_idx, param=params[param_idx:param_idx+2][0])
                cir.ry(qubits_idx, param=params[param_idx:param_idx+2][1])
                cir.cnot(qubits_idx)
                embedding_state = cir(embedding_state)
        
        for idx in range(num_qubits):
            param_idx = 2 * num_qubits * depth + idx
            
            cir = pq.Circuit(embedding_state.num_qubits)
            cir.rx(idx, param=params[param_idx][0])
            cir.ry(idx, param=params[param_idx][1])
            embedding_state = cir(embedding_state)
        
        return embedding_state

    def forward(self, batch_text: List[List[int]]) -> List[paddle.Tensor]:
        r"""
        The forward function to execute the model.

        Args:
            batch_text: The batch of input texts. Each of them is a list of int.

        Returns:
            Return a list which contains the predictions of the input texts.
        """
        predictions = []
        for text in batch_text:
            text_feature = [self.embedding_param[word] for word in text]
            for layer_idx in range(self.num_layers):
                queries = []
                keys = []
                values = []
                for char_idx in range(len(text_feature)):
                    embedding_state = self.__embedding_circuit(self.num_qubits, params=text_feature[char_idx])
                    query_state = self.query_circuits[layer_idx](embedding_state)
                    key_state = self.key_circuits[layer_idx](embedding_state)
                    value_state = self.value_circuits[layer_idx](embedding_state)
                    query = ExpecVal(self.ob_query)(query_state)
                    key = ExpecVal(self.ob_key)(key_state)
                    value = [ExpecVal(ob_item)(value_state) for ob_item in self.ob_value]
                    value = paddle.concat(value)
                    queries.append(query)
                    keys.append(key)
                    values.append(value)
                feature = []
                for char_idx in range(len(text_feature)):
                    query = queries[char_idx]
                    output = paddle.zeros_like(values[0])
                    alpha_sum = 0
                    for idx in range(len(keys)):
                        alpha = (keys[idx] - query) ** 2
                        alpha = paddle.exp(-1 * alpha)
                        output += alpha * values[idx]
                        alpha_sum += alpha
                    output = output / alpha_sum * np.pi
                    output = paddle.reshape(output, self.embedding_param[0].shape)
                    feature.append(output)
                text_feature = feature
            output = paddle.flatten(sum(text_feature) / len(text_feature))
            predictions.append(1 / (1 + paddle.exp(-output @ self.weight - self.bias)))
        return predictions

    def __circuit_list(self, num_layer, num_qubits, depth) -> List[pq.ansatz.Circuit]:
        r"""
        Generate a series of circuits.

        Args:
            num_layer: The number of the self-attention layers, which means the number of the circuits.
            num_qubits: The number of the qubits which the circuits contains.
            depth: The depth of the quantum circuits.

        Returns:
            A list of the generated circuits.
        """
        circuits = []
        for _ in range(num_layer):
            cir = pq.ansatz.Circuit(num_qubits)
            for _ in range(depth):
                for idx in range(num_qubits):
                    cir.rx(idx)
                    cir.ry(idx)
                    cir.rx((idx + 1) % num_qubits)
                    cir.ry((idx + 1) % num_qubits)
                    cir.cnot([idx, (idx + 1) % num_qubits])
            cir.rx('full')
            cir.ry('full')
            circuits.append(cir)
        return circuits


def deal_vocab(vocab_path: str) -> Dict[str, int]:
    r"""
    Get the map from the word to the index by the input vocabulary file.

    Args:
        vocab_path: The path of the vocabulary file.

    Returns:
        Return the map from the word to the corresponding index.
    """
    with open(vocab_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    word2idx = {word.strip(): idx for idx, word in enumerate(lines)}
    return word2idx


class TextDataset(Dataset):
    r"""
    The class to implement the text dataset.

    Args:
        file_path: The dataset file.
        word2idx: The map from the word to the corresponding index.
        pad_size: The size pad the text sequence to. Defaults to ``0``, which means no padding.
    """
    def __init__(self, file_path: str, word2idx: dict, pad_size: int = 0):
        super().__init__()
        self.contents = []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            text, label = line.strip().split('\t')
            text = [word2idx.get(word, 0) for word in text.split()]
            if pad_size != 0:
                if len(text) >= pad_size:
                    text = text[:pad_size]
                else:
                    text.extend([0] * (pad_size - len(text)))
            self.contents.append((text, int(label)))
        self.len_data = len(self.contents)

    def __getitem__(self, idx):
        return self.contents[idx]

    def __len__(self):
        return self.len_data


def build_iter(dataset: TextDataset, batch_size: int, shuffle: bool = False) -> list:
    r"""
    Build the iteration of the batch data.

    Args:
        dataset: The dataset to be built.
        batch_size: The number of the data in a batch.
        shuffle: Whether to randomly shuffle the order of the data. Defaults to ``False``.

    Returns:
        The built iteration which contains the batches of the data.
    """
    data_iter = []
    # 是否需要拼接成tensor
    if shuffle:
        random.shuffle(dataset.contents)
    for idx in range(0, len(dataset), batch_size):
        batch_data = dataset[idx: idx + batch_size]
        texts = [token_ids for token_ids, _ in batch_data]
        labels = [label for _, label in batch_data]
        data_iter.append((texts, labels))
    return data_iter


def train(
        model_name: str, dataset: str, num_qubits: int, num_layers: int,
        depth_ebd: int, depth_query: int, depth_key: int, depth_value: int,
        batch_size: int, num_epochs: int, learning_rate: float = 0.01,
        saved_dir: str = '', using_validation: bool = False,
        early_stopping: int = 1000,
) -> None:
    r"""
    The function of training the QSANN model.

    Args:
        model_name: The name of the model. It is the filename of the saved model.
        dataset: The dataset used to train the model, which should be a directory.
        num_qubits: The number of the qubits which the quantum circuit contains.
        num_layers: The number of the self-attention layers.
        depth_ebd: The depth of the embedding circuit.
        depth_query: The depth of the query circuit.
        depth_key: The depth of the key circuit.
        depth_value: The depth of the value circuit.
        batch_size: The size of the batch samplers.
        num_epochs: The number of the epochs to train the model.
        learning_rate: The learning rate used to update the parameters. Defaults to ``0.01`` .
        saved_dir: The directory to saved the trained model and the training log. Defaults to use the current path.
        using_validation: If the datasets contains the validation dataset.
            Defaults to ``False`` , which means the validation dataset is not included.
        early_stopping: Number of iterations with no improvement after which training will be stopped.
            Defaults to ``1000`` .
    """
    if not saved_dir:
        saved_dir = './'
    elif saved_dir[-1] != '/':
        saved_dir += '/'
    if dataset[-1] != '/':
        dataset += '/'
    logging.basicConfig(
        filename=f'{saved_dir}{model_name}.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
    )
    word2idx = deal_vocab(f'{dataset}vocab.txt')
    len_vocab = len(word2idx)
    train_dataset = TextDataset(file_path=f'{dataset}train.txt', word2idx=word2idx)
    if using_validation:
        dev_dataset = TextDataset(file_path=f'{dataset}dev.txt',  word2idx=word2idx)
    test_dataset = TextDataset(file_path=f'{dataset}test.txt', word2idx=word2idx)
    train_iter = build_iter(train_dataset, batch_size=batch_size, shuffle=True)
    if using_validation:
        dev_iter = build_iter(dev_dataset, batch_size=batch_size, shuffle=True)
    test_iter = build_iter(test_dataset, batch_size=batch_size, shuffle=True)
    model = QSANN(
        num_qubits=num_qubits, len_vocab=len_vocab, num_layers=num_layers,
        depth_ebd=depth_ebd, depth_query=depth_query, depth_key=depth_key, depth_value=depth_value,
    )
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    stopping_flag = False
    for epoch in range(num_epochs):
        p_bar = tqdm(
            total=len(train_iter),
            desc=f'Epoch[{epoch: 3d}]',
            ascii=True,
            dynamic_ncols=True,
        )
        for texts, labels in train_iter:
            p_bar.update(1)
            model.clear_gradients()
            predictions = model(texts)
            loss = sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels)) / len(labels)
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()
            if total_batch % 10 == 0:
                predictions = [0 if item < 0.5 else 1 for item in predictions]
                train_acc = sum(labels[idx] == predictions[idx] for idx in range(len(labels))) / len(labels)
                if using_validation:
                    with paddle.no_grad():
                        dev_loss, dev_acc = evaluate(model, dev_iter)
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
                        test_loss, test_acc = evaluate(model, test_iter)
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
        test(model, f'{saved_dir}/{model_name}.pdparams', test_iter)
    else:
        paddle.save(model.state_dict(), f'{saved_dir}/{model_name}.pdparams')
        with paddle.no_grad():
            test_loss, test_acc = evaluate(model, test_iter)
        msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
        logging.info(msg)
        print(msg)


def evaluate(model: paddle.nn.Layer, data_loader: list) -> Tuple[float, float]:
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
        for texts, labels in data_loader:
            predictions = model(texts)
            loss = sum((prediction - label) ** 2 for prediction, label in zip(predictions, labels))
            dev_loss += loss.item()
            labels_all.extend(labels)
            predictions = [0 if item < 0.5 else 1 for item in predictions]
            predicts_all.extend(predictions)
    dev_acc = sum(labels_all[idx] == predicts_all[idx] for idx in range(len(labels_all)))
    return dev_loss / len(labels_all), dev_acc / len(labels_all)


def test(model: paddle.nn.Layer, model_path: str, test_loader: list) -> None:
    r"""
    Use the test dataset to test the model.

    Args:
        model: The model to be tested.
        model_path: The file path of the models' file.
        test_loader: The dataloader of the test dataset.
    """
    model.set_state_dict(paddle.load(model_path))
    with paddle.no_grad():
        test_loss, test_acc = evaluate(model, test_loader)
    msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
    logging.info(msg)
    print(msg)


def inference(
        text: str, model_path: str, vocab_path: str, classes: List[str],
        num_qubits: int, num_layers: int, depth_ebd: int,
        depth_query: int, depth_key: int, depth_value: int
) -> str:
    r"""
    The inference function. Using the trained model to predict new data.

    Args:
        text: The path of the image to be predicted.
        model_path: The path of the model file.
        vocab_path: The path of the vocabulary file.
        classes: The classes of all the labels.
        num_qubits: The number of the qubits which the quantum circuit contains.
        num_layers: The number of the self-attention layers.
        depth_ebd: The depth of the embedding circuit.
        depth_query: The depth of the query circuit.
        depth_key: The depth of the key circuit.
        depth_value: The depth of the value circuit.

    Returns:
        Return the class which the model predicted.
    """
    word2idx = deal_vocab(vocab_path)
    model = QSANN(
        num_qubits=num_qubits, len_vocab=len(word2idx), num_layers=num_layers,
        depth_ebd=depth_ebd, depth_query=depth_query, depth_key=depth_key, depth_value=depth_value,
    )
    model.set_state_dict(paddle.load(model_path))
    model.eval()
    text = [word2idx.get(word, 0) for word in list(text)]
    prediction = model([text])
    prediction = 0 if prediction[0] < 0.5 else 1
    return classes[prediction]


if __name__ == '__main__':
    exit(0)
