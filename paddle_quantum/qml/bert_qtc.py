# !/usr/bin/env python3
# Copyright (c) 2023 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
The BERT-QTC(Quantum Temporal Convolution) model.
"""

import logging
import paddle
import paddle.nn
import paddle_quantum as pq
import paddlenlp as ppnlp
import paddle.nn.functional as F
from functools import partial
from tqdm import tqdm
from paddle.io import Dataset, DataLoader
from paddlenlp.transformers import BertModel, BertTokenizer
from typing import List, Tuple, Callable

ppnlp.utils.log.logger.disable()


class QuantumTemporalConvolution(paddle.nn.Layer):
    r"""
    The class of the quantum temporal convolution layer.

    Args:
        num_filter: The number of the qubits which the quantum circuit contains.
        kernel_size: The size of the kernel, which means the number of the qubits.
        circuit_depth: The depth of the quantum circuit, which means the number of repetitions of the circuit template.
        padding: The padding size. It will pad in the left and the right sides with elements of the same length if the input type is int.
          If the input type is list, it should be [pad_left, pad_right].
    """
    def __init__(self, num_filter: int, kernel_size: int, circuit_depth: int, padding: int):
        super().__init__()
        # Define the variational quantum circuit.
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pad = paddle.nn.Pad1D(padding=padding)
        if isinstance(padding, list):
            self.num_pad = padding[0] + padding[1]
        elif isinstance(padding, int):
            self.num_pad = padding * 2
        vqc_list = [
            pq.ansatz.layer.ComplexEntangledLayer(num_qubits=kernel_size, depth=circuit_depth) for _ in range(num_filter)]
        self.vqc = paddle.nn.LayerList(vqc_list)
        self.hamiltonian_list = [pq.Hamiltonian([[1, f'Z{idx:d}']]) for idx in range(kernel_size)]

    def forward(self, hidden_state: paddle.Tensor) -> paddle.Tensor:
        r"""The forward function to execute the model.

        Args:
            hidden_state: The feature obtained by the BERT model.

        Returns:
            The feature extracted by the quantum temporal convolution layer.
        """
        # The shape of the hidden_state should be [batch_size, hidden_size].
        batch_size, hidden_size = hidden_state.shape
        _hidden_size = hidden_size + self.num_pad - self.kernel_size - 1
        # Make the shape to [1, batch_size, hidden_size].
        hidden_state = paddle.unsqueeze(hidden_state, axis=0)
        hidden_state = self.pad(hidden_state)
        # Make the shape to [batch_size, hidden_size].
        hidden_state = paddle.squeeze(hidden_state, axis=0)
        init_state = pq.state.zero_state(num_qubits=self.kernel_size)
        batch_feature = []
        for batch_idx in range(batch_size):
            filter_feature = []
            for filter_idx in range(self.num_filter):
                for hidden_idx in range(_hidden_size):
                    _state = init_state
                    for qubit_idx in range(self.kernel_size):
                        _state = pq.gate.RX(qubits_idx=qubit_idx, 
                                            param=hidden_state[batch_idx][hidden_idx + qubit_idx])(_state)
                    _state = self.vqc[filter_idx](_state)
                    circuit_output = [pq.loss.ExpecVal(hamiltonian)(_state) for hamiltonian in self.hamiltonian_list]
                    circuit_output = paddle.concat(circuit_output)
                    filter_feature.append(circuit_output)
            batch_feature.append(paddle.stack(filter_feature, axis=0))
        # The shape of the return tensor is (batch_size, num_filter * _hidden_size, kernel_size)
        return paddle.stack(batch_feature, axis=0)


class Decoder(paddle.nn.Layer):
    r"""
    The decoder module of the BERT-QTC model. It contains quantum temporal convolution layer,
        global max pooling layer, and the linear layer.

    Args:
        num_filter: The number of the quantum temporal convolution filters.
        kernel_size: The size of the kernel, which means the number of the qubits.
        circuit_depth: The depth of the quantum circuit, which means the number of repetitions of the circuit template.
        padding: The padding size. It will pad in the left and the right sides with elements of the same length
            if the input type is int. If the input type is list, it should be [pad_left, pad_right].
        num_classes: The number of the classes which the model will classify.
        hidden_size: The size of the hidden state obtained through the BERT model.
    """
    def __init__(
            self, num_filter: int, kernel_size: int, circuit_depth: int, padding: int,
            num_classes: int, hidden_size: int
    ):
        super().__init__()
        # The quantum temporal convolution layer.
        self.qtc = QuantumTemporalConvolution(num_filter, kernel_size, circuit_depth, padding)
        if isinstance(padding, list):
            _hidden_size = hidden_size + padding[0] + padding[1] - kernel_size - 1
        elif isinstance(padding, int):
            _hidden_size = hidden_size + padding * 2 - kernel_size - 1
        # The global max pooling layer
        self.gmp = paddle.nn.MaxPool1D(kernel_size=[kernel_size])
        self.linear = paddle.nn.Linear(in_features=num_filter * _hidden_size, out_features=num_classes)

    def forward(self, bert_feature: paddle.Tensor) -> paddle.Tensor:
        r"""The forward function to execute the model.

        Args:
            bert_feature: The feature obtained by the BERT model.

        Returns:
            The logits of each class obtained by prediction.
        """
        # The shape of qtc_output is (batch_size, num_filter * _hidden_size, kernel_size).
        qtc_output = self.qtc(bert_feature)
        # The shape of gmp_output is (batch_size, num_filter * _hidden_size).
        gmp_output = paddle.squeeze(self.gmp(qtc_output), axis=2)
        # The shape of predict_logits is (batch_size, num_classes).
        predict_logits = self.linear(gmp_output)
        return predict_logits


class BERTQTC(paddle.nn.Layer):
    r"""The BERT-QTC model.

    The details can be referred to https://arxiv.org/abs/2203.03550 .

    Args:
        bert_model: The name of the pretrained BERT model.
        decoder: The decoder network.
    """
    def __init__(self, bert_model: str, decoder: paddle.nn.Layer):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.token_pad = ppnlp.data.Pad()
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.decoder = decoder

    def forward(self, token_ids: paddle.Tensor) -> paddle.Tensor:
        r"""The forward function to execute the BERT-QTC model.

        Args:
            token_ids: The id of the token converted by the sentences.

        Returns:
            The probability of each class obtained by prediction.
        """
        with paddle.no_grad():
            # The shape of bert_feature is (batch_size, hidden_state_size).
            bert_feature = self.bert_model(token_ids)[0][:, 0, :]
        bert_feature.stop_gradient = True
        predict_logits = self.decoder(bert_feature)
        # The shape of predict_logits is (batch_size, num_classes).
        return predict_logits


class TextDataset(Dataset):
    r"""
    The class to implement the text dataset.

    Args:
        file_path: The dataset file.
        bert_model: The name of the pretrained BERT model.
    """
    def __init__(self, file_path: str, bert_model: str):
        super().__init__()
        self.contents = []
        self.pad_func = ppnlp.data.Pad()
        tokenizer = BertTokenizer.from_pretrained(bert_model)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            text, label = line.strip().split('\t')
            token_ids = tokenizer(text)['input_ids']
            self.contents.append((token_ids, int(label)))
        self.len_data = len(self.contents)

    def __getitem__(self, idx: int):
        return self.contents[idx]

    def __len__(self):
        return self.len_data


def train(
        num_filter: int, kernel_size: int, circuit_depth: int, padding: int, num_classes: int,
        model_name: str, dataset: str, batch_size: int, num_epochs: int,
        bert_model: str = 'bert-base-chinese', hidden_size: int = 768,
        learning_rate: float = 0.01,
        saved_dir: str = '', using_validation: bool = False,
        early_stopping: int = 1000,
) -> None:
    r"""
    The function of training the BERT-QTC model.

    Args:
        num_filter: The number of the qubits which the quantum circuit contains.
        kernel_size: The size of the kernel, which means the number of the qubits.
        circuit_depth: The depth of the quantum circuit, which means the number of repetitions of the circuit template.
        padding: The padding size. It will pad in the left and the right sides with elements of the same length
            if the input type is int. If the input type is list, it should be [pad_left, pad_right].
        num_classes: The number of the classes which the model will classify.
        model_name: The name of the model. It is the filename of the saved model.
        dataset: The dataset used to train the model, which should be a directory.
        batch_size: The size of the batch samplers.
        num_epochs: The number of the epochs to train the model.
        bert_model: The name of the pretrained BERT model. Defaults to ``bert-base-chinese`` .
        hidden_size: The size of the hidden state obtained through the BERT model. Defaults to ``768`` .
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

    def collate_fn(batch_data: List[tuple], pad_func: Callable):
        batch_token_ids = [text for text, _ in batch_data]
        batch_labels = [label for _, label in batch_data]
        return paddle.to_tensor(pad_func(batch_token_ids)), paddle.to_tensor(batch_labels)

    train_dataset = TextDataset(file_path=f'{dataset}train.txt', bert_model=bert_model)
    if using_validation:
        dev_dataset = TextDataset(file_path=f'{dataset}dev.txt', bert_model=bert_model)
    test_dataset = TextDataset(file_path=f'{dataset}test.txt', bert_model=bert_model)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=partial(collate_fn, pad_func=train_dataset.pad_func),
        num_workers=4, use_buffer_reader=True)
    if using_validation:
        dev_loader = DataLoader(
            dataset=dev_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=partial(collate_fn, pad_func=dev_dataset.pad_func),
            num_workers=4, use_buffer_reader=True)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=partial(collate_fn, pad_func=test_dataset.pad_func),
        num_workers=4, use_buffer_reader=True)

    decoder = Decoder(
        num_filter=num_filter, kernel_size=kernel_size, circuit_depth=circuit_depth, padding=padding,
        num_classes=num_classes, hidden_size=hidden_size)
    model = BERTQTC(bert_model=bert_model, decoder=decoder)
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.decoder.parameters())
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
        for token_ids, labels in train_loader:
            p_bar.update(1)
            model.clear_gradients()
            logits = model(token_ids)
            # The shape of logits is (batch_size, num_classes).
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.minimize(
                loss, parameters=model.decoder.parameters(),
                no_grad_set=model.bert_model.parameters())
            opt.clear_grad()
            if total_batch % 10 == 0 and total_batch > 0:
                predictions = paddle.argmax(logits, axis=1).tolist()
                labels = labels.tolist()
                train_acc = sum(labels[idx] == predictions[idx] for idx in range(len(labels))) / len(labels)
                if using_validation:
                    with paddle.no_grad():
                        dev_loss, dev_acc = evaluate(model, dev_loader)
                        if dev_loss < dev_best_loss:
                            paddle.save(decoder.state_dict(), f'{saved_dir}/{model_name}.pdparams')
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
                        paddle.save(decoder.state_dict(), f'{saved_dir}{model_name}.pdparams')
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
        paddle.save(decoder.state_dict(), f'{saved_dir}/{model_name}.pdparams')
        with paddle.no_grad():
            test_loss, test_acc = evaluate(model, test_loader)
        msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
        logging.info(msg)
        print(msg)


def evaluate(model: paddle.nn.Layer, data_loader: DataLoader) -> Tuple[float, float]:
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
        for token_ids, labels in tqdm(data_loader):
            logits = model(token_ids)
            loss = paddle.nn.functional.cross_entropy(logits, labels, reduction='sum')
            labels = labels.tolist()
            dev_loss += loss.item()
            labels_all.extend(labels)
            predict = paddle.argmax(logits, axis=1)
            predicts_all.extend(predict.tolist())
    dev_acc = sum(labels_all[idx] == predicts_all[idx] for idx in range(len(labels_all)))
    return dev_loss / len(labels_all), dev_acc / len(labels_all)


def test(model: paddle.nn.Layer, model_path: str, test_loader: DataLoader) -> None:
    r"""
    Use the test dataset to test the model.

    Args:
        model: The model to be tested.
        model_path: The file path of the models' file.
        test_loader: The dataloader of the test dataset.
    """
    model.decoder.set_state_dict(paddle.load(model_path))
    test_loss, test_acc = evaluate(model, test_loader)
    msg = f"Test loss: {test_loss:3.5f}, acc: {test_acc:3.2%}"
    logging.info(msg)
    print(msg)


def inference(
        text: str, model_path: str, classes: List[str],
        num_filter: int, kernel_size: int, circuit_depth: int, padding: int,
        bert_model: str = 'bert-base-chinese', hidden_size: int = 768
) -> str:
    r"""
    The inference function. Using the trained model to predict new data.

    Args:
        text: The path of the image to be predicted.
        model_path: The path of the model file.
        classes: The classes of all the labels.
        num_filter: The number of the qubits which the quantum circuit contains.
        kernel_size: The size of the kernel, which means the number of the qubits.
        circuit_depth: The depth of the quantum circuit, which means the number of repetitions of the circuit template.
        padding: The padding size. It will pad in the left and the right sides with elements of the same length
            if the input type is int. If the input type is list, it should be [pad_left, pad_right].
        model_name: The name of the model. It is the filename of the saved model.
        dataset: The dataset used to train the model, which should be a directory.
        bert_model: The name of the pretrained BERT model. Defaults to ``bert-base-chinese`` .
        hidden_size: The size of the hidden state obtained through the BERT model. Defaults to ``768`` .

    Returns:
        Return the class which the model predicted.
    """
    num_classes = len(classes)
    decoder = Decoder(
        num_filter=num_filter, kernel_size=kernel_size, circuit_depth=circuit_depth, padding=padding,
        num_classes=num_classes, hidden_size=hidden_size
    )
    model = BERTQTC(bert_model=bert_model, decoder=decoder)
    model.decoder.set_state_dict(paddle.load(model_path))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    token_ids = tokenizer(text)['input_ids']
    logits = model(paddle.to_tensor([token_ids]))
    prediction = paddle.argmax(logits[0], axis=0)
    return classes[prediction]


if __name__ == '__main__':
    exit(0)
