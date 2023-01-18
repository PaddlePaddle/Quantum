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
General model templates for Quantum Neural Network
"""

import matplotlib.pyplot as plt
import time
import rich
from rich.progress import (
    Progress, TextColumn, BarColumn,
    SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
)
import numpy as np
import math
from typing import Iterable, List, Callable, Any, Tuple, Union

import paddle
from .ansatz import Sequential, Circuit
from .base import get_dtype
from .state import State
from .intrinsic import _get_float_dtype


__all__ = [
    'rcParams', 'reset_settings', 'random_batch',
    'NullScheduler', 'Model', 'OptModel', 'LearningModel', 'EncodingNetwork', 'EncodingModel'
]


# General settings for the QNN model
rcParams = {
    # settings for the optimizer
    'optimizer': 'Adam',  # optimizer in PaddlePaddle
    'scheduler': 'StepDecay',  # scheduler in PaddlePaddle: can be set to None.
    'learning_rate': 0.2,  # (initial) learning rate of the optimizer
    'scheduler_args': {100},  # arguments (in list or tuple) of the scheduler other than learning rate

    # settings for the training
    'num_itr': 200,  # number of iterations during the training
    'num_print': 10,  # number of messages printed during the training
    'print_digits': 6,  # number of decimal digits for printed number
    'print_style': 'uniform',  # the style of how printed iterations are arranged: can be 'exponential'

    # settings for the fitting
    'num_epoch': 5,  # number of epochs during the fitting
    'test_frequency': 1,  # frequency of evaluating test data set during the fitting
    'batch_size': None  # size of trained data for each epoch. Defaults to be half.
}


def reset_settings() -> None:
    r"""Set the current ``rcParams`` to Default settings.
    """
    global rcParams
    rcParams.update({
        'optimizer': 'Adam', 'scheduler': 'StepDecay', 'learning_rate': 0.1, 'scheduler_args': 20,
        'num_itr': 200, 'num_print': 10, 'print_digits': 6, 'print_style': 'uniform',
        'num_epoch': 5, 'num_tests': 5, 'batch_size': None
    })


def random_batch(data: Iterable, label: Iterable, batch_size: int = None) -> Tuple[List, List]:
    r"""Randomly return a data batch from dataset and label set
    
    Args:
        data: dataset
        label: label set
        batch_size: size of data batch. Defaults to be half of the data size.
        
    Returns:
        a random data batch
        
    """
    size_data = len(data)
    batch_idx = np.random.choice(list(range(size_data)), 
                                 size=size_data // 2 if batch_size is None else batch_size, 
                                 replace=False)
    
    data_batch, label_batch = [], []
    for idx in batch_idx:
        data_batch.append(data[idx])
        label_batch.append(label[idx])
        
    if isinstance(label, np.ndarray):
        return data_batch, np.concatenate(label_batch)
    if isinstance(label, paddle.Tensor):
        return data_batch, paddle.concat(label_batch)
    return data_batch, label_batch


class NullScheduler(paddle.optimizer.lr.LRScheduler):
    r"""An empty scheduler class, for users who expect no schedulers in Model. Can be activated by command
    
    .. code::
    
        from paddle_quantum.model import rcParams
        rcParams['scheduler'] = None
    
    """
    def get_lr(self):
        return self.base_lr


class Model(object):
    r"""General template for models of QNN
    
    Args:
        network: a quantum neural network
        name: name of this model. Default to be ``"Model"``
    
    """
    def __init__(self, network: paddle.nn.Layer, name: str = "Model") -> None:
        self.network = network
        self.name = name
        self.__prepared = False
        self.loss_data, self.metric_data = [], []
    
    def clear_data(self) -> None:
        r"""Clear the current data
        """
        self.loss_data, self.metric_data = [], []
    
    def parameters(self) -> List[paddle.fluid.framework.ParamBase]:
        r"""Return the parameters of this network
        """
        return self.network.parameters()
    
    def __validate_settings(self) -> None:
        r"""Assert whether rcParams can be a valid setting in Model
        """
        setting_list = ['num_itr', 'num_print', 'print_digits', 'num_epoch', 'test_frequency', 'batch_size']
        # assert non-negative
        if any((rcParams[key] is not None) and (rcParams[key] < 0) for key in setting_list):
            raise ValueError(
                "Received negative numbers: check rcParams.")
    
        # assert inequality
        if rcParams['num_print'] >= rcParams['num_itr']:
            raise ValueError(
                "The number of messages cannot be larger than the number of iterations: check rcParams.")
    
    def __print_itr_generation(self, num_itr: int, num_print: int):
        r"""Determine the list of iterations to be printed during the training process
        """
        if num_itr < 1:
            return []    
        
        if rcParams['print_style'] == 'exponential':
            # print list is generated by exponential distribution
            lamb = 4 * math.log(2) / num_itr
            poisson = lambda x: lamb * math.exp(-lamb * x)
            list_itr = list(range(1, num_itr))
            print_prob = [poisson(itr) for itr in list_itr]
            print_prob /= np.sum(print_prob)
            print_list = np.sort(np.random.choice(list_itr, size=num_print - 1, replace=False, p=print_prob)).tolist()
            print_list.append(num_itr)
        else:
            # print list is uniformly distributed
            print_ratio = num_itr / num_print
            print_list = list(filter(lambda itr: itr % print_ratio < 1, list(range(1, num_itr + 1))))
            print_list = print_list + [num_itr] if print_list[-1] != num_itr else print_list
        
        return print_list
    
    def __scheduler_step(self, scheduler_name: str) -> List:
        r""" Generate the argument for the scheduler.step()
        """
        return (lambda x: [x]) if scheduler_name == 'ReduceOnPlateau' else (lambda x: [])
    
    def prepare(self, loss_fcn: Callable[[Union[State, Circuit, Sequential], Any], Any],
                metric_fcn: Callable[[Union[Circuit, Sequential]], float] = None, 
                metric_name: str = None) -> None:
        r"""General function setup for QNN
        
        Args:
            loss_fcn: loss function for the QNN
            metric_fcn: metric function for the QNN, which does not mess with the training process. Defaults to be ``None``
            metric_name: name of the metric function. Defaults to be ``None``
            
        Raises:
            ValueError: the output datatype of metric function must be float
            
        Note:
            The prepare function will also take the settings of ``rcParams``
        
        """
        # we cannot generally check loss function since the label input is unknown
        self._loss_fcn = loss_fcn
        
        # sanity check for metric function
        if metric_fcn is not None:
            data = metric_fcn(self.network)
            if not isinstance(data, float):
                raise ValueError(
                    f"The output data of print function must be a float: received {type(data)}.")
            if metric_name is None:
                raise ValueError(
                    "The metric_name cannot be empty for a metric function.")
        
        self._metric_fcn, self._metric_name = metric_fcn, metric_name
        
        # setup the scheduler and optimizer
        self.__sch = NullScheduler(learning_rate=rcParams['learning_rate']) if rcParams['scheduler'] is None else \
            getattr(paddle.optimizer.lr, rcParams['scheduler'])(rcParams['learning_rate'], *rcParams['scheduler_args'])
        self.__step = self.__scheduler_step(rcParams['scheduler'])
        self.__opt = getattr(paddle.optimizer, rcParams["optimizer"])(learning_rate=self.__sch, parameters=self.parameters())
        
        # take the setting of rcParams
        self.__validate_settings()
        self.__print_list = self.__print_itr_generation(rcParams['num_itr'], rcParams['num_print'])
        self.__num_itr, self.__print_digits = rcParams['num_itr'], rcParams['print_digits']
        self.__num_epoch, self.__test_frequency, self.__batch_size = rcParams['num_epoch'], rcParams['test_frequency'], rcParams['batch_size']
        
        self.__prepared = True
        
    def check_prepared(self) -> None:
        r"""Assert whether Model is prepared for training
        """
        assert self.__prepared, \
                f"The model was not set properly: run the prepare function of {self.name} first."
    
    def __print_loss_message(self, loss: float, metric: float, metric_name: str, 
                            itr: int, digits: int, itr_max_digits: int) -> None:
        r"""Message print function used in loss training
        """
        loss = f'% .{digits}f' % loss
        
        if metric_name is not None:
            metric = f'% .{digits}f' % metric
            
        diff_digits = itr_max_digits - math.floor(math.log10(itr))
        itr = str(itr) + ''.join([' ' for _ in range(diff_digits)])
        rich.print(f"        iter: [magenta]{itr}[/magenta]     [i]loss[/i]: [red]{loss}[/red]",
                "" if metric_name is None else f"with [i]{metric_name}[/i] as [green]{metric}[/green]")

    def __print_summary_message(self, best_loss: float, best_loss_itr: int, 
                                best_metric: float, best_metric_itr: int, metric_name: str,
                                model_name: str, total_time: float, num_itr: int) -> None:
        r"""Message print function used after loss training
        """
        rich.print(f"    the [u]best [i]loss[/i][/u] is {best_loss} existing at iter [magenta]{best_loss_itr}[/magenta];")
        if metric_name is not None:
            rich.print(f"    the [u]best [i]{metric_name}[/i][/u] is {best_metric} existing at iter [magenta]{best_metric_itr}[/magenta];")

        avg_time = round(total_time / num_itr, 3)
        total_time = round(total_time, 3)
        rich.print(f"    {model_name} took [gold3]{total_time}[/gold3] seconds",
                    f"with [gold3]{avg_time}[/gold3] seconds per iteration in average.\n")
    
    def train(self, loss_generator: Callable[[Any], Any] = None) -> Union[List[float], Tuple[List[float], List[float]]]:
        r"""General template for QNN training
        
        Args:
            loss_generator: loss generator of the QNN, with ``Model.network`` as input. Defaults to ``None`` i.e. 
            use the loss function defined in ``Model.prepare``
        
        Returns:
            contains the following elements:

                - a list of loss values
                - a list of metric values if a metric function is given

        """
        loss_generator = self._loss_fcn if loss_generator is None else loss_generator
        num_itr, print_list, digits = self.__num_itr, self.__print_list.copy(), self.__print_digits
        metric_name = self._metric_name

        loss_list, metric_list = [], []
        best_loss, best_loss_itr = float('inf'), 0
        metric, best_metric, best_metric_itr = None, 0, 0

        itr_max_digits = math.floor(math.log10(num_itr))
        
        # start training
        print()
        start_time = time.time()
        with Progress(TextColumn("[progress.description]{task.description}"),
                      SpinnerColumn(),
                      BarColumn(),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                      TimeRemainingColumn(),
                      TimeElapsedColumn()) as progress:
            train_tqdm = progress.add_task(description="Training...", total=num_itr)
            for itr in range(1, num_itr + 1):
                
                # evaluate loss
                loss = loss_generator(self.network)
                loss.backward()
                
                # evaluate and update metric
                if metric_name is not None:
                    metric = self._metric_fcn(self.network)
                    metric_list.append(metric)
                    if best_metric <= metric:
                        best_metric, best_metric_itr = metric, itr
                
                self.__opt.minimize(loss)
                self.__opt.clear_grad()
                self.__sch.step(*self.__step(loss))

                # update loss
                loss = loss.item()
                loss_list.append(loss)
                if loss <= best_loss:
                    best_loss, best_loss_itr = loss, itr
                    
                if itr in print_list:
                    self.__print_loss_message(loss, metric, metric_name, itr, digits, itr_max_digits)
                    print_list.pop(0)
                    
                progress.advance(train_tqdm, advance=1)
        
        total_time = time.time() - start_time
        self.__print_summary_message(best_loss, best_loss_itr, best_metric, best_metric_itr,
                                     metric_name, self.name, total_time, num_itr)
        
        return loss_list if metric_name is None else (loss_list, metric_list)
    
    def evaluate(self, loss_generator: Callable[[Any], Any] = None) -> Union[float, Tuple[float, float]]:
        r"""General template for QNN evaluation
        
        Args:
            loss_generator: loss generator of the QNN, with ``Model.network`` as input. Defaults to ``None`` i.e. 
            use the loss function defined in ``Model.prepare``
        
        Returns:
            contains the following elements:

            - a loss value
            - a metric value if a metric function is given

        """
        loss_generator = self._loss_fcn if loss_generator is None else loss_generator
        loss = loss_generator(self.network).item()
        return loss if self._metric_name is None else (loss, self._metric_fcn(self.network))
    
    def fit(self, train_data: Iterable, train_label: Iterable, 
            test_data: Iterable, test_label: Iterable) -> None:
        r"""General template for QNN data fitting
        
        Args:
            train_data: data of the train set
            train_label: label of the train set
            test_data: data of the test set
            test_label: label of the test set
        
        """ 
        train_size, test_size = len(train_data), len(test_data)
        num_epoch, test_frequency = self.__num_epoch, self.__test_frequency, 
        batch_size = max(train_size // num_epoch, 2) if self.__batch_size is None else self.__batch_size
        
        assert batch_size <= train_size, \
            f"The batch size cannot be larger than the size of train dataset: \
                received {batch_size}, expected no larger than {train_size}"
        
        rich.print(f"\nFitting of {self.name} starts under setting: \n", 
                   f"    size of train data: [medium_violet_red]{train_size}[/medium_violet_red],",
                   f"size of test data: [medium_violet_red]{test_size}[/medium_violet_red],",
                   f"size of batches: [medium_violet_red]{batch_size}[/medium_violet_red]. \n")
        
        self.clear_data()
        for epoch in range(1, num_epoch + 1):
            data_batch, label_batch = random_batch(train_data, train_label, batch_size)
            rich.print(f"[bold bright_blue]Epoch {epoch}[/bold bright_blue] of {self.name} begins:")
            result = self.train_batch(data_batch, label_batch)
            
            if self._metric_name is None:
                self.loss_data.extend(result)
            else:
                self.loss_data.extend(result[0])
                self.metric_data.extend(result[1])

            if epoch % test_frequency < 1 or epoch == num_epoch:
                evaluation = self.eval_batch(test_data, test_label)
                evaluation = evaluation if self._metric_name is None else evaluation[0]
                rich.print(f"TEST:  the [i]loss[/i] of the test set is [u bold yellow]{evaluation}[/u bold yellow].\n")

    def plot(self, include_metric: bool, apply_log: bool, has_epoch: bool) -> None:
        r"""Plot the trained data
        
        Args:
            include_metric: whether include the metric data
            apply_log: whether apply the log function on the data
            has_epoch: whether the data list is composed by data in several epochs
        
        """
        loss_data, metric_data = np.array(self.loss_data), np.array(self.metric_data) if include_metric else None
        if apply_log:
            assert all(i > 0 for i in loss_data.flatten()), \
                "Cannot apply log function on non-positive loss: check your loss data"
            loss_data = np.log(loss_data)
            
            if include_metric:
                assert all(i > 0 for i in metric_data.flatten()), \
                    "Cannot apply log function on non-positive metric: check your metric data"
                metric_data = np.log(metric_data)
        
        data_size, min_loss, max_loss = len(loss_data), min(loss_data), max(loss_data)
        assert (not include_metric) or data_size == len(metric_data), \
            "The metric data size does not agree with the loss data."
        
        x_list = list(range(data_size))
        plt.figure(figsize=[16, 9])
        plt.xlabel("iterations")
        plt.plot(x_list, loss_data, color="blue", ls="-", label="loss")
        if include_metric:
            plt.plot(x_list, metric_data, color="red", ls="-", label=self._metric_name)
        if has_epoch and self.__num_epoch > 1:
            epoch_itr = list(range(0, data_size, self.__num_itr))
            epoch_itr.pop(0)
            for itr in epoch_itr:
                plt.axvline(itr, ls=':', color='black')
        plt.legend(prop={"size": 12})
        plt.show()


class OptModel(Model):
    r"""Class for optimization-based QNN model

    Args:
        circuit: a ``Circuit`` instance ready to be optimized
        name: name of this model. Default to be ``"OptModel"``
        
    """
    def __init__(self, circuit: Circuit, name: str = "OptModel") -> None:
        super().__init__(network=circuit, name=name)
        
    def prepare(self, loss_fcn: Callable[[Circuit, Any], paddle.Tensor], 
                metric_fcn: Callable[[Union[Circuit, Sequential]], float] = None, 
                metric_name: str = None, *loss_args: Any) -> None:
        r"""Prepare and check the function setup for optimized-based QNN
        
        Args:
            loss_fcn: loss function for the QNN
            metric_fcn: metric function for the QNN, which does not mess with the training process. Defaults to be ``None``
            metric_name: name of the metric function. Defaults to be ``None``
            loss_args: function arguments for loss_fcn other than QNN input
            
        Raises:
            ValueError: the output datatype of loss function must be paddle.Tensor
            
        Note:
            The prepare function will also take the settings of ``rcParams``
        
        """
        # since optimization-based model has no labels, we can check the validity of loss function
        data = paddle.squeeze(loss_fcn(self.network, *loss_args))
        if not isinstance(data, paddle.Tensor) and data.shape == []:
            raise ValueError(
                f"The output data of loss function must be a tensor: received type {type(data)} and shape {data.shape}.")
        
        # fix loss function by loss arguments
        fixed_loss_fcn = lambda network: loss_fcn(network, *loss_args)
        return super().prepare(fixed_loss_fcn, metric_fcn, metric_name)
        
    def optimize(self) -> Union[List[float], Tuple[List[float], List[float]]]:
        r"""Optimize the circuit in terms of the loss function
            
        Returns:
            contains the following elements:

            - a list of loss values
            - a list of metric values if a metric function is given
        
        """
        self.check_prepared()
        print(f"\nTraining of {self.name} begins:")
        result = super().train()
        
        # log data
        if self._metric_name is None:
            self.loss_data = result
        else:
            self.loss_data, self.metric_data = result
        return result
    
    def evaluate(self) -> Union[float, Tuple[float, float]]:
        r"""Evaluate the loss and metric value of the current QNN
        
        Returns:
            contains the following elements:

            - a loss value
            - a metric value if a metric function is given
        
        """
        self.check_prepared()
        return super().evaluate()

    def fit(self) -> None:
        r"""
        Raises:
            NotImplementedError: Optimization model does not have fit function

        """
        raise NotImplementedError(
            "Optimization-model does not have fit function: please use OptModel.optimize directly")

    def plot(self, include_metric: bool = True, apply_log: bool = False) -> None:
        r"""Plot the loss (and metric) data
        
        Args:
            include_metric: include the metric data. Defaults to be ``True``.
            apply_log: whether apply the log function on the data. Defaults to be ``False``.
        
        """
        if np.size(self.loss_data) == 0:
            raise ValueError(
                "The data of this model is empty: run OptModel.optimize first.")
        
        return super().plot(False if self._metric_name is None else include_metric, apply_log, False)


class LearningModel(Model):
    r"""Class for learning-based QNN model
    
    Args:
        circuit: a ``Circuit`` instance ready to be trained
        name: name of this model. Default to be ``"LearningModel"``
    
    """
    def __init__(self, circuit: Circuit, name: str = "LearningModel") -> None:
        super().__init__(network=circuit, name=name)
        self.__is_fitting = False # whether Model is fitting
    
    def prepare(self, loss_fcn: Callable[[State, Any, Any], Any], 
                metric_fcn: Callable[[Union[Circuit, Sequential]], float] = None, 
                metric_name: str = None, *loss_args: Any) -> None:
        r"""Prepare and check the function setup for learning-based QNN
        
        Args:
            loss_fcn: loss function for the output data of QNN
            metric_fcn: metric function for the QNN, which does not mess with the training process. Defaults to be ``None``
            metric_name: name of the metric function. Defaults to be ``None``
            loss_args: function arguments for loss_fcn other than QNN and label inputs
            
        Note:

            - The ``data`` input of this Model needs to be ``paddle_quantum.State``. Use ``EncodingModel`` if data is 
              expected to be encoded into quantum circuits
            - The prepare function will take the settings of ``rcParams``

        """
        # fix loss function by loss arguments
        fixed_loss_fcn = lambda output_states, label: loss_fcn(output_states, label, *loss_args)
        return super().prepare(fixed_loss_fcn, metric_fcn, metric_name)
        
    def train_batch(self, data: List[State], label: List[Any]) -> Union[List[float], Tuple[List[float], List[float]]]:
        r"""Train the circuit by input batch data
        
        Args:
            data: list of input ``State`` s
            label: expected label
            
        Returns:
            contains the following elements:

            - a list of loss values 
            - a list of metric values if a metric function is given

        """
        self.check_prepared()
        
        # define the network loss function
        def network_loss_fcn(cir: Circuit) -> paddle.Tensor:
            output_states = [cir(state) for state in data]
            return self._loss_fcn(output_states, label)
        
        if not self.__is_fitting:
            print(f"\nTraining of {self.name} begins:")
        return super().train(loss_generator=network_loss_fcn)
    
    def eval_batch(self, data: List[State], label: List[Any]) -> Union[float, Tuple[float, float]]:
        r"""Evaluate the circuit by input batch data
        
        Args:
            data: list of input ``State`` s
            label: expected label
        
        Returns:
            contains the following elements:
    
            - a loss value
            - a metric value if a metric function is given
        
        """
        self.check_prepared()
        # define the network loss function
        def network_loss_fcn(cir: Circuit) -> paddle.Tensor:
            output_states = [cir(state) for state in data]
            return self._loss_fcn(output_states, label)
        return super().evaluate(loss_generator=network_loss_fcn)
    
    def fit(self, train_data: List[State], train_label: Iterable, 
            test_data: List[State], test_label: Iterable) -> None:
        r"""Fit the circuit by input train_data
        
        Args:
            train_data: data of the train set
            train_label: label of the train set
            test_data: data of the test set
            test_label: label of the test set
        
        """ 
        self.check_prepared()
        self.__is_fitting = True
        
        train_size = len(train_data)
        assert isinstance(train_data[0], State), \
            f"The input data must be paddle_quantum.State: received {type(train_data[0])}"
        assert train_size == len(train_label), \
            f"The size of train data should be the same as that of labels: \
                received {len(train_label)}, expected {train_size}"
        assert len(test_data) == len(test_label), \
            f"The size of test data should be the same as that of labels: \
                received {len(test_label)}, expected {len(test_data)}"
        
        super().fit(train_data, train_label, test_data, test_label)
        self.__is_fitting = False
        
    def plot(self, include_metric: bool = True, apply_log: bool = False) -> None:
        r"""Plot the loss (and metric) data
        
        Args:
            include_metric: include the metric data. Defaults to be ``True``.
            apply_log: whether apply the log function on the data. Defaults to be ``False``.
        
        """
        if np.size(self.loss_data) == 0:
            raise ValueError(
                "The data of this model is empty: run LearningModel.fit first.")
        
        return super().plot(False if self._metric_name is None else include_metric, apply_log, True)


class EncodingNetwork(paddle.nn.Layer):
    r"""QNN for Encoding model
    
    Args:
        encoding_func: an encoding function that determines how to construct quantum circuits
        param_shape: the shape of input parameters
        initial_state: the initial state of circuits
        
    Note:
        Used for ``paddle_quantum.model.EncodingModel`` only.
    
    """
    def __init__(self, encoding_func: Callable[[Any, paddle.Tensor], Circuit],
                 param_shape: Iterable[int], initial_state: State) -> None:
        super().__init__()
        float_dtype = _get_float_dtype(get_dtype())
        theta = self.create_parameter(
                shape=param_shape, dtype=float_dtype,
                default_initializer=paddle.nn.initializer.Uniform(low=0, high=2 * math.pi),
        )
        self.add_parameter('theta', theta)
        
        self.encoding_func = encoding_func
        self.initial_state = initial_state
        
    def forward(self, input_data: List[Any]) -> List[State]:
        r"""Compute the output states corresponding to the input data
        
        Args:
            input_data: the list of input data that encodes circuits
            
        Returns:
            the output states from these circuits
        
        """
        return [self.encoding_func(data, self.theta)(self.initial_state) for data in input_data]


class EncodingModel(Model):
    r"""Class for encoding-based QNN model
    
    Args:
        encoding_fcn: an encoding function that determines how to construct quantum circuits by the encoding
    data and the parameters.
        param_shape: the shape of input parameters for ``encoding_func``
        initial_state: the initial state of circuits. Default to be ``None`` i.e. the zero state
        name: name of this model. Default to be ``"EncodingModel"``
        
    Note:
        Unlike LearningModel, the data of ``EncodingModel`` is encoded into quantum circuits instead of states.
        Therefore, ``EncodingModel`` requires the information of how circuits are encoded by input classical data.
        ``EncodingModel`` will automatically generate the parameters according to input ``param_shape``.
    
    """
    def __init__(self, encoding_fcn: Callable[[Any, paddle.Tensor], Circuit],
                 param_shape: Iterable[int], initial_state: State = None,
                 name: str = "EncodingModel") -> None:
        network = EncodingNetwork(encoding_fcn, param_shape, initial_state)
        
        super().__init__(network=network, name=name)
        self.__is_fitting = False # whether Model is fitting
    
    def prepare(self, loss_fcn: Callable[[State, Any, Any], Any], 
                metric_fcn: Callable[[Union[Circuit, Sequential]], float] = None, 
                metric_name: str = None, *loss_args: Any) -> None:
        r"""Prepare and check the function setup for encoding-based QNN
        
        Args:
            loss_fcn: loss function for the output data of QNN
            metric_fcn: metric function for the QNN, which does not mess with the training process. Defaults to be ``None``
            metric_name: name of the metric function. Defaults to be ``None``
            loss_args: function arguments for loss_fcn other than QNN and label inputs
            
        Note:
            The prepare function will take the settings of ``rcParams``

        """
        # fix loss function by loss arguments
        fixed_loss_fcn = lambda output_states, label: loss_fcn(output_states, label, *loss_args)
        return super().prepare(fixed_loss_fcn, metric_fcn, metric_name)
        
    def train_batch(self, data: Iterable, label: Iterable) -> Union[List[float], Tuple[List[float], List[float]]]:
        r"""Train the circuit by input batch data
        
        Args:
            data: list of input data
            label: expected label
            
        Returns:
            contains the following elements:

            - a list of loss values 
            - a list of metric values if a metric function is given
        
        """
        self.check_prepared()
        
        # define the network loss function
        def network_loss_fcn(network: paddle.nn.Layer) -> paddle.Tensor:
            return self._loss_fcn(network(data), label)
        
        if not self.__is_fitting:
            print(f"\nTraining of {self.name} begins:")
        return super().train(loss_generator=network_loss_fcn)
    
    def eval_batch(self, data: Iterable, label: Iterable) -> Union[float, Tuple[float, float]]:
        r"""Evaluate the circuit by input batch data
        
        Args:
            data: list of input data
            label: expected label
        
        Returns:
            contains the following elements:

            - a loss value
            - a metric value if a metric function is given
        
        """
        self.check_prepared()
        # define the network loss function
        def network_loss_fcn(network: paddle.nn.Layer) -> paddle.Tensor:
            return self._loss_fcn(network(data), label)
        return super().evaluate(loss_generator=network_loss_fcn)
    
    def fit(self, train_data: Iterable, train_label: Iterable, 
            test_data: Iterable, test_label: Iterable) -> None:
        r"""Fit the circuit by input train_data
        
        Args:
            train_data: data of the train set
            train_label: label of the train set
            test_data: data of the test set
            test_label: label of the test set
        
        """ 
        self.check_prepared()
        self.__is_fitting = True
        
        train_size = len(train_data)
        assert train_size == len(train_label), \
            f"The size of train data should be the same as that of labels: \
                received {len(train_label)}, expected {train_size}"
        assert len(test_data) == len(test_label), \
            f"The size of test data should be the same as that of labels: \
                received {len(test_label)}, expected {len(test_data)}"
        
        super().fit(train_data, train_label, test_data, test_label)
        self.__is_fitting = False
        
    def plot(self, include_metric: bool = True, apply_log: bool = False) -> None:
        r"""Plot the loss (and metric) data
        
        Args:
            include_metric: include the metric data. Defaults to be ``True``.
            apply_log: whether apply the log function on the data. Defaults to be ``False``.
        
        """
        if np.size(self.loss_data) == 0:
            raise ValueError(
                "The data of this model is empty: run EncodingModel.fit first.")
        
        return super().plot(False if self._metric_name is None else include_metric, apply_log, True)
