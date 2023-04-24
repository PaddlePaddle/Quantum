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
The VQR model.
"""

# system libs
# import os
from tqdm import tqdm
from typing import Optional, List, Tuple, Union, Callable
import logging

# common libs
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import metrics
SKLEARN_METRIC = metrics.__dict__
SKLEARN_REG_SCORER = ['explained_variance_score', 'max_error', 'mean_absolute_error', 
                      'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error', 
                      'mean_absolute_percentage_error', 'mean_pinball_loss', 'r2_score', 
                      'mean_tweedie_deviance', 'mean_poisson_deviance', 'mean_gamma_deviance', 
                      'd2_tweedie_score', 'd2_pinball_score', 'd2_absolute_error_score']
import warnings

# paddle libs
import paddle
import paddle_quantum as pq
from ..ansatz import Circuit
from ..state import State
from ..intrinsic import _type_fetch, _type_transform
from paddle_quantum.loss import Measure

# format figures
font_legend = font_manager.FontProperties(family='Times New Roman',
                                            # weight='bold',
                                            style='normal', size=12)
font_label = {"family": "Times New Roman", "size": 14}    

warnings.filterwarnings("ignore", category=Warning)
pq.set_backend('state_vector')
pq.set_dtype("complex128")


# loading dataset
def load_dataset(data_file: str, model_name: str):
    r"""Loading the Kaggle regression model

     You may obtain a copy of the License at https://www.kaggle.com/code/alirfat/starter-fish-market-8f18fa38-4/data .

    Args:
        data_file: Dataset file name.
        model_name: should be either ``linear`` or ``poly`` .

    Returns:
        Raw data.
    """
    if model_name not in ["linear", "poly"]:
        raise ValueError("Invalid regression model for the dataset.")

    df = pd.read_csv(data_file) # ./datasets/Fish.csv
    
    return df


# Kernel for evaluating inner product using PQ.
# Kernel 1 - direct encoding
def IPEstimator(circuit: Circuit, input_state: State, measure_idx: List[int] = [0]) -> paddle.Tensor:
    """Kernel-1 using direct encoded data state to evaluate inner product

    Args:
        circuit: Executed quantum circuit
        input_state: Input state. Defaults to None.

    Returns:
        The value of inner product
    """
    evolve_state = circuit(input_state)
    # measure the first qubit
    measure_instance = Measure()
    prob = measure_instance(evolve_state, qubits_idx=measure_idx, desired_result='0')
    result = (prob - 0.5) * 2

    return result


def _data_transform_(X: Union[paddle.Tensor, np.ndarray], y: Union[paddle.Tensor, np.ndarray]) -> Tuple[paddle.Tensor]:
    r"""Normalize classical data

    Args:
        X: Independent data in an array.
        y: Dependent data in an array.

    Returns:
        Normalized data
    """
    rawX, rawy = _type_transform(X, "numpy"), _type_transform(y, "numpy")

    Xdata_norm = normalize(rawX, axis=0)
    ydata_norm = rawy / np.linalg.norm(rawy)
    
    return _type_transform(Xdata_norm, "tensor"), _type_transform(ydata_norm, "tensor")


def _data_verifying_(num_qubits: int, X: Union[paddle.Tensor, np.ndarray], y: Union[paddle.Tensor, np.ndarray]) -> None:
    r"""Verifying the data dimension
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("Input dimension does not match!")
    if 2**num_qubits < y.shape[0]:
        raise ValueError("Insufficient number of qubits for a batch fitting.")


def _dtype_transform_(arg: Union[paddle.Tensor, np.ndarray], dt: str = "float32"):
    r"""dtype transforming
    """
    type_str = _type_fetch(arg)
    arg = _type_transform(arg, 'numpy').astype(dt)
    return _type_transform(arg, type_str)


def _logger_init_(log_name: str, filename: str):
    r"""Initialize a logger

    Args:
        log_name: logger name
        filename: filename
    """
    logger = logging.Logger(name=log_name)
    handler = logging.FileHandler(filename=filename)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


class QRegressionModel:
    r"""Regression model covering all classes.

    Args:
        data_file: The dataset.csv file.
        model_name: The regression model.
        x_feature: Independent variables from data labels.
        y_feature: Dependent feature values.
        num_variable: The number of variable initialized in the model.
        init_params: The initial parameters.
        num_qubits: The number of qubits in the estimator. Defaults to 6.
        learning_rate: The learning rate. Defaults to 0.1.
        iteration: The number optimization steps. Defaults to 100.
        language: The print language, Defaults to ``CN`` .
    """

    def __init__(self, data_file: str, model_name: str, x_feature: List[str],
                 y_feature: str, num_variable: int, init_params: List[float],
                 num_qubits: int = 6, learning_rate: float = 0.1, 
                 iteration: int = 100, language: str = 'CN') -> None:
        self.data = load_dataset(data_file, model_name)
        self.x_data = self.data[x_feature].values.astype("float32")
        self.y_data = self.data[y_feature].values.astype("float32").flatten()

        self.model_name = model_name
        self.x_feature = x_feature
        self.y_feature = y_feature
        self.num_variable = num_variable
        self.num_qubits = num_qubits
        self.lr = learning_rate
        self.itr = iteration
        self.lrg = language

        # initialize the regression model
        if model_name == "linear":
            if len(self.x_feature) != self.num_variable:
                raise ValueError(f"Invalid number of independent variables of 'linear': expected {len(self.x_feature)}, received {self.num_variable}.")
            model = LinearRegression(num_qubits = self.num_qubits + 1, num_x = self.num_variable)
            self.variable = [self.x_feature[i] for i in range(self.num_variable)]
        elif model_name == "poly":
            if len(self.x_feature) != 1:
                raise ValueError(f"Invalid number of independent variables of 'poly': expected 1, received {len(self.x_feature)}.")
            model = PolyRegression(num_qubits = self.num_qubits + 1, order = self.num_variable)
            self.variable = [f"{self.x_feature[0]}^{i+1}" for i in range(self.num_variable)]
        else:
            raise ValueError("Invalid model name. Should be either 'linear' or 'poly'.")
        
        # set parameters using set_params
        model.set_params(np.array(init_params, dtype="float32"))
        self.model = model
        self._outcome_printer_()

    def _outcome_printer_(self):
        if self.lrg == "CN":
            print(f"模型是否被训练：{self.model.status}。当前模型 R2 得分：{self.model.score(self.x_data, self.y_data):2.5f}。")
            print(f"拟合后的模型为：{self.y_feature[0]} = " + f"{self.model.reg_param[0].item():2.5f} + " + 
                    "".join([f"({self.model.reg_param[var_i + 1].item():2.5f}*{self.variable[var_i]}) + " for var_i in range(len(self.variable))])[:-3] + "。"
                 )
        elif self.lrg == "EN":
            print(f"Model status: {self.model.status}. Current regression R2 score: {self.model.score(self.x_data, self.y_data):2.5f}. ")
            print(f"The trained model: {self.y_feature[0]} = " + f"{self.model.reg_param[0].item():2.5f} + " + 
                    "".join([f"({self.model.reg_param[var_i + 1].item():2.5f}*{self.variable[var_i]}) + " for var_i in range(len(self.variable))])[:-3] + ". "
                 )
        else:
            print("The language style is not found. Translate to 'EN'. ")
            print(f"Model status: {self.model.status}. Current regression R2 score: {self.model.score(self.x_data, self.y_data):2.5f}. ")
            print(f"The trained model: {self.y_feature[0]} = " + f"{self.model.reg_param[0].item():2.5f} + " + 
                    "".join([f"({self.model.reg_param[var_i + 1].item():2.5f}*{self.variable[var_i]}) + " for var_i in range(len(self.variable))])[:-3] + ". "
                 )

    def regression_analyse(self):
        # optimize the model parameters to complete the linear regression analysis
        self.model.fit(self.x_data, self.y_data, learning_rate = self.lr, iteration = self.itr)
        self._outcome_printer_()
        
        fig = plt.figure(figsize=(6,4))
        # predict with optimized model
        if self.model_name == 'linear':

            fitted_predict = self.model.predict(self.x_data)
            # baseline y_true = y_pred
            x_base = np.linspace(0, 100, 50)
            y_base = x_base
            plt.scatter(fitted_predict, self.y_data, marker="^", 
                        label = "predict data", color="#800000")
            plt.plot(x_base, y_base, "--", color="#008000")  
            plt.xlabel("True value", fontdict=font_label)
            plt.ylabel("Predict", fontdict=font_label)

        elif self.model_name == "poly":

            x_base = np.linspace(0, 9, 50)
            fitted_predict = self.model.predict(x_base.reshape([50,1]))
            plt.plot(x_base, fitted_predict, "--", 
                        label = "predict data", color="#800000")
            plt.scatter(self.x_data, self.y_data, marker="*", 
                        label = "test data", color="#108090")
            plt.xlabel("Independent variable", fontdict=font_label)
            plt.ylabel("Dependent variable", fontdict=font_label)

        plt.title("Fish dataset", fontdict=font_label)
        plt.grid()
        plt.legend(prop=font_legend)
        plt.show()

        return self.model


class LinearRegression(paddle.nn.Layer):
    r"""Regression class for initializing a quantum linear regression model

    Args:
        num_qubits: The number of qubits which the quantum circuit contains.
        num_x: The the number of independent variables of data. Defaults to ``1``.
    """

    def __init__(self, num_qubits: int, num_x: int = 1) -> None:
        super(LinearRegression, self).__init__(name_scope = "LinearRegression")
        self.num_qubits = num_qubits
        self.num_x = num_x
        param = self.create_parameter([self.num_x + 1], 
                                          attr=None, dtype="float32", 
                                          is_bias=False, 
                                          default_initializer=None)
        self.add_parameter("param", param)
        self.status = False
    
    @property
    def reg_param(self) -> paddle.Tensor:
        r"""Flattened parameters in the Layer.
        """
        if len(self.parameters()) == 0:
            return []
        return paddle.concat([paddle.flatten(param) for param in self.parameters()])
    
    def set_params(self, new_params: Union[paddle.Tensor, np.ndarray]) -> None:
        r"""set parameters of the model.

        Args:
            params: New parameters
        """

        if not isinstance(new_params, paddle.Tensor):
            new_params = paddle.to_tensor(new_params, dtype='float32')
        new_params = paddle.flatten(new_params)

        if new_params.shape[0] != self.num_x + 1:
            raise ValueError(f"Incorrect number of params for the model: expect {self.num_x + 1}, received {new_params.shape[0]}")

        update_param = paddle.create_parameter(
                        shape=self.param.shape,
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(new_params.reshape(self.param.shape)),
                    )

        setattr(self, 'param', update_param)
    
    def _init_state_preparation_(self, X_data: paddle.Tensor, y_data: paddle.Tensor) -> State:
        r"""Generate an initial state for compute inner product
        """
        X_data, y_data = _dtype_transform_(X_data), _dtype_transform_(y_data)

        Phi_data = self.reg_param[0] * paddle.ones([y_data.shape[0]], dtype="float32")
        for i in range(self.num_x):
            Phi_data = Phi_data + self.reg_param[i + 1] * X_data.T[i]
        init_state = State((1/math.sqrt(2)) * paddle.concat((y_data, Phi_data)))
        return init_state

    def fit(self, X: Union[paddle.Tensor, np.ndarray], y: Union[paddle.Tensor, np.ndarray],
            learning_rate: float = 0.01, iteration: int = 200,
            saved_dir: str = '', model_name: str = 'linear'
            ) -> None:

        r"""Fitting method used for training the model

        Args:
            X: Independent data in a 2D array.
            y: Dependent data in an 1D array.
            learning_rate: Learning rate of optimization. Defaults to ``0.01``.
            iteration: Total training iteration. Defaults to ``200``.
            saved_dir: The path for saving the fitted data. Defaults to ``''``.
            model_name: The model name. Defaults to ``linear``.

        Returns:
            Trained model
        """
        if not saved_dir:
            saved_dir = './'
        elif saved_dir[-1] != '/':
            saved_dir += '/'
        filename=f'{saved_dir}{model_name}.log'
        linearlog = _logger_init_('linearlog', filename)
        msg = (
            f"\n####################################################\n"
            f"The model: {model_name} with following set-up:\n"
            f"####################################################\n"
            f"No. of qubits: {self.num_qubits - 1}; \n"
            f"Raw model: y=a0 + a1*x1 + a2*x2 + ... + ak*xk; \n"
            f"Optimizer: Adam\n"
            f"  Initial params: {self.reg_param.tolist()}; \n"
            f"  Learning rate: {learning_rate}; \n"
            f"  No. of iteration: {iteration}.\n"
        )
        linearlog.info(msg)
        
        # hyper parameters
        opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=self.parameters())

        # data processing
        X_data, y_data = _data_transform_(X, y)
        X_data, y_data = X_data[:2**(self.num_qubits-1), :], y_data[:2**(self.num_qubits-1)]

        # verifying input data
        _data_verifying_(self.num_qubits, X_data, y_data)

        # initialize circuit
        cir_inner_prod = Circuit(self.num_qubits)
        cir_inner_prod.h(0)

        p_bar = tqdm(
            total=iteration,
            ascii=True,
            dynamic_ncols=True,
        )
        linearlog.info("Optimization:")
        for _ in range(iteration):
            p_bar.update(1)
            # fitting data and training parameters
            init_state = self._init_state_preparation_(X_data, y_data)
            loss = (1 - IPEstimator(cir_inner_prod, init_state))**2
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()

            if _ % int(iteration * 0.1) == 0:
                msg = (
                    f"Train loss: {loss.item():2.5f}; "
                    f"The model has been fitted. Score: {self.score(X, y):2.5f}; "
                )
                linearlog.info(msg)
        
        p_bar.close()
        self.status = True
        deter_score = self.score(X, y)

        msg = (
            f"\n####################################################\n"
            f"Summary\n"
            f"####################################################\n"
            f"The fitting score: {deter_score:2f};\n"
            f"The model params: {self.reg_param.tolist()}.\n"
        )
        linearlog.info(msg)
        
        paddle.save(self.state_dict(), f'{saved_dir}/{model_name}.pdparams')
        msg = "The fitted model state_dict has been saved to '.pdparams' file."
        linearlog.info(msg)
        linearlog.info("="*25)

    def predict(self, X: Union[paddle.Tensor, np.ndarray]) -> Union[paddle.Tensor, np.ndarray]:

        r"""Predict value based on current model parameters

        Args:
            X: Independent data in a 2D array. Every column indicates an independent variable.
               Every row indicates a sample of data.
        Returns:
            predicted value
        """
        X = _dtype_transform_(X)
        type_str = _type_fetch(X)
        X = _type_transform(X, "tensor")

        predict_data = self.reg_param[0] * paddle.ones([X.shape[0]], dtype="float32")
        for i in range(self.num_x):
            predict_data = predict_data + self.reg_param[i + 1] * X.T[i]

        return _type_transform(predict_data, type_str)

    def score(self, X: Union[paddle.Tensor, np.ndarray], y: Union[paddle.Tensor, np.ndarray], metric: Union[str, Callable] = "r2_score") -> float:
        
        r"""Quantifying the quality of predictions given test set

        Args:
            X: Independent data in a 2D array. Every column indicates an independent variable.
               Every row indicates a sample of data.
            y: Dependent data in a 1D array.
            metric: The metric name for the quality. Defaults to ``r2``. If the metric is a callable function, the function should be in the expression
            function(y_true: np.ndarray, y_pred: np.ndarray) -> float.
        
        Returns:
            The model score. Based on sklearn.metric class.
        """

        if type(metric) == str:
            if metric not in SKLEARN_REG_SCORER:
                raise ValueError("The metric is not a valid sklearn.metrics.")
            else:
                scorer = SKLEARN_METRIC[metric]
        else:
            if type(metric) != Callable:
                raise ValueError("The metric is not a valid Callable metric.")
            else:
                scorer = metric

        X, y = _type_transform(X, "numpy"), _type_transform(y, "numpy")
        y_pred = self.predict(X)
        
        return scorer(y, y_pred)


class PolyRegression(paddle.nn.Layer):
    r"""Regression class for initializing a quantum polynomial regression model

    Args:
        num_qubits: The number of qubits which the quantum circuit contains.
        order: The order of the polynomial regression model. Defaults to ``1``.
    """
    def __init__(self, num_qubits: int, order: int = 1) -> None:
        super(PolyRegression, self).__init__(name_scope = "PolyRegression")
        self.num_qubits = num_qubits
        self.order = order
        param = self.create_parameter([self.order + 1], 
                                          attr=None, dtype="float32", 
                                          is_bias=False, 
                                          default_initializer=None)
        self.add_parameter("param", param)
        self.status = False
    
    @property
    def reg_param(self) -> paddle.Tensor:
        r"""Flattened parameters in the Layer.
        """
        if len(self.parameters()) == 0:
            return []

        return paddle.concat([paddle.flatten(param) for param in self.parameters()])

    def set_params(self, new_params: Union[paddle.Tensor, np.ndarray]) -> None:
        r"""set parameters of the model.

        Args:
            params: New parameters
        """

        if not isinstance(new_params, paddle.Tensor):
            new_params = paddle.to_tensor(new_params, dtype='float32')
        new_params = paddle.flatten(new_params)

        if new_params.shape[0] != self.order + 1:
            raise ValueError(f"Incorrect number of params for the model: expect {self.order + 1}, received {new_params.shape[0]}")

        update_param = paddle.create_parameter(
                        shape=self.param.shape,
                        dtype='float32',
                        default_initializer=paddle.nn.initializer.Assign(new_params.reshape(self.param.shape)),
                    )

        setattr(self, 'param', update_param)

    def _init_state_preparation_(self, X_data: paddle.Tensor, y_data: paddle.Tensor) -> State:
        r"""Generate an initial state for compute inner product
        """
        X_data, y_data = _dtype_transform_(X_data), _dtype_transform_(y_data)
        Phi_data = 0
        for i in range(self.order + 1):
            Phi_data += self.reg_param[i] * X_data.T[0] ** i
        init_state = State((1/math.sqrt(2)) * paddle.concat((y_data, Phi_data)))

        return init_state

    def fit(self, X: Union[paddle.Tensor, np.ndarray], y: Union[paddle.Tensor, np.ndarray],
            learning_rate: float = 0.01, iteration: int = 200,
            saved_dir: str = '', model_name: str = 'poly'
            ) -> None:

        r"""Fitting method used for training the model

        Args:
            X: Independent data in a 2D array.
            y: Dependent data in an 1D array.
            learning_rate: Learning rate of optimization. Defaults to ``0.01``.
            iteration: Total training iteration. Defaults to ``200``.
            saved_dir: The path for saving the fitted data. Defaults to ``''``.
            model_name: The model name. Defaults to ``poly``.

        Returns:
            Trained model
        """
        if not saved_dir:
            saved_dir = './'
        elif saved_dir[-1] != '/':
            saved_dir += '/'
        filename=f'{saved_dir}{model_name}.log'
        polylog = _logger_init_('polylog', filename)
        msg = (
            f"\n####################################################\n"
            f"The model: {model_name} with following set-up:\n"
            f"####################################################\n"
            f"No. of qubits: {self.num_qubits - 1}; \n"
            f"Raw model: y=a0 + a1*x^1 + a2*x^2 + ... + ak*x^k; \n"
            f"Optimizer: Adam\n"
            f"  Initial params: {self.reg_param.tolist()}; \n"
            f"  Learning rate: {learning_rate}; \n"
            f"  No. of iteration: {iteration}.\n"
        )
        polylog.info(msg)

        # hyper parameters
        opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=self.parameters())

        # data processing
        X_data, y_data = _data_transform_(X, y)
        X_data, y_data = X_data[:2**(self.num_qubits-1), :], y_data[:2**(self.num_qubits-1)]

        # verifying input data
        _data_verifying_(self.num_qubits, X_data, y_data)

        # initialize circuit
        cir_inner_prod = Circuit(self.num_qubits)
        cir_inner_prod.h(0)

        p_bar = tqdm(
            total=iteration,
            ascii=True,
            dynamic_ncols=True,
        )
        polylog.info("Optimization:")
        for _ in range(iteration):
            p_bar.update(1)
            # fitting data and training parameters
            init_state = self._init_state_preparation_(X_data, y_data)
            loss = (1 - IPEstimator(cir_inner_prod, init_state))**2
            loss.backward()
            opt.minimize(loss)
            opt.clear_grad()

            if _ % int(iteration * 0.1) == 0:
                msg = (
                    f"Train loss: {loss.item():2.5f}; "
                    f"The model has been fitted. Score: {self.score(X, y):2.5f}; "
                )
                polylog.info(msg)
        
        p_bar.close()
        self.status = True
        deter_score = self.score(X, y)

        msg = (
            f"\n####################################################\n"
            f"Summary\n"
            f"####################################################\n"
            f"The fitting score: {deter_score:2f};\n"
            f"The model params: {self.reg_param.tolist()}.\n"
        )
        polylog.info(msg)
        
        paddle.save(self.state_dict(), f'{saved_dir}/{model_name}.pdparams')
        msg = "The fitted model state_dict has been saved to '.pdparams' file."
        polylog.info(msg)
        polylog.info("="*25)

    def predict(self, X: Union[paddle.Tensor, np.ndarray]) -> Union[paddle.Tensor, np.ndarray]:
        r"""Predict value based on current model parameters

        Args:
            x: A sample of data in an array.

        Returns:
            predicted value
        """
        X = _dtype_transform_(X)
        type_str = _type_fetch(X)
        X = _type_transform(X, "tensor")

        predict_data = 0
        for i in range(self.order + 1):
            predict_data += self.reg_param[i] * X.T[0] ** i

        return _type_transform(predict_data, type_str)
    
    def score(self, X: Union[paddle.Tensor, np.ndarray], y: Union[paddle.Tensor, np.ndarray], metric: Union[str, Callable] = "r2_score") -> float:        
        r"""Quantifying the quality of predictions given test set

        Args:
            X: Independent data in a 2D array. Every column indicates an independent variable.
               Every row indicates a sample of data.
            y: Dependent data in a 1D array.
            metric: The metric name for the quality. Defaults to ``r2``. If the metric is a callable function, the function should be in the expression
            function(y_true: np.ndarray, y_pred: np.ndarray) -> float.
        
        Returns:
            The model score. Based on sklearn.metric class.
        """

        if type(metric) == str:
            if metric not in SKLEARN_REG_SCORER:
                raise ValueError("The metric is not a valid sklearn.metrics.")
            else:
                scorer = SKLEARN_METRIC[metric]
        else:
            if type(metric) != Callable:
                raise ValueError("The metric is not a valid Callable metric.")
            else:
                scorer = metric

        X, y = _type_transform(X, "numpy"), _type_transform(y, "numpy")
        y_pred = self.predict(X)
        
        return scorer(y, y_pred)


if __name__ == '__main__':
    exit(0)
