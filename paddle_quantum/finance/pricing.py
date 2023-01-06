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
Tools for QMC and option pricing.
"""

import logging
import math
import numpy as np
import paddle

from paddle_quantum.ansatz import Circuit
from paddle_quantum.intrinsic import get_dtype
from paddle_quantum.linalg import dagger
from paddle_quantum.loss import Measure
from paddle_quantum.qinfo import grover_generation, qft_generation
from typing import Tuple, Callable, List


__all__ = ['qae_cir', 'qae_alg', 'qmc_alg', 'EuroOptionEstimator']


def qae_cir(oracle: paddle.Tensor, num_ancilla: int) -> Circuit:
    r"""Create a QAE circuit based on the ``oracle`` given.

    Args:
        oracle: input oracle.
        num_ancilla: number of ancilla qubits used.
    
    Returns:
        a circuit used for quantum amplitude estimation.

    """
    assert num_ancilla > 0
    grover = grover_generation(oracle)
    sys_num_qubits = int(math.log2(oracle.shape[0]))
    
    # registers
    aux_reg = list(range(num_ancilla))
    sys_reg = list(range(num_ancilla, sys_num_qubits + num_ancilla))
    
    cir = Circuit(num_ancilla + sys_num_qubits)
    cir.superposition_layer(aux_reg)
    cir.oracle(oracle, sys_reg, latex_name=r'$\mathcal{A}$')

    for i in reversed(range(num_ancilla)):
        cir.control_oracle(grover, [i] + sys_reg, 
                           latex_name= r'$\mathcal{Q}^{2^' + str(num_ancilla - 1 - i) + r'}$')
        grover = grover @ grover
        
    cir.oracle(dagger(qft_generation(num_ancilla)), aux_reg, latex_name=r'$QFT^\dagger$')
    
    return cir
    

def qae_alg(oracle: paddle.Tensor, num_ancilla: int) -> Tuple[Circuit, paddle.Tensor]:
    r"""Quantum amplitude estimation (QAE) algorithm
    
    Args:
        oracle: an :math:`n`-qubit oracle :math:`\mathcal{A}`.
        num_ancilla: number of ancilla qubits used.

    Returns:
        a QAE circuit and :math:`|\sin(2\pi\theta)|`
        
    Note:
        :math:`\mathcal{A}` satisfies
        
    .. math::
    
        \mathcal{A}|0^{\otimes n}\rangle=\cos(2\pi\theta)|0\rangle|\psi\rangle+\sin(2\pi\theta)|1\rangle|\phi\rangle.
        
    """
    complex_dtype = get_dtype()
    op_M = Measure()
    aux_reg = list(range(num_ancilla))
    
    oracle = oracle.cast(complex_dtype)
    cir = qae_cir(oracle, num_ancilla)
    state = cir()
    measured_result = paddle.argmax(op_M(state, qubits_idx=aux_reg))
    estimated_sin = paddle.abs(paddle.sin(measured_result / (2 ** num_ancilla) * math.pi))
    
    return cir, estimated_sin


def __standardize_fcn(fcn: Callable[[float], float], list_input: List[float]) -> Tuple[float, float, List[float]]:
    r"""Make input ``fcn`` discretized and normalized.
    
    Args:
        fcn: input function, can be continuous.
        domain: list of input data of ``fcn``.
        
    Returns:
        maximum, minimum of output data, and a list of output data of normalized ``fcn``
        
    """
    list_output = [fcn(x) for x in list_input]
    maximum, minimum = max(list_output), min(list_output)
    list_output = (np.array(list_output, dtype='float64') - minimum) / (maximum - minimum)
    return maximum, minimum, list_output.tolist()


def __u_d(list_output: List[float]) -> paddle.Tensor:
    dimension = len(list_output)
    complex_dtype = get_dtype()
    
    def reflection(x: float) -> np.ndarray:
        return np.array([[math.sqrt(1 - x), math.sqrt(x)], 
                         [math.sqrt(x), -1 * math.sqrt(1 - x)]], dtype=complex_dtype)
    
    def ket_bra(j: int) -> np.ndarray:
        mat = np.zeros([dimension, dimension])
        mat[j, j] += 1
        return mat
    
    mat = sum(np.kron(reflection(list_output[j]), ket_bra(j)) for j in range(dimension))
    return paddle.to_tensor(mat, dtype=complex_dtype)


def __u_p(list_prob: List[float]) -> paddle.Tensor:
    dimension = len(list_prob)
    list_prob = np.array(list_prob)
    
    # construct basis
    max_index = np.argmax(list_prob).item()
    mat = np.eye(dimension)
    mat[:, max_index] = np.sqrt(list_prob)
    mat[:, [max_index, 0]] = mat[:, [0, max_index]]

    # qr decomposition
    Q, _ = np.linalg.qr(mat)
    return paddle.to_tensor(Q * -1, dtype=get_dtype())


def qmc_alg(fcn: Callable[[float], float], list_prob: List[float], 
            num_ancilla: int = 6) -> Tuple[Circuit, paddle.Tensor]:
    r"""Quantum Monte Carlo (QMC) algorithm.
    
    Args:
        fcn: real-valued function :math:`f` applied to a random variable :math:`X`.
        list_prob: probability distribution of :math:`X`, where the j-th probability corresponds to the j-th outcome.
        num_ancilla: number of ancilla qubits used. Defaults to be ``6``.
    
    Returns:
        a QAE circuit and an estimation of :math:`\mathbb{E}[f(X)]`.
    
    """
    num_qubits = math.ceil(math.log2(len(list_prob)))
    dimension = 2 ** num_qubits
    list_input = list(range(dimension))
    list_prob += [0] * (dimension - len(list_prob))

    f_max, f_min, list_standard_output = __standardize_fcn(fcn, list_input)
    oracle = __u_d(list_standard_output) @ paddle.kron(paddle.eye(2), __u_p(list_prob))
    cir, val = qae_alg(oracle, num_ancilla)
    return cir, (val ** 2) * (f_max - f_min) + f_min


class EuroOptionEstimator(object):
    r"""European option pricing estimator.
    
    Args:
        initial_price: initial price of the asset.
        strike_price: pre-fixed price of the asset.
        interest_rate: risk-free interest rate.
        volatility: dispersion of returns for the asset.
        maturity_date: date when option is expired.
        degree_of_estimation: degree of price estimation. Defaults to be ``5``.
        
    Note:
        Option price is evaluated under 
        the `Black-Scholes-Merton model <https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model>`_.

    
    """
    def __init__(self, initial_price: float, strike_price: float, 
                 interest_rate: float, volatility: float, maturity_date: float, 
                 degree_of_estimation: int = 5) -> None:
        
        logging.basicConfig(
            filename='./euro_pricing.log',
            filemode='w',
            format='%(asctime)s %(levelname)s %(message)s',
            level=logging.INFO
        )
        
        logging.info(
            "Received the following information of this asset:"
        )
        logging.info(
            f"  initial price {initial_price}, strike price: {strike_price},"
        )
        logging.info(
            f"  interest_rate: {interest_rate}, volatility: {volatility}, maturity date: {maturity_date}."
        )
        logging.info("Begin initialization.")
        
        x_max = 5 * math.sqrt(maturity_date)
        sample_points = np.linspace(start=-x_max, stop=x_max, num=2**degree_of_estimation)
        
        logging.info(
            f"  uniformly sampling {len(sample_points)} points between -{x_max:<.3f} and {x_max:<.3f};")

        bs_model_fcn = lambda x: \
            initial_price * math.exp(volatility * x + (interest_rate - 0.5 * (volatility ** 2)) * maturity_date) - strike_price
        normal_pdf = lambda x: \
            np.exp(-1 * (x ** 2) / 2 / maturity_date) / math.sqrt(2 * math.pi * maturity_date)
        list_prob = normal_pdf(sample_points)
        
        self.__fcn = lambda j: max(0, bs_model_fcn(sample_points[j])) * math.exp(-1 * interest_rate * maturity_date)
        
        logging.info(
            "  the function of random variable has been set up;")
        
        self.__list_prob = (list_prob / np.sum(list_prob)).tolist()
        
        logging.info(
            "  the probability distribution of random variable has been set up;")
        
        self.__cir = None
        
        logging.info(
            "  the quantum circuit has been set up.")
        
    def estimate(self) -> float:
        r"""Estimate the European option price using Quantum Monte Carlo (QMC) methods.
        
        Returns:
            Risk-neutral price of the given asset.
        
        """
        logging.info("Begin estimation.")
        cir, estimated_expectance = qmc_alg(self.__fcn, self.__list_prob)
        self.__cir = cir
        estimated_expectance = estimated_expectance.item()
        logging.info(
            f"  retrieve estimation result {estimated_expectance:<.5f} from QMC algorithm.")
        return estimated_expectance
    
    def plot(self, dpi: int = 100) -> None:
        r"""Plot the quantum circuit used in pricing.
        
        Args:
            dpi: image clarity of the plotted circuit. Defaults to be ``200``.
        
        """
        if self.__cir is None:
            raise UserWarning(
                "You need to estimate the option first before plotting the circuit.")
        self.__cir.plot(dpi=dpi)
