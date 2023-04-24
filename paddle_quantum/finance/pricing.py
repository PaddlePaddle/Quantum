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
Tools for QMC, option pricing and credit analysis.
"""

import math
import numpy as np
from scipy.special import erf, erfinv

import paddle
from paddle_quantum.ansatz import Circuit
from paddle_quantum.intrinsic import get_dtype, _get_float_dtype
from paddle_quantum.linalg import dagger, NKron
from paddle_quantum.loss import Measure
from paddle_quantum.qinfo import grover_generation, qft_generation
from functools import reduce
import operator

import logging
from itertools import product
from typing import Tuple, Callable, List, Union, Optional


__all__ = ['qae_cir', 'qae_alg', 'qmc_alg', 'EuroOptionEstimator', 'CreditRiskAnalyzer']


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
    float_dtype = _get_float_dtype(complex_dtype)
    op_M = Measure()
    aux_reg = list(range(num_ancilla))
    
    oracle = oracle.cast(complex_dtype)
    cir = qae_cir(oracle, num_ancilla)
    state = cir()
    measured_result = paddle.argmax(op_M(state, qubits_idx=aux_reg)).cast(float_dtype)
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
        Option price is evaluated under the [Black-Scholes-Merton model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model).
    
    """
    def __init__(self, initial_price: float, strike_price: float, 
                 interest_rate: float, volatility: float, maturity_date: float, 
                 degree_of_estimation:  Optional[int] = 5) -> None:
        
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
    
    def plot(self, dpi: int = 200) -> None:
        r"""Plot the quantum circuit used in pricing.
        
        Args:
            dpi: image clarity of the plotted circuit. Defaults to be ``200``.
        
        """
        if self.__cir is None:
            raise UserWarning(
                "You need to estimate the option first before plotting the circuit.")
        self.__cir.plot(dpi=dpi)


def __u_t(list_output: np.ndarray) -> paddle.Tensor:
    num_assets, size = list_output.shape
    complex_dtype = get_dtype()
    
    def reflection(x: float) -> np.ndarray:
        return np.array([[math.sqrt(1 - x), math.sqrt(x)], 
                         [math.sqrt(x), -1 * math.sqrt(1 - x)]], dtype=complex_dtype)
    
    def ket_bra(j: int) -> np.ndarray:
        mat = np.zeros([size, size])
        mat[j, j] += 1
        return mat
    
    mat = sum(np.kron(NKron(*[reflection(list_output[k, j]) for k in range(num_assets)]), 
                      ket_bra(j)) for j in range(size))
    
    return paddle.to_tensor(mat, dtype=complex_dtype)

def __u_comp(lgd_list: List[int], y: float) -> paddle.Tensor:
    num_assets = len(lgd_list)
    dimension = 2 ** num_assets
    complex_dtype = get_dtype()
    
    def threshold_matrix(x: float) -> np.ndarray:
        x = np.binary_repr(x, width=num_assets)
        loss = sum(int(x[i]) * lgd_list[i] for i in range(num_assets))
        return np.array([[1, 0], [0, -1]]) if loss > y else np.array([[0, 1], [1, 0]])
    
    def ket_bra(j: int) -> np.ndarray:
        mat = np.zeros([dimension, dimension])
        mat[j, j] += 1
        return mat
    
    mat = sum(np.kron(threshold_matrix(j), ket_bra(j)) for j in range(dimension))
    return paddle.to_tensor(mat, dtype=complex_dtype)

def cra_oracle(prob: np.ndarray, param: np.ndarray, lgd: List[int], threshold: float) -> paddle.Tensor:
    r"""Construct the oracle for credit risk analysis.
    
    Args:
        prob: probability distribution of the latent random variable
        param: parameters for Bernoulli variables of assets
        lgd: losses given default (LGD) of assets
        threshold: guessed VaR
        
    Returns:
        The grover operator ready to be placed into QAE circuit.
    
    """
    num_sample, num_assets = len(prob), len(lgd)
    Up = __u_p(prob)
    Ut = __u_t(param)
    A = Ut @ paddle.kron(paddle.eye(2 ** num_assets), Up)
    Uc = __u_comp(lgd, threshold)
    return paddle.kron(Uc, paddle.eye(num_sample)) @ paddle.kron(paddle.eye(2), A)

class CreditRiskAnalyzer(object):
    r"""Simulator for credit risk analysis (CRA) speeded up by quantum algorithms
    
    Args:
        num_assets: number of assets.
        base_default_prob: basic default probabilities of assets.
        sensitivity: sensitivities of assets.
        lgd: losses given default (LGD) of assets.
        confidence_level: level of confidence of computed result.
        degree_of_simulation: degree of simulation of CRA, determining the number of samples of the latent variable. Defaults to be ``4``.
        even_sample: whether sample the latent variable by evenly-spaced probability. Defaults to be ``True``.
        
    Note:
        CRA is simulated under the model given by [Tarca & Silvio](https://arxiv.org/abs/1412.1183).
        That is, the latent random variable for the market is assumed to follow a standard normal distribution.
        
        
    """
    def __init__(self, num_assets: int, base_default_prob: np.ndarray, sensitivity: np.ndarray, 
                 lgd: np.ndarray, confidence_level: float, 
                 degree_of_simulation:  Optional[int] = 4, even_sample: Optional[bool] = True) -> None:
        logging.basicConfig(
            filename='./risk_analysis.log',
            filemode='w',
            format='%(asctime)s %(levelname)s %(message)s',
            level=logging.INFO
        )
        base_default_prob, sensitivity, lgd = np.array(base_default_prob), np.array(sensitivity), np.array(lgd)
        self.__assert_and_log(num_assets, base_default_prob, sensitivity, lgd, confidence_level)
        
        # generate latent variable
        num_samples = 2 ** degree_of_simulation
        logging.info(
            f"Taking {num_samples} samples from the normal distribution."
        )
        outcome_z_list = self.__quantile_normal(np.linspace(1 / num_samples, 1, num_samples, endpoint=False)) \
            if even_sample else np.linspace(-2, 2, num_samples)
        prob_z_list = self.__pdf_normal(outcome_z_list)
        prob_z_list /= sum(prob_z_list)
        
        # generate Bernoulli variables
        bernoulli_param_list = np.zeros([num_assets, num_samples])
        for k, j in product(range(num_assets), range(num_samples)):
            bernoulli_param_list[k, j] = self.__prob_generation(outcome_z_list[j], base_default_prob[k], sensitivity[k])
        logging.info(
            f"Bernoulli random variables of {num_assets} assets are created."
        )
        
        self.prob_z_list = prob_z_list
        self.bernoulli_param_list = bernoulli_param_list
        self.lgd = lgd
        self.confidence_level = confidence_level
        self.__log_exact()
        
        logging.info(
            f"{1 + num_assets + degree_of_simulation + 6} qubits are required to execute quantum algorithms."
        )
        
    def estimate_var(self) -> float:
        r"""Estimate VaR using the QAE algorithm.
        
        Returns:
            the estimated Value at Risk of these credit assets
        
        """
        return self.__bisection_search()
        
    def __assert_and_log(self, num_assets: int, 
                         base_default_prob: np.ndarray, sensitivity: np.ndarray, 
                         lgd: np.ndarray, confidence_level: float) -> None:
        r"""Assert and log the input data of this model.
        """
        logging.info(
            f"Received the following information of {num_assets} assets:"
        )

        np.set_printoptions(precision=4)
        assert base_default_prob.size == num_assets, \
                f"The shape of input default_prob is incorrect: received {base_default_prob.shape}, expect {[num_assets]}."
        logging.info(
            f"      list of default probabilities {base_default_prob},"
        )
        assert all(base_default_prob >= 0) and all(base_default_prob <= 1), \
            "The data of input default_prob is incorrect: data must be within [0, 1]."

        assert sensitivity.size == num_assets, \
                f"The shape of input sensitivity is incorrect: received {sensitivity.shape}, expect {[num_assets]}."
        logging.info(
            f"      list of sensitivities {sensitivity},"
        )
        assert all(sensitivity >= 0) and all(sensitivity <= 1), \
            "The data of input sensitivity is incorrect: data must be within [0, 1]."

        assert lgd.size == num_assets, \
                f"The shape of input lgd is incorrect: received {lgd.shape}, expect {[num_assets]}."
        logging.info(
            f"      list of losses given default {lgd}."
        )

        assert 0 <= confidence_level <= 1, \
                f"The level of confidence must be within [0, 1]: received {confidence_level :.2f}"
        confidence_level = np.array(confidence_level)
        logging.info(
            f"Assets are analyzed with level of confidence {confidence_level * 100} %."
        )
        
    def __log_exact(self) -> None:
        r"""Log the exact values of VaR and CVaR of these assets.
        """
        num_assets = len(self.lgd)
        logging.info(
            f"Exact values of some properties of these {num_assets} assets:"
        )
        
        # generate random variable of total loss
        default_prob_list = self.bernoulli_param_list @ self.prob_z_list
        adhere_prob_list = (1 - self.bernoulli_param_list) @ self.prob_z_list
        outcome_list = [np.binary_repr(x, num_assets) for x in range(2 ** num_assets)]
        
        value_list = [sum(int(outcome[i]) * self.lgd[i] for i in range(num_assets)) 
              for outcome in outcome_list]
        prob_list = [reduce(operator.mul, [adhere_prob_list[i] if outcome[i] == '0' else default_prob_list[i] 
                                           for i in range(num_assets)], 1) for outcome in outcome_list]
        logging.info(
            f"      expected loss:                      {np.dot(prob_list, value_list) :.5f}"
        )
        
        # sort two lists by value_list
        prob_list = np.array([x for _, x in sorted(zip(value_list, prob_list))])
        value_list = np.array(sorted(value_list))
        cdf_list = self.__cdf_variable(prob_list, value_list, value_list)
        
        # compute the exact values
        exact_var = min(value_list[cdf_list >= self.confidence_level])
        logging.info(
            f"      Value at Risk:                      {exact_var :.5f}"
        )
        
        prob_var = self.__cdf_variable(prob_list, value_list, float(exact_var))
        logging.info(
            f"      probability to be lower than VaR:   {prob_var :.5f}"
        )

        # TODO: add the estimation of CVaR using quantum algorithms 
        # indices = value_list > exact_var
        # exact_cvar = np.dot(prob_list[indices], value_list[indices]) / sum(prob_list[indices])
        # logging.info(
        #     f"      Conditional Value at Risk:          {exact_cvar :.5f}"
        # )
        
    def __pdf_normal(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Apply the pdf of normal distribution.
        """
        return 1 / math.sqrt(2 * math.pi) * np.exp(-0.5 * (x ** 2))

    def __cdf_normal(self,x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Apply the cdf of normal distribution.
        """
        return 0.5 * (1 + erf(x / math.sqrt(2)))

    def __quantile_normal(self,prob: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Apply the quantile of normal distribution.
        """
        return math.sqrt(2) * erfinv(2 * prob - 1)

    def __prob_generation(self, z: float, base_default_prob: float, sensitivity: float) -> float:
        r""" Calculate the parameter of the Bernoulli variable of a given asset.
        """
        rho, rho_0 = sensitivity, base_default_prob
        return self.__cdf_normal((self.__quantile_normal(rho_0) - math.sqrt(rho) * z) / math.sqrt(1 - rho))
    
    def __cdf_variable(self, distribution: np.ndarray, outcome: np.ndarray, 
                       x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""Calculate Pr(X <= x) given the description of X.
        """
        return sum(distribution[outcome <= x]) if isinstance(x, float) else \
            np.array([sum(distribution[outcome <= i]) for i in x])
    
    def __eval_prob(self, guess_var: int) -> float:
        r"""Evaluate the probability that the total loss is less than ``guess_var``
        """
        logging.info(
            f"      Setting up Grover operator with guessed VaR {guess_var}"
        )
        grover = cra_oracle(self.prob_z_list, self.bernoulli_param_list, self.lgd, guess_var)
        
        logging.info(
            "      Running QAE algorithm..."
        )
        guess_amplitude = qae_alg(grover, 6)[1].item()
        return guess_amplitude ** 2
    
    def __bisection_search(self) -> float:
        r"""Bisection method to search the minimum loss of assets that happens with probability greater than ``confidence_level``.
        """
        confidence_level = self.confidence_level
        print("-----------------------------------------------------------------------------------")
        print(f"Begin bisection search for VaR with confidence level >= {confidence_level * 100:.1f}%.")
        print("-----------------------------------------------------------------------------------")
        print("Lower guess: level            Middle guess: level            Upper guess: level    ")
        
        low_guess, up_guess = -1, math.floor(np.sum(np.abs(self.lgd)).item())
        low_prob, up_prob = 0, 1
        logging.info(
            "Starting classical bisection search..."
        )
        
        while up_guess > low_guess + 1:
            # extract middle index
            mid_guess = int((low_guess + up_guess) // 2)
            mid_prob = self.__eval_prob(mid_guess)
            
            # print message
            print("    %2d     : %.3f                 %2d    : %.3f                 %2d     : %.3f    " % 
                (low_guess, low_prob, mid_guess, mid_prob, up_guess, up_prob))
            
            # update lower or upper guess
            if mid_prob < confidence_level:
                low_guess, low_prob = mid_guess, mid_prob
            else:
                up_guess, up_prob = mid_guess, mid_prob
        
        final_guess, final_prob = [up_guess, up_prob] if mid_prob < confidence_level else \
                                  [mid_guess, mid_prob]
        print("-----------------------------------------------------------------------------------")
        print(f"Estimated VaR is {final_guess} with confidence level {final_prob * 100:.1f}%.")
        print("-----------------------------------------------------------------------------------")
        logging.info(
            f"Receive VaR {final_guess} with confidence level {final_prob}."
        )
        return final_guess
