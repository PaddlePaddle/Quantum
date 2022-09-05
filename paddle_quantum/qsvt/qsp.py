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
import numpy as np
from math import pi, acos
import paddle
from paddle_quantum.ansatz.circuit import Circuit
from numpy.polynomial.polynomial import Polynomial, polytrim
from typing import List, Tuple
from .qsp_utils import clean_small_error


"""
    Libraries for Quantum Signal Processing
        referring to paper https://arxiv.org/abs/1806.01838

"""


# ----------------------------- belows are support and test functions ------------------------


def signal_unitary(signal_x: float) -> np.ndarray:
    r"""signal unitary :math:`W(x)` in paper https://arxiv.org/abs/1806.01838
    
    Args:
        signal_x: variable x in [-1, 1]
        
    Returns:
        matrix W(x = x)
    
    """
    assert  -1 <= signal_x <= 1, "x must be in domain [-1, 1]"
    return np.array([[signal_x, 1j * np.sqrt(1 - signal_x ** 2)], 
                     [1j * np.sqrt(1 - signal_x ** 2), signal_x]])


def poly_parity_verification(poly_p: Polynomial, k: int, error: float = 1e-6) -> bool:
    r"""verify whether :math:`P` has parity-(k mod 2), i.e., the second condition of theorem 3 holds
    
    Args:
        poly_p: polynomial :math:`P(x)`
        k: parameter that determine parity
        error: tolerated error, defaults to `1e-6`
        
    Returns:
        determine whether :math:`P` has parity-(k mod 2)
    
    """
    parity = k % 2
    P_coef = poly_p.coef
    
    for i in range(poly_p.degree()):
        if i % 2 != parity:
            if np.abs(P_coef[i]) > error:
                return False
            P_coef[i] = 0  # this element should be 0
    return True


def normalization_verification(poly_p: Polynomial, poly_q: Polynomial, trials: int = 10, error: float = 1e-2) -> bool:
    r"""verify whether polynomials :math:`P(x)` and :math:`Q(x)` satisfy the normalization condition :math:`|P|^2 + (1 - x^2)|Q|^2 = 1`, i.e., the third condition of Theorem 3.
    
    Args:
        poly_p: polynomial :math:`P(x)`
        poly_q: polynomial :math:`Q(x)`
        trials: number of tests, defaults to `10`
        error: tolerated error, defaults to `1e-2`
        
    Returns:
        determine whether :math:`|P|^2 + (1 - x^2)|Q|^2 = 1`
    
    """
    P_conj = Polynomial(np.conj(poly_p.coef))
    Q_conj = Polynomial(np.conj(poly_q.coef))

    test_poly = poly_p * P_conj + Polynomial([1, 0, -1]) * poly_q * Q_conj
    for _ in range(trials):
        x = np.random.rand() * 2 - 1  # sample x from [-1, 1]
        y = test_poly(x)
        if abs(np.real(y) - 1) > error or abs(np.imag(y)) > error:
            print(np.real(y) - 1)
            return False
    return True


def angle_phi_verification(phi: float, poly_p: Polynomial, poly_p_hat: Polynomial, poly_q_hat: Polynomial, 
                            trials: int = 10, error: float = 0.01) -> bool:
    r"""verify :math:`\phi` during the iteration of finding :math:`\Phi`
    
    Args:
        phi: rotation angle :math:`\phi`
        poly_p: polynomial :math:`P(x)`
        poly_p_hat: updated polynomial :math:`\hat{P}`
        poly_q_hat: updated polynomial :math:`\hat{Q}`
        trials: number of tests, defaults to `10`
        error: tolerated error, defaults to `0.01`
        
    Returns:
        determine whether the equation (6) in paper https://arxiv.org/abs/1806.01838 holds
    
    """
    
    def block_encoding_new_p(x):
        return np.array([[poly_p_hat(x), 1j * poly_q_hat(x) * np.sqrt(1 - x ** 2)], 
                         [1j * np.conj(poly_q_hat(x)) * np.sqrt(1 - x ** 2), np.conj(poly_p_hat(x))]])

    rz = np.array([[np.exp(1j * phi), 0], 
                   [0, np.exp(-1j * phi)]])
    
    for _ in range(trials):
        x = np.random.rand() * 2 - 1 # sample x from [-1, 1]
        matrix = np.matmul(np.matmul(block_encoding_new_p(x), signal_unitary(x)), rz)
        y = matrix[0, 0]
        if np.abs(y - poly_p(x)) > error:
            print(y)
            print(poly_p(x))
            return False
    return True
    

def processing_unitary(list_matrices: List[np.ndarray], signal_x: float) -> np.ndarray:
    r"""processing unitary :math:`W_\Phi(x)`, see equation 1 in paper https://arxiv.org/abs/1806.01838
    
    Args:
        list_matrices: array of phi's matrices
        signal_x: input signal x in [-1, 1]
        
    Returns:
        matrix :math:`W_\Phi(x)`
    
    """
    assert -1 <= signal_x <= 1, "x must be in domain [-1, 1]"
    
    W = signal_unitary(signal_x) 
    M = list_matrices[0]
    for i in range(1, len(list_matrices)):
        M = np.matmul(M, np.matmul(W, list_matrices[i]))
    return M


def Phi_verification(list_phi: np.ndarray, poly_p: Polynomial, trials: int = 100, error: float = 1e-6) -> bool:
    r"""verify the final :math:`\Phi`
    
    Args:
        list_phi: array of :math:`\phi`'s
        poly_p: polynomial :math:`P(x)`
        trials: number of tests, defaults to `100`
        error: tolerated error, defaults to `1e-6`
        
    Returns:
        determine whether :math:`W_\Phi(x)` is a block encoding of :math:`P(x)`
    
    """
    def rz(theta: float) -> np.ndarray:
        return np.array([[np.exp(1j * theta), 0], 
                         [0, np.exp(-1j * theta)]])
        
    matrix_phi = list(map(rz, list_phi))

    for _ in range(trials):
        x = np.random.rand() * 2 - 1 # sample x from [-1, 1]
        y = processing_unitary(matrix_phi, x)[0, 0]
        if np.abs(y - poly_p(x)) > error:
            print(y - poly_p(x), error)
            return False
    return True


# ----------------------------- belows are Phi-generation functions ------------------------
    


def update_polynomial(poly_p: Polynomial, poly_q: Polynomial, phi: float) -> Tuple[Polynomial, Polynomial]:
    r"""update :math:`P, Q` by given phi according to proof in theorem 3
    
    Args:
        poly_p: polynomial :math:`P(x)`
        poly_q: polynomial :math:`Q(x)`
        phi: derived :math:`phi`
        
    Returns:
        updated :math:`P(x), Q(x)`
    
    """
    poly_1 = Polynomial([0, 1]) # x
    poly_2 = Polynomial([1, 0, -1]) # 1 - x^2
    
    # P = e^{-i phi} x P + e^{i phi} (1 - x^2) Q
    # Q = e^{i phi} x Q - e^{-i phi} P
    P_new = np.exp(-1j * phi) * poly_1 * poly_p + np.exp(1j * phi) * poly_2 * poly_q
    Q_new = np.exp(1j * phi) * poly_1 * poly_q - np.exp(-1j * phi) * poly_p

    # clean the error that is lower than 0.001
    P_new = Polynomial(polytrim(P_new.coef, 0.001))
    Q_new = Polynomial(polytrim(Q_new.coef, 0.001))
    
    # clean the error further, 
    P_new.coef = clean_small_error(P_new.coef)
    Q_new.coef = clean_small_error(Q_new.coef)
    
    if P_new.degree() >= poly_p.degree() and np.abs(P_new.coef[-1]) < np.abs(P_new.coef[-3]):
        P_new.coef = np.delete(np.delete(P_new.coef, -1), -1)
    if Q_new.degree() >= poly_q.degree() > 0 and np.abs(Q_new.coef[-1]) < np.abs(Q_new.coef[-3]):
        Q_new.coef = np.delete(np.delete(Q_new.coef, -1), -1)
    
    # used for debug, can be removed in formal version    
    assert P_new.degree() < poly_p.degree(), print(P_new, '\n', poly_p)
    assert Q_new.degree() < poly_q.degree() or poly_q.degree() == 0, print(Q_new, '\n', poly_q)
    assert poly_parity_verification(P_new, poly_p.degree() - 1)
    assert poly_parity_verification(Q_new, poly_q.degree() - 1)
    assert normalization_verification(P_new, Q_new), print(P_new, '\n', Q_new)
    assert angle_phi_verification(phi, poly_p, P_new, Q_new)

    return P_new, Q_new
    

def alg_find_Phi(poly_p: Polynomial, poly_q: Polynomial, length: int) -> np.ndarray:
    r"""The algorithm of finding phi's by theorem 3
    
    Args:
        poly_p: polynomial :math:`P(x)`
        poly_q: polynomial :math:`Q(x)`
        length: length of returned array
        
    Returns:
        array of phi's
    
    """
    n = poly_p.degree()
    m = poly_q.degree()
    
    # condition check for theorem 3 
    assert n <= length, "the condition for P's degree is not satisfied"
    assert m <= max(0, length - 1), "the condition for Q's degree is not satisfied"
    assert poly_parity_verification(poly_p, length), "the condition for P's parity is not satisfied"
    assert poly_parity_verification(poly_q, length - 1), "the condition for Q's parity is not satisfied"
    assert normalization_verification(poly_p, poly_q), "the third equation for P, Q is not satisfied"
    
    i = length
    Phi = np.zeros([length + 1])

    while n > 0:
        # assign phi
        Phi[i] = np.log(poly_p.coef[n] / poly_q.coef[m]) * -1j / 2 
        
        if Phi[i] == 0:
            Phi[i] = np.pi

        # update step
        poly_p, poly_q = update_polynomial(poly_p, poly_q, Phi[i])

        n = poly_p.degree()
        m = poly_q.degree()
        i = i - 1
    
    for j in range(1, i):
        Phi[j] = (-1) ** (j - 1) * pi / 2
    Phi[0] = -1j * np.log(poly_p.coef[0])

    return Phi


# ----------------------------- belows are Q-generation functions ------------------------


def poly_A_hat_generation(poly_p: Polynomial) -> Polynomial: 
    r"""function for :math:`\hat{A}` generation
    
    Args:
        poly_p: polynomial :math:`P(x)`
        
    Returns:
        polynomial :math:`\hat{A}(y) = 1 - P(x)P^*(x)`, with :math:`y = x^2`
    
    """
    P_conj = Polynomial(np.conj(poly_p.coef))
    A = 1 - poly_p * P_conj
    A_coef = A.coef
    coef = [A_coef[0]]
    for i in range(1, poly_p.degree() + 1):
        coef.append(A_coef[2 * i])
        
    return Polynomial(np.array(coef))


def poly_A_hat_decomposition(A_hat: Polynomial, error: float = 0.001) -> Tuple[float, List[float]]:
    r"""function for :math:`\hat{A}` decomposition
    
    Args:
        A_hat: polynomial :math:`\hat{A}(x)`
        error: tolerated error, defaults to `0.001`
        
    Returns:
        Tuple: including the following elements
        - leading coefficient of :math:`\hat{A}`
        - list of roots of :math:`\hat{A}` such that there exist no two roots that are complex conjugates
    
    """
    leading_coef = A_hat.coef[A_hat.degree()]
    
    # remove one 1 and 0 (if k is even) from this list
    roots = [i for i in A_hat.roots() if not ((np.abs(np.real(i) - 1) < error
                                               and np.abs(np.imag(i)) < error) or np.abs(i) < error)]
    
    # Note that root function in numpy return roots in complex conjugate pairs
    # Now elements in roots are all in pairs
    output_roots = [roots[i] for i in range(len(roots)) if i % 2 == 0]
    
    return leading_coef, output_roots


def poly_Q_generation(leading_coef: float, roots: List[float], parity: int) -> Polynomial:
    r"""function for polynomial :math:`Q` generation
    
    Args:
        leading_coef: leading coefficient of :math:`\hat{A}`
        roots: filtered list of roots of :math:`\hat{A}`
        parity: parity that affects decomposition
        
    Returns:
        polynomial :math:`Q`

    """
    a = np.sqrt(-1 * leading_coef)
    
    poly_q = Polynomial([a])
    for i in range(len(roots)):
        poly_q = poly_q * Polynomial([-1 * roots[i], 0, 1])
    
    if parity % 2 == 0:
        poly_q = poly_q * Polynomial([0, 1]) 
        return poly_q
    return poly_q
    

def alg_find_Q(poly_p: Polynomial, k: int) -> Polynomial:
    r"""The algorithm of finding :math:`Q` by theorem 4 in paper https://arxiv.org/abs/1806.01838
    
    Args:
        poly_p: polynomial :math:`P(x)`
        k: length of returned array
        
    Returns:
        polynomial :math:`Q(x)`
    
    """
    n = poly_p.degree()
    
    # condition check for theorem 3 
    assert n <= k, "the condition for P's degree is not satisfied"
    assert poly_parity_verification(poly_p, k), "the condition for P's parity is not satisfied"
    
    A_hat = poly_A_hat_generation(poly_p)
    leading_coef, roots = poly_A_hat_decomposition(A_hat)
    poly_q = poly_Q_generation(leading_coef, roots, k)
     
    return poly_q    


# ----------------------------- belows are final functions ------------------------


def quantum_signal_processing(poly_p: Polynomial, length: int = None) -> np.ndarray:
    r""" Compute :math:`\Phi` that transfer a block encoding of x to a block encoding of :math:`P(x)` by :math:`W_\Phi(x)`
    
    Args:
        poly_p: polynomial :math:`P(x)`
        length: length of returned array, defaults to `None` to be the degree of  :math:`P(x)`
        
    Returns:
        array of :math:`\phi`'s
    
    """
    if length is None:
        length = poly_p.degree()
    
    Q = alg_find_Q(poly_p, length)
    Phi = alg_find_Phi(poly_p, Q, length)
    
    assert Phi_verification(Phi, poly_p)
        
    return Phi

def reflection_based_quantum_signal_processing(P: Polynomial) -> np.ndarray:
    r""" Reflection-based quantum signal processing, compute Phi that transfer a block encoding of x to a block encoding of :math:`P(x)` with :math:`R_\Phi(x)`. Refer to Corollary 8 in the paper.
    
    Args:
        P: polynomial :math:`P(x)`
        
    Returns:
        array of :math:`phi`'s
    
    """
    Phi = quantum_signal_processing(P)
    k = P.degree()
    Phi_new = np.zeros([k])
    
    Phi_new[0] = Phi[0] + Phi[k] + (k - 1) * np.pi / 2
    for i in range(1, k):
        Phi_new[i] = Phi[i] - np.pi / 2
    
    # assertion
    phi_sum = 0
    phi_alternate = 0
    for i in range(k):
        phi_sum += Phi_new[i]
        phi_alternate += ((-1) ** i) * Phi_new[i]
        
    assert np.abs(P(1) - np.exp(1j * phi_sum)) < 10 ** (-8)
    assert np.abs(P(-1) - ((-1) ** k) * np.exp(1j * phi_sum)) < 10 ** (-8)
    if k % 2 == 0:
        assert np.abs(P(0) - np.exp(-1j * phi_alternate)) < 10 ** (-8)
    
    return Phi_new

    
# ----------------------------- below is the class for QSP ------------------------


class ScalarQSP(object):
    def __init__(self, poly_p: Polynomial, length: int = None) -> None:
        r"""Initialize a class that is used for QSP in single qubit

        Args:
            poly_p: Polynomial P(x)
            k: length of array Phi - 1

        """
        if length is None:
            length = poly_p.degree()
    
        self.poly_p = poly_p
        self.poly_q = alg_find_Q(poly_p, length)
        self.Phi = paddle.to_tensor(quantum_signal_processing(poly_p, length))

    def block_encoding(self, signal_x: float) -> Circuit:
        r"""generate a block encoding of :math:`P(x)` by quantum circuit

        Args:
            x: input parameter

        Returns:
            a quantum circuit of unitary that is the block encoding of :math:`P(x)`

        """
        assert -1 <= signal_x <= 1, "x must be in domain [-1, 1]"
        
        Phi = self.Phi
        cir = Circuit()
        signal_x = -2 * acos(signal_x)
                        
        for i in range(1, len(Phi)):
            cir.rz(0, param=-2 * Phi[-i])
            cir.rx(0, param=signal_x)
        cir.rz(0, param=-2 * Phi[0])
        return cir
    
    def block_encoding_matrix(self, signal_x: float) -> paddle.Tensor:
        r"""generate a block encoding of :math:`P(x)` for verification

        Args:
            x: input parameter

        Returns:
            a block encoding unitary of :math:`P(x)`

        """
        assert -1 <= signal_x <= 1, "x must be in domain [-1, 1]"
        matrix = np.array([[self.poly_p(signal_x), 1j * self.poly_q(signal_x) * np.sqrt(1 - signal_x ** 2)],
                           [1j * np.conj(self.poly_q(signal_x)) * np.sqrt(1 - signal_x ** 2), np.conj(self.poly_p(signal_x))]])
        return paddle.to_tensor(matrix)
    