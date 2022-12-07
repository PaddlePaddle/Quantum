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
from paddle_quantum.ansatz import Circuit

from copy import copy
from typing import Optional, Tuple, List
from math import atan, cos, sin
from .laurent import Laurent


r"""
QPP angle solver for trigonometric QSP, see Lemma 3 in paper https://arxiv.org/abs/2205.07848 for more details.
"""


__all__ = ['qpp_angle_finder', 'qpp_angle_approximator']


def qpp_angle_finder(P: Laurent, Q: Laurent) -> Tuple[List[float], List[float]]:
    r"""Find the corresponding set of angles for a Laurent pair `P`, `Q`.
    
    Args:
        P: a Laurent poly.
        Q: a Laurent poly.
        
    Returns:
        contains the following elements:
        - list_theta: angles for :math:`R_Y` gates;
        - list_phi: angles for :math:`R_Z` gates.
    
    """
    # input check
    P, Q = copy(P), copy(Q)
    condition_test(P, Q)
    
    list_theta = []
    list_phi = []
    
    # backup P for output check
    P_copy = copy(P)
    
    L = P.deg
    while L > 0:
        theta, phi = update_angle([P.coef[-1], P.coef[0], Q.coef[-1], Q.coef[0]])
        
        list_theta.append(theta)
        list_phi.append(phi)
        
        P, Q = update_polynomial(P, Q, theta, phi)
        L = P.deg
    
    # decide theta[0], phi[0] and global phase alpha
    p_0, q_0 = P.coef[0], Q.coef[0]
    alpha, theta, phi = yz_decomposition(np.array([[p_0, -q_0], [np.conj(q_0), np.conj(p_0)]]))
    list_theta.append(theta)
    list_phi.append(phi)
    
    # test outputs, by 5 random data points in [-pi, pi]
    err_list = []
    list_x = (np.random.rand(5) * 2 - 1) * np.pi 
    for x in list_x:
        experiment_y = matrix_generation(list_theta, list_phi, x, alpha)[0, 0] 
        actual_y = P_copy(np.exp(1j * x / 2))
        
        err = np.abs(experiment_y + actual_y) if np.abs(experiment_y / actual_y + 1) < 1e-2 else np.abs(experiment_y - actual_y)
        if err > 0.1:
            print(experiment_y)
            print(actual_y)
            raise ValueError( 
                f"oversized error: {err}, check your code")
        err_list.append(err)
    print(f"computations of angles for QPP are completed with mean error {np.mean(err_list)}")
    
    return list_theta, list_phi


def qpp_angle_approximator(P: Laurent, Q: Laurent) -> Tuple[List[float], List[float]]:
    r"""Approximate the corresponding set of angles for a Laurent pair `P`, `Q`.
    
    Args:
        P: a Laurent poly.
        Q: a Laurent poly.
        
    Returns:
        contains the following elements:
        - list_theta: angles for :math:`R_Y` gates;
        - list_phi: angles for :math:`R_Z` gates.
    
    Note:
        unlike `yzzyz_angle_finder`, 
        `yzzyz_angle_approximator` assumes that the only source of error is the precision (which is not generally true).

    """
    list_theta = []
    list_phi = []
    
    # backup P for output check
    P_copy = copy(P)
    
    L = P.deg
    while L > 0:
        theta, phi = update_angle([P.coef[-1], P.coef[0], Q.coef[-1], Q.coef[0]])
        
        list_theta.append(theta)
        list_phi.append(phi)
        
        P, Q = update_polynomial(P, Q, theta, phi, verify=False)
        
        L -= 1
        P, Q = P.reduced_poly(L), Q.reduced_poly(L)
    
    # decide theta[0], phi[0] and global phase alpha
    p_0, q_0 = P.coef[0], Q.coef[0]
    alpha, theta, phi = yz_decomposition(np.array([[p_0, -q_0], [np.conj(q_0), np.conj(p_0)]]))
    list_theta.append(theta)
    list_phi.append(phi)
    
    # test outputs, by 5 random data points in [-pi, pi]
    err_list = []
    list_x = (np.random.rand(5) * 2 - 1) * np.pi 
    for x in list_x:
        experiment_y = matrix_generation(list_theta, list_phi, x, alpha)[0, 0] 
        actual_y = P_copy(np.exp(1j * x / 2))
        
        err = np.abs(experiment_y + actual_y) if np.abs(experiment_y / actual_y + 1) < 1e-2 else np.abs(experiment_y - actual_y)
        err_list.append(err)
    print(f"Computations of angles for QPP are completed with mean error {np.mean(err_list)}")
    
    return list_theta, list_phi


# ------------------------------------------------- Split line -------------------------------------------------
r"""
    Belows are support functions for angles computation.
"""


def update_angle(coef: List[complex]) -> Tuple[float, float]:
    r"""Compute angles by `coef` from `P` and `Q`.
    
    Args:
        coef: the first and last terms from `P` and `Q`.
        
    Returns:
        `theta` and `phi`.
    
    """
    # with respect to the first and last terms of P and Q, respectively
    p_d, p_nd, q_d, q_nd = coef[0], coef[1], coef[2], coef[3]
    if p_d != 0 and q_d != 0:
        val = -1 * p_d / q_d
        return atan(np.abs(val)) * 2, np.real(np.log(val / np.abs(val)) / (-1j))
    
    elif np.abs(p_d) < 1e-25 and np.abs(q_d) < 1e-25:
        val = q_nd / p_nd
        return atan(np.abs(val)) * 2, np.real(np.log(val / np.abs(val)) / (-1j))
    
    elif np.abs(p_d) < 1e-25 and np.abs(q_nd) < 1e-25:
        return 0, 0
    
    elif np.abs(p_nd) < 1e-25 and np.abs(q_d) < 1e-25:
        return np.pi, 0
    
    raise ValueError(
        f"Coef error: check these four coef {[p_d, p_nd, q_d, q_nd]}")


def update_polynomial(P: Laurent, Q: Laurent, theta: float, phi: float, verify: Optional[bool] = True) -> Tuple[Laurent, Laurent]:
    r"""Update `P` and `Q` by `theta` and `phi`.
    
    Args:
        P: a Laurent poly.
        Q: a Laurent poly.
        theta: a param.
        phi: a param.
        verify: whether verify the correctness of computation, defaults to be `True`.
    
    Returns:
        updated `P` and `Q`.
        
    """
    
    phi_hf = phi / 2
    theta_hf = theta / 2
    
    X = Laurent([0, 0, 1])
    inv_X = Laurent([1, 0, 0])
    
    new_P = (X * P * np.exp(1j * phi_hf) * cos(theta_hf)) + (X * Q * np.exp(-1j * phi_hf) * sin(theta_hf))
    new_Q = (inv_X * Q * np.exp(-1j * phi_hf) * cos(theta_hf)) - (inv_X * P * np.exp(1j * phi_hf) * sin(theta_hf))
    
    if not verify:
        return new_P, new_Q
    
    condition_test(new_P, new_Q)
    return new_P, new_Q

    
def condition_test(P: Laurent, Q: Laurent) -> None:
    r"""Check whether `P` and `Q` satisfy:
        - deg(`P`) = deg(`Q`);
        - `P` 和 `Q` have the same parity;
        - :math:`PP^* + QQ^* = 1`.
    
    Args:
        P: a Laurent poly.
        Q: a Laurent poly.
    
    """
    L = P.deg
    
    if L != Q.deg:
        print("The last and first terms of P: ", P.coef[0], P.coef[-1])
        print("The last and first terms of Q: ", Q.coef[0], Q.coef[-1])
        raise ValueError(f"P's degree {L} does not agree with Q's degree {Q.deg}")
    
    if P.parity != Q.parity or P.parity != L % 2:
        print(f"P's degree is {L}")
        raise ValueError(f"P's parity {P.parity} and Q's parity {Q.parity}) should be both {L % 2}")
    
    poly_one = (P * P.conj) + (Q * Q.conj)
    if poly_one != 1:
        print(f"P's degree is {L}")
        print("the last and first terms of PP* + QQ*: ", poly_one.coef[0], poly_one.coef[-1])
        raise ValueError("PP* + QQ* != 1: check your code")
    

def matrix_generation(list_theta: List[float], list_phi: List[float], x: float, alpha: Optional[float] = 0) -> np.ndarray:
    r"""Return the matrix generated by sets of angles.
    
    Args:
        list_theta: angles for :math:`R_Y` gates.
        list_phi: angles for :math:`R_Z` gates.
        x: input of polynomial P
        alpha: global phase
        
    Returns:
        unitary matrix generated by YZZYZ circuit
        
    """
    assert len(list_theta) == len(list_phi)
    L = len(list_theta) - 1
    
    cir = Circuit(1)
    for i in range(L):
        cir.rz(0, param=list_phi[i])
        cir.ry(0, param=list_theta[i])
        cir.rz(0, param=x)  # input x
    cir.rz(0, param=list_phi[-1])
    cir.ry(0, param=list_theta[-1])
    
    return cir.unitary_matrix().numpy() * alpha


def yz_decomposition(U: np.ndarray) -> Tuple[complex, float, float]:
    r"""Return the YZ decomposition of U.
    
    Args:
        U: single-qubit unitary.

    Returns:
        `alpha`, `theta`, `phi` st. :math:`U[0, 0] = \alpha R_Y(\theta) R_Z(\phi) [0, 0]`.
    
    """
    a, b = U[0, 0], U[0, 1]
    x, y, p, q = np.real(a), np.imag(a), np.real(b), np.imag(b)
    
    phi = np.pi if x == p == 0 else np.arctan(-y / x) - np.arctan(-q / p)
    theta = 2 * np.arctan(np.sqrt((p ** 2 + q ** 2) / (x ** 2 + y ** 2)))
    
    alpha = -a / (cos(phi / 2) * cos(theta / 2) - 1j * sin(phi / 2) * cos(theta / 2))
    assert np.abs(np.abs(alpha) - 1) < 1e-5, f"Calculation error for absolute global phase {np.abs(alpha)}, check your code."
    
    if np.abs((cos(theta / 2) * np.exp(1j * phi / 2)) * alpha / a + 1) < 1e-6:
        alpha = -alpha
    return alpha, theta, phi
    