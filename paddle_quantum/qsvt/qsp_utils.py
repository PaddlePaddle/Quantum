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
from numpy.polynomial.polynomial import Polynomial
import numpy as np
from typing import Tuple, Optional
import paddle
from scipy.linalg import expm


r"""
    Tools for Polynomial & Tensor in qsvt Modules
"""


# ----------------------------- belows are polynomial tools ------------------------


def clean_small_error(array: np.ndarray) -> np.ndarray:
    r"""clean relatively small quantity

    Args:
        array: target array

    Returns:
        cleaned array

    """

    def compare_and_clean(a: float, b: float) -> Tuple[float, float]:
        r"""discard tiny or relatively tiny real or imaginary parts of elements"""
        if a == 0 or b == 0:
            return a, b

        a_abs = np.abs(a)
        b_abs = np.abs(b)

        abs_error = 10 ** (-2)
        rel_error = 10 ** (-4)

        if a_abs < abs_error and a_abs / b_abs < rel_error:
            return 0, b

        if b_abs < abs_error and b_abs / a_abs < rel_error:
            return a, 0

        return a, b

    for i in range(len(array)):
        real, imag = compare_and_clean(np.real(array[i]), np.imag(array[i]))
        array[i] = real + 1j * imag

    return array


def poly_norm(poly: Polynomial, p: Optional[int] = 1) -> float:
    r"""calculate the p-norm of a polynomial

    Args:
        poly: the target polynomial
        p: order of norm, 0 means to be infinity

    Returns:
        p-norm of the target polynomial

    """
    coef = poly.coef
    if p == 0:
        return np.max(coef)

    coef_pow = list(map(lambda x: np.abs(x) ** p, coef))
    return np.power(np.sum(coef_pow), 1 / p)


def poly_real(poly: Polynomial) -> Polynomial:
    r"""return the real part of a polynomial

    Args:
        poly: the target polynomial

    Returns:
        the real part of poly

    """
    return Polynomial(np.real(poly.coef))


def poly_imag(poly: Polynomial) -> Polynomial:
    r"""return the imaginary part of the polynomial

    Args:
        poly: the target polynomial

    Returns:
        the imaginary part of poly

    """
    return Polynomial(np.imag(poly.coef))


# ----------------------------- belows are polynomial and tensor tools ------------------------


def poly_matrix(poly: Polynomial, matrix_A: paddle.Tensor) -> paddle.Tensor:
    r"""calculate the polynomial of a matrix, poly(matrix_A)

    Args:
        poly: the polynomial
        matrix_A: the matrix

    Returns:
        poly(matrix_A)

    """
    coef = paddle.to_tensor(poly.coef)
    k = poly.degree()

    N = matrix_A.shape[0]
    I = paddle.eye(N)

    matrix = paddle.cast(paddle.zeros([N, N]), "complex128")
    matrix_temp = I
    for i in range(0, k + 1):
        for j in range(i):
            matrix_temp = matrix_temp @ matrix_A
        matrix += coef[i] * matrix_temp
        matrix_temp = I

    return matrix


def exp_matrix(t: float, matrix_A: paddle.Tensor) -> paddle.Tensor:
    r"""calculate :math:`e^{itA}`

    Args:
        t: time constant
        A: the target matrix

    Returns:
        :math:`e^{itA}`

    """
    matrix_A = matrix_A.cast("complex128").numpy()
    return paddle.to_tensor(expm(1j * t * matrix_A), dtype="complex128")
