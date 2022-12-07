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
import warnings
from numpy.polynomial.polynomial import Polynomial, polyfromroots
from scipy.special import jv as Bessel
from typing import Any, Callable, List, Optional, Tuple, Union

r"""
Definition of ``Laurent`` class and its functions
"""


__all__ = ['Laurent', 'revise_tol', 'remove_abs_error', 'random_laurent_poly', 
           'sqrt_generation', 'Q_generation', 'pair_generation', 
           'laurent_generator', 'hamiltonian_laurent', 'ln_laurent', 'power_laurent']


TOL = 1e-30 # the error tolerance for Laurent polynomial, default to be machinery


class Laurent(object):
    r"""Class for Laurent polynomial defined as 
    :math:`P:\mathbb{C}[X, X^{-1}] \to \mathbb{C} :x \mapsto \sum_{j = -L}^{L} p_j X^j`.
    
    Args:
        coef: list of coefficients of Laurent poly, arranged as :math:`\{p_{-L}, ..., p_{-1}, p_0, p_1, ..., p_L\}`.

    """
    def __init__(self, coef: np.ndarray) -> None:
        if not isinstance(coef, np.ndarray):
            coef = np.array(coef)
        coef = coef.astype('complex128')
        coef = remove_abs_error(np.squeeze(coef) if len(coef.shape) > 1 else coef)
        assert len(coef.shape) == 1 and coef.shape[0] % 2 == 1
        
        # if the first and last terms are both 0, remove them
        while len(coef) > 1 and coef[0] == coef[-1] == 0:
            coef = coef[1:-1]
        
        # decide degree of this poly
        L = (len(coef) - 1) // 2 if len(coef) > 1 else 0
        
        # rearrange the coef in order p_0, ..., p_L, p_{-L}, ..., p_{-1},
        #   then we can call ``poly_coef[i]`` to retrieve p_i, this order is for internal use only
        coef = coef.tolist()
        poly_coef = np.array(coef[L:] + coef[:L]).astype('complex128')
        
        self.deg = L
        self.__coef = poly_coef
        
    def __call__(self, X: Union[int, float, complex]) -> complex:
        r"""Evaluate the value of P(X).
        """
        if X == 0:
            return self.__coef[0]

        return sum(self.__coef[i] * (X ** i) for i in range(-self.deg, self.deg + 1))
    
    @property
    def coef(self) -> np.ndarray:
        r"""The coefficients of this polynomial in ascending order (of indices).
        """
        return ascending_coef(self.__coef)
    
    @property
    def conj(self) -> 'Laurent':
        r"""The conjugate of this polynomial i.e. :math:`P(x) = \sum_{j = -L}^{L} p_{-j}^* X^j`.
        """
        coef = np.copy(self.__coef)
        for i in range(1, self.deg + 1):
            coef[i], coef[-i] = coef[-i], coef[i]
        coef = np.conj(coef)
        return Laurent(ascending_coef(coef))
    
    @property
    def roots(self) -> List[complex]:
        r"""List of roots of this polynomial.
        """
        # create the corresponding (common) polynomial with degree 2L
        P = Polynomial(self.coef)
        roots = P.roots().tolist()
        return sorted(roots, key=lambda x: np.abs(x))
    
    @property
    def norm(self) -> float:
        r"""The square sum of absolute value of coefficients of this polynomial.
        """
        return np.sum(np.square(np.abs(self.__coef)))
    
    @property
    def max_norm(self) -> float:
        r"""The maximum of absolute value of coefficients of this polynomial.
        """
        list_x = np.exp(-1j * np.arange(-np.pi, np.pi + 0.005, 0.005) / 2)
        return max(np.abs(self(x)) for x in list_x)
    
    @property
    def parity(self) -> int:
        r""" Parity of this polynomial.
        """
        coef = np.copy(self.__coef)

        even = not any(i % 2 != 0 and coef[i] != 0 for i in range(-self.deg, self.deg + 1))
        odd = not any(i % 2 != 1 and coef[i] != 0 for i in range(-self.deg, self.deg + 1))

        if even:
            return 0
        return 1 if odd else None
    
    def __copy__(self) -> 'Laurent':
        r"""Copy of Laurent polynomial.
        """
        return Laurent(ascending_coef(self.__coef))
    
    def __add__(self, other: Any) -> 'Laurent':
        r"""Addition of Laurent polynomial.
        
        Args:
            other: scalar or a Laurent polynomial :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`.
        
        """
        coef = np.copy(self.__coef)

        if isinstance(other, (int, float, complex)):
            coef[0] += other

        elif isinstance(other, Laurent):
            if other.deg > self.deg:
                return other + self

            deg_diff = self.deg - other.deg

            # retrieve the coef of Q
            q_coef = other.coef
            q_coef = np.concatenate([q_coef[other.deg:], np.zeros(2 * deg_diff),
                                     q_coef[:other.deg]]).astype('complex128')
            coef += q_coef

        else:
            raise TypeError(
                f"does not support the addition between Laurent and {type(other)}.")

        return Laurent(ascending_coef(coef))
    
    def __mul__(self, other: Any) -> 'Laurent':
        r"""Multiplication of Laurent polynomial.
        
        Args:
            other: scalar or a Laurent polynomial :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`.
        
        """
        p_coef = np.copy(self.__coef)

        if isinstance(other, (int, float, complex)):
            new_coef = p_coef * other

        elif isinstance(other, Laurent):
            # retrieve the coef of Q
            q_coef = other.coef.tolist()
            q_coef = np.array(q_coef[other.deg:] + q_coef[:other.deg]).astype('complex128')

            L = self.deg + other.deg # deg of new poly
            new_coef = np.zeros([2 * L + 1]).astype('complex128')

            # (P + Q)[X^n] = \sum_{j, k st. j + k = n} p_j q_k
            for j in range(-self.deg, self.deg + 1):
                for k in range(-other.deg, other.deg + 1):
                    new_coef[j + k] += p_coef[j] * q_coef[k]

        else:
            raise TypeError(
                f"does not support the multiplication between Laurent and {type(other)}.")

        return Laurent(ascending_coef(new_coef))
    
    def __sub__(self, other: Any) -> 'Laurent':
        r"""Subtraction of Laurent polynomial.
        
        Args:
            other: scalar or a Laurent polynomial :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`.
        
        """
        return self.__add__(other=other * -1)
    
    def __eq__(self, other: Any) -> bool:
        r"""Equality of Laurent polynomial.
        
        Args:
            other: a Laurent polynomial :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`.
        
        """
        if isinstance(other, (int, float, complex)):
            p_coef = self.__coef
            constant_term = p_coef[0]
            return self.deg == 0 and np.abs(constant_term - other) < 1e-6

        elif isinstance(other, Laurent):
            p_coef = self.coef
            q_coef = other.coef
            return self.deg == other.deg and np.max(np.abs(p_coef - q_coef)) < 1e-6

        else:
            raise TypeError(
                f"does not support the equality between Laurent and {type(other)}.")
    
    def __str__(self) -> str:
        r"""Print of Laurent polynomial.
        
        """
        coef = np.around(self.__coef, 3)
        L = self.deg
        
        print_str = "info of this Laurent poly is as follows\n"
        print_str += f"   - constant: {coef[0]}    - degree: {L}\n"
        if L > 0:
            print_str += f"   - coef of terms from pos 1 to pos {L}: {coef[1:L + 1]}\n"
            print_str += f"   - coef of terms from pos -1 to pos -{L}: {np.flip(coef[L + 1:])}\n"
        return print_str
    
    def is_parity(self, p: int) -> Tuple[bool, complex]:
        r"""Whether this Laurent polynomial has parity :math:`p % 2`.
        
        Args:
            p: parity.
        
        Returns
            contains the following elements:
            * whether parity is p % 2;
            * if not, then return the the (maximum) absolute coef term breaking such parity;
            * if not, then return the the (minimum) absolute coef term obeying such parity.
        
        """
        p %= 2
        coef = np.copy(self.__coef)

        disagree_coef = []
        agree_coef = []
        for i in range(-self.deg, self.deg + 1):
            c = coef[i]
            if i % 2 != p and c != 0:
                disagree_coef.append(c)
            elif i % 2 == p:
                agree_coef.append(c)

        return (False, max(np.abs(disagree_coef)), min(np.abs(agree_coef))) if disagree_coef else (True, None, None)
    
    def reduced_poly(self, target_deg: int) -> 'Laurent':
        r"""Generate :math:`P'(x) = \sum_{j = -D}^{D} p_j X^j`, where :math:`D \leq L` is `target_deg`.
        
        Args:
            target_deg: the degree of returned polynomial
        
        """
        coef = self.coef
        L = self.deg
        return Laurent(coef[L - target_deg:L + 1 + target_deg]) if target_deg <= L else Laurent(coef)


# ------------------------------------------------- Split line -------------------------------------------------
r"""
    Belows are support functions for `Laurent` class
"""


def revise_tol(t: float) -> None:
    r"""Revise the value of `TOL`.
    """
    global TOL
    assert t > 0
    TOL = t


def ascending_coef(coef: np.ndarray) -> np.ndarray:
    r"""Transform the coefficients of a polynomial in ascending order (of indices).
    
    Args:
        coef: list of coefficients arranged as :math:`\{ p_0, ..., p_L, p_{-L}, ..., p_{-1} \}`.
        
    Returns:
        list of coefficients arranged as :math:`\{ p_{-L}, ..., p_{-1}, p_0, p_1, ..., p_L \}`.
    
    """
    L = int((len(coef) - 1) / 2)
    coef = coef.tolist()
    return np.array(coef[L + 1:] + coef[:L + 1])


def remove_abs_error(data: np.ndarray, tol: Optional[float] = None) -> np.ndarray:
    r"""Remove the error in data array.
    
    Args:
        data: data array.
        tol: error tolerance.
        
    Returns:
        sanitized data.
        
    """
    data_len = len(data)
    tol = TOL if tol is None else tol

    for i in range(data_len):
        if np.abs(np.real(data[i])) < tol:
            data[i] = 1j * np.imag(data[i])
        elif np.abs(np.imag(data[i])) < tol:
            data[i] = np.real(data[i])
        
        if np.abs(data[i]) < tol:
            data[i] = 0
    return data


# ------------------------------------------------- Split Line -------------------------------------------------
r"""
    Belows are some functions using `Laurent` class
"""


def random_laurent_poly(deg: int, parity: Optional[int] = None, is_real: Optional[bool] = False) -> Laurent:
    r"""Randomly generate a Laurent polynomial.
    
    Args:
        deg: degree of this poly.
        parity: parity of this poly, defaults to be `None`.
        is_real: whether coefficients of this poly are real, defaults to be `False`.
        
    Returns:
        a Laurent poly with norm less than or equal to 1.
    
    """
    real = np.random.rand(deg * 2 + 1) * 2 - 1
    imag = np.zeros(deg * 2 + 1) if is_real else np.random.rand(deg * 2 + 1) * 2 - 1
    
    coef = real + 1j * imag
    coef /= np.sum(np.abs(coef))
    
    if parity is not None:
        coef = coef.tolist()
        coef = coef[deg:] + coef[:deg]
        for i in range(-deg, deg + 1):
            if i % 2 != parity:
                coef[i] = 0
        coef = np.array(coef[deg + 1:] + coef[:deg + 1])
                
    return Laurent(coef)
    

def sqrt_generation(A: Laurent) -> Laurent:
    r"""Generate the "square root" of a Laurent polynomial :math:`A`.
    
    Args:
        A: a Laurent polynomial.
        
    Returns:
        a Laurent polynomial :math:`Q` such that :math:`QQ^* = A`.
        
    Notes:
        More details are in Lemma S1 of the paper https://arxiv.org/abs/2209.14278.  
    
    """
    leading_coef = A.coef[-1]
    roots = A.roots

    roots_dict = dict({})
    def has_key(y: complex) -> Tuple[bool, complex]:
        r"""Test whether `y` is the key of roots_dict.
        
        Returns:
            contains the following elements:
            * boolean for whether `y` is the key of roots_dict.
            * the key matched (under certain error tolerance) or `None`.
        
        """
        list_key = list(roots_dict.keys())
        for key in list_key:
            # you can adjust this tolerance if the below assertion fails
            if np.abs(key - y) < 1e-6: 
                return True, key
        return False, None

    # begin filtering roots
    inv_roots = []
    for x in roots:
        inv_x = 1 / np.conj(x)

        is_key, key = has_key(x)

        # if x match with a existed key, save x
        if is_key:
            assert not has_key(inv_x)[0], \
                f"{x} and {inv_x} should not be in the same list, check your code; it is perhaps a precision problem."
            roots_dict[key] += 1

        # if neither x nor its inverse conjugate match, save x
        elif not has_key(inv_x)[0]:
            roots_dict[x] = 1

        # otherwise (i.e. inv_x with a existed key, save x), filter x
        else:
            inv_roots.append(x)

    # 1/x^* should be filtered from the list of roots, now update the roots of Q
    Q_roots = []
    for key in roots_dict:
        for _ in range(roots_dict[key]):
            Q_roots.append(key)

    # be careful that the number of filtered roots should be identical with that of the saved roots
    if len(Q_roots) != len(inv_roots):
        warnings.warn(
            "\nError occurred in square root decomposition of polynomial: " +
            f"# of total, saved and filtered roots are {len(roots)}, {len(Q_roots)}, {len(inv_roots)}." +
            "\n     Will force equal size of saved and filtered root list to mitigate the error")
        excess_roots, Q_roots = Q_roots[len(inv_roots):], Q_roots[:len(inv_roots)]
        excess_roots.sort(key=lambda x: np.real(x)) # sort by real part
        
        for i in range(len(excess_roots) // 2):
            Q_roots.append(excess_roots[2 * i])
            inv_roots.append(excess_roots[2 * i + 1])            
    inv_roots = np.array(inv_roots)

    # construct Q
    Q_coef = polyfromroots(Q_roots) * np.sqrt(leading_coef * np.prod(inv_roots))
    Q = Laurent(Q_coef)

    # final output test
    if Q * Q.conj != A:
        warnings.warn(
            f"\ncomputation error: QQ* != A, check your code \n degree of Q: {Q.deg}, degree of A: {A.deg}")

    return Q


def Q_generation(P: Laurent) -> Laurent:
    r"""Generate a Laurent complement for Laurent polynomial :math:`P`.
    
    Args:
        P: a Laurent poly with parity :math:`L` and degree :math:`L`.
        
    Returns:
        a Laurent poly :math:`Q` st. :math:`PP^* + QQ^* = 1`, with parity :math:`L` and degree :math:`L`.
    
    """
    assert P.parity is not None and P.parity == P.deg % 2, \
        "this Laurent poly does not satisfy the requirement for parity"
    assert P.max_norm < 1, \
        f"the max norm {P.max_norm} of this Laurent poly should be smaller than 1"
    
    Q2 = P * P.conj * -1 + 1
    Q = sqrt_generation(Q2)
        
    is_parity, max_diff, min_val = Q.is_parity(P.parity)
    if not is_parity:
        warnings.warn(
            f"\nQ's parity {Q.parity} does not agree with P's parity {P.parity}, max err is {max_diff}, min val is {min_val}")

    return Q


def pair_generation(f: Laurent) -> Laurent:
    r""" Generate Laurent pairs for Laurent polynomial :math:`f`.
    
    Args:
        f: a real-valued and even Laurent polynomial, with max_norm smaller than 1.
    
    Returns:
        Laurent polys :math:`P, Q` st. :math:`P = \sqrt{1 + f / 2}, Q = \sqrt{1 - f / 2}`.
    
    """
    assert f.max_norm < 1, \
        f"the max norm {f.max_norm} of this Laurent poly should be smaller than 1"
    assert f.parity == 0, \
        "the parity of this Laurent poly should be 0"
    expect_parity = (f.deg // 2) % 2
    P, Q = sqrt_generation((f + 1) * 0.5), sqrt_generation((f * (-1) + 1) * 0.5)

    is_parity, max_diff, min_val = P.is_parity(expect_parity)
    if not is_parity:
        warnings.warn(
            f"\nP's parity {P.parity} does not agree with {expect_parity}, max err is {max_diff}, min val is {min_val}", UserWarning)

    is_parity, max_diff, min_val = Q.is_parity(expect_parity)
    if not is_parity:
        warnings.warn(
            f"\nQ's parity {Q.parity} does not agree with {expect_parity}, max err is {max_diff}, min val is {min_val}", UserWarning)
    return P, Q


# ------------------------------------------------- Split line -------------------------------------------------
r"""
    Belows are tools for trigonometric approximation.
"""


def laurent_generator(fn: Callable[[np.ndarray], np.ndarray], dx: float, deg: int, L: float) -> Laurent:
    r"""Generate a Laurent polynomial (with :math:`X = e^{ix / 2}`) approximating `fn`.
    
    Args:
        fn: function to be approximated.
        dx: sampling frequency of data points.
        deg: degree of Laurent poly.
        L: half of approximation width.
    
    Returns:
        a Laurent polynomial approximating `fn` in interval :math:`[-L, L]` with degree `deg`.
    
    """
    assert dx > 0 and L > 0 and deg >= 0

    N = 2 * L / dx
    coef = np.zeros(deg + 1, dtype=np.complex128)
    xk = np.arange(-L, L + dx, dx)

    # Calculate the coefficients for each term
    for mi in range(deg + 1):
        n = mi - deg / 2
        coef[mi] = 1 / N * sum(fn(xk) * np.exp(-1j * n * np.pi * xk / L))

    coef = np.array([coef[i // 2] if i % 2 == 0 else 0 for i in range(deg * 2 + 1)])
    
    return Laurent(coef)


def deg_finder(fn: Callable[[np.ndarray], np.ndarray], 
               delta: Optional[float] = 0.00001 * np.pi, l: Optional[float] = np.pi) -> int:
    r"""Find a degree such that the Laurent polynomial generated from `laurent_generator` has max_norm smaller than 1.
    
    Args:
        fn: function to be approximated.
        dx: sampling frequency of data points, defaults to be :math:`0.00001 \pi`.
        L: half of approximation width, defaults to be :math:`\pi`.
    
    Returns:
        the degree of approximation:
        
    Notes:
        used to fix the problem of function `laurent_generator`.
    
    """
    deg = 50
    acc = 1
    P = laurent_generator(fn, delta, deg, l)
    while P.max_norm > 1:
        deg += acc * 50
        P = laurent_generator(fn, delta, deg, l)
        acc += 1
        assert deg <= 10000, "degree too large"
    return deg


def step_laurent(deg: int) -> Laurent:
    r"""Generate a Laurent polynomial approximating the step function.
    
    Args:
        deg: (even) degree of the output Laurent poly.
    
    Returns:
        a Laurent poly approximating :math:`f(x) = 0.5` if :math:`x <= 0` else :math:`0`.
    
    Notes:
        used in Hamiltonian energy solver
    
    """
    assert deg % 2 == 0
    deg //= 2
    coef = np.zeros(2 * deg + 1).astype('complex128')
    for n in range(-deg, deg + 1):
        if n != 0:
            coef[n] = (0.025292684335809737j +
                         0.11253953951963828j * np.exp(-np.pi * 1j * n) -
                         0.13783222385544802j * np.exp(np.pi * 1j * n)) / n
        else:
            coef[n] = 0.7865660924854931
    coef = ascending_coef(coef)
    coef = np.array([coef[i // 2] if i % 2 == 0 else 0 for i in range(4 * deg + 1)])
    return Laurent(coef)


def hamiltonian_laurent(t: float, deg: int) -> Laurent:
    r"""Generate a Laurent polynomial approximating the Hamiltonian evolution function.
    
    Args:
        t: evolution constant (time).
        deg: (even) degree of the output Laurent poly.
    
    Returns:
        a Laurent poly approximating :math:`e^{it \cos(x)}`.
        
    Note:
        - originated from the Jacobi-Anger expansion: :math:`y(x) = \sum_n i^n Bessel(n, x) e^{inx}`;
        - used in Hamiltonian simulation.
    
    """
    assert deg % 2 == 0
    deg //= 2
    coef = np.zeros(2 * deg + 1).astype('complex128')
    for n in range(-deg, deg + 1):
        coef[n] = (1j ** n) * Bessel(n, t)

    coef = ascending_coef(coef)
    coef = np.array([coef[i // 2] if i % 2 == 0 else 0 for i in range(4 * deg + 1)])
    return Laurent(coef)


def ln_laurent(deg: int, t: float) -> Laurent:
    r"""Generate a Laurent polynomial approximating the ln function.
    
    Args:
        deg: degree of Laurent polynomial that is a factor of 4.
        t: normalization factor.
        
    Returns:
        a Laurent poly approximating :math:`ln(cos(x)^2) / t`.
        
    Notes:
        used in von Neumann entropy estimation.
        
    """
    assert deg % 4 == 0
    deg //= 2
    coef = np.zeros(2 * deg + 1).astype('complex128')

    for k in range(1, deg + 1):
        for j in range(2 * k + 1):
            coef[2 * j - 2 * k] += ((-1) ** (k + j + 1)) / t / k * (0.25 ** k) * comb(2 * k, j)

    coef = ascending_coef(coef)
    coef = np.array([coef[i // 2] if i % 2 == 0 else 0 for i in range(4 * deg + 1)])
    return Laurent(coef)


def comb(n: float, k: int) -> float:
    r"""Compute nCr(n, k).
    """
    prod = 1
    for i in range(k):
        prod *= (n - i) / (k - i)
    return prod

    
def power_laurent(deg: int, alpha: float, t: float) -> Laurent:
    r"""Generate a Laurent polynomial approximating the power function.
    
    Args:
        deg: degree of Laurent polynomial that is a factor of 4.
        alpha: the # of power.
        t: normalization factor.
        
    Returns:
        a Laurent poly approximating :math:`(cos(x)^2)^{\alpha / 2} / t`.
        
    """
    assert deg % 4 == 0 and alpha != 0 and alpha > -1
    alpha /= 2
    deg //= 2
    coef = np.zeros(2 * deg + 1).astype('complex128')

    for k in range(deg + 1):
        for j in range(2 * k + 1):
            coef[2 * j - 2 * k] += ((-1) ** j) / t * comb(alpha, k) * (0.25 ** k) * comb(2 * k, j)

    coef = ascending_coef(coef)
    coef = np.array([coef[i // 2] if i % 2 == 0 else 0 for i in range(4 * deg + 1)])
    return Laurent(coef)
