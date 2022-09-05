# !/usr/bin/env python3
# Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
Trotter Hamiltonian time evolution circuit module.
"""


from collections import defaultdict
from typing import Optional, Union, Iterable
import paddle_quantum
from paddle_quantum import Hamiltonian
import warnings
import numpy as np
import re
import paddle
from .intrinsic import _get_float_dtype
from .ansatz import Circuit

float_dtype = _get_float_dtype(paddle_quantum.get_dtype())
PI = paddle.to_tensor(np.pi, dtype=float_dtype)


def construct_trotter_circuit(
        circuit: Circuit,
        hamiltonian: Hamiltonian,
        tau: float,
        steps: int,
        method: Optional[str] = 'suzuki',
        order: Optional[int] = 1,
        grouping: Optional[str] = None,
        coefficient: Optional[Union[np.ndarray, paddle.Tensor]] = None,
        permutation: Optional[np.ndarray] = None
):
    r"""Add time-evolving circuits to a user-specified circuit.
    
    This circuit could approximate the time-evolving operator of a system given its Hamiltonian H,
    i.e., :math:`U_{\rm cir}~ e^{-iHt}`.

    Args:
        circuit: Circuit object to which a time evolution circuit will be added.
        hamiltonian: Hamiltonian of the system whose time evolution is to be simulated.
        tau: Evolution time of each trotter block.
        steps: Number of trotter blocks that will be added in total.
            (Hint: ``steps * tau`` should be the total evolution time.)
        method: How the time evolution circuit will be constructed. Defaults to ``'suzuki'``, i.e., using
            Trotter-Suzuki decomposition. Set to ``'custom'`` to use a customized simulation strategy.
            (Needs to be specified with arguments permutation and coefficient.)
        order: Order of the Trotter-Suzuki decomposition. Only works when ``method='suzuki'``. Defaults to 1.
        grouping: Whether the Hamiltonian's ordering will be rearranged in a user-specified way. Supports ``'xyz'``
            and ``'even_odd'`` grouping methods. Defaults to None.
        coefficient: Custom coefficients corresponding to terms of the Hamiltonian. Only works for ``method='custom'``. Defaults to None.
        permutation: Custom permutation of the Hamiltonian. Only works for ``method='custom'``. Defaults to None.
    
    Raises:
        ValueError: The order of the trotter-suzuki decomposition should be either 1, 2 or 2k (k an integer)
        ValueError: Shape of the permutation and coefficient array don\'t match
        ValueError: Grouping method ``grouping`` is not supported, valid key words: 'xyz', 'even_odd'
        ValueError: The method ``method`` is not supported, valid method keywords: 'suzuki', 'custom'

    Hint:
        For a more detailed explanation of how this function works, users may refer to the tutorials on Paddle Quantum's website: https://qml.baidu.com/tutorials/overview.html.
    """
    # check the legitimacy of the inputs (coefficient and permutation)
    def check_input_legitimacy(arg_in):
        if not isinstance(arg_in, np.ndarray) and not isinstance(arg_in, paddle.Tensor):
            arg_out = np.array(arg_in)
        else:
            arg_out = arg_in

        if arg_out.ndim == 1 and isinstance(arg_out, np.ndarray):
            arg_out = arg_out.reshape(1, arg_out.shape[0])
        elif arg_out.ndim == 1 and isinstance(arg_out, paddle.Tensor):
            arg_out = arg_out.reshape([1, arg_out.shape[0]])

        return arg_out

    # check compatibility between input method and customization arguments
    if (permutation is not None or coefficient is not None) and (method != 'custom'):
        warning_message = 'method {} is not compatible with customized permutation ' \
                          'or coefficient and will be overlooked'.format(method)
        method = 'custom'
        warnings.warn(warning_message, RuntimeWarning)

    # check the legitimacy of inputs
    if method == 'suzuki':
        if order > 2 and order % 2 != 0 and type(order) != int:
            raise ValueError('The order of the trotter-suzuki decomposition should be either 1, 2 or 2k (k an integer)'
                             ', got order = %i' % order)

    # check and reformat inputs for 'custom' mode
    elif method == 'custom':
        # check permutation
        if permutation is not None:
            permutation = np.array(check_input_legitimacy(permutation), dtype=int)
            # give a warning for using permutation and grouping at the same time
            if grouping:
                warning_message = 'Using specified permutation and automatic grouping {} at the same time, the ' \
                                  'permutation will act on the grouped Hamiltonian'.format(grouping)
                warnings.warn(warning_message, RuntimeWarning)
        # check coefficient
        if coefficient is not None:
            coefficient = check_input_legitimacy(coefficient)
        # if the permutation is not specified, then set it to [[1, 2, ...], ...]
        if coefficient is not None and permutation is None:
            permutation = np.arange(hamiltonian.n_terms) if coefficient.ndim == 1 \
                else np.arange(hamiltonian.n_terms).reshape(1, hamiltonian.n_terms).repeat(coefficient.shape[0], axis=0)
            permutation = np.arange(hamiltonian.n_terms).reshape(1, hamiltonian.n_terms)
            permutation = permutation.repeat(coefficient.shape[0], axis=0)
        # if the coefficient is not specified, set a uniform (normalized) coefficient
        if permutation is not None and coefficient is None:
            coefficient = 1 / len(permutation) * np.ones_like(permutation)
        # the case that the shapes of input coefficient and permutations don't match
        if tuple(permutation.shape) != tuple(coefficient.shape):
            # case not-allowed
            if permutation.shape[1] != coefficient.shape[1]:
                raise ValueError('Shape of the permutation and coefficient array don\'t match, got {} and {}'.format(
                    tuple(permutation.shape), tuple(coefficient.shape)
                ))
            # cases can be fixed by repeating one of the two inputs
            elif permutation.shape[0] != coefficient.shape[0] and permutation[0] == 1:
                permutation = permutation.repeat(coefficient.shape[0])
            elif permutation.shape[0] != coefficient.shape[0] and coefficient[0] == 1:
                if isinstance(coefficient, paddle.Tensor):
                    coefficient = paddle.stack([coefficient for _ in range(permutation.shape[0])])\
                        .reshape([permutation.shape[0], permutation.shape[1]])
                elif isinstance((coefficient, np.ndarray)):
                    coefficient = coefficient.repeat(permutation.shape[0])

    # group the hamiltonian according to the input
    if not grouping:
        grouped_hamiltonian = [hamiltonian]
    elif grouping == 'xyz':
        grouped_hamiltonian = __group_hamiltonian_xyz(hamiltonian=hamiltonian)
    elif grouping == 'even_odd':
        grouped_hamiltonian = __group_hamiltonian_even_odd(hamiltonian=hamiltonian)
    else:
        raise ValueError("Grouping method %s is not supported, valid key words: 'xyz', 'even_odd'" % grouping)

    # apply trotter blocks
    for step in range(steps):
        if method == 'suzuki':
            _add_trotter_block(circuit=circuit, tau=tau, grouped_hamiltonian=grouped_hamiltonian, order=order)
        elif method == 'custom':
            _add_custom_block(circuit=circuit, tau=tau, grouped_hamiltonian=grouped_hamiltonian,
                              custom_coefficients=coefficient, permutation=permutation)
        else:
            raise ValueError("The method %s is not supported, valid method keywords: 'suzuki', 'custom'" % method)


def __get_suzuki_num(order):
    r"""compute the Trotter number of the suzuki product formula with the order of ``order``
    """
    if order == 1 or order == 2:
        n_suzuki = order
    elif order > 2 and order % 2 == 0:
        n_suzuki = 2 * 5 ** (order // 2 - 1)
    else:
        raise ValueError('The order of the trotter-suzuki decomposition should be either 1, 2 or 2k (k an integer)'
                         ', got order = %i' % order)

    return n_suzuki


def __sort_pauli_word(pauli_word, site):
    r"""reordering the ``pauli_word`` by the value of ``site``, return the new pauli_word and site after sort.

    Note:
        This is an intrinsic function, user do not need to call this directly
    """
    sort_index = np.argsort(np.array(site))
    return ''.join(np.array(list(pauli_word))[sort_index].tolist()), np.array(site)[sort_index]


def _add_trotter_block(circuit, tau, grouped_hamiltonian, order):
    r"""add a Trotter block, i.e. :math:`e^{-iH\tau}`, use Trotter-Suzuki decomposition to expand it.

    Args:
        circuit: target circuit to add the Trotter block
        tau: evolution time of this Trotter block
        grouped_hamiltonian: list of Hamiltonian objects, this function uses these as the basic terms of Trotter-Suzuki expansion by default
        order: The order of Trotter-Suzuki expansing

    Note:
        About how to use grouped_hamiltonian: 
        For example, consider Trotter-Suzuki decomposition of the second order S2(t), if grouped_hamiltonian = [H_1, H_2], it will add Trotter circuit
        with (H_1, t/2)(H_2, t/2)(H_2, t/2)(H_1, t/2). Specifically, if user does not pre-grouping the Hamiltonians and put a single Hamiltonian object,
        this function will make canonical decomposition according to the order of this Hamiltonian: for second order, if put a single Hamiltonian H, 
        the circuit will be added with (H[0:-1:1], t/2)(H[-1:0:-1], t/2)

    Warning:
        This function is usually an intrinsic function, it does not check or correct the input. 
        To build time evolution circuit, function ``construct_trotter_circuit()`` is recommanded
    """
    if order == 1:
        __add_first_order_trotter_block(circuit, tau, grouped_hamiltonian)
    elif order == 2:
        __add_second_order_trotter_block(circuit, tau, grouped_hamiltonian)
    else:
        __add_higher_order_trotter_block(circuit, tau, grouped_hamiltonian, order)
    pass


def _add_custom_block(circuit, tau, grouped_hamiltonian, custom_coefficients, permutation):
    r"""Add a custom Trotter block

    Args:
        circuit: target circuit to add the Trotter block
        tau: evolution time of this Trotter block
        grouped_hamiltonian: list of Hamiltonian objects, this function uses these as the basic terms of Trotter-Suzuki expansion by default
        order: The order of Trotter-Suzuki expansing
        permutation: custom permutation
        custom_coefficients: custom coefficients

    Warning:
        This function is usually an intrinsic function, it does not check or correct the input. 
        To build time evolution circuit, function ``construct_trotter_circuit()`` is recommanded
    """

    # combine the grouped hamiltonian into one single hamiltonian
    hamiltonian = sum(grouped_hamiltonian, Hamiltonian([]))

    # apply trotter circuit according to coefficient and
    h_coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
    for i in range(permutation.shape[0]):
        for term_index in range(permutation.shape[1]):
            custom_coeff = custom_coefficients[i][term_index]
            term_index = permutation[i][term_index]
            pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
            coeff = h_coeffs[term_index] * custom_coeff
            add_n_pauli_gate(circuit, 2 * tau * coeff, pauli_word, site)


def __add_first_order_trotter_block(circuit, tau, grouped_hamiltonian, reverse=False, optimization=False):
    r"""Add a time evolution block of the first order Trotter-Suzuki decompositon

    Notes:
        This is an intrinsic function, user do not need to call this directly
    """
    if not reverse:
        for hamiltonian in grouped_hamiltonian:
            assert isinstance(hamiltonian, Hamiltonian)
            if optimization:
                # Combine XX, YY, ZZ of the same site in the original Hamiltonian quantity
                grouped_hamiltonian = []
                coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
                grouped_terms_indices = []
                left_over_terms_indices = []
                d = defaultdict(list)
                # Merge XX,YY,ZZ of the same site
                for term_index in range(len(coeffs)):
                    site = sites[term_index]
                    pauli_word = pauli_words[term_index]
                    for pauli in ['XX', 'YY', 'ZZ']:
                        assert isinstance(pauli_word, str), "Each pauli word should be a string type"
                        if (pauli_word == pauli or pauli_word == pauli.lower()):
                            key = tuple(sorted(site))
                            d[key].append((pauli, term_index))
                            if len(d[key]) == 3:
                                terms_indices_to_be_grouped = [x[1] for x in d[key]]
                                grouped_terms_indices.extend(terms_indices_to_be_grouped)
                                grouped_hamiltonian.append(hamiltonian[terms_indices_to_be_grouped])
                # Other remaining sites
                for term_index in range(len(coeffs)):
                    if term_index not in grouped_terms_indices:
                        left_over_terms_indices.append(term_index)
                if len(left_over_terms_indices):
                    for term_index in left_over_terms_indices:
                        grouped_hamiltonian.append(hamiltonian[term_index])
                # Get the new Hamiltonian
                res = grouped_hamiltonian[0]
                for i in range(1, len(grouped_hamiltonian)):
                    res += grouped_hamiltonian[i]
                hamiltonian = res
            # decompose the Hamiltonian into 3 lists
            coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
            # apply rotational gate of each term
            # for term_index in range(len(coeffs)):
            #     # get the sorted pauli_word and site (an array of qubit indices) according to their qubit indices
            #     pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
            #     add_n_pauli_gate(circuit, 2 * tau * coeffs[term_index], pauli_word, site)
            term_index = 0
            while term_index < len(coeffs):
                if optimization and term_index+3 <= len(coeffs) and \
                        len(set(y for x in sites[term_index:term_index+3] for y in x)) == 2 and\
                        set(pauli_words[term_index:term_index+3]) == {'XX', 'YY', 'ZZ'}:
                    optimal_circuit(circuit, [tau*i for i in coeffs[term_index:term_index+3]], sites[term_index])
                    term_index += 3
                else:
                    # get the sorted pauli_word and site (an array of qubit indices) according to their qubit indices
                    pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
                    add_n_pauli_gate(circuit, 2 * tau * coeffs[term_index], pauli_word, site)
                    term_index += 1
    # in the reverse mode, if the Hamiltonian is a single element list, reverse the order its each term
    else:
        if len(grouped_hamiltonian) == 1:
            coeffs, pauli_words, sites = grouped_hamiltonian[0].decompose_with_sites()
            for term_index in reversed(range(len(coeffs))):
                pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
                add_n_pauli_gate(circuit, 2 * tau * coeffs[term_index], pauli_word, site)
        # otherwise, if it is a list of multiple Hamiltonian, only reverse the order of that list
        else:
            for hamiltonian in reversed(grouped_hamiltonian):
                assert isinstance(hamiltonian, Hamiltonian)
                coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
                for term_index in range(len(coeffs)):
                    pauli_word, site = __sort_pauli_word(pauli_words[term_index], sites[term_index])
                    add_n_pauli_gate(circuit, 2 * tau * coeffs[term_index], pauli_word, site)


def optimal_circuit(circuit: paddle_quantum.ansatz.Circuit, theta: Union[paddle.Tensor, float], which_qubits: Iterable):
    """Add an optimized circuit with the Hamiltonian 'XXYYZZ'.

    Args:
        circuit: Circuit where the gates are to be added.
        theta: Three rotation angles.
        which_qubits: List of the index of the qubit that each Pauli operator acts on.
    """
    p = paddle.to_tensor(np.pi / 2, dtype=float_dtype)
    x, y, z = theta
    alpha = paddle.to_tensor(3 * p - 4 * x * p + 2 * x, dtype=float_dtype)
    beta = paddle.to_tensor(-3 * p + 4 * y * p - 2 * y, dtype=float_dtype)
    gamma = paddle.to_tensor(2 * z - p, dtype=float_dtype)
    which_qubits.sort()
    a, b = which_qubits
    
    circuit.rz(b, param=p)
    circuit.cnot([b, a])
    circuit.rz(a, param=gamma)
    circuit.ry(b, param=alpha)
    circuit.cnot([a, b])
    circuit.ry(b, param=beta)
    circuit.cnot([b, a])
    circuit.rz(a, param=-p)


def __add_second_order_trotter_block(circuit, tau, grouped_hamiltonian):
    r"""Add a time evolution block of the second order Trotter-Suzuki decompositon

    Notes:
        This is an intrinsic function, user do not need to call this directly
    """
    __add_first_order_trotter_block(circuit, tau / 2, grouped_hamiltonian)
    __add_first_order_trotter_block(circuit, tau / 2, grouped_hamiltonian, reverse=True)


def __add_higher_order_trotter_block(circuit, tau, grouped_hamiltonian, order):
    r"""Add a time evolution block of the higher order (2k) Trotter-Suzuki decompositon 

    Notes:
        This is an intrinsic function, user do not need to call this directly
    """
    assert order % 2 == 0
    p_values = get_suzuki_p_values(order)
    if order - 2 != 2:
        for p in p_values:
            __add_higher_order_trotter_block(circuit, p * tau, grouped_hamiltonian, order - 2)
    else:
        for p in p_values:
            __add_second_order_trotter_block(circuit, p * tau, grouped_hamiltonian)


def add_n_pauli_gate(
        circuit: paddle_quantum.ansatz.Circuit, theta: Union[paddle.Tensor, float],
        pauli_word: str, which_qubits: Iterable
):
    r"""Add a rotation gate for a tensor product of Pauli operators, for example :math:`e^{-\theta/2 * X \otimes I \otimes X \otimes Y}`.

    Args:
        circuit: Circuit where the gates are to be added.
        theta: Rotation angle.
        pauli_word: Pauli operators in a string format, e.g., ``"XXZ"``.
        which_qubits: List of the index of the qubit that each Pauli operator in the ``pauli_word`` acts on.

    Raises:
        ValueError: The ``which_qubits`` should be either ``list``, ``tuple``, or ``np.ndarray``.
    """
    if isinstance(which_qubits, tuple) or isinstance(which_qubits, list):
        which_qubits = np.array(which_qubits)
    elif not isinstance(which_qubits, np.ndarray):
        raise ValueError('which_qubits should be either a list, tuple or np.ndarray')

    if not isinstance(theta, paddle.Tensor):
        theta = paddle.to_tensor(theta, dtype=float_dtype)
    # if it is a single-Pauli case, apply the single qubit rotation gate accordingly
    if len(which_qubits) == 1:
        if re.match(r'X', pauli_word[0], flags=re.I):
            circuit.rx(which_qubits[0], param=theta)
        elif re.match(r'Y', pauli_word[0], flags=re.I):
            circuit.ry(which_qubits[0], param=theta)
        elif re.match(r'Z', pauli_word[0], flags=re.I):
            circuit.rz(which_qubits[0], param=theta)

    # if it is a multiple-Pauli case, implement a Pauli tensor rotation
    # we use a scheme described in 4.7.3 of Nielson & Chuang, that is, basis change + tensor Z rotation
    # (tensor Z rotation is 2 layers of CNOT and a Rz rotation)
    else:
        which_qubits.sort()

        # Change the basis for qubits on which the acting operators are not 'Z'
        for qubit_index in range(len(which_qubits)):
            if re.match(r'X', pauli_word[qubit_index], flags=re.I):
                circuit.h([which_qubits[qubit_index]])
            elif re.match(r'Y', pauli_word[qubit_index], flags=re.I):
                circuit.rx(which_qubits[qubit_index], param=PI / 2)

        # Add a Z tensor n rotational gate
        for i in range(len(which_qubits) - 1):
            circuit.cnot([which_qubits[i], which_qubits[i + 1]])
        circuit.rz(which_qubits[-1], param=theta)
        for i in reversed(range(len(which_qubits) - 1)):
            circuit.cnot([which_qubits[i], which_qubits[i + 1]])

        # Change the basis for qubits on which the acting operators are not 'Z'
        for qubit_index in range(len(which_qubits)):
            if re.match(r'X', pauli_word[qubit_index], flags=re.I):
                circuit.h([which_qubits[qubit_index]])
            elif re.match(r'Y', pauli_word[qubit_index], flags=re.I):
                circuit.rx(which_qubits[qubit_index], param=-PI / 2)


def __group_hamiltonian_xyz(hamiltonian):
    r"""Decompose the Hamiltonian as X, Y, Z, and remainder term, return the list of them.

    Args:
        hamiltonian: Hamiltonian class in Paddle Quantum

    Notes:
        X, (Y, Z) means the terms whose Pauli word only include X, (Y, Z). For example, 'XXXY' would be a remainder term.
    """
    grouped_hamiltonian = []
    coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
    grouped_terms_indices = []
    left_over_terms_indices = []
    for pauli in ['X', 'Y', 'Z']:
        terms_indices_to_be_grouped = []
        for term_index in range(len(coeffs)):
            pauli_word = pauli_words[term_index]
            assert isinstance(pauli_word, str), "Each pauli word should be a string type"
            if pauli_word.count(pauli) == len(pauli_word) or pauli_word.count(pauli.lower()) == len(pauli_word):
                terms_indices_to_be_grouped.append(term_index)
        grouped_terms_indices.extend(terms_indices_to_be_grouped)
        grouped_hamiltonian.append(hamiltonian[terms_indices_to_be_grouped])

    for term_index in range(len(coeffs)):
        if term_index not in grouped_terms_indices:
            left_over_terms_indices.append(term_index)
    if len(left_over_terms_indices):
        for term_index in left_over_terms_indices:
            grouped_hamiltonian.append(hamiltonian[term_index])
    return grouped_hamiltonian


def __group_hamiltonian_even_odd(hamiltonian):
    r"""Decompose the Hamiltonian into odd and even parts.

    Args:
        hamiltonian (Hamiltonian): Hamiltonian class in Paddle Quantum

    Warning:
        Note this decomposition cannot make sure the mutual commutativity among odd terms or even terms. Use this method incorrectly would cause larger
        Trotter error.
        Please check whether Hamiltonian could be odd-even decomposed before you call this method. For example, 1-D Hamiltonian with nearest neighbor 
        interaction could be odd-even decomposed.
    """
    grouped_hamiltonian = []
    coeffs, pauli_words, sites = hamiltonian.decompose_with_sites()
    grouped_terms_indices = []
    left_over_terms_indices = []

    for offset in range(2):
        terms_indices_to_be_grouped = []
        for term_index in range(len(coeffs)):
            if not isinstance(sites[term_index], np.ndarray):
                site = np.array(sites[term_index])
            else:
                site = sites[term_index]
            site.sort()
            if site.min() % 2 == offset:
                terms_indices_to_be_grouped.append(term_index)
        grouped_terms_indices.extend(terms_indices_to_be_grouped)
        grouped_hamiltonian.append(hamiltonian[terms_indices_to_be_grouped])

    for term_index in range(len(coeffs)):
        if term_index not in grouped_terms_indices:
            left_over_terms_indices.append(term_index)

    if len(left_over_terms_indices):
        grouped_hamiltonian.append(hamiltonian[left_over_terms_indices])
    return grouped_hamiltonian


def get_suzuki_permutation(length: int, order: int) -> np.ndarray:
    r"""Calculate the permutation array corresponding to the Suzuki decomposition.

    Args:
        length: Number of terms in the Hamiltonian, i.e., how many terms to be permuted.
        order: Order of the Suzuki decomposition.

    Returns:
        Permutation array.
    """
    if order == 1:
        return np.arange(length)
    if order == 2:
        return np.vstack([np.arange(length), np.arange(length - 1, -1, -1)])
    else:
        return np.vstack([get_suzuki_permutation(length=length, order=order - 2) for _ in range(5)])


def get_suzuki_p_values(k: int) -> list:
    r"""Calculate the parameter p(k) in the Suzuki recurrence relationship.

    Args:
        k: Order of the Suzuki decomposition.

    Returns:
        A list of length five of form [p, p, (1 - 4 * p), p, p].
    """
    p = 1 / (4 - 4 ** (1 / (k - 1)))
    return [p, p, (1 - 4 * p), p, p]


def get_suzuki_coefficients(length: int, order: int) -> np.ndarray:
    r"""Calculate the coefficient array corresponding to the Suzuki decomposition.

    Args:
        length: Number of terms in the Hamiltonian, i.e., how many terms to be permuted.
        order: Order of the Suzuki decomposition.

    Returns:
        Coefficient array.
    """
    if order == 1:
        return np.ones(length)
    if order == 2:
        return np.vstack([1 / 2 * np.ones(length), 1 / 2 * np.ones(length)])
    else:
        p_values = get_suzuki_p_values(order)
        return np.vstack([get_suzuki_coefficients(length=length, order=order - 2) * p_value
                          for p_value in p_values])


def get_1d_heisenberg_hamiltonian(
        length: int,
        j_x: float = 1.,
        j_y: float = 1.,
        j_z: float = 1.,
        h_z: float or np.ndarray = 0.,
        periodic_boundary_condition: bool = True
) -> Hamiltonian:
    r"""Generate the Hamiltonian of a one-dimensional Heisenberg chain.

    Args:
        length: Chain length.
        j_x: Coupling strength Jx on the x direction. Defaults to ``1.``.
        j_y: Coupling strength Jy on the y direction. Defaults to ``1.``.
        j_z: Coupling strength Jz on the z direction. Defaults to ``1.``.
        h_z: Magnet field along z-axis. A uniform field will be added for single float input. Defaults to ``0.``.
        periodic_boundary_condition: Whether to consider the periodic boundary, i.e., l + 1 = 0. Defaults to ``True``.

    Returns:
        Hamiltonian of this Heisenberg chain.
    """
    # Pauli words for Heisenberg interactions and their coupling strength
    interactions = ['XX', 'YY', 'ZZ']
    interaction_strength = [j_x, j_y, j_z]
    pauli_str = []  # The Pauli string defining the Hamiltonian

    # add terms (0, 1), (1, 2), ..., (n - 1, n) by adding [j_x, 'X0, X1'], ... into the Pauli string
    for i in range(length - 1):
        for interaction_idx in range(len(interactions)):
            term_str = ''
            interaction = interactions[interaction_idx]
            for idx_word in range(len(interaction)):
                term_str += interaction[idx_word] + str(i + idx_word)
                if idx_word != len(interaction) - 1:
                    term_str += ', '
            pauli_str.append([interaction_strength[interaction_idx], term_str])

    # add interactions on (0, n) for closed periodic boundary condition
    if periodic_boundary_condition:
        boundary_sites = [0, length - 1]
        for interaction_idx in range(len(interactions)):
            term_str = ''
            interaction = interactions[interaction_idx]
            for idx_word in range(len(interaction)):
                term_str += interaction[idx_word] + str(boundary_sites[idx_word])
                if idx_word != len(interaction) - 1:
                    term_str += ', '
            pauli_str.append([interaction_strength[interaction_idx], term_str])

    # add magnetic field, if h_z is a single value, then add a uniform field on each site
    if isinstance(h_z, np.ndarray) or isinstance(h_z, list) or isinstance(h_z, tuple):
        assert len(h_z) == length, 'length of the h_z array do not match the length of the system'
        for i in range(length):
            pauli_str.append([h_z[i], 'Z' + str(i)])
    elif h_z:
        for i in range(length):
            pauli_str.append([h_z, 'Z' + str(i)])

    # instantiate a Hamiltonian object with the Pauli string
    h = Hamiltonian(pauli_str)
    return h
