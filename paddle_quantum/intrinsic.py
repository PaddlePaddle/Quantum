# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

import math
import numpy as np
from numpy import binary_repr

import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import ComplexVariable
from paddle.complex.tensor.math import elementwise_mul, elementwise_add


def dic_between2and10(n):
    r"""
    :param n: number of qubits
    :return: dictionary between binary and decimal

    for example: if n=3, the dictionary is
    dic2to10: {'000': 0, '011': 3, '010': 2, '111': 7, '100': 4, '101': 5, '110': 6, '001': 1}
    dic10to2: ['000', '001', '010', '011', '100', '101', '110', '111']

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    dic2to10 = {}
    dic10to2 = [None] * 2 ** n

    for i in range(2 ** n):
        binary_text = binary_repr(i, width=n)
        dic2to10[binary_text] = i
        dic10to2[i] = binary_text

    return dic2to10, dic10to2  # the returned dic will have 2 ** n value


def single_H_vec_i(H, target_vec):
    r"""
    If H = 'x0,z1,z2,y3,z5', target_vec = '100111', then it returns H * target_vec, which we record as [1j, '000011']

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    op_list = H.split(',')
    coef = 1 + 0*1j  # Coefficient for the vector
    new_vec = list(target_vec)
    for op in op_list:
        pos = int(op[1:])
        if op[0] == 'x':
            new_vec[pos] = '0' if target_vec[pos] == '1' else '1'
        elif op[0] == 'y':
            new_vec[pos] = '0' if target_vec[pos] == '1' else '1'
            coef *= 1j if target_vec[pos] == '0' else -1j
        elif op[0] == 'z':
            new_vec[pos] = target_vec[pos]
            coef *= 1 if target_vec[pos] == '0' else -1

    return [coef, ''.join(new_vec)]


def single_H_vec(H, vec):
    r"""
    If vec is a paddle variable [a1, a2, a3, a4], and H = 'x0,z1', then it returns H * vec,
    which is [a3, -a4, a1, -a2]

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    old_vec = vec.numpy()
    new_vec = np.zeros(len(old_vec)) + 0j
    dic2to10, dic10to2 = dic_between2and10(int(math.log(len(old_vec), 2)))
    # Iterate through all vectors in the computational basis
    for i in range(len(old_vec)):
        # If old_vec[i] is 0, the result is 0
        if old_vec[i] != 0:
            coef, target_update = single_H_vec_i(H, dic10to2[i])
            index_update = dic2to10[target_update]
            new_vec[index_update] = coef * old_vec[i]
    return fluid.dygraph.to_variable(new_vec)


def H_vec(H, vec):
    r"""
    If H = [[0.2, 'x0,z1'], [0.6, 'x0'], [0.1, 'z1'], [-0.7, 'y0,y1']], then it returns H * vec

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    coefs = fluid.dygraph.to_variable(np.array([coef for coef, Hi in H], dtype=np.float64))
    # Convert all strings to lowercase
    H_list = [Hi.lower() for coef, Hi in H]
    result = fluid.layers.zeros(shape=vec.shape, dtype='float64')
    for i in range(len(coefs)):
        xi = elementwise_mul(coefs[i], single_H_vec(H_list[i], vec))
        result = elementwise_add(result, xi)
    return result


def vec_expecval(H, vec):
    r"""
    It returns expectation value of H with respect to vector vec

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    vec_conj = ComplexVariable(vec.real, -vec.imag)
    result = paddle.complex.sum(elementwise_mul(vec_conj, H_vec(H, vec)))
    return result
