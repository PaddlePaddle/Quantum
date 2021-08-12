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

import math
from functools import wraps
import numpy as np
from numpy import binary_repr
import re
import paddle
from paddle import multiply, add, to_tensor, matmul, real, trace
from paddle_quantum.simulator import StateTransfer
from paddle_quantum import utils


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
    op_list = re.split(r',\s*', H.lower())
    coef = 1 + 0*1j  # Coefficient for the vector
    new_vec = list(target_vec)
    for op in op_list:
        if len(op) >= 2:
            pos = int(op[1:])
        elif op[0] != 'i':
            raise ValueError('only operator "I" can be used without identifying its position')
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
    return to_tensor(new_vec)


def H_vec(H, vec):
    r"""
    If H = [[0.2, 'x0,z1'], [0.6, 'x0'], [0.1, 'z1'], [-0.7, 'y0,y1']], then it returns H * vec

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    coefs = to_tensor(np.array([coef for coef, Hi in H], dtype=np.float64))
    # Convert all strings to lowercase
    H_list = [Hi.lower() for coef, Hi in H]
    result = paddle.zeros(shape=vec.shape, dtype='float64')
    for i in range(len(coefs)):
        xi = multiply(coefs[i], single_H_vec(H_list[i], vec))
        result = add(result, xi)
    return result


def vec_expecval(H, vec):
    r"""
    It returns expectation value of H with respect to vector vec

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    vec_conj = paddle.conj(vec)
    result = paddle.sum(multiply(vec_conj, H_vec(H, vec)))
    return result


def transfer_by_history(state, history, params):
    r"""
    It transforms the input state according to the history given.

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    for history_ele in history:
        if history_ele['gate'] != 'channel':
            which_qubit = history_ele['which_qubits']
            parameter =  [params[i] for i in history_ele['theta']] if history_ele['theta'] else None
            
            if history_ele['gate'] in {'s', 't', 'ry', 'rz', 'rx', 'sdg', "tdg"}:
                state = StateTransfer(state, 'u', which_qubit, params=parameter)
            elif history_ele['gate'] == 'MS_gate':
                state = StateTransfer(state, 'RXX_gate', which_qubit, params=parameter)
            elif history_ele['gate'] in {'crx', 'cry', 'crz'}:
                state = StateTransfer(state, 'CU', which_qubit, params=parameter)
            else:
                state = StateTransfer(state, history_ele['gate'], which_qubit, params=parameter)
    return state


def apply_channel(func):
    r"""
    Decorator for channels.

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    @wraps(func)
    def inner(self, *args):
        """
        args should include channel parameters and which_qubit
        """
        which_qubit = args[-1]
        assert 0 <= which_qubit < self.n, "the qubit's index should >= 0 and < n(the number of qubit)"
        self._UAnsatz__has_channel = True
        ops = func(self, *args)
        self._UAnsatz__history.append({'gate': 'channel', 'operators': ops, 'which_qubits': [which_qubit]})

    return inner