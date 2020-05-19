# Copyright (c) 2020 Paddle Quantum Authors. All Rights Reserved.
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

"""
circuit
"""

from numpy import binary_repr, eye, identity

from paddle.complex import kron as pp_kron
from paddle.complex import matmul

from paddle.fluid import dygraph

from paddle_quantum.utils import rotation_x, rotation_y, rotation_z

__all__ = [
    "dic_between2and10",
    "base_2_change",
    "cnot_construct",
    "identity_generator",
    "single_gate_construct",
    "UAnsatz",
]


def dic_between2and10(n):
    """
    :param n: number of qubits
    :return: dictionary between binary and decimal

    for example: if n=3, the dictionary is
    dic2to10: {'000': 0, '011': 3, '010': 2, '111': 7, '100': 4, '101': 5, '110': 6, '001': 1}
    dic10to2: ['000', '001', '010', '011', '100', '101', '110', '111']
    """

    dic2to10 = {}
    dic10to2 = [None] * 2**n

    for i in range(2**n):
        binary_text = binary_repr(i, width=n)
        dic2to10[binary_text] = i
        dic10to2[i] = binary_text

    return dic2to10, dic10to2  # the returned dic will have 2 ** n value


def base_2_change(string_n, i_ctrl, j_target):
    """
    :param string_n: an n-bit string
    :param i_ctrl: i-th bit is control, 'int'
    :param j_target: j-th bit is target, 'int'
    :return: if i-th bit is 1, j-th bit takes inverse
    """

    string_n_list = list(string_n)
    if string_n_list[i_ctrl] == "1":
        string_n_list[j_target] = str((int(string_n_list[j_target]) + 1) % 2)
    return "".join(string_n_list)


def cnot_construct(n, ctrl):
    """
    cnot_construct: The controlled-NOT gate
    :param n: number of qubits
    :param ctrl: a shape [2] vector, in 'int' type. ctrl[0]-th qubit controls the ctrl[1]-qubit
    :return: cnot module
    """

    mat = eye(2**n)
    dummy_mat = eye(2**n)
    dic2to10, dic10to2 = dic_between2and10(n)
    """ for example: if n=3, the dictionary is
    dic2to10: {'000': 0, '011': 3, '010': 2, '111': 7, '100': 4, '101': 5, '110': 6, '001': 1}
    dic10to2: ['000', '001', '010', '011', '100', '101', '110', '111']
    """

    for row in range(2**n):
        """ for each decimal index 'row', transform it into binary,
            and use 'base_2_change()' function to compute the new binary index 'row'.
            Lastly, use 'dic2to10' to transform this new 'row' into decimal.

            For instance, n=3, ctrl=[1,3]. if row = 5,
            its process is 5 -> '101' -> '100' -> 4
        """
        new_string_base_2 = base_2_change(dic10to2[row], ctrl[0] - 1,
                                          ctrl[1] - 1)
        new_int_base_10 = dic2to10[new_string_base_2]
        mat[row] = dummy_mat[new_int_base_10]

    return mat.astype("complex64")


def identity_generator(n):
    """
    identity_generator
    """

    idty_np = identity(2**n, dtype="float32")
    idty = dygraph.to_variable(idty_np)

    return idty


def single_gate_construct(mat, n, which_qubit):
    """
    :param mat: the input matrix
    :param n: number of qubits
    :param which_qubit: indicate which qubit the matrix is placed on
    :return: the kronecker product of the matrix and identity
    """

    idty = identity_generator(n - 1)

    if which_qubit == 1:
        mat = pp_kron(mat, idty)

    elif which_qubit == n:
        mat = pp_kron(idty, mat)

    else:
        I_top = identity_generator(which_qubit - 1)
        I_bot = identity_generator(n - which_qubit)
        mat = pp_kron(pp_kron(I_top, mat), I_bot)

    return mat


class UAnsatz:
    """
    UAnsatz: ansatz for the parameterized quantum circuit or quantum neural network.
    """

    def __init__(self, n, input_state=None):
        """
        :param input_state: if the input_state is 'None', the self.mat is a matrix,
                            if the input_state is a row complex vector, the self.mat is a row vector
        """

        self.n = n
        self.state = input_state if input_state is not None else identity_generator(
            self.n)

    def rx(self, theta, which_qubit):
        """
        Rx: the single qubit X rotation
        """

        transform = single_gate_construct(
            rotation_x(theta), self.n, which_qubit)
        self.state = matmul(self.state, transform)

    def ry(self, theta, which_qubit):
        """
        Ry: the single qubit Y rotation
        """

        transform = single_gate_construct(
            rotation_y(theta), self.n, which_qubit)
        self.state = matmul(self.state, transform)

    def rz(self, theta, which_qubit):
        """
        Rz: the single qubit Z rotation
        """

        transform = single_gate_construct(
            rotation_z(theta), self.n, which_qubit)
        self.state = matmul(self.state, transform)

    def cnot(self, control):
        """
        :param control: [1,3], the 1st qubit controls 3rd qubit
        :return: cnot module
        """

        cnot = dygraph.to_variable(cnot_construct(self.n, control))
        self.state = matmul(self.state, cnot)
