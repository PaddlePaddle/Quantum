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

import warnings
import copy
import math
import re
import matplotlib.pyplot as plt
from functools import reduce
from collections import defaultdict
import numpy as np
import paddle
from paddle_quantum.simulator import transfer_state, init_state_gen, measure_state
from paddle import imag, real, reshape, kron, matmul, trace
from paddle_quantum.utils import partial_trace, dagger, pauli_str_to_matrix
from paddle_quantum import shadow
from paddle_quantum.intrinsic import *
from paddle_quantum.state import density_op

__all__ = [
    "UAnsatz",
    "swap_test"
]


class UAnsatz:
    r"""基于 PaddlePaddle 的动态图机制实现量子电路的 ``class`` 。

    用户可以通过实例化该 ``class`` 来搭建自己的量子电路。

    Attributes:
        n (int): 该电路的量子比特数
    """

    def __init__(self, n):
        r"""UAnsatz 的构造函数，用于实例化一个 UAnsatz 对象

        Args:
            n (int): 该电路的量子比特数
        """
        self.n = n
        self.__has_channel = False
        self.__state = None
        self.__run_mode = ''
        # Record parameters in the circuit
        self.__param = [paddle.to_tensor(np.array([0.0])),
                        paddle.to_tensor(np.array([math.pi / 2])), paddle.to_tensor(np.array([-math.pi / 2])),
                        paddle.to_tensor(np.array([math.pi / 4])), paddle.to_tensor(np.array([-math.pi / 4]))]
        # Record history of adding gates to the circuit
        self.__history = []

    def __add__(self, cir):
        r"""重载加法 ‘+’ 运算符，用于拼接两个维度相同的电路

        Args:
            cir (UAnsatz): 拼接到现有电路上的电路
        
        Returns:
            UAnsatz: 拼接后的新电路
        
        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz

            print('cir1: ')
            cir1 = UAnsatz(2)
            cir1.superposition_layer()
            print(cir1)

            print('cir2: ')
            cir2 = UAnsatz(2)
            cir2.cnot([0,1])
            print(cir2)

            print('cir3: ')
            cir3 = cir1 + cir2
            print(cir3)
        ::

            cir1: 
            --H--
                
            --H--
                
            cir2: 
            --*--
              |  
            --x--
                
            cir3: 
            --H----*--
                   |  
            --H----x--

        """
        assert self.n == cir.n, "two circuits does not have the same dimension"

        # Construct a new circuit that adds the two together
        cir_out = UAnsatz(self.n)
        cir_out.__param = copy.copy(self.__param)
        cir_out.__history = copy.copy(self.__history)
        cir_out._add_history(cir.__history, cir.__param)

        return cir_out

    def _get_history(self):
        r"""获取当前电路加门的历史

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        return self.__history, self.__param

    def _add_history(self, histories, param):
        r"""往当前 UAnsatz 里直接添加历史

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        if type(histories) is dict:
            histories = [histories]

        for history_ele in histories:
            param_idx = history_ele['theta']
            if param_idx is None:
                self.__history.append(copy.copy(history_ele))
            else:
                new_param_idx = []
                curr_idx = len(self.__param)
                for idx in param_idx:
                    self.__param.append(param[idx])
                    new_param_idx.append(curr_idx)
                    curr_idx += 1
                self.__history.append({'gate': history_ele['gate'],
                                       'which_qubits': history_ele['which_qubits'],
                                       'theta': new_param_idx})

    def get_run_mode(self):
        r"""获取当前电路的运行模式。

        Returns:
            string: 当前电路的运行模式，态矢量或者是密度矩阵

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            import numpy as np

            cir = UAnsatz(5)
            cir.superposition_layer()
            cir.run_state_vector()

            print(cir.get_run_mode())

        ::

            state_vector
        """
        return self.__run_mode

    def get_state(self):
        r"""获取当前电路运行后的态

        Returns:
            paddle.Tensor: 当前电路运行后的态

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            import numpy as np

            cir = UAnsatz(5)
            cir.superposition_layer()
            cir.run_state_vector()

            print(cir.get_state())

        ::

            Tensor(shape=[4], dtype=complex128, place=CPUPlace, stop_gradient=True,
                   [(0.4999999999999999+0j), (0.4999999999999999+0j), (0.4999999999999999+0j), (0.4999999999999999+0j)])
        """
        return self.__state

    def _count_history(self):
        r"""calculate how many blocks needed for printing

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        # Record length of each section
        length = [5]
        n = self.n
        # Record current section number for every qubit
        qubit = [0] * n
        # Number of sections
        qubit_max = max(qubit)
        # Record section number for each gate
        gate = []
        history = self.__history

        for current_gate in history:
            # Single-qubit gates with no params to print
            if current_gate['gate'] in {'h', 's', 't', 'x', 'y', 'z', 'u', 'sdg', 'tdg'}:
                curr_qubit = current_gate['which_qubits'][0]
                gate.append(qubit[curr_qubit])
                qubit[curr_qubit] = qubit[curr_qubit] + 1
                # A new section is added
                if qubit[curr_qubit] > qubit_max:
                    length.append(5)
                    qubit_max = qubit[curr_qubit]
            # Gates with params to print
            elif current_gate['gate'] in {'rx', 'ry', 'rz'}:
                curr_qubit = current_gate['which_qubits'][0]
                gate.append(qubit[curr_qubit])
                if length[qubit[curr_qubit]] == 5:
                    length[qubit[curr_qubit]] = 13
                qubit[curr_qubit] = qubit[curr_qubit] + 1
                if qubit[curr_qubit] > qubit_max:
                    length.append(5)
                    qubit_max = qubit[curr_qubit]
            # Two-qubit gates or Three-qubit gates
            elif current_gate['gate'] in {'CNOT', 'SWAP', 'RXX_gate', 'RYY_gate', 'RZZ_gate', 'MS_gate', 'cy', 'cz',
                                          'CU', 'crx', 'cry', 'crz'} or current_gate['gate'] in {'CSWAP', 'CCX'}:
                a = max(current_gate['which_qubits'])
                b = min(current_gate['which_qubits'])
                ind = max(qubit[b: a + 1])
                gate.append(ind)
                if length[ind] < 13 and current_gate['gate'] in {'RXX_gate', 'RYY_gate', 'RZZ_gate', 'crx', 'cry',
                                                                 'crz'}:
                    length[ind] = 13
                for j in range(b, a + 1):
                    qubit[j] = ind + 1
                if ind + 1 > qubit_max:
                    length.append(5)
                    qubit_max = ind + 1

        return length, gate

    def __str__(self):
        r"""实现画电路的功能

        Returns:
            string: 用来print的字符串

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            import numpy as np

            cir = UAnsatz(5)
            cir.superposition_layer()
            rotations = paddle.to_tensor(np.random.uniform(-2, 2, size=(3, 5, 1)))
            cir.real_entangled_layer(rotations, 3)

            print(cir)
        ::

            The printed circuit is:

            --H----Ry(-0.14)----*-------------------X----Ry(-0.77)----*-------------------X--
                                |                   |                 |                   |  
            --H----Ry(-1.00)----X----*--------------|----Ry(-0.83)----X----*--------------|--
                                     |              |                      |              |  
            --H----Ry(-1.88)---------X----*---------|----Ry(-0.98)---------X----*---------|--
                                          |         |                           |         |  
            --H----Ry(1.024)--------------X----*----|----Ry(-0.37)--------------X----*----|--
                                               |    |                                |    |  
            --H----Ry(1.905)-------------------X----*----Ry(-1.82)-------------------X----*--
        """
        length, gate = self._count_history()
        history = self.__history
        n = self.n
        # Ignore the unused section
        total_length = sum(length) - 5

        print_list = [['-' if i % 2 == 0 else ' '] * total_length for i in range(n * 2)]

        for i, current_gate in enumerate(history):
            if current_gate['gate'] in {'h', 's', 't', 'x', 'y', 'z', 'u'}:
                # Calculate starting position ind of current gate
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'][0] * 2][ind + length[sec] // 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'sdg'}:
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'][0] * 2][
                    ind + length[sec] // 2 - 1: ind + length[sec] // 2 + 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'tdg'}:
                sec = gate[i]
                ind = sum(length[:sec])
                print_list[current_gate['which_qubits'][0] * 2][
                    ind + length[sec] // 2 - 1: ind + length[sec] // 2 + 2] = current_gate['gate'].upper()
            elif current_gate['gate'] in {'rx', 'ry', 'rz'}:
                sec = gate[i]
                ind = sum(length[:sec])
                line = current_gate['which_qubits'][0] * 2
                param = self.__param[current_gate['theta'][2 if current_gate['gate'] == 'rz' else 0]]
                print_list[line][ind + 2] = 'R'
                print_list[line][ind + 3] = current_gate['gate'][1]
                print_list[line][ind + 4] = '('
                print_list[line][ind + 5: ind + 10] = format(float(param.numpy()), '.3f')[:5]
                print_list[line][ind + 10] = ')'
            # Two-qubit gates
            elif current_gate['gate'] in {'CNOT', 'SWAP', 'RXX_gate', 'RYY_gate', 'RZZ_gate', 'MS_gate', 'cz', 'cy',
                                          'CU', 'crx', 'cry', 'crz'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit = current_gate['which_qubits'][0]
                tqubit = current_gate['which_qubits'][1]
                if current_gate['gate'] in {'CNOT', 'SWAP', 'cy', 'cz', 'CU'}:
                    print_list[cqubit * 2][ind + length[sec] // 2] = \
                        '*' if current_gate['gate'] in {'CNOT', 'cy', 'cz', 'CU'} else 'x'
                    print_list[tqubit * 2][ind + length[sec] // 2] = \
                        'x' if current_gate['gate'] in {'SWAP', 'CNOT'} else current_gate['gate'][1]
                elif current_gate['gate'] == 'MS_gate':
                    for qubit in {cqubit, tqubit}:
                        print_list[qubit * 2][ind + length[sec] // 2 - 1] = 'M'
                        print_list[qubit * 2][ind + length[sec] // 2] = '_'
                        print_list[qubit * 2][ind + length[sec] // 2 + 1] = 'S'
                elif current_gate['gate'] in {'RXX_gate', 'RYY_gate', 'RZZ_gate'}:
                    param = self.__param[current_gate['theta'][0]]
                    for line in {cqubit * 2, tqubit * 2}:
                        print_list[line][ind + 2] = 'R'
                        print_list[line][ind + 3: ind + 5] = current_gate['gate'][1:3].lower()
                        print_list[line][ind + 5] = '('
                        print_list[line][ind + 6: ind + 10] = format(float(param.numpy()), '.2f')[:4]
                        print_list[line][ind + 10] = ')'
                elif current_gate['gate'] in {'crx', 'cry', 'crz'}:
                    param = self.__param[current_gate['theta'][2 if current_gate['gate'] == 'crz' else 0]]
                    print_list[cqubit * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit * 2][ind + 2] = 'R'
                    print_list[tqubit * 2][ind + 3] = current_gate['gate'][2]
                    print_list[tqubit * 2][ind + 4] = '('
                    print_list[tqubit * 2][ind + 5: ind + 10] = format(float(param.numpy()), '.3f')[:5]
                    print_list[tqubit * 2][ind + 10] = ')'
                start_line = min(cqubit, tqubit)
                end_line = max(cqubit, tqubit)
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
            # Three-qubit gates
            elif current_gate['gate'] in {'CSWAP'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit = current_gate['which_qubits'][0]
                tqubit1 = current_gate['which_qubits'][1]
                tqubit2 = current_gate['which_qubits'][2]
                start_line = min(current_gate['which_qubits'])
                end_line = max(current_gate['which_qubits'])
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
                if current_gate['gate'] in {'CSWAP'}:
                    print_list[cqubit * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit1 * 2][ind + length[sec] // 2] = 'x'
                    print_list[tqubit2 * 2][ind + length[sec] // 2] = 'x'
            elif current_gate['gate'] in {'CCX'}:
                sec = gate[i]
                ind = sum(length[:sec])
                cqubit1 = current_gate['which_qubits'][0]
                cqubit2 = current_gate['which_qubits'][1]
                tqubit = current_gate['which_qubits'][2]
                start_line = min(current_gate['which_qubits'])
                end_line = max(current_gate['which_qubits'])
                for k in range(start_line * 2 + 1, end_line * 2):
                    print_list[k][ind + length[sec] // 2] = '|'
                if current_gate['gate'] in {'CCX'}:
                    print_list[cqubit1 * 2][ind + length[sec] // 2] = '*'
                    print_list[cqubit2 * 2][ind + length[sec] // 2] = '*'
                    print_list[tqubit * 2][ind + length[sec] // 2] = 'X'

        print_list = list(map(''.join, print_list))
        return_str = '\n'.join(print_list)

        return return_str

    def run_state_vector(self, input_state=None, store_state=True):
        r"""运行当前的量子电路，输入输出的形式为态矢量。

        Warning:
            该方法只能运行无噪声的电路。

        Args:
            input_state (Tensor, optional): 输入的态矢量，默认为 :math:`|00...0\rangle`
            store_state (Bool, optional): 是否存储输出的态矢量，默认为 ``True`` ，即存储

        Returns:
            Tensor: 量子电路输出的态矢量

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.state import vec
            n = 2
            theta = np.ones(3)

            input_state = paddle.to_tensor(vec(0, n))
            theta = paddle.to_tensor(theta)
            cir = UAnsatz(n)
            cir.h(0)
            cir.ry(theta[0], 1)
            cir.rz(theta[1], 1)
            output_state = cir.run_state_vector(input_state).numpy()
            print(f"The output state vector is {output_state}")

        ::

            The output state vector is [[0.62054458+0.j 0.18316521+0.28526291j 0.62054458+0.j 0.18316521+0.28526291j]]
        """
        # Throw a warning when cir has channel
        if self.__has_channel:
            warnings.warn('The noiseless circuit will be run.', RuntimeWarning)
        state = init_state_gen(self.n, 0) if input_state is None else input_state
        old_shape = state.shape
        assert reduce(lambda x, y: x * y, old_shape) == 2 ** self.n, \
            'The length of the input vector is not right'
        state = reshape(state, (2 ** self.n,))

        state_conj = paddle.conj(state)
        assert paddle.abs(real(paddle.sum(paddle.multiply(state_conj, state))) - 1) < 1e-8, \
            'Input state is not a normalized vector'

        state = transfer_by_history(state, self.__history, self.__param)

        if store_state:
            self.__state = state
            # Add info about which function user called
            self.__run_mode = 'state_vector'

        return reshape(state, old_shape)

    def run_density_matrix(self, input_state=None, store_state=True):
        r"""运行当前的量子电路，输入输出的形式为密度矩阵。

        Args:
            input_state (Tensor, optional): 输入的密度矩阵，默认为 :math:`|00...0\rangle \langle00...0|`
            store_state (bool, optional): 是否存储输出的密度矩阵，默认为 ``True`` ，即存储

        Returns:
            Tensor: 量子电路输出的密度矩阵

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.state import density_op
            n = 1
            theta = np.ones(3)

            input_state = paddle.to_tensor(density_op(n))
            theta = paddle.to_tensor(theta)
            cir = UAnsatz(n)
            cir.rx(theta[0], 0)
            cir.ry(theta[1], 0)
            cir.rz(theta[2], 0)
            density_matrix = cir.run_density_matrix(input_state).numpy()
            print(f"The output density matrix is\n{density_matrix}")

        ::

            The output density matrix is
            [[0.64596329+0.j         0.47686058+0.03603751j]
            [0.47686058-0.03603751j 0.35403671+0.j        ]]
        """
        state = paddle.to_tensor(density_op(self.n)) if input_state is None else input_state
        assert state.shape == [2 ** self.n, 2 ** self.n], \
            "The dimension is not right"

        if not self.__has_channel:
            state = matmul(self.U, matmul(state, dagger(self.U)))
        else:
            dim = 2 ** self.n
            shape = (dim, dim)
            num_ele = dim ** 2
            identity = paddle.eye(dim, dtype='float64')
            identity = paddle.cast(identity, 'complex128')
            identity = reshape(identity, [num_ele])

            u_start = 0
            i = 0
            for i, history_ele in enumerate(self.__history):
                if history_ele['gate'] == 'channel':
                    # Combine preceding unitary operations
                    unitary = transfer_by_history(identity, self.__history[u_start:i], self.__param)
                    sub_state = paddle.zeros(shape, dtype='complex128')
                    # Sum all the terms corresponding to different Kraus operators
                    for op in history_ele['operators']:
                        pseudo_u = reshape(transfer_state(unitary, op, history_ele['which_qubits']), shape)
                        sub_state += matmul(pseudo_u, matmul(state, dagger(pseudo_u)))
                    state = sub_state
                    u_start = i + 1
            # Apply unitary operations left
            unitary = reshape(transfer_by_history(identity, self.__history[u_start:(i + 1)], self.__param), shape)
            state = matmul(unitary, matmul(state, dagger(unitary)))

        if store_state:
            self.__state = state
            # Add info about which function user called
            self.__run_mode = 'density_matrix'

        return state

    def reset_state(self, state, which_qubits):
        r"""对当前电路中的量子态的部分量子比特进行重置。

        Args:
            state (paddle.Tensor): 输入的量子态，表示要把选定的量子比特重置为该量子态
            which_qubits (list): 需要被重置的量子比特编号
        """
        qubits_list = which_qubits
        n = self.n
        m = len(qubits_list)
        assert max(qubits_list) <= n, "qubit index out of range"

        origin_seq = list(range(0, n))
        target_seq = [idx for idx in origin_seq if idx not in qubits_list]
        target_seq = qubits_list + target_seq

        swapped = [False] * n
        swap_list = list()
        for idx in range(0, n):
            if not swapped[idx]:
                next_idx = idx
                swapped[next_idx] = True
                while not swapped[target_seq[next_idx]]:
                    swapped[target_seq[next_idx]] = True
                    swap_list.append((next_idx, target_seq[next_idx]))
                    next_idx = target_seq[next_idx]

        cir0 = UAnsatz(n)
        for a, b in swap_list:
            cir0.swap([a, b])

        cir1 = UAnsatz(n)
        swap_list.reverse()
        for a, b in swap_list:
            cir1.swap([a, b])

        _state = self.__state

        if self.__run_mode == 'state_vector':
            raise NotImplementedError('This feature is not implemented yet.')
        elif self.__run_mode == 'density_matrix':
            _state = cir0.run_density_matrix(_state)
            _state = partial_trace(_state, 2 ** m, 2 ** (n - m), 1)
            _state = kron(state, _state)
            _state = cir1.run_density_matrix(_state)
        else:
            raise ValueError("Can't recognize the mode of quantum state.")
        self.__state = _state

    @property
    def U(self):
        r"""量子电路的酉矩阵形式。

        Warning:
            该属性只限于无噪声的电路。

        Returns:
            Tensor: 当前电路的酉矩阵表示

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 2
            cir = UAnsatz(2)
            cir.h(0)
            cir.cnot([0, 1])
            unitary_matrix = cir.U
            print("The unitary matrix of the circuit for Bell state preparation is\n", unitary_matrix.numpy())

        ::

            The unitary matrix of the circuit for Bell state preparation is
            [[ 0.70710678+0.j  0.        +0.j  0.70710678+0.j  0.        +0.j]
            [ 0.        +0.j  0.70710678+0.j  0.        +0.j  0.70710678+0.j]
            [ 0.        +0.j  0.70710678+0.j  0.        +0.j -0.70710678+0.j]
            [ 0.70710678+0.j  0.        +0.j -0.70710678+0.j  0.        +0.j]]
        """
        # Throw a warning when cir has channel
        if self.__has_channel:
            warnings.warn('The unitary matrix of the noiseless circuit will be given.', RuntimeWarning)
        dim = 2 ** self.n
        shape = (dim, dim)
        num_ele = dim ** 2
        state = paddle.eye(dim, dtype='float64')
        state = paddle.cast(state, 'complex128')
        state = reshape(state, [num_ele])
        state = transfer_by_history(state, self.__history, self.__param)

        return reshape(state, shape)

    def __input_which_qubits_check(self, which_qubits):
        r"""实现3个功能：

        1. 检查 which_qubits 长度有无超过 qubits 的个数, (应小于等于qubits)
        2. 检查 which_qubits 有无重复的值
        3. 检查 which_qubits 的每个值有无超过量子 qubits 的序号, (应小于qubits,从 0 开始编号)

        Args:
            which_qubits (list) : 用于编码的量子比特

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        which_qubits_len = len(which_qubits)
        set_list = set(which_qubits)
        assert which_qubits_len <= self.n, \
            "the length of which_qubit_list should less than the number of qubits"
        assert which_qubits_len == len(set_list), \
            "the which_qubits can not have duplicate elements"
        for qubit_idx in which_qubits:
            assert qubit_idx < self.n, \
                "the value of which_qubit_list should less than the number of qubits"

    def basis_encoding(self, x, which_qubits=None, invert=False):
        r"""将输入的经典数据使用基态编码的方式编码成量子态。

        在 basis encoding 中，输入的经典数据只能包括 0 或 1。如输入数据为 1101，则编码后的量子态为 :math:`|1101\rangle` 。
        这里假设量子态在编码前为全 0 的态，即 :math:`|00\ldots 0\rangle` 。

        Args:
            x (Tensor): 待编码的向量
            which_qubits (list): 用于编码的量子比特
            invert (bool): 添加的是否为编码电路的逆电路，默认为 ``False`` ，即添加正常的编码电路
        """
        x = paddle.flatten(x)
        x = paddle.cast(x, dtype="int32")
        assert x.size <= self.n, \
            "the number of classical data should less than or equal to the number of qubits"
        if which_qubits is None:
            which_qubits = list(range(self.n))
        else:
            self.__input_which_qubits_check(which_qubits)
            assert x.size <= len(which_qubits), \
                "the number of classical data should less than or equal to the number of 'which_qubits'"

        for idx, element in enumerate(x):
            if element:
                self.x(which_qubits[idx])

    def amplitude_encoding(self, x, mode, which_qubits=None):
        r"""将输入的经典数据使用振幅编码的方式编码成量子态。

        Args:
            x (Tensor): 待编码的向量
            which_qubits (list): 用于编码的量子比特
            mode (str): 生成的量子态的表示方式，``"state_vector"`` 代表态矢量表示， ``"density_matrix"`` 代表密度矩阵表示

        Returns:
            Tensor: 一个形状为 ``(2 ** n, )`` 或 ``(2 ** n, 2 ** n)`` 的张量，表示编码之后的量子态。

        """
        assert x.size <= 2 ** self.n, \
            "the number of classical data should less than or equal to the number of qubits"

        if which_qubits is None:
            which_qubits_len = math.ceil(math.log2(x.size))
            which_qubits = list(range(which_qubits_len))
        else:
            self.__input_which_qubits_check(which_qubits)
            which_qubits_len = len(which_qubits)
        assert x.size <= 2 ** which_qubits_len, \
            "the number of classical data should <= 2^(which_qubits)"
        assert x.size > 2 ** (which_qubits_len - 1), \
            "the number of classical data should >= 2^(which_qubits-1)"

        def calc_location(location_of_bits_list):
            r"""递归计算需要参与编码的量子态展开后的序号
            方式：全排列，递归计算

            Args:
                location_of_bits_list (list): 标识了指定 qubits 的序号值，如指定编码第3个qubit(序号2)，
                    则它处在展开后的 2**(3-1)=4 位置上。

            Returns:
                list : 标识了将要参与编码的量子位展开后的序号
            """
            if len(location_of_bits_list) <= 1:
                result_list = [0, location_of_bits_list[0]]
            else:
                current_tmp = location_of_bits_list[0]
                inner_location_of_qubits_list = calc_location(location_of_bits_list[1:])
                current_list_len = len(inner_location_of_qubits_list)
                for each in range(current_list_len):
                    inner_location_of_qubits_list.append(inner_location_of_qubits_list[each] + current_tmp)
                result_list = inner_location_of_qubits_list

            return result_list

        def encoding_location_list(which_qubits):
            r"""计算每一个经典数据将要编码到量子态展开后的哪一个位置

            Args:
                which_qubits (list): 标识了参与编码的量子 qubits 的序号, 此参数与外部 which_qubits 参数应保持一致

            Returns:
                (list) : 将要参与编码的量子 qubits 展开后的序号，即位置序号
            """
            location_of_bits_list = []
            for each in range(len(which_qubits)):
                tmp = 2 ** (self.n - which_qubits[each] - 1)
                location_of_bits_list.append(tmp)
            result_list = calc_location(location_of_bits_list)

            return sorted(result_list)

        # Get the specific position of the code, denoted by sequence number (list)
        location_of_qubits_list = encoding_location_list(which_qubits)
        # Classical data preprocessing
        x = paddle.flatten(x)
        length = paddle.norm(x, p=2)
        # Normalization
        x = paddle.divide(x, length)
        # Create a quantum state with all zero amplitudes
        zero_tensor = paddle.zeros((2 ** self.n,), x.dtype)
        # The value of the encoded amplitude is filled into the specified qubits
        for i in range(len(x)):
            zero_tensor[location_of_qubits_list[i]] = x[i]
        # The quantum state that stores the result
        result_tensor = zero_tensor
        if mode == "state_vector":
            result_tensor = paddle.cast(result_tensor, dtype="complex128")
        elif mode == "density_matrix":
            result_tensor = paddle.reshape(result_tensor, (2 ** self.n, 1))
            result_tensor = matmul(result_tensor, dagger(result_tensor))
        else:
            raise ValueError("the mode should be state_vector or density_matrix")

        return result_tensor

    def angle_encoding(self, x, encoding_gate, which_qubits=None, invert=False):
        r"""将输入的经典数据使用角度编码的方式进行编码。

        Args:
            x (Tensor): 待编码的向量
            encoding_gate (str): 编码要用的量子门，可以是 ``"rx"`` 、 ``"ry"`` 和 ``"rz"``
            which_qubits (list): 用于编码的量子比特
            invert (bool): 添加的是否为编码电路的逆电路，默认为 ``False`` ，即添加正常的编码电路
        """
        assert x.size <= self.n, \
            "the number of classical data should be equal to the number of qubits"
        if which_qubits is None:
            which_qubits = list(range(self.n))
        else:
            self.__input_which_qubits_check(which_qubits)
            assert x.size <= len(which_qubits), \
                "the number of classical data should less than or equal to the number of 'which_qubits'"

        x = paddle.flatten(x)
        if invert:
            x = -x

        def add_encoding_gate(theta, which, gate):
            if gate == "rx":
                self.rx(theta, which)
            elif gate == "ry":
                self.ry(theta, which)
            elif gate == "rz":
                self.rz(theta, which)
            else:
                raise ValueError("the encoding_gate should be rx, ry, or rz")

        for idx, element in enumerate(x):
            add_encoding_gate(element[0], which_qubits[idx], encoding_gate)

    def iqp_encoding(self, x, num_repeats=1, pattern=None, invert=False):
        r"""将输入的经典数据使用 IQP 编码的方式进行编码。

        Args:
            x (Tensor): 待编码的向量
            num_repeats (int): 编码层的层数
            pattern (list): 量子比特的纠缠方式
            invert (bool): 添加的是否为编码电路的逆电路，默认为 ``False`` ，即添加正常的编码电路
        """
        assert x.size <= self.n, \
            "the number of classical data should be equal to the number of qubits"
        num_x = x.size
        x = paddle.flatten(x)
        if pattern is None:
            pattern = list()
            for idx0 in range(0, self.n):
                for idx1 in range(idx0 + 1, self.n):
                    pattern.append((idx0, idx1))

        while num_repeats > 0:
            num_repeats -= 1
            if invert:
                for item in pattern:
                    self.cnot(list(item))
                    self.rz(-x[item[0]] * x[item[1]], item[1])
                    self.cnot(list(item))
                for idx in range(0, num_x):
                    self.rz(-x[idx], idx)
                for idx in range(0, num_x):
                    self.h(idx)
            else:
                for idx in range(0, num_x):
                    self.h(idx)
                for idx in range(0, num_x):
                    self.rz(x[idx], idx)
                for item in pattern:
                    self.cnot(list(item))
                    self.rz(x[item[0]] * x[item[1]], item[1])
                    self.cnot(list(item))

    """
    Common Gates
    """

    def rx(self, theta, which_qubit):
        r"""添加关于 x 轴的单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix}
                \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            theta = paddle.to_tensor(theta)
            num_qubits = 1
            cir = UAnsatz(num_qubits)
            which_qubit = 0
            cir.rx(theta[0], which_qubit)

        """
        assert 0 <= which_qubit < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'rx', 'which_qubits': [which_qubit], 'theta': [curr_idx, 2, 1]})
        self.__param.append(theta)

    def crx(self, theta, which_qubit):
        r"""添加关于 x 轴的控制单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{align}
                CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes rx\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                    0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
            \end{align}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (list): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            theta = paddle.to_tensor(theta)
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            which_qubit = [0, 1]
            cir.crx(theta[0], which_qubit)

        """
        assert 0 <= which_qubit[0] < self.n and 0 <= which_qubit[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert which_qubit[0] != which_qubit[1], \
            "the control qubit is the same as the target qubit"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'crx', 'which_qubits': which_qubit, 'theta': [curr_idx, 2, 1]})
        self.__param.append(theta)

    def ry(self, theta, which_qubit):
        r"""添加关于 y 轴的单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix}
                \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            theta = paddle.to_tensor(theta)
            num_qubits = 1
            cir = UAnsatz(num_qubits)
            which_qubit = 0
            cir.ry(theta[0], which_qubit)
        """
        assert 0 <= which_qubit < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'ry', 'which_qubits': [which_qubit], 'theta': [curr_idx, 0, 0]})
        self.__param.append(theta)

    def cry(self, theta, which_qubit):
        r"""添加关于 y 轴的控制单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{align}
                CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes rx\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                    0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
            \end{align}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (list): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            theta = paddle.to_tensor(theta)
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            which_qubit = [0, 1]
            cir.cry(theta[0], which_qubit)
        """
        assert 0 <= which_qubit[0] < self.n and 0 <= which_qubit[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert which_qubit[0] != which_qubit[1], \
            "the control qubit is the same as the target qubit"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'cry', 'which_qubits': which_qubit, 'theta': [curr_idx, 0, 0]})
        self.__param.append(theta)

    def rz(self, theta, which_qubit):
        r"""添加关于 z 轴的单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            theta = paddle.to_tensor(theta)
            num_qubits = 1
            cir = UAnsatz(num_qubits)
            which_qubit = 0
            cir.rz(theta[0], which_qubit)
        """
        assert 0 <= which_qubit < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'rz', 'which_qubits': [which_qubit], 'theta': [0, 0, curr_idx]})
        self.__param.append(theta)

    def crz(self, theta, which_qubit):
        r"""添加关于 z 轴的控制单量子比特旋转门。

        其矩阵形式为：

        .. math::

            \begin{align}
                CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes rx\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\theta}
                \end{bmatrix}
            \end{align}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (list): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            theta = paddle.to_tensor(theta)
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            which_qubit = [0, 1]
            cir.crz(theta[0], which_qubit)
        """
        assert 0 <= which_qubit[0] < self.n and 0 <= which_qubit[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert which_qubit[0] != which_qubit[1], \
            "the control qubit is the same as the target qubit"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'crz', 'which_qubits': which_qubit, 'theta': [0, 0, curr_idx]})
        self.__param.append(theta)

    def cnot(self, control):
        r"""添加一个 CNOT 门。

        对于 2 量子比特的量子电路，当 ``control`` 为 ``[0, 1]`` 时，其矩阵形式为：

        .. math::

            \begin{align}
                CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的量子比特的编号，``control[0]`` 为控制位，``control[1]`` 为目标位，
                其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.cnot([0, 1])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert control[0] != control[1], \
            "the control qubit is the same as the target qubit"
        self.__history.append({'gate': 'CNOT', 'which_qubits': control, 'theta': None})

    def cy(self, control):
        r"""添加一个 cy 门。

        对于 2 量子比特的量子电路，当 ``control`` 为 ``[0, 1]`` 时，其矩阵形式为：

        .. math::

            \begin{align}
                CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & -1j \\
                    0 & 0 & 1j & 0
                \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的量子比特的编号，``control[0]`` 为控制位，``control[1]`` 为目标位，
                其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.cy([0, 1])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert control[0] != control[1], \
            "the control qubit is the same as the target qubit"
        self.__history.append({'gate': 'cy', 'which_qubits': control, 'theta': None})

    def cz(self, control):
        r"""添加一个 cz 门。

        对于 2 量子比特的量子电路，当 ``control`` 为 ``[0, 1]`` 时，其矩阵形式为：

        .. math::

            \begin{align}
                CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & -1
                \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的量子比特的编号，``control[0]`` 为控制位，``control[1]`` 为目标位，
                其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.cz([0, 1])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert control[0] != control[1], \
            "the control qubit is the same as the target qubit"
        self.__history.append({'gate': 'cz', 'which_qubits': control, 'theta': None})

    def cu(self, theta, phi, lam, control):
        r"""添加一个控制 U 门。

        对于 2 量子比特的量子电路，当 ``control`` 为 ``[0, 1]`` 时，其矩阵形式为：

        .. math::

            \begin{align}
                CU
                &=
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & \cos\frac\theta2 &-e^{i\lambda}\sin\frac\theta2 \\
                    0 & 0 & e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                \end{bmatrix}
            \end{align}

        Args:
            theta (Tensor): 旋转角度 :math:`\theta` 。
            phi (Tensor): 旋转角度 :math:`\phi` 。
            lam (Tensor): 旋转角度 :math:`\lambda` 。
            control (list): 作用在的量子比特的编号，``control[0]`` 为控制位，``control[1]`` 为目标位，
                其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            num_qubits = 2
            cir = UAnsatz(num_qubits)
            theta = paddle.to_tensor(np.array([np.pi], np.float64), stop_gradient=False)
            phi = paddle.to_tensor(np.array([np.pi / 2], np.float64), stop_gradient=False)
            lam = paddle.to_tensor(np.array([np.pi / 4], np.float64), stop_gradient=False)
            cir.cu(theta, phi, lam, [0, 1])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert control[0] != control[1], \
            "the control qubit is the same as the target qubit"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'CU', 'which_qubits': control, 'theta': [curr_idx, curr_idx + 1, curr_idx + 2]})
        self.__param.extend([theta, phi, lam])

    def swap(self, control):
        r"""添加一个 SWAP 门。

        其矩阵形式为：

        .. math::

            \begin{align}
                SWAP =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的量子比特的编号，``control[0]`` 和 ``control[1]`` 是想要交换的位，
                其值都应该在 :math:`[0, n)`范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.swap([0, 1])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert control[0] != control[1], \
            "the indices needed to be swapped should not be the same"
        self.__history.append({'gate': 'SWAP', 'which_qubits': control, 'theta': None})

    def cswap(self, control):
        r"""添加一个 CSWAP (Fredkin) 门。

        其矩阵形式为：

        .. math::

            \begin{align}
                SWAP =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
                \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的量子比特的编号，``control[0]`` 为控制位，``control[1]`` 和 ``control[2]`` 是想要交换的目标位，
                其值都应该在 :math:`[0, n)`范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 3
            cir = UAnsatz(num_qubits)
            cir.cswap([0, 1, 2])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n and 0 <= control[2] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert control[0] != control[1] and control[0] != control[
            2], "the control qubit is the same as the target qubit"
        assert control[1] != control[2], "the indices needed to be swapped should not be the same"
        self.__history.append({'gate': 'CSWAP', 'which_qubits': control, 'theta': None})

    def ccx(self, control):
        r"""添加一个 CCX (Toffoli) 门。

        其矩阵形式为：

        .. math::

            \begin{align}
                CCX =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的量子比特的编号， ``control[0]`` 和 ``control[1]`` 为控制位， ``control[2]`` 为目标位，
                当控制位值都为1时在该比特位作用X门。其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 3
            cir = UAnsatz(num_qubits)
            cir.ccx([0, 1, 2])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n and 0 <= control[2] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert control[0] != control[2] and control[1] != control[2], \
            "the control qubits should not be the same as the target qubit"
        assert control[0] != control[1], \
            "two control qubits should not be the same"
        self.__history.append({'gate': 'CCX', 'which_qubits': control, 'theta': None})

    def x(self, which_qubit):
        r"""添加单量子比特 X 门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix}
                0 & 1 \\
                1 & 0
            \end{bmatrix}

        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 1
            cir = UAnsatz(num_qubits)
            which_qubit = 0
            cir.x(which_qubit)
            cir.run_state_vector()
            print(cir.measure(shots = 0))

        ::

            {'0': 0.0, '1': 1.0}
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        self.__history.append({'gate': 'x', 'which_qubits': [which_qubit], 'theta': None})

    def y(self, which_qubit):
        r"""添加单量子比特 Y 门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix}
                0 & -i \\
                i & 0
            \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 1
            cir = UAnsatz(num_qubits)
            which_qubit = 0
            cir.y(which_qubit)
            cir.run_state_vector()
            print(cir.measure(shots = 0))

        ::

            {'0': 0.0, '1': 1.0}
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        self.__history.append({'gate': 'y', 'which_qubits': [which_qubit], 'theta': None})

    def z(self, which_qubit):
        r"""添加单量子比特 Z 门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix}
                1 & 0 \\
                0 & -1
            \end{bmatrix}

        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 1
            cir = UAnsatz(num_qubits)
            which_qubit = 0
            cir.z(which_qubit)
            cir.run_state_vector()
            print(cir.measure(shots = 0))

        ::

            {'0': 1.0, '1': 0.0}
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        self.__history.append({'gate': 'z', 'which_qubits': [which_qubit], 'theta': None})

    def h(self, which_qubit):
        r"""添加一个单量子比特的 Hadamard 门。

        其矩阵形式为：

        .. math::

            H = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                    1&1\\
                    1&-1
                \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        self.__history.append({'gate': 'h', 'which_qubits': [which_qubit], 'theta': None})

    def s(self, which_qubit):
        r"""添加一个单量子比特的 S 门。

        其矩阵形式为：

        .. math::

            S =
                \begin{bmatrix}
                    1&0\\
                    0&i
                \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        self.__history.append({'gate': 's', 'which_qubits': [which_qubit], 'theta': [0, 0, 1]})

    def sdg(self, which_qubit):
        r"""添加一个单量子比特的 S dagger 门。

        其矩阵形式为：

        .. math::

            S^\dagger =
                \begin{bmatrix}
                    1&0\\
                    0&-i
                \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        self.__history.append({'gate': 'sdg', 'which_qubits': [which_qubit], 'theta': [0, 0, 2]})

    def t(self, which_qubit):
        r"""添加一个单量子比特的 T 门。

        其矩阵形式为：

        .. math::

            T =
                \begin{bmatrix}
                    1&0\\
                    0&e^\frac{i\pi}{4}
                \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        self.__history.append({'gate': 't', 'which_qubits': [which_qubit], 'theta': [0, 0, 3]})

    def tdg(self, which_qubit):
        r"""添加一个单量子比特的 T dagger 门。

        其矩阵形式为：

        .. math::

            T^\dagger =
                \begin{bmatrix}
                    1&0\\
                    0&e^\frac{-i\pi}{4}
                \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        self.__history.append({'gate': 'tdg', 'which_qubits': [which_qubit], 'theta': [0, 0, 4]})

    def u3(self, theta, phi, lam, which_qubit):
        r"""添加一个单量子比特的旋转门。

        其矩阵形式为：

        .. math::

            \begin{align}
                U3(\theta, \phi, \lambda) =
                    \begin{bmatrix}
                        \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                        e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                    \end{bmatrix}
            \end{align}

        Args:
              theta (Tensor): 旋转角度 :math:`\theta` 。
              phi (Tensor): 旋转角度 :math:`\phi` 。
              lam (Tensor): 旋转角度 :math:`\lambda` 。
              which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n (the number of qubit)"
        curr_idx = len(self.__param)
        self.__history.append(
            {'gate': 'u', 'which_qubits': [which_qubit], 'theta': [curr_idx, curr_idx + 1, curr_idx + 2]})
        self.__param.extend([theta, phi, lam])

    def rxx(self, theta, which_qubits):
        r"""添加一个 RXX 门。

        其矩阵形式为：

        .. math::

            \begin{align}
                RXX(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
                    \end{bmatrix}
            \end{align}

        Args:
            theta (Tensor): 旋转角度
            which_qubits (list): 作用在的两个量子比特的编号，其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.rxx(paddle.to_tensor(np.array([np.pi/2])), [0, 1])
        """
        assert 0 <= which_qubits[0] < self.n and 0 <= which_qubits[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert which_qubits[0] != which_qubits[1], "the indices of two qubits should be different"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'RXX_gate', 'which_qubits': which_qubits, 'theta': [curr_idx]})
        self.__param.append(theta)

    def ryy(self, theta, which_qubits):
        r"""添加一个 RYY 门。

        其矩阵形式为：

        .. math::

            \begin{align}
                RYY(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        i\sin\frac{\theta}{2} & 0 & 0 & cos\frac{\theta}{2}
                    \end{bmatrix}
            \end{align}

        Args:
            theta (Tensor): 旋转角度
            which_qubits (list): 作用在的两个量子比特的编号，其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.ryy(paddle.to_tensor(np.array([np.pi/2])), [0, 1])
        """
        assert 0 <= which_qubits[0] < self.n and 0 <= which_qubits[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert which_qubits[0] != which_qubits[1], "the indices of two qubits should be different"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'RYY_gate', 'which_qubits': which_qubits, 'theta': [curr_idx]})
        self.__param.append(theta)

    def rzz(self, theta, which_qubits):
        r"""添加一个 RZZ 门。

        其矩阵形式为：

        .. math::

            \begin{align}
                RZZ(\theta) =
                    \begin{bmatrix}
                        e^{-i\frac{\theta}{2}} & 0 & 0 & 0 \\
                        0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
                        0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
                        0 & 0 & 0 & e^{-i\frac{\theta}{2}}
                    \end{bmatrix}
            \end{align}

        Args:
            theta (Tensor): 旋转角度
            which_qubits (list): 作用在的两个量子比特的编号，其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.rzz(paddle.to_tensor(np.array([np.pi/2])), [0, 1])
        """
        assert 0 <= which_qubits[0] < self.n and 0 <= which_qubits[1] < self.n, \
            "the qubit should >= 0 and < n (the number of qubit)"
        assert which_qubits[0] != which_qubits[1], "the indices of two qubits should be different"
        curr_idx = len(self.__param)
        self.__history.append({'gate': 'RZZ_gate', 'which_qubits': which_qubits, 'theta': [curr_idx]})
        self.__param.append(theta)

    def ms(self, which_qubits):
        r"""添加一个 Mølmer-Sørensen (MS) 门，用于离子阱设备。

        其矩阵形式为：

        .. math::

            \begin{align}
                MS = RXX(-\frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                    \begin{bmatrix}
                        1 & 0 & 0 & i \\
                        0 & 1 & i & 0 \\
                        0 & i & 1 & 0 \\
                        i & 0 & 0 & 1
                    \end{bmatrix}
            \end{align}

        Args:
            which_qubits (list): 作用在的两个量子比特的编号，其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        Note:
            参考文献 https://arxiv.org/abs/quant-ph/9810040

        ..  code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.ms([0, 1])
        """
        assert 0 <= which_qubits[0] < self.n and 0 <= which_qubits[1] < self.n, \
            "the qubit should >= 0 and < n(the number of qubit)"
        assert which_qubits[0] != which_qubits[1], "the indices of two qubits should be different"
        self.__history.append({'gate': 'MS_gate', 'which_qubits': which_qubits, 'theta': [2]})

    def universal_2_qubit_gate(self, theta, which_qubits):
        r"""添加 2-qubit 通用门，这个通用门需要 15 个参数。

        Args:
            theta (Tensor): 2-qubit 通用门的参数，其维度为 ``(15, )``
            which_qubits(list): 作用的量子比特编号

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 2
            theta = paddle.to_tensor(np.ones(15))
            cir = UAnsatz(n)
            cir.universal_2_qubit_gate(theta, [0, 1])
            cir.run_state_vector()
            print(cir.measure(shots = 0))

        ::

            {'00': 0.4306256106527819, '01': 0.07994547866706268, '10': 0.07994547866706264, '11': 0.40948343201309334}
        """

        assert len(theta.shape) == 1, 'The shape of theta is not right'
        assert len(theta) == 15, 'This Ansatz accepts 15 parameters'
        assert len(which_qubits) == 2, "You should add this gate on two qubits"

        a, b = which_qubits

        self.u3(theta[0], theta[1], theta[2], a)
        self.u3(theta[3], theta[4], theta[5], b)
        self.cnot([b, a])
        self.rz(theta[6], a)
        self.ry(theta[7], b)
        self.cnot([a, b])
        self.ry(theta[8], b)
        self.cnot([b, a])
        self.u3(theta[9], theta[10], theta[11], a)
        self.u3(theta[12], theta[13], theta[14], b)

    def __u3qg_U(self, theta, which_qubits):
        r"""
        用于构建 universal_3_qubit_gate
        """
        self.cnot(which_qubits[1:])
        self.ry(theta[0], which_qubits[1])
        self.cnot(which_qubits[:2])
        self.ry(theta[1], which_qubits[1])
        self.cnot(which_qubits[:2])
        self.cnot(which_qubits[1:])
        self.h(which_qubits[2])
        self.cnot([which_qubits[1], which_qubits[0]])
        self.cnot([which_qubits[0], which_qubits[2]])
        self.cnot(which_qubits[1:])
        self.rz(theta[2], which_qubits[2])
        self.cnot(which_qubits[1:])
        self.cnot([which_qubits[0], which_qubits[2]])

    def __u3qg_V(self, theta, which_qubits):
        r"""
        用于构建 universal_3_qubit_gate
        """
        self.cnot([which_qubits[2], which_qubits[0]])
        self.cnot(which_qubits[:2])
        self.cnot([which_qubits[2], which_qubits[1]])
        self.ry(theta[0], which_qubits[2])
        self.cnot(which_qubits[1:])
        self.ry(theta[1], which_qubits[2])
        self.cnot(which_qubits[1:])
        self.s(which_qubits[2])
        self.cnot([which_qubits[2], which_qubits[0]])
        self.cnot(which_qubits[:2])
        self.cnot([which_qubits[1], which_qubits[0]])
        self.h(which_qubits[2])
        self.cnot([which_qubits[0], which_qubits[2]])
        self.rz(theta[2], which_qubits[2])
        self.cnot([which_qubits[0], which_qubits[2]])

    def universal_3_qubit_gate(self, theta, which_qubits):
        r"""添加 3-qubit 通用门，这个通用门需要 81 个参数。

        Args:
            theta (Tensor): 3-qubit 通用门的参数，其维度为 ``(81, )``
            which_qubits(list): 作用的量子比特编号

        Note:
            参考: https://cds.cern.ch/record/708846/files/0401178.pdf

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 3
            theta = paddle.to_tensor(np.ones(81))
            cir = UAnsatz(n)
            cir.universal_3_qubit_gate(theta, [0, 1, 2])
            cir.run_state_vector()
            print(cir.measure(shots = 0))

        ::

            {'000': 0.06697926831547105, '001': 0.13206788591381013, '010': 0.2806525391078656, '011': 0.13821526515701105, '100': 0.1390530116439897, '101': 0.004381404333075108, '110': 0.18403296778911565, '111': 0.05461765773966483}
        """
        assert len(which_qubits) == 3, "You should add this gate on three qubits"
        assert len(theta) == 81, "The length of theta is supposed to be 81"

        psi = reshape(x=theta[: 60], shape=[4, 15])
        phi = reshape(x=theta[60:], shape=[7, 3])
        self.universal_2_qubit_gate(psi[0], which_qubits[:2])
        self.u3(phi[0][0], phi[0][1], phi[0][2], which_qubits[2])

        self.__u3qg_U(phi[1], which_qubits)

        self.universal_2_qubit_gate(psi[1], which_qubits[:2])
        self.u3(phi[2][0], phi[2][1], phi[2][2], which_qubits[2])

        self.__u3qg_V(phi[3], which_qubits)

        self.universal_2_qubit_gate(psi[2], which_qubits[:2])
        self.u3(phi[4][0], phi[4][1], phi[4][2], which_qubits[2])

        self.__u3qg_U(phi[5], which_qubits)

        self.universal_2_qubit_gate(psi[3], which_qubits[:2])
        self.u3(phi[6][0], phi[6][1], phi[6][2], which_qubits[2])

    def pauli_rotation_gate_partial(self, ind, gate_name):
        r"""计算传入的泡利旋转门的偏导。

        Args:
            ind (int): 该门在本电路中的序号
            gate_name (string): 门的名字

        Return:
            UAnsatz: 用电路表示的该门的偏导

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            cir = UAnsatz(2)
            theta = paddle.to_tensor([np.pi, np.pi/2, np.pi/4], 'float64')
            cir.rx(theta[0], 0)
            cir.ryy(theta[1], [1, 0])
            cir.rz(theta[2], 1)
            print(cir.pauli_rotation_gate_partial(0, 'rx'))

        ::

            ------------x----Rx(3.142)----Ryy(1.57)---------------
                        |                     |                   
            ------------|-----------------Ryy(1.57)----Rz(0.785)--
                        |                                         
            --H---SDG---*--------H--------------------------------
        """
        history, param = self._get_history()
        assert ind <= len(history), "The index number should be less than or equal to %d" % len(history)
        assert gate_name in {'rx', 'ry', 'rz', 'RXX_gate', 'RYY_gate', 'RZZ_gate'}, "Gate not supported."
        assert gate_name == history[ind]['gate'], "Gate name incorrect."

        n = self.n
        new_circuit = UAnsatz(n + 1)
        new_circuit._add_history(history[:ind], param)
        new_circuit.h(n)
        new_circuit.sdg(n)
        if gate_name in {'rx', 'RXX_gate'}:
            new_circuit.cnot([n, history[ind]['which_qubits'][0]])
            if gate_name == 'RXX_gate':
                new_circuit.cnot([n, history[ind]['which_qubits'][1]])
        elif gate_name in {'ry', 'RYY_gate'}:
            new_circuit.cy([n, history[ind]['which_qubits'][0]])
            if gate_name == 'RYY_gate':
                new_circuit.cy([n, history[ind]['which_qubits'][1]])
        elif gate_name in {'rz', 'RZZ_gate'}:
            new_circuit.cz([n, history[ind]['which_qubits'][0]])
            if gate_name == 'RZZ_gate':
                new_circuit.cz([n, history[ind]['which_qubits'][1]])
        new_circuit.h(n)
        new_circuit._add_history(history[ind: len(history)], param)

        return new_circuit

    def control_rotation_gate_partial(self, ind, gate_name):
        r"""计算传入的控制旋转门的偏导。

        Args:
            ind (int): 该门在本电路中的序号
            gate_name (string): 门的名字

        Return:
            List: 用两个电路表示的该门的偏导

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            cir = UAnsatz(2)
            theta = paddle.to_tensor([np.pi, np.pi/2, np.pi/4], 'float64')
            cir.rx(theta[0], 0)
            cir.ryy(theta[1], [1, 0])
            cir.crz(theta[2], [0, 1])
            print(cir.control_rotation_gate_partial(2, 'crz')[0])
            print(cir.control_rotation_gate_partial(2, 'crz')[1])

        ::

            --Rx(3.142)----Ryy(1.57)-------------*------
                               |                 |      
            ---------------Ryy(1.57)----z----Rz(0.785)--
                                        |               
            ------H-----------SDG-------*--------H------

            --Rx(3.142)----Ryy(1.57)----z-------------*------
                               |        |             |      
            ---------------Ryy(1.57)----|----z----Rz(0.785)--
                                        |    |               
            ------H------------S--------*----*--------H------
        """
        history, param = self._get_history()
        assert ind <= len(history), "The index number should be less than or equal to %d" % len(history)
        assert gate_name in {'crx', 'cry', 'crz'}, "Gate not supported."
        assert gate_name == history[ind]['gate'], "Gate name incorrect."

        n = self.n
        new_circuit = [UAnsatz(n + 1) for j in range(2)]
        for k in range(2):
            new_circuit[k]._add_history(history[:ind], param)
            new_circuit[k].h(n)
            new_circuit[k].sdg(n) if k == 0 else new_circuit[k].s(n)
            if k == 1:
                new_circuit[k].cz([n, history[ind]['which_qubits'][1]])
            if gate_name == 'crx':
                new_circuit[k].cnot([n, history[ind]['which_qubits'][0]])
            elif gate_name == 'cry':
                new_circuit[k].cy([n, history[ind]['which_qubits'][0]])
            elif gate_name == 'crz':
                new_circuit[k].cz([n, history[ind]['which_qubits'][0]])
            new_circuit[k].h(n)
            new_circuit[k]._add_history(history[ind: len(history)], param)

        return new_circuit

    def u3_partial(self, ind_history, ind_gate):
        r"""计算传入的 u3 门的一个参数的偏导。

        Args:
            ind_history (int): 该门在本电路中的序号
            ind_gate (int): u3 门参数的 index，可以是 0 或 1 或 2

        Return:
            UAnsatz: 用电路表示的该门的一个参数的偏导

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            cir = UAnsatz(2)
            theta = paddle.to_tensor([np.pi, np.pi/2, np.pi/4], 'float64')
            cir.u3(theta[0], theta[1], theta[2], 0)
            print(cir.u3_partial(0, 0))

        ::

            ------------z----U--
                        |       
            ------------|-------
                        |       
            --H---SDG---*----H--
        """
        history, param = self._get_history()
        assert ind_history <= len(history), "The index number should be less than or equal to %d" % len(history)
        assert ind_gate in {0, 1, 2}, "U3 gate has only three parameters, please choose from {0, 1, 2}"
        assert history[ind_history]['gate'] == 'u', "Not a u3 gate."

        n = self.n
        new_circuit = UAnsatz(n + 1)
        assert ind_gate in {0, 1, 2}, "ind must be in {0, 1, 2}"
        new_circuit._add_history(history[:ind_history], param)
        if ind_gate == 0:
            new_circuit.h(n)
            new_circuit.sdg(n)
            new_circuit.cz([n, history[ind_history]['which_qubits'][0]])
            new_circuit.h(n)
            new_circuit._add_history(history[ind_history], param)
        elif ind_gate == 1:
            new_circuit.h(n)
            new_circuit.sdg(n)
            new_circuit.rz(self.__param[history[ind_history]['theta'][2]], history[ind_history]['which_qubits'][0])
            new_circuit.cy([n, history[ind_history]['which_qubits'][0]])
            new_circuit.ry(self.__param[history[ind_history]['theta'][0]], history[ind_history]['which_qubits'][0])
            new_circuit.rz(self.__param[history[ind_history]['theta'][1]], history[ind_history]['which_qubits'][0])
            new_circuit.h(n)
        elif ind_gate == 2:
            new_circuit.h(n)
            new_circuit.sdg(n)
            new_circuit.rz(self.__param[history[ind_history]['theta'][2]], history[ind_history]['which_qubits'][0])
            new_circuit.ry(self.__param[history[ind_history]['theta'][0]], history[ind_history]['which_qubits'][0])
            new_circuit.cz([n, history[ind_history]['which_qubits'][0]])
            new_circuit.rz(self.__param[history[ind_history]['theta'][1]], history[ind_history]['which_qubits'][0])
            new_circuit.h(n)
        new_circuit._add_history(history[ind_history + 1: len(history)], param)

        return new_circuit

    def cu3_partial(self, ind_history, ind_gate):
        r"""计算传入的 cu 门的一个参数的偏导。

        Args:
            ind_history (int): 该门在本电路中的序号
            ind_gate (int): cu 门参数的 index，可以是 0 或 1 或 2

        Return:
            UAnsatz: 用电路表示的该门的一个参数的偏导

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            cir = UAnsatz(2)
            theta = paddle.to_tensor([np.pi, np.pi/2, np.pi/4], 'float64')
            cir.cu(theta[0], theta[1], theta[2], [0, 1])
            print(cir.cu3_partial(0, 0)[0])
            print(cir.cu3_partial(0, 0)[1])

        ::

            -----------------x--
                             |  
            ------------z----U--
                        |       
            --H---SDG---*----H--

            ------------z---------x--
                        |         |  
            ------------|----z----U--
                        |    |       
            --H----S----*----*----H--
        """
        history, param = self._get_history()
        assert ind_history <= len(history), "The index number should be less than or equal to %d" % len(history)
        assert ind_gate in {0, 1, 2}, "CU gate has only three parameters, please choose from {0, 1, 2}"
        assert history[ind_history]['gate'] == 'CU', "Not a CU gate."

        n = self.n
        new_circuit = [UAnsatz(n + 1) for j in range(2)]
        assert ind_gate in {0, 1, 2}, "ind must be in {0, 1, 2}"
        for k in range(2):
            new_circuit[k]._add_history(history[:ind_history], param)
            if ind_gate == 0:
                new_circuit[k].h(n)
                new_circuit[k].sdg(n) if k == 0 else new_circuit[k].s(n)
                if k == 1:
                    new_circuit[k].cz([n, history[ind_history]['which_qubits'][0]])
                new_circuit[k].cz([n, history[ind_history]['which_qubits'][1]])
                new_circuit[k].h(n)
                new_circuit[k]._add_history([history[ind_history]], param)
            elif ind_gate == 1:
                new_circuit[k].h(n)
                new_circuit[k].sdg(n) if k == 0 else new_circuit[k].s(n)
                new_circuit[k].crz(self.__param[history[ind_history]['theta'][2]], history[ind_history]['which_qubits'])
                if k == 1:
                    new_circuit[k].cz([n, history[ind_history]['which_qubits'][0]])
                new_circuit[k].cy([n, history[ind_history]['which_qubits'][0]])
                new_circuit[k].cry(self.__param[history[ind_history]['theta'][0]], history[ind_history]['which_qubits'])
                new_circuit[k].crz(self.__param[history[ind_history]['theta'][1]], history[ind_history]['which_qubits'])
                new_circuit[k].h(n)
            elif ind_gate == 2:
                new_circuit[k].h(n)
                new_circuit[k].sdg(n) if k == 0 else new_circuit[k].s(n)
                new_circuit[k].crz(self.__param[history[ind_history]['theta'][2]], history[ind_history]['which_qubits'])
                new_circuit[k].cry(self.__param[history[ind_history]['theta'][0]], history[ind_history]['which_qubits'])
                if k == 1:
                    new_circuit[k].cz([n, history[ind_history]['which_qubits'][0]])
                new_circuit[k].cz([n, history[ind_history]['which_qubits'][0]])
                new_circuit[k].crz(self.__param[history[ind_history]['theta'][1]], history[ind_history]['which_qubits'])
                new_circuit[k].h(n)

            new_circuit[k]._add_history(history[ind_history + 1: len(history)], param)

        return new_circuit

    def linear_combinations_gradient(self, H, shots=0):
        r"""用 linear combination 的方法计算电路中所有需要训练的参数的梯度。损失函数默认为计算哈密顿量的期望值。

        Args:
            H (list or Hamiltonian): 损失函数中用到的记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
            shots (int, optional): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值

        Return:
            Tensor: 该电路中所有需要训练的参数的梯度

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz

            def U_theta(theta, N, D):
                cir = UAnsatz(N)
                cir.complex_entangled_layer(theta[:D], D)
                for i in range(N):
                    cir.ry(theta=theta[D][i][0], which_qubit=i)
                cir.run_state_vector()
                return cir

            H = [[1.0, 'z0,z1']]
            theta = paddle.uniform(shape=[2, 2, 3], dtype='float64', min=0.0, max=np.pi * 2)
            theta.stop_gradient = False
            circuit = U_theta(theta, 2, 1)
            gradient = circuit.linear_combinations_gradient(H, shots=0)
            print(gradient)

        ::

            Tensor(shape=[8], dtype=float64, place=CPUPlace, stop_gradient=True,
                   [ 0.        , -0.11321444, -0.22238044,  0.        ,  0.04151700,  0.44496212, -0.19465690,  0.96022600])
        """
        history, param = self._get_history()
        grad = []

        if not isinstance(H, list):
            H = H.pauli_str
        H = copy.deepcopy(H)
        for i in H:
            i[1] += ',z' + str(self.n)

        for i, history_i in enumerate(history):
            if history_i['gate'] == 'rx' and self.__param[history_i['theta'][0]].stop_gradient is False:
                new_circuit = self.pauli_rotation_gate_partial(i, 'rx')
                if self.__run_mode == 'state_vector':
                    new_circuit.run_state_vector()
                elif self.__run_mode == 'density_matrix':
                    new_circuit.run_density_matrix()
                grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
            elif history_i['gate'] == 'ry' and self.__param[history_i['theta'][0]].stop_gradient is False:
                new_circuit = self.pauli_rotation_gate_partial(i, 'ry')
                if self.__run_mode == 'state_vector':
                    new_circuit.run_state_vector()
                elif self.__run_mode == 'density_matrix':
                    new_circuit.run_density_matrix()
                grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
            elif history_i['gate'] == 'rz' and self.__param[history_i['theta'][2]].stop_gradient is False:
                new_circuit = self.pauli_rotation_gate_partial(i, 'rz')
                if self.__run_mode == 'state_vector':
                    new_circuit.run_state_vector()
                elif self.__run_mode == 'density_matrix':
                    new_circuit.run_density_matrix()
                grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
            elif history_i['gate'] == 'crx' and self.__param[history_i['theta'][0]].stop_gradient is False:
                new_circuit = self.control_rotation_gate_partial(i, 'crx')
                for k in new_circuit:
                    if self.__run_mode == 'state_vector':
                        k.run_state_vector()
                    elif self.__run_mode == 'density_matrix':
                        k.run_density_matrix()
                gradient = paddle.to_tensor(np.mean([circuit.expecval(H, shots) for circuit in new_circuit]), 'float64')
                grad.append(gradient)
            elif history_i['gate'] == 'cry' and self.__param[history_i['theta'][0]].stop_gradient is False:
                new_circuit = self.control_rotation_gate_partial(i, 'cry')
                for k in new_circuit:
                    if self.__run_mode == 'state_vector':
                        k.run_state_vector()
                    elif self.__run_mode == 'density_matrix':
                        k.run_density_matrix()
                gradient = paddle.to_tensor(np.mean([circuit.expecval(H, shots) for circuit in new_circuit]), 'float64')
                grad.append(gradient)
            elif history_i['gate'] == 'crz' and self.__param[history_i['theta'][2]].stop_gradient is False:
                new_circuit = self.control_rotation_gate_partial(i, 'crz')
                for k in new_circuit:
                    if self.__run_mode == 'state_vector':
                        k.run_state_vector()
                    elif self.__run_mode == 'density_matrix':
                        k.run_density_matrix()
                gradient = paddle.to_tensor(np.mean([circuit.expecval(H, shots) for circuit in new_circuit]), 'float64')
                grad.append(gradient)
            elif history_i['gate'] == 'RXX_gate' and self.__param[history_i['theta'][0]].stop_gradient is False:
                new_circuit = self.pauli_rotation_gate_partial(i, 'RXX_gate')
                if self.__run_mode == 'state_vector':
                    new_circuit.run_state_vector()
                elif self.__run_mode == 'density_matrix':
                    new_circuit.run_density_matrix()
                grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
            elif history_i['gate'] == 'RYY_gate' and self.__param[history_i['theta'][0]].stop_gradient is False:
                new_circuit = self.pauli_rotation_gate_partial(i, 'RYY_gate')
                if self.__run_mode == 'state_vector':
                    new_circuit.run_state_vector()
                elif self.__run_mode == 'density_matrix':
                    new_circuit.run_density_matrix()
                grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
            elif history_i['gate'] == 'RZZ_gate' and self.__param[history_i['theta'][0]].stop_gradient is False:
                new_circuit = self.pauli_rotation_gate_partial(i, 'RZZ_gate')
                if self.__run_mode == 'state_vector':
                    new_circuit.run_state_vector()
                elif self.__run_mode == 'density_matrix':
                    new_circuit.run_density_matrix()
                grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
            elif history_i['gate'] == 'u':
                if not self.__param[history_i['theta'][0]].stop_gradient:
                    new_circuit = self.u3_partial(i, 0)
                    if self.__run_mode == 'state_vector':
                        new_circuit.run_state_vector()
                    elif self.__run_mode == 'density_matrix':
                        new_circuit.run_density_matrix()
                    grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
                if not self.__param[history_i['theta'][1]].stop_gradient:
                    new_circuit = self.u3_partial(i, 1)
                    if self.__run_mode == 'state_vector':
                        new_circuit.run_state_vector()
                    elif self.__run_mode == 'density_matrix':
                        new_circuit.run_density_matrix()
                    grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
                if not self.__param[history_i['theta'][2]].stop_gradient:
                    new_circuit = self.u3_partial(i, 2)
                    if self.__run_mode == 'state_vector':
                        new_circuit.run_state_vector()
                    elif self.__run_mode == 'density_matrix':
                        new_circuit.run_density_matrix()
                    grad.append(paddle.to_tensor(new_circuit.expecval(H, shots), 'float64'))
            elif history_i['gate'] == 'CU':
                if not self.__param[history_i['theta'][0]].stop_gradient:
                    new_circuit = self.cu3_partial(i, 0)
                    for k in new_circuit:
                        if self.__run_mode == 'state_vector':
                            k.run_state_vector()
                        elif self.__run_mode == 'density_matrix':
                            k.run_density_matrix()
                    gradient = paddle.to_tensor(np.mean([circuit.expecval(H, shots) for circuit in new_circuit]), 'float64')
                    grad.append(gradient)
                if not self.__param[history_i['theta'][1]].stop_gradient:
                    new_circuit = self.cu3_partial(i, 1)
                    for k in new_circuit:
                        if self.__run_mode == 'state_vector':
                            k.run_state_vector()
                        elif self.__run_mode == 'density_matrix':
                            k.run_density_matrix()
                    gradient = paddle.to_tensor(np.mean([circuit.expecval(H, shots) for circuit in new_circuit]), 'float64')
                    grad.append(gradient)
                if not self.__param[history_i['theta'][2]].stop_gradient:
                    new_circuit = self.cu3_partial(i, 2)
                    for k in new_circuit:
                        if self.__run_mode == 'state_vector':
                            k.run_state_vector()
                        elif self.__run_mode == 'density_matrix':
                            k.run_density_matrix()
                    gradient = paddle.to_tensor(np.mean([circuit.expecval(H, shots) for circuit in new_circuit]), 'float64')
                    grad.append(gradient)
        grad = paddle.concat(grad)

        return grad

    """
    Measurements
    """

    def __process_string(self, s, which_qubits):
        r"""该函数基于 which_qubits 返回 s 的一部分
        This functions return part of string s baesd on which_qubits
        If s = 'abcdefg', which_qubits = [0,2,5], then it returns 'acf'

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        new_s = ''.join(s[j] for j in which_qubits)
        return new_s

    def __process_similiar(self, result):
        r"""该函数基于相同的键合并值。
        This functions merges values based on identical keys.
        If result = [('00', 10), ('01', 20), ('11', 30), ('11', 40), ('11', 50), ('00', 60)],
            then it returns {'00': 70, '01': 20, '11': 120}

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        data = defaultdict(int)
        for idx, val in result:
            data[idx] += val

        return dict(data)

    def __measure_hist(self, result, which_qubits, shots):
        r"""将测量的结果以柱状图的形式呈现。

        Note:
            这是内部函数，你并不需要直接调用到该函数。

        Args:
              result (dictionary): 测量结果
              which_qubits (list): 测量的量子比特，如测量所有则是 ``None``
              shots(int): 测量次数

        Returns
            dict: 测量结果

        """
        n = self.n if which_qubits is None else len(which_qubits)
        assert n < 6, "Too many qubits to plot"

        ylabel = "Measured Probabilities"
        if shots == 0:
            shots = 1
            ylabel = "Probabilities"

        state_list = [np.binary_repr(index, width=n) for index in range(0, 2 ** n)]
        freq = []
        for state in state_list:
            freq.append(result.get(state, 0.0) / shots)

        plt.bar(range(2 ** n), freq, tick_label=state_list)
        plt.xticks(rotation=90)
        plt.xlabel("Qubit State")
        plt.ylabel(ylabel)
        plt.show()

        return result

    # Which_qubits is list-like
    def measure(self, which_qubits=None, shots=2 ** 10, plot=False):
        r"""对量子电路输出的量子态进行测量。

        Warning:
            当 ``plot`` 为 ``True`` 时，当前量子电路的量子比特数需要小于 6 ，否则无法绘制图片，会抛出异常。

        Args:
            which_qubits (list, optional): 要测量的qubit的编号，默认全都测量
            shots (int, optional): 该量子电路输出的量子态的测量次数，默认为 1024 次；若为 0，则返回测量结果的精确概率分布
            plot (bool, optional): 是否绘制测量结果图，默认为 ``False`` ，即不绘制

        Returns:
            dict: 测量的结果

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            cir = UAnsatz(2)
            cir.h(0)
            cir.cnot([0,1])
            cir.run_state_vector()
            result = cir.measure(shots = 2048, which_qubits = [1])
            print(f"The results of measuring qubit 1 2048 times are {result}")

        ::

            The results of measuring qubit 1 2048 times are {'0': 964, '1': 1084}

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            cir = UAnsatz(2)
            cir.h(0)
            cir.cnot([0,1])
            cir.run_state_vector()
            result = cir.measure(shots = 0, which_qubits = [1])
            print(f"The probability distribution of measurement results on qubit 1 is {result}")

        ::

            The probability distribution of measurement results on qubit 1 is {'0': 0.4999999999999999, '1': 0.4999999999999999}
        """
        if self.__run_mode == 'state_vector':
            state = self.__state
        elif self.__run_mode == 'density_matrix':
            # Take the diagonal of the density matrix as a probability distribution
            diag = np.diag(self.__state.numpy())
            state = paddle.to_tensor(np.sqrt(diag))
        else:
            # Raise error
            raise ValueError("no state for measurement; please run the circuit first")

        if shots == 0:  # Returns probability distribution over all measurement results
            dic2to10, dic10to2 = dic_between2and10(self.n)
            result = {}
            for i in range(2 ** self.n):
                result[dic10to2[i]] = (real(state)[i] ** 2 + imag(state)[i] ** 2).numpy()[0]

            if which_qubits is not None:
                new_result = [(self.__process_string(key, which_qubits), value) for key, value in result.items()]
                result = self.__process_similiar(new_result)
        else:
            if which_qubits is None:  # Return all the qubits
                result = measure_state(state, shots)
            else:
                assert all([e < self.n for e in which_qubits]), 'Qubit index out of range'
                which_qubits.sort()  # Sort in ascending order

                collapse_all = measure_state(state, shots)
                new_collapse_all = [(self.__process_string(key, which_qubits), value) for key, value in
                                    collapse_all.items()]
                result = self.__process_similiar(new_collapse_all)

        return result if not plot else self.__measure_hist(result, which_qubits, shots)

    def measure_in_bell_basis(self, which_qubits, shots=0):
        r"""对量子电路输出的量子态进行贝尔基测量。

        Args:
            which_qubits(list): 要测量的量子比特
            shots(int): 测量的采样次数，默认为0，表示计算解析解

        Returns:
            list: 测量得到四个贝尔基的概率
        """
        assert which_qubits[0] != which_qubits[1], "You have to measure two different qubits."
        which_qubits.sort()
        i, j = which_qubits
        qubit_num = self.n
        input_state = self.__state
        mode = self.__run_mode
        cir = UAnsatz(qubit_num)
        cir.cnot([i, j])
        cir.h(i)

        if mode == 'state_vector':
            output_state = cir.run_state_vector(input_state).numpy()
        elif mode == 'density_matrix':
            output_density_matrix = cir.run_density_matrix(input_state).numpy()
            output_state = np.sqrt(np.diag(output_density_matrix))
        else:
            raise ValueError("Can't recognize the mode of quantum state.")

        prob_amplitude = np.abs(output_state).tolist()
        prob_amplitude = [item ** 2 for item in prob_amplitude]

        prob_array = [0] * 4
        for i in range(2 ** qubit_num):
            binary = bin(i)[2:]
            binary = '0' * (qubit_num - len(binary)) + binary
            target_qubits = str()
            for qubit_idx in which_qubits:
                target_qubits += binary[qubit_idx]
            prob_array[int(target_qubits, base=2)] += prob_amplitude[i]

        if shots == 0:
            result = prob_array
        else:
            result = [0] * 4
            samples = np.random.choice(list(range(4)), shots, p=prob_array)
            for item in samples:
                result[item] += 1
            result = [item / shots for item in result]

        return result

    def expecval(self, H, shots=0):
        r"""量子电路输出的量子态关于可观测量 H 的期望值。

        Hint:
            如果想输入的可观测量的矩阵为 :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I` ，
                则 ``H`` 的 ``list`` 形式为 ``[[0.7, 'Z0, X1'], [0.2, 'Z1']]`` 。

        Args:
            H (Hamiltonian or list): 可观测量的相关信息
            shots (int, optional): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值

        Returns:
            Tensor: 量子电路输出的量子态关于 ``H`` 的期望值

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz

            n = 5
            experiment_shots = 2**10
            H_info = [[0.1, 'x1'], [0.2, 'y0,z4']]
            theta = paddle.ones([3], dtype='float64')

            cir = UAnsatz(n)
            cir.rx(theta[0], 0)
            cir.rz(theta[1], 1)
            cir.rx(theta[2], 2)
            cir.run_state_vector()

            result_1 = cir.expecval(H_info, shots = experiment_shots).numpy()
            result_2 = cir.expecval(H_info, shots = 0).numpy()

            print(f'The expectation value obtained by {experiment_shots} measurements is {result_1}')
            print(f'The accurate expectation value of H is {result_2}')

        ::

            The expectation value obtained by 1024 measurements is [-0.16328125]
            The accurate expectation value of H is [-0.1682942]
        """
        expec_val = 0
        if not isinstance(H, list):
            H = H.pauli_str
        if shots == 0:
            if self.__run_mode == 'state_vector':
                expec_val = real(vec_expecval(H, self.__state))
            elif self.__run_mode == 'density_matrix':
                state = self.__state
                H_mat = paddle.to_tensor(pauli_str_to_matrix(H, self.n))
                expec_val = real(trace(matmul(state, H_mat)))
            else:
                # Raise error
                raise ValueError("no state for measurement; please run the circuit first")
        else:
            for term in H:
                expec_val += term[0] * _local_H_prob(self, term[1], shots=shots)
            expec_val = paddle.to_tensor(expec_val, 'float64')

        return expec_val

    """
    Circuit Templates
    """

    def superposition_layer(self):
        r"""添加一层 Hadamard 门。

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            cir = UAnsatz(2)
            cir.superposition_layer()
            cir.run_state_vector()
            result = cir.measure(shots = 0)
            print(f"The probability distribution of measurement results on both qubits is {result}")

        ::

            The probability distribution of measurement results on both qubits is
                {'00': 0.2499999999999999, '01': 0.2499999999999999,
                '10': 0.2499999999999999, '11': 0.2499999999999999}
        """
        for i in range(self.n):
            self.h(i)

    def weak_superposition_layer(self):
        r"""添加一层旋转角度为 :math:`\pi/4` 的 Ry 门。

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            cir = UAnsatz(2)
            cir.weak_superposition_layer()
            cir.run_state_vector()
            result = cir.measure(shots = 0)
            print(f"The probability distribution of measurement results on both qubits is {result}")

        ::

            The probability distribution of measurement results on both qubits is
                {'00': 0.7285533905932737, '01': 0.12500000000000003,
                '10': 0.12500000000000003, '11': 0.021446609406726238}
        """
        _theta = paddle.to_tensor(np.array([np.pi / 4]))  # Used in fixed Ry gate
        for i in range(self.n):
            self.ry(_theta, i)

    def linear_entangled_layer(self, theta, depth, which_qubits=None):
        r"""添加 ``depth`` 层包含 Ry 门，Rz 门和 CNOT 门的线性纠缠层。

        Attention:
            ``theta`` 的维度为 ``(depth, n, 2)`` ，最低维内容为对应的 ``ry`` 和 ``rz`` 的参数， ``n`` 为作用的量子比特数量。

        Args:
            theta (Tensor): Ry 门和 Rz 门的旋转角度
            depth (int): 纠缠层的深度
            which_qubits (list): 作用的量子比特编号

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 2
            DEPTH = 3
            theta = paddle.ones([DEPTH, 2, 2], dtype='float64')
            cir = UAnsatz(n)
            cir.linear_entangled_layer(theta, DEPTH, [0, 1])
            cir.run_state_vector()
            result = cir.measure(shots = 0)
            print(f"The probability distribution of measurement results on both qubits is {result}")

        ::

            The probability distribution of measurement results on both qubits is
                {'00': 0.646611169077063, '01': 0.06790630495474384,
                '10': 0.19073671025717626, '11': 0.09474581571101756}
        """
        # reformat 1D theta list
        theta_flat = paddle.flatten(theta)
        width = len(which_qubits) if which_qubits is not None else self.n
        assert len(theta_flat) == depth * width * 2, 'the size of theta is not right'
        theta = paddle.reshape(theta_flat, [depth, width, 2])

        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 2, 'the shape of theta is not right'
        # assert theta.shape[1] == self.n, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        if which_qubits is None:
            which_qubits = np.arange(self.n)

        for repeat in range(depth):
            for i, q in enumerate(which_qubits):
                self.ry(theta[repeat][i][0], q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i], which_qubits[i + 1]])
            for i, q in enumerate(which_qubits):
                self.rz(theta[repeat][i][1], q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i + 1], which_qubits[i]])

    def real_entangled_layer(self, theta, depth, which_qubits=None):
        r"""添加 ``depth`` 层包含 Ry 门和 CNOT 门的强纠缠层。

        Note:
            这一层量子门的数学表示形式为实数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, n, 1)``， ``n`` 为作用的量子比特数量。

        Args:
            theta (Tensor): Ry 门的旋转角度
            depth (int): 纠缠层的深度
            which_qubits (list): 作用的量子比特编号

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 2
            DEPTH = 3
            theta = paddle.ones([DEPTH, 2, 1], dtype='float64')
            cir = UAnsatz(n)
            cir.real_entangled_layer(paddle.to_tensor(theta), DEPTH, [0, 1])
            cir.run_state_vector()
            result = cir.measure(shots = 0)
            print(f"The probability distribution of measurement results on both qubits is {result}")

        ::

            The probability distribution of measurement results on both qubits is
                {'00': 2.52129874867343e-05, '01': 0.295456784923382,
                '10': 0.7045028818254718, '11': 1.5120263659845063e-05}
        """
        # reformat 1D theta list
        theta_flat = paddle.flatten(theta)
        width = len(which_qubits) if which_qubits is not None else self.n
        assert len(theta_flat) == depth * width, 'the size of theta is not right'
        theta = paddle.reshape(theta_flat, [depth, width, 1])

        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 1, 'the shape of theta is not right'
        # assert theta.shape[1] == self.n, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        if which_qubits is None:
            which_qubits = np.arange(self.n)

        for repeat in range(depth):
            for i, q in enumerate(which_qubits):
                self.ry(theta[repeat][i][0], q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i], which_qubits[i + 1]])
            self.cnot([which_qubits[-1], which_qubits[0]])

    def complex_entangled_layer(self, theta, depth, which_qubits=None):
        r"""添加 ``depth`` 层包含 U3 门和 CNOT 门的强纠缠层。

        Note:
            这一层量子门的数学表示形式为复数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, n, 3)`` ，最低维内容为对应的 ``u3`` 的参数 ``(theta, phi, lam)``， ``n`` 为作用的量子比特数量。

        Args:
            theta (Tensor): U3 门的旋转角度
            depth (int): 纠缠层的深度
            which_qubits (list): 作用的量子比特编号

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 2
            DEPTH = 3
            theta = paddle.ones([DEPTH, 2, 3], dtype='float64')
            cir = UAnsatz(n)
            cir.complex_entangled_layer(paddle.to_tensor(theta), DEPTH, [0, 1])
            cir.run_state_vector()
            result = cir.measure(shots = 0)
            print(f"The probability distribution of measurement results on both qubits is {result}")

        ::

            The probability distribution of measurement results on both qubits is
                {'00': 0.15032627279218896, '01': 0.564191201239618,
                '10': 0.03285998070292556, '11': 0.25262254526526823}
        """
        # reformat 1D theta list
        theta_flat = paddle.flatten(theta)
        width = len(which_qubits) if which_qubits is not None else self.n
        assert len(theta_flat) == depth * width * 3, 'the size of theta is not right'
        theta = paddle.reshape(theta_flat, [depth, width, 3])

        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 3, 'the shape of theta is not right'
        # assert theta.shape[1] == self.n, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        if which_qubits is None:
            which_qubits = np.arange(self.n)

        for repeat in range(depth):
            for i, q in enumerate(which_qubits):
                self.u3(theta[repeat][i][0], theta[repeat][i][1], theta[repeat][i][2], q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i], which_qubits[i + 1]])
            self.cnot([which_qubits[-1], which_qubits[0]])

    def __add_real_block(self, theta, position):
        r"""
        Add a real block to the circuit in (position). theta is a one dimensional tensor

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert len(theta) == 4, 'the length of theta is not right'
        assert 0 <= position[0] < self.n and 0 <= position[1] < self.n, 'position is out of range'
        self.ry(theta[0], position[0])
        self.ry(theta[1], position[1])

        self.cnot([position[0], position[1]])

        self.ry(theta[2], position[0])
        self.ry(theta[3], position[1])

    def __add_complex_block(self, theta, position):
        r"""
        Add a complex block to the circuit in (position). theta is a one dimensional tensor

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert len(theta) == 12, 'the length of theta is not right'
        assert 0 <= position[0] < self.n and 0 <= position[1] < self.n, 'position is out of range'
        self.u3(theta[0], theta[1], theta[2], position[0])
        self.u3(theta[3], theta[4], theta[5], position[1])

        self.cnot([position[0], position[1]])

        self.u3(theta[6], theta[7], theta[8], position[0])
        self.u3(theta[9], theta[10], theta[11], position[1])

    def __add_real_layer(self, theta, position):
        r"""
        Add a real layer on the circuit. theta is a two dimensional tensor.
        position is the qubit range the layer needs to cover.

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert theta.shape[1] == 4 and theta.shape[0] == (position[1] - position[0] + 1) / 2, \
            'the shape of theta is not right'
        for i in range(position[0], position[1], 2):
            self.__add_real_block(theta[int((i - position[0]) / 2)], [i, i + 1])

    def __add_complex_layer(self, theta, position):
        r"""
        Add a complex layer on the circuit. theta is a two dimensional tensor.
        position is the qubit range the layer needs to cover.

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert theta.shape[1] == 12 and theta.shape[0] == (position[1] - position[0] + 1) / 2, \
            'the shape of theta is not right'
        for i in range(position[0], position[1], 2):
            self.__add_complex_block(theta[int((i - position[0]) / 2)], [i, i + 1])

    def real_block_layer(self, theta, depth):
        r"""添加 ``depth`` 层包含 Ry 门和 CNOT 门的弱纠缠层。

        Note:
            这一层量子门的数学表示形式为实数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, n-1, 4)`` 。

        Args:
            theta (Tensor): Ry 门的旋转角度
            depth (int): 纠缠层的深度

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 4
            DEPTH = 3
            theta = paddle.ones([DEPTH, n - 1, 4], dtype='float64')
            cir = UAnsatz(n)
            cir.real_block_layer(paddle.to_tensor(theta), DEPTH)
            cir.run_density_matrix()
            print(cir.measure(shots = 0, which_qubits = [0]))

        ::

            {'0': 0.9646724056906162, '1': 0.035327594309385896}
        """
        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'The dimension of theta is not right'
        _depth, m, block = theta.shape
        assert depth > 0, 'depth must be greater than zero'
        assert _depth == depth, 'the depth of parameters has a mismatch'
        assert m == self.n - 1 and block == 4, 'The shape of theta is not right'

        if self.n % 2 == 0:
            for i in range(depth):
                self.__add_real_layer(theta[i][:int(self.n / 2)], [0, self.n - 1])
                self.__add_real_layer(theta[i][int(self.n / 2):], [1, self.n - 2]) if self.n > 2 else None
        else:
            for i in range(depth):
                self.__add_real_layer(theta[i][:int((self.n - 1) / 2)], [0, self.n - 2])
                self.__add_real_layer(theta[i][int((self.n - 1) / 2):], [1, self.n - 1])

    def complex_block_layer(self, theta, depth):
        r"""添加 ``depth`` 层包含 U3 门和 CNOT 门的弱纠缠层。

        Note:
            这一层量子门的数学表示形式为复数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, n-1, 12)`` 。

        Args:
            theta (Tensor): U3 门的角度信息
            depth (int): 纠缠层的深度

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 4
            DEPTH = 3
            theta = paddle.ones([DEPTH, n - 1, 12], dtype='float64')
            cir = UAnsatz(n)
            cir.complex_block_layer(paddle.to_tensor(theta), DEPTH)
            cir.run_density_matrix()
            print(cir.measure(shots = 0, which_qubits = [0]))

        ::

            {'0': 0.5271554811768046, '1': 0.4728445188231988}
        """
        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'The dimension of theta is not right'
        assert depth > 0, 'depth must be greater than zero'
        _depth, m, block = theta.shape
        assert _depth == depth, 'the depth of parameters has a mismatch'
        assert m == self.n - 1 and block == 12, 'The shape of theta is not right'

        if self.n % 2 == 0:
            for i in range(depth):
                self.__add_complex_layer(theta[i][:int(self.n / 2)], [0, self.n - 1])
                self.__add_complex_layer(theta[i][int(self.n / 2):], [1, self.n - 2]) if self.n > 2 else None
        else:
            for i in range(depth):
                self.__add_complex_layer(theta[i][:int((self.n - 1) / 2)], [0, self.n - 2])
                self.__add_complex_layer(theta[i][int((self.n - 1) / 2):], [1, self.n - 1])

    def finite_difference_gradient(self, H, delta, shots=0):
        r"""用差分法估计电路中参数的梯度。损失函数默认为计算哈密顿量的期望值。

        Args:
            H (list or Hamiltonian): 记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
            delta (float): 差分法中的 delta
            shots (int, optional): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值

        Returns:
            Tensor: 电路中所有可训练参数的梯度

        代码示例:

        .. code-block:: python

            import paddle
            import numpy as np
            from paddle_quantum.circuit import UAnsatz

            H = [[1.0, 'z0,z1']]
            theta = paddle.to_tensor(np.array([6.186, 5.387, 1.603, 1.998]), stop_gradient=False)

            cir = UAnsatz(2)
            cir.ry(theta[0], 0)
            cir.ry(theta[1], 1)
            cir.cnot([0, 1])
            cir.cnot([1, 0])
            cir.ry(theta[2], 0)
            cir.ry(theta[3], 1)
            cir.run_state_vector()

            gradients = cir.finite_difference_gradient(H, delta=0.01, shots=0)
            print(gradients)

        ::

            Tensor(shape=[4], dtype=float64, place=CPUPlace, stop_gradient=False,
                   [0.01951135, 0.56594233, 0.37991172, 0.35337436])
        """
        grad = []
        for i, theta_i in enumerate(self.__param):
            if theta_i.stop_gradient:
                continue
            self.__param[i] += delta / 2
            self.run_state_vector()
            expec_plu = self.expecval(H, shots)
            self.__param[i] -= delta
            self.run_state_vector()
            expec_min = self.expecval(H, shots)
            self.__param[i] += delta / 2
            self.run_state_vector()
            grad.append(paddle.to_tensor((expec_plu - expec_min) / delta, 'float64'))
            self.__param[i].stop_gradient = False
        grad = paddle.concat(grad)
        grad.stop_gradient = False

        return grad

    def param_shift_gradient(self, H, shots=0):
        r"""用 parameter-shift 方法计算电路中参数的梯度。损失函数默认为计算哈密顿量的期望值。

        Args:
            H (list or Hamiltonian): 记录哈密顿量信息的列表或 ``Hamiltonian`` 类的对象
            shots (int, optional): 测量次数；默认为 0，表示返回期望值的精确值，即测量无穷次后的期望值

        Returns:
            Tensor: 电路中所有可训练参数的梯度

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz

            H = [[1.0, 'z0,z1']]
            theta = paddle.to_tensor(np.array([6.186, 5.387, 1.603, 1.998]), stop_gradient=False)

            cir = UAnsatz(2)
            cir.ry(theta[0], 0)
            cir.ry(theta[1], 1)
            cir.cnot([0, 1])
            cir.cnot([1, 0])
            cir.ry(theta[2], 0)
            cir.ry(theta[3], 1)
            cir.run_state_vector()

            gradients = cir.param_shift_gradient(H, shots=0)
            print(gradients)

        ::

            Tensor(shape=[4], dtype=float64, place=CPUPlace, stop_gradient=False,
                   [0.01951143, 0.56594470, 0.37991331, 0.35337584])
        """
        r = 1 / 2
        grad = []
        for i, theta_i in enumerate(self.__param):
            if theta_i.stop_gradient:
                continue
            self.__param[i] += np.pi / (4 * r)
            self.run_state_vector()
            f_plu = self.expecval(H, shots)
            self.__param[i] -= 2 * np.pi / (4 * r)
            self.run_state_vector()
            f_min = self.expecval(H, shots)
            self.__param[i] += np.pi / (4 * r)
            self.run_state_vector()
            grad.append(paddle.to_tensor(r * (f_plu - f_min), 'float64'))
            self.__param[i].stop_gradient = False
        grad = paddle.concat(grad)
        grad.stop_gradient = False

        return grad

    def get_param(self):
        r"""得到电路参数列表中的可训练的参数。

        Returns:
            list: 电路中所有可训练的参数
        """
        param = []
        for theta in self.__param:
            if not theta.stop_gradient:
                param.append(theta)
        assert len(param) != 0, "circuit does not contain trainable parameters"
        param = paddle.concat(param)
        param.stop_gradient = False
        return param

    def update_param(self, new_param):
        r"""用得到的新参数列表更新电路参数列表中的可训练的参数。
        
        Args:
            new_param (list): 新的参数列表

        Returns:
            Tensor: 更新后电路中所有训练的参数
        """
        j = 0
        for i in range(len(self.__param)):
            if not self.__param[i].stop_gradient:
                self.__param[i] = paddle.to_tensor(new_param[j], 'float64')
                self.__param[i].stop_gradient = False
                j += 1
        self.run_state_vector()
        return self.__param

    """
    Channels
    """

    @apply_channel
    def amplitude_damping(self, gamma, which_qubit):
        r"""添加振幅阻尼信道。

        其 Kraus 算符为：

        .. math::

            E_0 =
            \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
            \end{bmatrix},
            E_1 =
            \begin{bmatrix}
                0 & \sqrt{\gamma} \\
                0 & 0
            \end{bmatrix}.

        Args:
            gamma (float): 减振概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            gamma = 0.1
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.amplitude_damping(gamma, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.5       +0.j 0.        +0.j 0.        +0.j 0.47434165+0.j]
             [0.        +0.j 0.05      +0.j 0.        +0.j 0.        +0.j]
             [0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j]
             [0.47434165+0.j 0.        +0.j 0.        +0.j 0.45      +0.j]]
        """
        assert 0 <= gamma <= 1, 'the parameter gamma should be in range [0, 1]'

        e0 = paddle.to_tensor([[1, 0], [0, np.sqrt(1 - gamma)]], dtype='complex128')
        e1 = paddle.to_tensor([[0, np.sqrt(gamma)], [0, 0]], dtype='complex128')

        return [e0, e1]

    @apply_channel
    def generalized_amplitude_damping(self, gamma, p, which_qubit):
        r"""添加广义振幅阻尼信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{p}
            \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
            \end{bmatrix},
            E_1 = \sqrt{p} \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix},\\
            E_2 = \sqrt{1-p} \begin{bmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{bmatrix},
            E_3 = \sqrt{1-p} \begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}.

        Args:
            gamma (float): 减振概率，其值应该在 :math:`[0, 1]` 区间内
            p (float): 激发概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            gamma = 0.1
            p = 0.2
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.generalized_amplitude_damping(gamma, p, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.46      +0.j 0.        +0.j 0.        +0.j 0.47434165+0.j]
             [0.        +0.j 0.01      +0.j 0.        +0.j 0.        +0.j]
             [0.        +0.j 0.        +0.j 0.04      +0.j 0.        +0.j]
             [0.47434165+0.j 0.        +0.j 0.        +0.j 0.49      +0.j]]
        """
        assert 0 <= gamma <= 1, 'the parameter gamma should be in range [0, 1]'
        assert 0 <= p <= 1, 'The parameter p should be in range [0, 1]'

        e0 = paddle.to_tensor(np.sqrt(p) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype='complex128'))
        e1 = paddle.to_tensor(np.sqrt(p) * np.array([[0, np.sqrt(gamma)], [0, 0]]), dtype='complex128')
        e2 = paddle.to_tensor(np.sqrt(1 - p) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]], dtype='complex128'))
        e3 = paddle.to_tensor(np.sqrt(1 - p) * np.array([[0, 0], [np.sqrt(gamma), 0]]), dtype='complex128')

        return [e0, e1, e2, e3]

    @apply_channel
    def phase_damping(self, gamma, which_qubit):
        r"""添加相位阻尼信道。

        其 Kraus 算符为：

        .. math::

            E_0 =
            \begin{bmatrix}
                1 & 0 \\
                0 & \sqrt{1-\gamma}
            \end{bmatrix},
            E_1 =
            \begin{bmatrix}
                0 & 0 \\
                0 & \sqrt{\gamma}
            \end{bmatrix}.

        Args:
            gamma (float): phase damping 信道的参数，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            p = 0.1
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.phase_damping(p, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.5       +0.j 0.        +0.j 0.        +0.j 0.47434165+0.j]
             [0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j]
             [0.        +0.j 0.        +0.j 0.        +0.j 0.        +0.j]
             [0.47434165+0.j 0.        +0.j 0.        +0.j 0.5       +0.j]]
        """
        assert 0 <= gamma <= 1, 'the parameter gamma should be in range [0, 1]'

        e0 = paddle.to_tensor([[1, 0], [0, np.sqrt(1 - gamma)]], dtype='complex128')
        e1 = paddle.to_tensor([[0, 0], [0, np.sqrt(gamma)]], dtype='complex128')

        return [e0, e1]

    @apply_channel
    def bit_flip(self, p, which_qubit):
        r"""添加比特反转信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{1-p} I,
            E_1 = \sqrt{p} X.

        Args:
            p (float): 发生 bit flip 的概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            p = 0.1
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.bit_flip(p, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.45+0.j 0.  +0.j 0.  +0.j 0.45+0.j]
             [0.  +0.j 0.05+0.j 0.05+0.j 0.  +0.j]
             [0.  +0.j 0.05+0.j 0.05+0.j 0.  +0.j]
             [0.45+0.j 0.  +0.j 0.  +0.j 0.45+0.j]]
        """
        assert 0 <= p <= 1, 'the probability p of a bit flip should be in range [0, 1]'

        e0 = paddle.to_tensor([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]], dtype='complex128')
        e1 = paddle.to_tensor([[0, np.sqrt(p)], [np.sqrt(p), 0]], dtype='complex128')

        return [e0, e1]

    @apply_channel
    def phase_flip(self, p, which_qubit):
        r"""添加相位反转信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{1 - p} I,
            E_1 = \sqrt{p} Z.

        Args:
            p (float): 发生 phase flip 的概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            p = 0.1
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.phase_flip(p, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.5+0.j 0. +0.j 0. +0.j 0.4+0.j]
             [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
             [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
             [0.4+0.j 0. +0.j 0. +0.j 0.5+0.j]]
        """
        assert 0 <= p <= 1, 'the probability p of a phase flip should be in range [0, 1]'

        e0 = paddle.to_tensor([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]], dtype='complex128')
        e1 = paddle.to_tensor([[np.sqrt(p), 0], [0, -np.sqrt(p)]], dtype='complex128')

        return [e0, e1]

    @apply_channel
    def bit_phase_flip(self, p, which_qubit):
        r"""添加比特相位反转信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{1 - p} I,
            E_1 = \sqrt{p} Y.

        Args:
            p (float): 发生 bit phase flip 的概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            p = 0.1
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.bit_phase_flip(p, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[ 0.45+0.j  0.  +0.j  0.  +0.j  0.45+0.j]
             [ 0.  +0.j  0.05+0.j -0.05+0.j  0.  +0.j]
             [ 0.  +0.j -0.05+0.j  0.05+0.j  0.  +0.j]
             [ 0.45+0.j  0.  +0.j  0.  +0.j  0.45+0.j]]
        """
        assert 0 <= p <= 1, 'the probability p of a bit phase flip should be in range [0, 1]'

        e0 = paddle.to_tensor([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]], dtype='complex128')
        e1 = paddle.to_tensor([[0, -1j * np.sqrt(p)], [1j * np.sqrt(p), 0]], dtype='complex128')

        return [e0, e1]

    @apply_channel
    def depolarizing(self, p, which_qubit):
        r"""添加去极化信道。

        其 Kraus 算符为：

        .. math::

            E_0 = \sqrt{1-p} I,
            E_1 = \sqrt{p/3} X,
            E_2 = \sqrt{p/3} Y,
            E_3 = \sqrt{p/3} Z.

        Args:
            p (float): depolarizing 信道的参数，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            p = 0.1
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.depolarizing(p, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.46666667+0.j 0.        +0.j 0.        +0.j 0.43333333+0.j]
             [0.        +0.j 0.03333333+0.j 0.        +0.j 0.        +0.j]
             [0.        +0.j 0.        +0.j 0.03333333+0.j 0.        +0.j]
             [0.43333333+0.j 0.        +0.j 0.        +0.j 0.46666667+0.j]]
        """
        assert 0 <= p <= 1, 'the parameter p should be in range [0, 1]'

        e0 = paddle.to_tensor([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]], dtype='complex128')
        e1 = paddle.to_tensor([[0, np.sqrt(p / 3)], [np.sqrt(p / 3), 0]], dtype='complex128')
        e2 = paddle.to_tensor([[0, -1j * np.sqrt(p / 3)], [1j * np.sqrt(p / 3), 0]], dtype='complex128')
        e3 = paddle.to_tensor([[np.sqrt(p / 3), 0], [0, -np.sqrt(p / 3)]], dtype='complex128')

        return [e0, e1, e2, e3]

    @apply_channel
    def pauli_channel(self, p_x, p_y, p_z, which_qubit):
        r"""添加泡利信道。

        Args:
            p_x (float): 泡利矩阵 X 的对应概率，其值应该在 :math:`[0, 1]` 区间内
            p_y (float): 泡利矩阵 Y 的对应概率，其值应该在 :math:`[0, 1]` 区间内
            p_z (float): 泡利矩阵 Z 的对应概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        Note:
            三个输入的概率加起来需要小于等于 1。

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            p_x = 0.1
            p_y = 0.2
            p_z = 0.3
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.pauli_channel(p_x, p_y, p_z, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[ 0.35+0.j  0.  +0.j  0.  +0.j  0.05+0.j]
             [ 0.  +0.j  0.15+0.j -0.05+0.j  0.  +0.j]
             [ 0.  +0.j -0.05+0.j  0.15+0.j  0.  +0.j]
             [ 0.05+0.j  0.  +0.j  0.  +0.j  0.35+0.j]]
        """
        prob_list = [p_x, p_y, p_z]
        assert sum(prob_list) <= 1, 'the sum of probabilities should be smaller than or equal to 1 '
        X = np.array([[0, 1], [1, 0]], dtype='complex128')
        Y = np.array([[0, -1j], [1j, 0]], dtype='complex128')
        Z = np.array([[1, 0], [0, -1]], dtype='complex128')
        I = np.array([[1, 0], [0, 1]], dtype='complex128')

        op_list = [X, Y, Z]
        for i, prob in enumerate(prob_list):
            assert 0 <= prob <= 1, 'the parameter p' + str(i + 1) + ' should be in range [0, 1]'
            op_list[i] = paddle.to_tensor(np.sqrt(prob_list[i]) * op_list[i])
        op_list.append(paddle.to_tensor(np.sqrt(1 - sum(prob_list)) * I))

        return op_list

    @apply_channel
    def reset(self, p, q, which_qubit):
        r"""添加重置信道。有 p 的概率将量子态重置为 :math:`|0\rangle` 并有 q 的概率重置为 :math:`|1\rangle`。

        其 Kraus 算符为：

        .. math::

            E_0 =
            \begin{bmatrix}
                \sqrt{p} & 0 \\
                0 & 0
            \end{bmatrix},
            E_1 =
            \begin{bmatrix}
                0 & \sqrt{p} \\
                0 & 0
            \end{bmatrix},\\
            E_2 =
            \begin{bmatrix}
                0 & 0 \\
                \sqrt{q} & 0
            \end{bmatrix},
            E_3 =
            \begin{bmatrix}
                0 & 0 \\
                0 & \sqrt{q}
            \end{bmatrix},\\
            E_4 = \sqrt{1-p-q} I.

        Args:
            p (float): 重置为 :math:`|0\rangle`的概率，其值应该在 :math:`[0, 1]` 区间内
            q (float): 重置为 :math:`|1\rangle`的概率，其值应该在 :math:`[0, 1]` 区间内
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        Note:
            两个输入的概率加起来需要小于等于 1。

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            p = 1
            q = 0
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.reset(p, q, 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.5+0.j 0. +0.j 0. +0.j 0. +0.j]
             [0. +0.j 0.5+0.j 0. +0.j 0. +0.j]
             [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
             [0. +0.j 0. +0.j 0. +0.j 0. +0.j]]
        """
        assert p + q <= 1, 'the sum of probabilities should be smaller than or equal to 1 '

        e0 = paddle.to_tensor([[np.sqrt(p), 0], [0, 0]], dtype='complex128')
        e1 = paddle.to_tensor([[0, np.sqrt(p)], [0, 0]], dtype='complex128')
        e2 = paddle.to_tensor([[0, 0], [np.sqrt(q), 0]], dtype='complex128')
        e3 = paddle.to_tensor([[0, 0], [0, np.sqrt(q)]], dtype='complex128')
        e4 = paddle.to_tensor([[np.sqrt(1 - (p + q)), 0], [0, np.sqrt(1 - (p + q))]], dtype='complex128')

        return [e0, e1, e2, e3, e4]

    @apply_channel
    def thermal_relaxation(self, t1, t2, time, which_qubit):
        r"""添加热弛豫信道，模拟超导硬件上的 T1 和 T2 混合过程。

        Args:
            t1 (float): :math:`T_1` 过程的弛豫时间常数，单位是微秒
            t2 (float): :math:`T_2` 过程的弛豫时间常数，单位是微秒
            time (float): 弛豫过程中量子门的执行时间，单位是纳秒
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        Note:
            时间常数必须满足 :math:`T_2 \le T_1`，参考文献 https://arxiv.org/abs/2101.02109

        代码示例:

        .. code-block:: python

            from paddle_quantum.circuit import UAnsatz
            N = 2
            t1 = 30
            t2 = 20
            tg = 200
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.thermal_relaxation(t1, t2, tg, 0)
            cir.thermal_relaxation(t1, t2, tg, 1)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.5   +0.j 0.    +0.j 0.    +0.j 0.4901+0.j]
             [0.    +0.j 0.0033+0.j 0.    +0.j 0.    +0.j]
             [0.    +0.j 0.    +0.j 0.0033+0.j 0.    +0.j]
             [0.4901+0.j 0.    +0.j 0.    +0.j 0.4934+0.j]]

        """
        assert 0 <= t2 <= t1, 'Relaxation time constants are not valid as 0 <= T2 <= T1!'
        assert 0 <= time, 'Invalid gate time!'

        # Change time scale
        time = time / 1000
        # Probability of resetting the state to |0>
        p_reset = 1 - np.exp(-time / t1)
        # Probability of phase flip
        p_z = (1 - p_reset) * (1 - np.exp(-time / t2) * np.exp(time / t1)) / 2
        # Probability of identity
        p_i = 1 - p_reset - p_z

        e0 = paddle.to_tensor([[np.sqrt(p_i), 0], [0, np.sqrt(p_i)]], dtype='complex128')
        e1 = paddle.to_tensor([[np.sqrt(p_z), 0], [0, -np.sqrt(p_z)]], dtype='complex128')
        e2 = paddle.to_tensor([[np.sqrt(p_reset), 0], [0, 0]], dtype='complex128')
        e3 = paddle.to_tensor([[0, np.sqrt(p_reset)], [0, 0]], dtype='complex128')

        return [e0, e1, e2, e3]

    @apply_channel
    def customized_channel(self, ops, which_qubit):
        r"""添加自定义的量子信道。

        Args:
            ops (list): 表示信道的 Kraus 算符的列表
            which_qubit (int): 该信道作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子电路的量子比特数

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            N = 2
            k1 = paddle.to_tensor([[1, 0], [0, 0]], dtype='complex128')
            k2 = paddle.to_tensor([[0, 0], [0, 1]], dtype='complex128')
            cir = UAnsatz(N)
            cir.h(0)
            cir.cnot([0, 1])
            cir.customized_channel([k1, k2], 0)
            final_state = cir.run_density_matrix()
            print(final_state.numpy())

        ::

            [[0.5+0.j 0. +0.j 0. +0.j 0. +0.j]
             [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
             [0. +0.j 0. +0.j 0. +0.j 0. +0.j]
             [0. +0.j 0. +0.j 0. +0.j 0.5+0.j]]
        """
        completeness = paddle.to_tensor([[0, 0], [0, 0]], dtype='complex128')
        for op in ops:
            assert isinstance(op, paddle.Tensor), 'The input operators should be Tensors.'
            assert op.shape == [2, 2], 'The shape of each operator should be [2, 2].'
            assert op.dtype.name == 'COMPLEX128', 'The dtype of each operator should be COMPLEX128.'
            completeness += matmul(dagger(op), op)
        assert np.allclose(completeness.numpy(),
                           np.eye(2, dtype='complex128')), 'Kraus operators should satisfy completeness.'

        return ops

    def shadow_trace(self, hamiltonian, sample_shots, method='CS'):
        r"""估计可观测量 :math:`H` 的期望值 :math:`\text{trace}(H\rho)` 。

        Args:
            hamiltonian (Hamiltonian): 可观测量
            sample_shots (int): 采样次数
            method (str, optional): 使用 shadow 来进行估计的方法，可选 "CS"、"LBCS"、"APS" 三种方法，默认为 "CS"

        代码示例:

        .. code-block:: python

            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.utils import Hamiltonian
            from paddle_quantum.state import vec_random

            n_qubit = 2
            sample_shots = 1000
            state = vec_random(n_qubit)
            ham = [[0.1, 'x1'], [0.2, 'y0']]
            ham = Hamiltonian(ham)

            cir = UAnsatz(n_qubit)
            input_state = cir.run_state_vector(paddle.to_tensor(state))
            trace_cs = cir.shadow_trace(ham, sample_shots, method="CS")
            trace_lbcs = cir.shadow_trace(ham, sample_shots, method="LBCS")
            trace_aps = cir.shadow_trace(ham, sample_shots, method="APS")

            print('trace CS = ', trace_cs)
            print('trace LBCS = ', trace_lbcs)
            print('trace APS = ', trace_aps)

        ::

            trace CS =  -0.09570000000000002
            trace LBCS =  -0.0946048044954126
            trace APS =  -0.08640438803809354
        """
        if not isinstance(hamiltonian, list):
            hamiltonian = hamiltonian.pauli_str
        state = self.__state
        num_qubits = self.n
        mode = self.__run_mode
        if method == "LBCS":
            result, beta = shadow.shadow_sample(state, num_qubits, sample_shots, mode, hamiltonian, method)
        else:
            result = shadow.shadow_sample(state, num_qubits, sample_shots, mode, hamiltonian, method)

        def prepare_hamiltonian(hamiltonian, num_qubits):
            r"""改写可观测量 ``[[0.3147,'y2'], [-0.5484158742278,'x2,z1'],...]`` 的形式

            Args:
                hamiltonian (list): 可观测量的相关信息
                num_qubits (int): 量子比特数目

            Returns:
                list: 可观测量的形式改写为[[0.3147,'iiy'], [-0.5484158742278,'izx'],...]

            Note:
                这是内部函数，你并不需要直接调用到该函数。
            """
            new_hamiltonian = list()
            for idx, (coeff, pauli_str) in enumerate(hamiltonian):
                pauli_str = re.split(r',\s*', pauli_str.lower())
                pauli_term = ['i'] * num_qubits
                for item in pauli_str:
                    if len(item) > 1:
                        pauli_term[int(item[1:])] = item[0]
                    elif item[0].lower() != 'i':
                        raise ValueError('Expecting I for ', item[0])
                new_term = [coeff, ''.join(pauli_term)]
                new_hamiltonian.append(new_term)
            return new_hamiltonian

        hamiltonian = prepare_hamiltonian(hamiltonian, num_qubits)

        sample_pauli_str = [item for item, _ in result]
        sample_measurement_result = [item for _, item in result]
        coeff_terms = list()
        pauli_terms = list()
        for coeff, pauli_term in hamiltonian:
            coeff_terms.append(coeff)
            pauli_terms.append(pauli_term)

        pauli2idx = {'x': 0, 'y': 1, 'z': 2}

        def estimated_weight_cs(sample_pauli_str, pauli_term):
            r"""定义 CS 算法中的对测量的权重估计函数

            Args:
                sample_pauli_str (str): 随机选择的 pauli 项
                pauli_term (str): 可观测量的 pauli 项

            Returns:
                int: 返回估计的权重值

            Note:
                这是内部函数，你并不需要直接调用到该函数。
            """
            result = 1
            for i in range(num_qubits):
                if sample_pauli_str[i] == 'i' or pauli_term[i] == 'i':
                    continue
                elif sample_pauli_str[i] == pauli_term[i]:
                    result *= 3
                else:
                    result = 0
            return result

        def estimated_weight_lbcs(sample_pauli_str, pauli_term, beta):
            r"""定义 LBCS 算法中的权重估计函数

            Args:
                sample_pauli_str (str): 随机选择的 pauli 项
                pauli_term (str): 可观测量的 pauli 项
                beta (list): 所有量子位上关于 pauli 的概率分布

            Returns:
                float: 返回函数数值

            Note:
                这是内部函数，你并不需要直接调用到该函数。
            """
            # beta is 2-d, and the shape looks like (len, 3)
            assert len(sample_pauli_str) == len(pauli_term)
            result = 1
            for i in range(num_qubits):
                # The probability distribution is different at each qubit
                score = 0
                idx = pauli2idx[sample_pauli_str[i]]
                if sample_pauli_str[i] == 'i' or pauli_term[i] == 'i':
                    score = 1
                elif sample_pauli_str[i] == pauli_term[i] and beta[i][idx] != 0:
                    score = 1 / beta[i][idx]
                result *= score
            return result

        def estimated_value(pauli_term, measurement_result):
            r"""满足条件的测量结果本征值的乘积

            Args:
                pauli_term (str): 可观测量的 pauli 项
                measurement_result (list): 测量结果

            Returns:
                int: 返回测量结果本征值的乘积

            Note:
                这是内部函数，你并不需要直接调用到该函数。
            """
            value = 1
            for idx in range(num_qubits):
                if pauli_term[idx] != 'i' and measurement_result[idx] == '1':
                    value *= -1
            return value

        # Define the functions required by APS
        def is_covered(pauli, pauli_str):
            r"""判断可观测量的 pauli 项是否被随机选择的 pauli 项所覆盖

            Args:
                pauli (str): 可观测量的 pauli 项
                pauli_str (str): 随机选择的 pauli 项

            Note:
                这是内部函数，你并不需要直接调用到该函数。
            """
            for qubit_idx in range(num_qubits):
                if not pauli[qubit_idx] in ('i', pauli_str[qubit_idx]):
                    return False
            return True

        def update_pauli_estimator(hamiltonian, pauli_estimator, pauli_str, measurement_result):
            r"""用于更新 APS 算法下当前可观测量 pauli 项 P 的最佳估计 tr( P \rho)，及 P 被覆盖的次数

            Args:
                hamiltonian (list): 可观测量的相关信息
                pauli_estimator (dict): 用于记录最佳估计与被覆盖次数
                pauli_str (list): 随机选择的 pauli 项
                measurement_result (list): 对随机选择的 pauli 项测量得到的结果

            Note:
                这是内部函数，你并不需要直接调用到该函数。
            """
            for coeff, pauli_term in hamiltonian:
                last_estimator = pauli_estimator[pauli_term]['value'][-1]
                if is_covered(pauli_term, pauli_str):
                    value = estimated_value(pauli_term, measurement_result)  
                    chose_number = pauli_estimator[pauli_term]['times']
                    new_estimator = 1 / (chose_number + 1) * (chose_number * last_estimator + value)
                    pauli_estimator[pauli_term]['times'] += 1
                    pauli_estimator[pauli_term]['value'].append(new_estimator)
                else:
                    pauli_estimator[pauli_term]['value'].append(last_estimator)

        trace_estimation = 0
        if method == "CS":
            for sample_idx in range(sample_shots):
                estimation = 0
                for i in range(len(pauli_terms)):
                    value = estimated_value(pauli_terms[i], sample_measurement_result[sample_idx])
                    weight = estimated_weight_cs(sample_pauli_str[sample_idx], pauli_terms[i])
                    estimation += coeff_terms[i] * weight * value
                trace_estimation += estimation
            trace_estimation /= sample_shots
        elif method == "LBCS":
            for sample_idx in range(sample_shots):
                estimation = 0
                for i in range(len(pauli_terms)):
                    value = estimated_value(pauli_terms[i], sample_measurement_result[sample_idx])
                    weight = estimated_weight_lbcs(sample_pauli_str[sample_idx], pauli_terms[i], beta)
                    estimation += coeff_terms[i] * weight * value
                trace_estimation += estimation
            trace_estimation /= sample_shots
        elif method == "APS":
            # Create a search dictionary for easy storage
            pauli_estimator = dict()
            for coeff, pauli_term in hamiltonian:
                pauli_estimator[pauli_term] = {'times': 0, 'value': [0]}
            for sample_idx in range(sample_shots):
                update_pauli_estimator(
                    hamiltonian,
                    pauli_estimator,
                    sample_pauli_str[sample_idx],
                    sample_measurement_result[sample_idx]
                )
            for sample_idx in range(sample_shots):
                estimation = 0
                for coeff, pauli_term in hamiltonian:
                    estimation += coeff * pauli_estimator[pauli_term]['value'][sample_idx + 1]
                trace_estimation = estimation

        return trace_estimation


def _local_H_prob(cir, hamiltonian, shots=1024):
    r"""
    构造出 Pauli 测量电路并测量 ancilla，处理实验结果来得到 ``H`` (只有一项)期望值的实验测量值。

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    # Add one ancilla, which we later measure and process the result
    new_cir = UAnsatz(cir.n + 1)
    input_state = paddle.kron(cir.run_state_vector(store_state=False), init_state_gen(1))
    # Used in fixed Rz gate
    _theta = paddle.to_tensor(np.array([-np.pi / 2]))

    op_list = hamiltonian.split(',')
    # Set up pauli measurement circuit
    for op in op_list:
        element = op[0]
        if len(op) > 1:
            index = int(op[1:])
        elif op[0].lower() != 'i':
            raise ValueError('Expecting {} to be {}'.format(op, 'I'))
        if element.lower() == 'x':
            new_cir.h(index)
            new_cir.cnot([index, cir.n])
        elif element.lower() == 'z':
            new_cir.cnot([index, cir.n])
        elif element.lower() == 'y':
            new_cir.rz(_theta, index)
            new_cir.h(index)
            new_cir.cnot([index, cir.n])

    new_cir.run_state_vector(input_state)
    prob_result = new_cir.measure(shots=shots, which_qubits=[cir.n])
    if shots > 0:
        if len(prob_result) == 1:
            if '0' in prob_result:
                result = (prob_result['0']) / shots
            else:
                result = -(prob_result['1']) / shots
        else:
            result = (prob_result['0'] - prob_result['1']) / shots
    else:
        result = (prob_result['0'] - prob_result['1'])

    return result


def swap_test(n):
    r"""构造用 Swap Test 测量两个量子态之间差异的电路。

    Args:
        n (int): 待比较的两个态的量子比特数

    Returns:
        UAnsatz: Swap Test 的电路

    代码示例:

    .. code-block:: python

        import paddle
        import numpy as np
        from paddle_quantum.state import vec
        from paddle_quantum.circuit import UAnsatz, swap_test
        from paddle_quantum.utils import NKron

        n = 2
        ancilla = vec(0, 1)
        psi = vec(1, n)
        phi = vec(0, n)
        input_state = NKron(ancilla, psi, phi)

        cir = swap_test(n)
        cir.run_state_vector(paddle.to_tensor(input_state))
        result = cir.measure(which_qubits=[0], shots=8192, plot=True)
        probability = result['0'] / 8192
        inner_product = (probability - 0.5) * 2
        print(f"The inner product is {inner_product}")

    ::

        The inner product is 0.006591796875
    """
    cir = UAnsatz(2 * n + 1)
    cir.h(0)
    for i in range(n):
        cir.cswap([0, i + 1, i + n + 1])
    cir.h(0)

    return cir
