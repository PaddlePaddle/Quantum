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

import gc
import math
from functools import reduce
from collections import defaultdict
import numpy as np
from numpy import binary_repr, eye, identity
import matplotlib.pyplot as plt

from Simulator.main import StateTranfer, init_state_gen, measure_state

import paddle
from paddle.complex import kron as pp_kron
from paddle.complex import reshape as complex_reshape
from paddle.complex import matmul, transpose, trace
from paddle.complex.tensor.math import elementwise_mul

import paddle.fluid as fluid
from paddle.fluid import dygraph
from paddle.fluid.layers import reshape, cast, eye, zeros
from paddle.fluid.framework import ComplexVariable

from paddle_quantum.utils import hermitian, pauli_str_to_matrix
from paddle_quantum.intrinsic import *
from paddle_quantum.state import density_op

__all__ = [
    "UAnsatz",
    "H_prob"
]


class UAnsatz:
    r"""基于Paddle的动态图机制实现量子线路的 ``class`` 。

    用户可以通过实例化该 ``class`` 来搭建自己的量子线路。

    Attributes:
        n (int): 该线路的量子比特数
    """

    def __init__(self, n):
        r"""UAnsatz的构造函数，用于实例化一个UAnsatz对象

        Args:
            n (int): 该线路的量子比特数

        """
        self.n = n
        self.__state = None
        self.__run_state = ''
        # Record history of adding gates to the circuit
        self.__history = []

    def run_state_vector(self, input_state=None, store_state=True):
        r"""运行当前的量子线路，输入输出的形式为态矢量。
        
        Args:
            input_state (ComplexVariable, optional): 输入的态矢量，默认为 :math:`|00...0\rangle`
            store_state (Bool, optional): 是否存储输出的态矢量，默认为 ``True`` ，即存储
        
        Returns:
            ComplexVariable: 量子线路输出的态矢量
        
        代码示例:

        .. code-block:: python
        
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            n = 2
            theta = np.ones(3)
            input_state = np.ones(2**n)+0j
            input_state = input_state / np.linalg.norm(input_state)
            with fluid.dygraph.guard():
                
                input_state_var = fluid.dygraph.to_variable(input_state)
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.rx(theta[0], 0)
                cir.ry(theta[1], 1)
                cir.rz(theta[2], 1)
                vec = cir.run_state_vector(input_state_var).numpy()
                print(f"运行后的向量是 {vec}")

        ::

            运行后的向量是 [0.17470783-0.09544332j 0.59544332+0.32529217j 0.17470783-0.09544332j 0.59544332+0.32529217j]
        """
        state = init_state_gen(self.n, 0) if input_state is None else input_state
        old_shape = state.shape
        assert reduce(lambda x, y: x * y, old_shape) == 2 ** self.n, 'The length of the input vector is not right'
        state = complex_reshape(state, (2 ** self.n,))

        state_conj = ComplexVariable(state.real, -state.imag)
        assert fluid.layers.abs(paddle.complex.sum(elementwise_mul(state_conj, state)).real - 1) < 1e-8, \
            'Input state is not a normalized vector'

        for history_ele in self.__history:
            if history_ele[0] == 'u':
                state = StateTranfer(state, 'u', history_ele[1], history_ele[2])
            elif history_ele[0] in {'x', 'y', 'z', 'h'}:
                state = StateTranfer(state, history_ele[0], history_ele[1], params=history_ele[2])
            elif history_ele[0] == 'CNOT':
                state = StateTranfer(state, 'CNOT', history_ele[1])

        if store_state:
            self.__state = state
            # Add info about which function user called
            self.__run_state = 'state_vector'

        return complex_reshape(state, old_shape)

    def run_density_matrix(self, input_state=None, store_state=True):
        r"""运行当前的量子线路，输入输出的形式为密度矩阵。
        
        Args:
            input_state (ComplexVariable, optional): 输入的密度矩阵，默认为 :math:`|00...0\rangle \langle00...0|`
            store_state (bool, optional): 是否存储输出的密度矩阵，默认为 ``True`` ，即存储
        
        Returns:
            ComplexVariable: 量子线路输出的密度矩阵

        代码示例:

        .. code-block:: python
        
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            n = 1
            theta = np.ones(3)
            input_state = np.diag(np.arange(2**n))+0j
            input_state = input_state / np.trace(input_state)
            with fluid.dygraph.guard():
                
                input_state_var = fluid.dygraph.to_variable(input_state)
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.rx(theta[0], 0)
                cir.ry(theta[1], 0)
                cir.rz(theta[2], 0)
                density = cir.run_density_matrix(input_state_var).numpy()
                print(f"密度矩阵是\n{density}")

        ::

            密度矩阵是
            [[ 0.35403671+0.j         -0.47686058-0.03603751j]
            [-0.47686058+0.03603751j  0.64596329+0.j        ]]
        """
        state = dygraph.to_variable(density_op(self.n)) if input_state is None else input_state

        assert state.real.shape == [2 ** self.n, 2 ** self.n], "The dimension is not right"
        state = matmul(self.U, matmul(state, hermitian(self.U)))

        if store_state:
            self.__state = state
            # Add info about which function user called
            self.__run_state = 'density_matrix'

        return state

    @property
    def U(self):
        r"""量子线路的酉矩阵形式。
        
        Returns:
            ComplexVariable: 当前线路的酉矩阵表示

        代码示例:

        .. code-block:: python
        
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            n = 2
            with fluid.dygraph.guard():
                cir = UAnsatz(2)
                cir.h(0)
                cir.cnot([0, 1])
                matrix = cir.U
                print("生成贝尔态电路的酉矩阵表示为\n",matrix.numpy())

        ::

            生成贝尔态电路的酉矩阵表示为
            [[ 0.70710678+0.j  0.        +0.j  0.70710678+0.j  0.        +0.j]
            [ 0.        +0.j  0.70710678+0.j  0.        +0.j  0.70710678+0.j]
            [ 0.        +0.j  0.70710678+0.j  0.        +0.j -0.70710678+0.j]
            [ 0.70710678+0.j  0.        +0.j -0.70710678+0.j  0.        +0.j]]
        """
        state = ComplexVariable(eye(2 ** self.n, dtype='float64'), zeros([2 ** self.n, 2 ** self.n], dtype='float64'))
        shape = (2 ** self.n, 2 ** self.n)
        num_ele = reduce(lambda x, y: x * y, shape)
        state = ComplexVariable(reshape(state.real, [num_ele]), reshape(state.imag, [num_ele]))

        for history_ele in self.__history:
            if history_ele[0] == 'u':
                state = StateTranfer(state, 'u', history_ele[1], history_ele[2])
            elif history_ele[0] in {'x', 'y', 'z', 'h'}:
                state = StateTranfer(state, history_ele[0], history_ele[1], params=history_ele[2])
            elif history_ele[0] == 'CNOT':
                state = StateTranfer(state, 'CNOT', history_ele[1])

        return ComplexVariable(reshape(state.real, shape), reshape(state.imag, shape))

    """
    Common Gates
    """

    def rx(self, theta, which_qubit):
        r"""添加关于x轴的单量子比特旋转门。
 
        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}
 
        Args:
            theta (Variable): 旋转角度
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
 
        ..  code-block:: python

            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 0
                cir.rx(theta[0], which_qubit)
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [theta,
                                                    dygraph.to_variable(np.array([-math.pi / 2])),
                                                    dygraph.to_variable(np.array([math.pi / 2]))]])

    def ry(self, theta, which_qubit):
        r"""添加关于y轴的单量子比特旋转门。

        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Variable): 旋转角度
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数

        ..  code-block:: python
        
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 0
                cir.ry(theta[0], which_qubit)
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [theta,
                                                    dygraph.to_variable(np.array([0.0])),
                                                    dygraph.to_variable(np.array([0.0]))]])

    def rz(self, theta, which_qubit):
        r"""添加关于y轴的单量子比特旋转门。
 
        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} e^{-\frac{i\theta}{2}} & 0 \\ 0 & e^{\frac{i\theta}{2}} \end{bmatrix}
 
        Args:
            theta (Variable): 旋转角度
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
 
        ..  code-block:: python
        
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            theta = np.array([np.pi], np.float64)
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 0
                cir.ry(theta[0], which_qubit)
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [dygraph.to_variable(np.array([0.0])),
                                                    dygraph.to_variable(np.array([0.0])),
                                                    theta]])

    def cnot(self, control):
        r"""添加一个CNOT门。
 
        对于2量子比特的量子线路，当control为 ``[0, 1]`` 时，其矩阵形式为：
        
        .. math::
        
            \begin{align}
            CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
            &=\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}
            \end{align}
 
        Args:
            control (list): 作用在的qubit的编号，``control[0]`` 为控制位，``control[1]`` 为目标位，其值都应该在 :math:`[0, n)`范围内， :math:`n` 为该量子线路的量子比特数

        ..  code-block:: python
        
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            with fluid.dygraph.guard():
                cir = UAnsatz(num_qubits)
                cir.cnot([0, 1])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n,\
            "the qubit should >= 0 and < n(the number of qubit)"
        assert control[0] != control[1], "the control qubit is the same as the target qubit"
        self.__history.append(['CNOT', control])

    def x(self, which_qubit):
        r"""添加单量子比特X门。
 
        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
 
        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
 
        .. code-block:: python
            
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            with fluid.dygraph.guard():
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 0
                cir.x(which_qubit)
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['x', [which_qubit], None])

    def y(self, which_qubit):
        r"""添加单量子比特Y门。
 
        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}
 
        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
 
        .. code-block:: python
            
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            with fluid.dygraph.guard():
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 0
                cir.y(which_qubit)
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['y', [which_qubit], None])

    def z(self, which_qubit):
        r"""添加单量子比特Z门。
 
        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}
 
        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
 
        .. code-block:: python
        
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            with fluid.dygraph.guard():
                num_qubits = 1
                cir = UAnsatz(num_qubits)
                which_qubit = 0
                cir.z(which_qubit)
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['z', [which_qubit], None])

    def h(self, which_qubit):
        r"""添加一个单量子比特的Hadamard门。

        具体形式为

        .. math::
        
            H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1&1\\1&-1 \end{bmatrix}

        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['h', [which_qubit], None])

    def s(self, which_qubit):
        r"""添加一个单量子比特的S门。

        具体形式为

        .. math::
        
            S = \begin{bmatrix} 1&0\\0&i \end{bmatrix}

        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [dygraph.to_variable(np.array([0.0])),
                                                    dygraph.to_variable(np.array([0.0])),
                                                    dygraph.to_variable(np.array([math.pi / 2]))]])

    def t(self, which_qubit):
        r"""添加一个单量子比特的T门。

        具体形式为

        .. math::

            T = \begin{bmatrix} 1&0\\0&e^\frac{i\pi}{4} \end{bmatrix}

        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [dygraph.to_variable(np.array([0.0])),
                                                    dygraph.to_variable(np.array([0.0])),
                                                    dygraph.to_variable(np.array([math.pi / 4]))]])

    def u3(self, theta, phi, lam, which_qubit):
        r"""添加一个单量子比特的旋转门。

        具体形式为

        .. math::
        
            \begin{align}
            U3(\theta, \phi, \lambda) =
            \begin{bmatrix}
                \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
            \end{bmatrix}
            \end{align}

        Args:
              theta (Variable): 旋转角度 :math:`\theta` 。
              phi (Variable): 旋转角度 :math:`\phi` 。
              lam (Variable): 旋转角度 :math:`\lambda` 。
              which_qubit (int): 作用在的qubit的编号，其值应该在[0, n)范围内，n为该量子线路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [theta, phi, lam]])

    """
    Measurements
    """

    def __process_string(self, s, which_qubits):
        r"""
        This functions return part of string s baesd on which_qubits
        If s = 'abcdefg', which_qubits = [0,2,5], then it returns 'acf'

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        new_s = ''.join(s[j] for j in which_qubits)
        return new_s

    def __process_similiar(self, result):
        r"""
        This functions merges values based on identical keys.
        If result = [('00', 10), ('01', 20), ('11', 30), ('11', 40), ('11', 50), ('00', 60)], then it returns {'00': 70, '01': 20, '11': 120}

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
        r"""对量子线路输出的量子态进行测量。

        Warning:
            当plot为True时，当前量子线路的量子比特数需要小于6，否则无法绘制图片，会抛出异常。

        Args:
            which_qubits (list, optional): 要测量的qubit的编号，默认全都测量
            shots (int, optional): 该量子线路输出的量子态的测量次数，默认为1024次；若为0，则输出测量期望值的精确值
            plot (bool, optional): 是否绘制测量结果图，默认为 ``False`` ，即不绘制
        
        Returns:
            dict: 测量的结果

        代码示例:

        .. code-block:: python
        
            import paddle
            from paddle_quantum.circuit import UAnsatz
            with paddle.fluid.dygraph.guard():
                cir = UAnsatz(2)
                cir.h(0)
                cir.cnot([0,1])
                cir.run_state_vector()
                result = cir.measure(shots = 2048, which_qubits = [1])
                print(f"测量第1号量子比特2048次的结果是{result}")

        ::

            测量第1号量子比特2048次的结果是{'0': 964, '1': 1084}

        .. code-block:: python
        
            import paddle
            from paddle_quantum.circuit import UAnsatz
            with paddle.fluid.dygraph.guard():
                cir = UAnsatz(2)
                cir.h(0)
                cir.cnot([0,1])
                cir.run_state_vector()
                result = cir.measure(shots = 0, which_qubits = [1])
                print(f"测量第1号量子比特的概率结果是{result}")

        ::

            测量第1号量子比特的概率结果是{'0': 0.4999999999999999, '1': 0.4999999999999999}
        """
        if self.__run_state == 'state_vector':
            state = self.__state
        elif self.__run_state == 'density_matrix':
            # Take the diagonal of the density matrix as a probability distribution
            diag = np.diag(self.__state.numpy())
            state = fluid.dygraph.to_variable(np.sqrt(diag))
        else:
            # Raise error
            raise ValueError("no state for measurement; please run the circuit first")

        if shots == 0:  # Returns probability distribution over all measurement results
            dic2to10, dic10to2 = dic_between2and10(self.n)
            result = {}
            for i in range(2 ** self.n):
                result[dic10to2[i]] = (state.real[i] ** 2 + state.imag[i] ** 2).numpy()[0]

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

    def expecval(self, H):
        r"""量子线路输出的量子态关于可观测量H的期望值。

        Hint:
            如果想输入的可观测量的矩阵为 :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I` 。则 ``H`` 应为 ``[[0.7, 'z0,x1'], [0.2, 'z1']]`` 。
        Args:
            H (list): 可观测量的相关信息
        Returns:
            Variable: 量子线路输出的量子态关于H的期望值

        代码示例:
        
        .. code-block:: python
            
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            n = 5
            H_info = [[0.1, 'x1'], [0.2, 'y0,z4']]
            theta = np.ones(3)
            input_state = np.ones(2**n)+0j
            input_state = input_state / np.linalg.norm(input_state)
            with fluid.dygraph.guard():
                input_state_var = fluid.dygraph.to_variable(input_state)
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.rx(theta[0], 0)
                cir.rz(theta[1], 1)
                cir.rx(theta[2], 2)
                cir.run_state_vector(input_state_var)
                expect_value = cir.expecval(H_info).numpy()
                print(f'计算得到的{H_info}期望值是{expect_value}')
        
        ::
        
            计算得到的[[0.1, 'x1'], [0.2, 'y0,z4']]期望值是[0.05403023]
        
        .. code-block:: python
            
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            n = 5
            H_info = [[0.1, 'x1'], [0.2, 'y0,z4']]
            theta = np.ones(3)
            input_state = np.diag(np.arange(2**n))+0j
            input_state = input_state / np.trace(input_state)
            with fluid.dygraph.guard():
                input_state_var = fluid.dygraph.to_variable(input_state)
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.rx(theta[0], 0)
                cir.ry(theta[1], 1)
                cir.rz(theta[2], 2)
                cir.run_density_matrix(input_state_var)
                expect_value = cir.expecval(H_info).numpy()
                print(f'计算得到的{H_info}期望值是{expect_value}')
        
        ::

            计算得到的[[0.1, 'x1'], [0.2, 'y0,z4']]期望值是[-0.02171538]
        """
        if self.__run_state == 'state_vector':
            return vec_expecval(H, self.__state).real
        elif self.__run_state == 'density_matrix':
            state = self.__state
            H_mat = fluid.dygraph.to_variable(pauli_str_to_matrix(H, self.n))
            return trace(matmul(state, H_mat)).real
        else:
            # Raise error
            raise ValueError("no state for measurement; please run the circuit first")

    """
    Circuit Templates
    """

    def superposition_layer(self):
        r"""添加一层Hadamard门。

        代码示例:

        .. code-block:: python
        
            import paddle
            from paddle_quantum.circuit import UAnsatz
            with paddle.fluid.dygraph.guard():
                cir = UAnsatz(2)
                cir.superposition_layer()
                cir.run_state_vector()
                result = cir.measure(shots = 0)
                print(f"测量全部量子比特的结果是{result}")

        ::

            测量全部量子比特结果是{'00': 0.2499999999999999, '01': 0.2499999999999999, '10': 0.2499999999999999, '11': 0.2499999999999999}
        """
        for i in range(self.n):
            self.h(i)

    def weak_superposition_layer(self):
        r"""添加一层Hadamard的平方根门，即 :math:`\sqrt{H}` 门。

        代码示例:

        .. code-block:: python
        
            import paddle
            from paddle_quantum.circuit import UAnsatz
            with paddle.fluid.dygraph.guard():
                cir = UAnsatz(2)
                cir.weak_superposition_layer()
                cir.run_state_vector()
                result = cir.measure(shots = 0)
                print(f"测量全部量子比特的结果是{result}")

        ::

            测量全部量子比特的结果是{'00': 0.7285533905932737, '01': 0.12500000000000003, '10': 0.12500000000000003, '11': 0.021446609406726238}
        """
        _theta = fluid.dygraph.to_variable(np.array([np.pi / 4]))  # Used in fixed Ry gate
        for i in range(self.n):
            self.ry(_theta, i)

    def real_entangled_layer(self, theta, depth):
        r"""添加一层包含Ry门的强纠缠层。

        Note:
            这一层量子门的数学表示形式为实数酉矩阵。
        
        Attention:
            ``theta`` 的维度为 ``(depth, n, 1)``
        
        Args:
            theta (Variable): Ry门的旋转角度
            depth (int): 纠缠层的深度

        代码示例:

        .. code-block:: python
        
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            n = 2
            DEPTH = 3
            theta = np.ones([DEPTH, n, 1])
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.real_entangled_layer(fluid.dygraph.to_variable(theta), DEPTH)
                cir.run_state_vector()
                print(cir.measure(shots = 0))
        
        ::

            {'00': 2.52129874867343e-05, '01': 0.295456784923382, '10': 0.7045028818254718, '11': 1.5120263659845063e-05}
        """
        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 1, 'the shape of theta is not right'
        assert theta.shape[1] == self.n, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        for repeat in range(depth):
            for i in range(self.n):
                self.ry(theta=theta[repeat][i][0], which_qubit=i)
            for i in range(self.n - 1):
                self.cnot(control=[i, i + 1])
            self.cnot([self.n - 1, 0])

    def complex_entangled_layer(self, theta, depth):
        r"""添加一层包含U3门的强纠缠层。

        Note:
            这一层量子门的数学表示形式为复数酉矩阵。
        
        Attention:
            ``theta`` 的维度为 ``(depth, n, 3)`` ，最低维内容为对应的 ``u3`` 的参数 ``(theta, phi, lam)``
        
        Args:
            theta (Variable): U3门的旋转角度
            depth (int): 纠缠层的深度

        代码示例:

        .. code-block:: python
        
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            n = 2
            DEPTH = 3
            theta = np.ones([DEPTH, n, 1])
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.complex_entangled_layer(fluid.dygraph.to_variable(theta), DEPTH)
                cir.run_state_vector()
                print(cir.measure(shots = 0))
        
        ::

            {'00': 0.15032627279218896, '01': 0.564191201239618, '10': 0.03285998070292556, '11': 0.25262254526526823}
        """
        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 3, 'the shape of theta is not right'
        assert theta.shape[1] == self.n, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        for repeat in range(depth):
            for i in range(self.n):
                self.u3(theta[repeat][i][0], theta[repeat][i][1], theta[repeat][i][2], i)
            for i in range(self.n - 1):
                self.cnot([i, i + 1])
            self.cnot([self.n - 1, 0])

    def universal_2_qubit_gate(self, theta):
        r"""添加2-qubit通用门，这个通用门需要15个参数。

        Attention:
            只适用于量子比特数为2的量子线路。
        
        Args:
            theta (Variable): 2-qubit通用门的参数，其维度为 ``(15, )``

        代码示例:

        .. code-block:: python
        
            import numpy as np
            from paddle import fluid
            from paddle_quantum.circuit import UAnsatz
            n = 2
            theta = np.ones(15)
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.universal_2_qubit_gate(fluid.dygraph.to_variable(theta))
                cir.run_state_vector()
                print(cir.measure(shots = 0))
        
        ::

            {'00': 0.4306256106527819, '01': 0.07994547866706268, '10': 0.07994547866706264, '11': 0.40948343201309334}
        """
        assert self.n == 2, 'It only works on 2-qubit circuit'
        assert len(theta.shape) == 1, 'The shape of theta is not right'
        assert len(theta) == 15, 'This Ansatz accepts 15 parameters'

        self.u3(theta[0], theta[1], theta[2], 0)
        self.u3(theta[3], theta[4], theta[5], 1)

        self.cnot([1, 0])

        self.rz(theta[6], 0)
        self.ry(theta[7], 1)

        self.cnot([0, 1])

        self.ry(theta[8], 1)

        self.cnot([1, 0])

        self.u3(theta[9], theta[10], theta[11], 0)
        self.u3(theta[12], theta[13], theta[14], 1)

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
        Add a real layer on the circuit. theta is a two dimensional tensor. position is the qubit range the layer needs to cover

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert theta.shape[1] == 4 and theta.shape[0] == (position[1] - position[0] + 1) / 2,\
            'the shape of theta is not right'
        for i in range(position[0], position[1], 2):
            self.__add_real_block(theta[int((i - position[0]) / 2)], [i, i + 1])

    def __add_complex_layer(self, theta, position):
        r"""
        Add a complex layer on the circuit. theta is a two dimensional tensor. position is the qubit range the layer needs to cover

        Note:
            这是内部函数，你并不需要直接调用到该函数。
        """
        assert theta.shape[1] == 12 and theta.shape[0] == (position[1] - position[0] + 1) / 2,\
            'the shape of theta is not right'
        for i in range(position[0], position[1], 2):
            self.__add_complex_block(theta[int((i - position[0]) / 2)], [i, i + 1])

    def real_block_layer(self, theta, depth):
        r"""添加一层包含Ry门的弱纠缠层。

        Note:
            这一层量子门的数学表示形式为实数酉矩阵。
        
        Attention:
            ``theta`` 的维度为 ``(depth, n-1, 4)``
        
        Args:
            theta(Variable): Ry门的旋转角度
            depth(int): 纠缠层的深度

        代码示例:

        .. code-block:: python
        
            n = 4
            DEPTH = 3
            theta = np.ones([DEPTH, n-1, 4])  
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.real_block_layer(fluid.dygraph.to_variable(theta), DEPTH)
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
        r"""添加一层包含U3门的弱纠缠层。
        
        Note:
            这一层量子门的数学表示形式为复数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, n-1, 12)``
        
        Args:
            theta (Variable): U3门的角度信息
            depth (int): 纠缠层的深度

        代码示例:

        .. code-block:: python
        
            n = 4
            DEPTH = 3
            theta = np.ones([DEPTH, n-1, 12])  
            with fluid.dygraph.guard():
                theta = fluid.dygraph.to_variable(theta)
                cir = UAnsatz(n)
                cir.complex_block_layer(fluid.dygraph.to_variable(theta), DEPTH)
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


def local_H_prob(cir, hamiltonian, shots=1024):
    r"""
    构造出Pauli测量电路并测量ancilla，处理实验结果来得到 ``H`` (只有一项)期望值的实验测量值。

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    # Add one ancilla, which we later measure and process the result
    new_cir = UAnsatz(cir.n + 1)
    input_state = pp_kron(cir.run_state_vector(store_state=False), init_state_gen(1))
    # Used in fixed Rz gate
    _theta = fluid.dygraph.to_variable(np.array([-np.pi / 2]))

    op_list = hamiltonian.split(',')
    # Set up pauli measurement circuit
    for op in op_list:
        element = op[0]
        index = int(op[1:])
        if element == 'x':
            new_cir.h(index)
            new_cir.cnot([index, cir.n])
        elif element == 'z':
            new_cir.cnot([index, cir.n])
        elif element == 'y':
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
                result = (prob_result['1']) / shots
        else:
            result = (prob_result['0'] - prob_result['1']) / shots
    else:
        result = (prob_result['0'] - prob_result['1'])

    return result


def H_prob(cir, H, shots=1024):
    r"""构造Pauli测量电路并测量关于H的期望值。
    
    Args:
        cir (UAnsatz): UAnsatz的一个实例化对象
        H (list): 记录哈密顿量信息的列表
        shots (int, optional): 默认为1024，表示测量次数；若为0，则表示返回测量期望值的精确值，即测量无穷次后的期望值
    
    Returns:
        float: 测量得到的H的期望值
    
    代码示例:

    .. code-block:: python
        
        import numpy as np
        from paddle import fluid
        from paddle_quantum.circuit import UAnsatz, H_prob
        n = 4
        theta = np.ones(3)
        experiment_shots = 2**10
        H_info = [[0.1, 'x2'], [0.3, 'y1,z3']]
        input_state = np.ones(2**n)+0j
        input_state = input_state / np.linalg.norm(input_state)
        with fluid.dygraph.guard():
            theta = fluid.dygraph.to_variable(theta)
            cir = UAnsatz(n)
            cir.rx(theta[0], 0)
            cir.ry(theta[1], 1)
            cir.rz(theta[2], 1)
            result_1 = H_prob(cir, H_info, shots = experiment_shots)
            result_2 = H_prob(cir, H_info, shots = 0)
            print(f'消耗 {experiment_shots} 次测量后的期望值实验值是 {result_1}')
            print(f'H期望值精确值是 {result_2}')

    ::

        消耗 1024 次测量后的期望值实验值是 0.2326171875
        H期望值精确值是 0.21242202548207134
    """
    expval = 0
    for term in H:
        expval += term[0] * local_H_prob(cir, term[1], shots=shots)
    return expval
