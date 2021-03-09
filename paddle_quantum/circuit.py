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

import gc
import math
from functools import reduce
from collections import defaultdict
import numpy as np
from numpy import binary_repr, eye, identity
import matplotlib.pyplot as plt

from paddle_quantum.simulator import StateTransfer, init_state_gen, measure_state

import paddle
from paddle import kron as pp_kron
from paddle import reshape as complex_reshape
from paddle import matmul, transpose, trace, real, imag
from paddle import multiply

from paddle import reshape, cast, eye, zeros

from paddle_quantum.utils import dagger, pauli_str_to_matrix
from paddle_quantum.intrinsic import *
from paddle_quantum.state import density_op

__all__ = [
    "UAnsatz",
    "H_prob"
]


class UAnsatz:
    r"""基于 PaddlePaddle 的动态图机制实现量子线路的 ``class`` 。

    用户可以通过实例化该 ``class`` 来搭建自己的量子线路。

    Attributes:
        n (int): 该线路的量子比特数
    """

    def __init__(self, n):
        r"""UAnsatz 的构造函数，用于实例化一个 UAnsatz 对象

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
            input_state (Tensor, optional): 输入的态矢量，默认为 :math:`|00...0\rangle`
            store_state (Bool, optional): 是否存储输出的态矢量，默认为 ``True`` ，即存储
        
        Returns:
            Tensor: 量子线路输出的态矢量
        
        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            from paddle_quantum.state import vec
            n = 2
            theta = np.ones(3)

            input_state = paddle.to_tensor(vec(n))
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
        state = init_state_gen(self.n, 0) if input_state is None else input_state
        old_shape = state.shape
        assert reduce(lambda x, y: x * y, old_shape) == 2 ** self.n, 'The length of the input vector is not right'
        state = complex_reshape(state, (2 ** self.n,))

        state_conj = paddle.conj(state)
        assert paddle.abs(paddle.real(paddle.sum(multiply(state_conj, state))) - 1) < 1e-8, \
            'Input state is not a normalized vector'

        for history_ele in self.__history:
            if history_ele[0] == 'u':
                state = StateTransfer(state, 'u', history_ele[1], history_ele[2])
            elif history_ele[0] in {'x', 'y', 'z', 'h'}:
                state = StateTransfer(state, history_ele[0], history_ele[1], params=history_ele[2])
            elif history_ele[0] == 'SWAP':
                state = StateTransfer(state, 'SWAP', history_ele[1])
            elif history_ele[0] == 'CNOT':
                state = StateTransfer(state, 'CNOT', history_ele[1])

        if store_state:
            self.__state = state
            # Add info about which function user called
            self.__run_state = 'state_vector'

        return complex_reshape(state, old_shape)

    def run_density_matrix(self, input_state=None, store_state=True):
        r"""运行当前的量子线路，输入输出的形式为密度矩阵。
        
        Args:
            input_state (Tensor, optional): 输入的密度矩阵，默认为 :math:`|00...0\rangle \langle00...0|`
            store_state (bool, optional): 是否存储输出的密度矩阵，默认为 ``True`` ，即存储
        
        Returns:
            Tensor: 量子线路输出的密度矩阵

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

        assert paddle.real(state).shape == [2 ** self.n, 2 ** self.n], "The dimension is not right"
        state = matmul(self.U, matmul(state, dagger(self.U)))

        if store_state:
            self.__state = state
            # Add info about which function user called
            self.__run_state = 'density_matrix'

        return state

    @property
    def U(self):
        r"""量子线路的酉矩阵形式。
        
        Returns:
            Tensor: 当前线路的酉矩阵表示

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
        state = eye(2 ** self.n, dtype='float64')
        state = paddle.cast(state, 'complex128')

        shape = (2 ** self.n, 2 ** self.n)
        num_ele = reduce(lambda x, y: x * y, shape)
        state = paddle.reshape(state, [num_ele])

        for history_ele in self.__history:
            if history_ele[0] == 'u':
                state = StateTransfer(state, 'u', history_ele[1], history_ele[2])
            elif history_ele[0] in {'x', 'y', 'z', 'h'}:
                state = StateTransfer(state, history_ele[0], history_ele[1], params=history_ele[2])
            elif history_ele[0] == 'SWAP':
                state = StateTransfer(state, 'SWAP', history_ele[1])
            elif history_ele[0] == 'CNOT':
                state = StateTransfer(state, 'CNOT', history_ele[1])

        return paddle.reshape(state, shape)

    """
    Common Gates
    """

    def rx(self, theta, which_qubit):
        r"""添加关于 x 轴的单量子比特旋转门。

        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数

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
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [theta,
                                                    paddle.to_tensor(np.array([-math.pi / 2])),
                                                    paddle.to_tensor(np.array([math.pi / 2]))]])

    def ry(self, theta, which_qubit):
        r"""添加关于 y 轴的单量子比特旋转门。

        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数

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
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [theta,
                                                    paddle.to_tensor(np.array([0.0])),
                                                    paddle.to_tensor(np.array([0.0]))]])

    def rz(self, theta, which_qubit):
        r"""添加关于 z 轴的单量子比特旋转门。

        其矩阵形式为：
        
        .. math::

            \begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数

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
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [paddle.to_tensor(np.array([0.0])),
                                                    paddle.to_tensor(np.array([0.0])),
                                                    theta]])

    def cnot(self, control):
        r"""添加一个 CNOT 门。

        对于 2 量子比特的量子线路，当 ``control`` 为 ``[0, 1]`` 时，其矩阵形式为：

        .. math::
        
            \begin{align}
            CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
            &=\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的 qubit 的编号，``control[0]`` 为控制位，``control[1]`` 为目标位，其值都应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数

        ..  code-block:: python
        
            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.cnot([0, 1])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n,\
            "the qubit should >= 0 and < n(the number of qubit)"
        assert control[0] != control[1], "the control qubit is the same as the target qubit"
        self.__history.append(['CNOT', control])

    def swap(self, control):
        r"""添加一个 SWAP 门。

        其矩阵形式为：

        .. math::

            \begin{align}
            SWAP &=\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
            \end{align}

        Args:
            control (list): 作用在的 qubit 的编号，``control[0]`` 和 ``control[1]`` 是想要交换的位，其值都应该在 :math:`[0, n)`范围内， :math:`n` 为该量子线路的量子比特数

        ..  code-block:: python

            import numpy as np
            import paddle 
            from paddle_quantum.circuit import UAnsatz
            num_qubits = 2
            cir = UAnsatz(num_qubits)
            cir.swap([0, 1])
        """
        assert 0 <= control[0] < self.n and 0 <= control[1] < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        assert control[0] != control[1], "the indices needed to be swapped should not be the same"
        self.__history.append(['SWAP', control])

    def x(self, which_qubit):
        r"""添加单量子比特 X 门。

        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数

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
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['x', [which_qubit], None])

    def y(self, which_qubit):
        r"""添加单量子比特 Y 门。

        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数

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
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['y', [which_qubit], None])

    def z(self, which_qubit):
        r"""添加单量子比特 Z 门。

        其矩阵形式为：
        
        .. math::
        
            \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

        Args:
            which_qubit (int): 作用在的qubit的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数

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
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['z', [which_qubit], None])

    def h(self, which_qubit):
        r"""添加一个单量子比特的 Hadamard 门。

        其矩阵形式为：

        .. math::
        
            H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1&1\\1&-1 \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['h', [which_qubit], None])

    def s(self, which_qubit):
        r"""添加一个单量子比特的 S 门。

        其矩阵形式为：

        .. math::
        
            S = \begin{bmatrix} 1&0\\0&i \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [paddle.to_tensor(np.array([0.0])),
                                                    paddle.to_tensor(np.array([0.0])),
                                                    paddle.to_tensor(np.array([math.pi / 2]))]])

    def t(self, which_qubit):
        r"""添加一个单量子比特的 T 门。

        其矩阵形式为：

        .. math::

            T = \begin{bmatrix} 1&0\\0&e^\frac{i\pi}{4} \end{bmatrix}

        Args:
            which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [paddle.to_tensor(np.array([0.0])),
                                                    paddle.to_tensor(np.array([0.0])),
                                                    paddle.to_tensor(np.array([math.pi / 4]))]])

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
              which_qubit (int): 作用在的 qubit 的编号，其值应该在 :math:`[0, n)` 范围内， :math:`n` 为该量子线路的量子比特数
        """
        assert 0 <= which_qubit < self.n, "the qubit should >= 0 and < n(the number of qubit)"
        self.__history.append(['u', [which_qubit], [theta, phi, lam]])

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

        Note:
            参考: https://cds.cern.ch/record/708846/files/0401178.pdf

        Args:
            theta (Tensor): 3-qubit 通用门的参数，其维度为 ``(81, )``
            which_qubits(list): 作用的量子比特编号

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

        psi = paddle.reshape(x=theta[: 60], shape=[4, 15])
        phi = paddle.reshape(x=theta[60:], shape=[7, 3])
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
            当 ``plot`` 为 ``True`` 时，当前量子线路的量子比特数需要小于 6 ，否则无法绘制图片，会抛出异常。

        Args:
            which_qubits (list, optional): 要测量的qubit的编号，默认全都测量
            shots (int, optional): 该量子线路输出的量子态的测量次数，默认为 1024 次；若为 0，则返回测量结果的精确概率分布
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
        if self.__run_state == 'state_vector':
            state = self.__state
        elif self.__run_state == 'density_matrix':
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

    def expecval(self, H):
        r"""量子线路输出的量子态关于可观测量 H 的期望值。

        Hint:
            如果想输入的可观测量的矩阵为 :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I` 。则 ``H`` 应为 ``[[0.7, 'z0,x1'], [0.2, 'z1']]`` 。
        Args:
            H (list): 可观测量的相关信息
        Returns:
            Tensor: 量子线路输出的量子态关于 H 的期望值

        代码示例:
        
        .. code-block:: python
            
            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 5
            H_info = [[0.1, 'x1'], [0.2, 'y0,z4']]
            theta = paddle.to_tensor(np.ones(3))
            cir = UAnsatz(n)
            cir.rx(theta[0], 0)
            cir.rz(theta[1], 1)
            cir.rx(theta[2], 2)
            cir.run_state_vector()
            expect_value = cir.expecval(H_info).numpy()
            print(f'Calculated expectation value of {H_info} is {expect_value}')

        ::

            Calculated expectation value of [[0.1, 'x1'], [0.2, 'y0,z4']] is [-0.1682942]

        """
        if self.__run_state == 'state_vector':
            return real(vec_expecval(H, self.__state))
        elif self.__run_state == 'density_matrix':
            state = self.__state
            H_mat = paddle.to_tensor(pauli_str_to_matrix(H, self.n))
            return real(trace(matmul(state, H_mat)))
        else:
            # Raise error
            raise ValueError("no state for measurement; please run the circuit first")

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

            The probability distribution of measurement results on both qubits is {'00': 0.2499999999999999, '01': 0.2499999999999999, '10': 0.2499999999999999, '11': 0.2499999999999999}
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

            The probability distribution of measurement results on both qubits is {'00': 0.7285533905932737, '01': 0.12500000000000003, '10': 0.12500000000000003, '11': 0.021446609406726238}
        """
        _theta = paddle.to_tensor(np.array([np.pi / 4]))  # Used in fixed Ry gate
        for i in range(self.n):
            self.ry(_theta, i)

    def real_entangled_layer(self, theta, depth, which_qubits=None):
        r"""添加 ``depth`` 层包含 Ry 门和 CNOT 门的强纠缠层。

        Note:
            这一层量子门的数学表示形式为实数酉矩阵。

        Attention:
            ``theta`` 的维度为 ``(depth, n, 1)``

        Args:
            theta (Tensor): Ry 门的旋转角度
            depth (int): 纠缠层的深度
            which_qubits(list): 作用的量子比特编号

        代码示例:

        .. code-block:: python

            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 2
            DEPTH = 3
            theta = np.ones([DEPTH, n, 1])
            theta = paddle.to_tensor(theta)
            cir = UAnsatz(n)
            cir.real_entangled_layer(paddle.to_tensor(theta), DEPTH, [0, 1])
            cir.run_state_vector()
            print(cir.measure(shots = 0))
        
        ::

            {'00': 2.52129874867343e-05, '01': 0.295456784923382, '10': 0.7045028818254718, '11': 1.5120263659845063e-05}
        """
        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 1, 'the shape of theta is not right'
        # assert theta.shape[1] == self.n, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        if which_qubits is None:
            which_qubits = np.arange(self.n)

        for repeat in range(depth):
            for i, q in enumerate(which_qubits):
                self.ry(theta=theta[repeat][i][0], which_qubit=q)
            for i in range(len(which_qubits) - 1):
                self.cnot([which_qubits[i], which_qubits[i + 1]])
            self.cnot([which_qubits[-1], which_qubits[0]])

    def complex_entangled_layer(self, theta, depth, which_qubits=None):
        r"""添加 ``depth`` 层包含 U3 门和 CNOT 门的强纠缠层。

        Note:
            这一层量子门的数学表示形式为复数酉矩阵。
        
        Attention:
            ``theta`` 的维度为 ``(depth, n, 3)`` ，最低维内容为对应的 ``u3`` 的参数 ``(theta, phi, lam)``
        
        Args:
            theta (Tensor): U3 门的旋转角度
            depth (int): 纠缠层的深度
            which_qubits(list): 作用的量子比特编号

        代码示例:

        .. code-block:: python
        
            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 2
            DEPTH = 3
            theta = np.ones([DEPTH, n, 3])
            theta = paddle.to_tensor(theta)
            cir = UAnsatz(n)
            cir.complex_entangled_layer(paddle.to_tensor(theta), DEPTH, [0, 1])
            cir.run_state_vector()
            print(cir.measure(shots = 0))
        
        ::

            {'00': 0.15032627279218896, '01': 0.564191201239618, '10': 0.03285998070292556, '11': 0.25262254526526823}
        """
        assert self.n > 1, 'you need at least 2 qubits'
        assert len(theta.shape) == 3, 'the shape of theta is not right'
        assert theta.shape[2] == 3, 'the shape of theta is not right'
        # assert theta.shape[1] == self.n, 'the shape of theta is not right'
        assert theta.shape[0] == depth, 'the depth of theta has a mismatch'

        if which_qubits is None:
            which_qubits = np.arange(self.n)

        for repeat in range(depth):
            for i, q in enumerate(which_qubits):
                self.u3(theta[repeat][i][0], theta[repeat][i][1], theta[repeat][i][2],  which_qubit=q)
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
        r"""添加 ``depth`` 层包含 Ry 门和 CNOT 门的弱纠缠层。

        Note:
            这一层量子门的数学表示形式为实数酉矩阵。
        
        Attention:
            ``theta`` 的维度为 ``(depth, n-1, 4)``
        
        Args:
            theta(Tensor): Ry 门的旋转角度
            depth(int): 纠缠层的深度

        代码示例:

        .. code-block:: python
        
            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 4
            DEPTH = 3
            theta = np.ones([DEPTH, n-1, 4])
            theta = paddle.to_tensor(theta)
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
            ``theta`` 的维度为 ``(depth, n-1, 12)``

        Args:
            theta (Tensor): U3 门的角度信息
            depth (int): 纠缠层的深度

        代码示例:

        .. code-block:: python
        
            import numpy as np
            import paddle
            from paddle_quantum.circuit import UAnsatz
            n = 4
            DEPTH = 3
            theta = np.ones([DEPTH, n-1, 12])
            theta = paddle.to_tensor(theta)
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


def __local_H_prob(cir, hamiltonian, shots=1024):
    r"""
    构造出 Pauli 测量电路并测量 ancilla，处理实验结果来得到 ``H`` (只有一项)期望值的实验测量值。

    Note:
        这是内部函数，你并不需要直接调用到该函数。
    """
    # Add one ancilla, which we later measure and process the result
    new_cir = UAnsatz(cir.n + 1)
    input_state = pp_kron(cir.run_state_vector(store_state=False), init_state_gen(1))
    # Used in fixed Rz gate
    _theta = paddle.to_tensor(np.array([-np.pi / 2]))

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
    r"""构造 Pauli 测量电路并测量关于 H 的期望值。

    Args:
        cir (UAnsatz): UAnsatz 的一个实例化对象
        H (list): 记录哈密顿量信息的列表
        shots (int, optional): 默认为 1024，表示测量次数；若为 0，则表示返回测量期望值的精确值，即测量无穷次后的期望值

    Returns:
        float: 测量得到的H的期望值
    
    代码示例:

    .. code-block:: python
        
        import numpy as np
        import paddle
        from paddle_quantum.circuit import UAnsatz, H_prob
        n = 4
        experiment_shots = 2**10
        H_info = [[0.1, 'x2'], [0.3, 'y1,z3']]

        theta = paddle.to_tensor(np.ones(3))
        cir = UAnsatz(n)
        cir.rx(theta[0], 0)
        cir.ry(theta[1], 1)
        cir.rz(theta[2], 1)
        result_1 = H_prob(cir, H_info, shots = experiment_shots)
        result_2 = H_prob(cir, H_info, shots = 0)
        print(f'The expectation value obtained by {experiment_shots} measurements is {result_1}')
        print(f'The accurate expectation value of H is {result_2}')

    ::

        The expectation value obtained by 1024 measurements is 0.2177734375
        The accurate expectation value of H is 0.21242202548207134
    """
    expval = 0
    for term in H:
        expval += term[0] * __local_H_prob(cir, term[1], shots=shots)
    return expval
