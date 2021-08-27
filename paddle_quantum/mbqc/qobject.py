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

"""
此模块包含量子信息处理的常用对象，如量子态、量子电路、测量模式等。
"""

from numpy import log2, sqrt
from paddle import Tensor, to_tensor, t, conj, matmul

__all__ = [
    "State",
    "Circuit",
    "Pattern"
]


class State:
    r"""定义量子态。

    Attributes:
        vector (Tensor): 量子态的列向量
        system (list): 量子态的系统标签列表
    """

    def __init__(self, vector=None, system=None):
        r"""构造函数，用于实例化一个 ``"State"`` 量子态对象。

        Args:
            vector (Tensor, optional): 量子态的列向量
            system (list, optional): 量子态的系统标签列表
        """
        if vector is None and system is None:
            self.vector = to_tensor([1], dtype='float64')  # A trivial state
            self.system = []
            self.length = 1  # Length of state vector
            self.size = 0  # Number of qubits
        elif vector is not None and system is not None:
            assert isinstance(vector, Tensor), "'vector' should be a 'Tensor'."
            assert vector.shape[0] >= 1 and vector.shape[1] == 1, "'vector' should be of shape [x, 1] with x >= 1."
            assert isinstance(system, list), "'system' should be a list."
            self.vector = vector
            self.system = system
            self.length = self.vector.shape[0]  # Length of state vector
            self.size = int(log2(self.length))  # Number of qubits
            assert self.size == len(self.system), "dimension of vector and system do not match."

        else:
            raise ValueError("we should either input both 'vector' and 'system' or input nothing.")
        self.state = [self.vector, self.system]
        self.norm = sqrt(matmul(t(conj(self.vector)), self.vector).numpy())

    def __str__(self):
        r"""打印该 ``State`` 类的信息，便于用户查看。
        """
        class_type_str = "State"
        vector_str = str(self.vector.numpy())
        system_str = str(self.system)
        length_str = str(self.length)
        size_str = str(self.size)
        print_str = class_type_str + "(" + \
                    "size=" + size_str + ", " + \
                    "system=" + system_str + ", " + \
                    "length=" + length_str + ", " + \
                    "vector=\r\n" + vector_str + ")"
        return print_str


class Circuit:
    r"""定义量子电路。

    Note:
        该类与 ``UAnsatz`` 类似，用户可以仿照 ``UAnsatz`` 电路的调用方式对此类进行实例化，完成电路图的构建。

    Warning:
        当前版本仅支持 ``H, X, Y, Z, S, T, Rx, Ry, Rz, Rz_5, U, CNOT, CNOT_15, CZ`` 中的量子门以及测量操作。

    Attributes:
        width (int): 电路的宽度（比特数）
    """

    def __init__(self, width):
        r"""``Circuit`` 的构造函数，用于实例化一个 ``Circuit`` 对象。

        Args:
            width (int): 电路的宽度（比特数）
        """
        assert isinstance(width, int), "circuit 'width' must be a int."
        self.__history = []  # A list to record the circuit information
        self.__measured_qubits = []  # A list to record the measurement indices in the circuit
        self.__width = width  # The width of circuit

    def h(self, which_qubit):
        r"""添加 ``Hadamard`` 门。

        其矩阵形式为：

        .. math::

            \frac{1}{\sqrt{2}}\begin{bmatrix} 1&1\\1&-1 \end{bmatrix}

        Args:
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        .. code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.h(which_qubit)
            print(cir.get_circuit())

        ::

            [['h', [0], None]]
        """
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['h', [which_qubit], None])

    def x(self, which_qubit):
        r"""添加 ``Pauli X`` 门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

        Args:
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        .. code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.x(which_qubit)
            print(cir.get_circuit())

        ::

            [['x', [0], None]]
        """
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['x', [which_qubit], None])

    def y(self, which_qubit):
        r"""添加 ``Pauli Y`` 门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}

        Args:
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        .. code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.y(which_qubit)
            print(cir.get_circuit())

        ::

            [['y', [0], None]]
        """
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['y', [which_qubit], None])

    def z(self, which_qubit):
        r"""添加 ``Pauli Z`` 门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

        Args:
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        .. code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.z(which_qubit)
            print(cir.get_circuit())

        ::

            [['z', [0], None]]
        """
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['z', [which_qubit], None])

    def s(self, which_qubit):
        r"""添加 ``S`` 门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} 1&0\\0& i \end{bmatrix}

        Args:
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        .. code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.s(which_qubit)
            print(cir.get_circuit())

        ::

            [['s', [0], None]]
        """
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['s', [which_qubit], None])

    def t(self, which_qubit):
        r"""添加 ``T`` 门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} 1&0\\0& e^{i\pi/ 4} \end{bmatrix}

        Args:
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        .. code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.t(which_qubit)
            print(cir.get_circuit())

        ::

            [['t', [0], None]]
        """
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['t', [which_qubit], None])

    def rx(self, theta, which_qubit):
        r"""添加关于 x 轴的旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        ..  code-block:: python

            from paddle import to_tensor
            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            angle = to_tensor([1], dtype='float64')
            cir.rx(angle, which_qubit)
            print(cir.get_circuit())

        ::

            [['rx', [0], Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True, [1.])]]
        """
        assert isinstance(theta, Tensor) and theta.shape == [1], "'theta' must be a 'Tensor' of shape [1]."
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['rx', [which_qubit], theta])

    def ry(self, theta, which_qubit):
        r"""添加关于 y 轴的旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        ..  code-block:: python

            from paddle import to_tensor
            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            angle = to_tensor([1], dtype='float64')
            cir.ry(angle, which_qubit)
            print(cir.get_circuit())

        ::

            [['ry', [0], Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True, [1.])]]
        """
        assert isinstance(theta, Tensor) and theta.shape == [1], "'theta' must be a 'Tensor' of shape [1]."
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['ry', [which_qubit], theta])

    def rz(self, theta, which_qubit):
        r"""添加关于 z 轴的旋转门。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        ..  code-block:: python

            from paddle import to_tensor
            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            angle = to_tensor([1], dtype='float64')
            cir.rz(angle, which_qubit)
            print(cir.get_circuit())

        ::

            [['rz', [0], Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True, [1.])]]
        """
        assert isinstance(theta, Tensor) and theta.shape == [1], "'theta' must be a 'Tensor' of shape [1]."
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['rz', [which_qubit], theta])

    def rz_5(self, theta, which_qubit):
        r"""添加关于 z 轴的旋转门（该旋转门对应的测量模式由五个量子比特构成）。

        其矩阵形式为：

        .. math::

            \begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}

        Args:
            theta (Tensor): 旋转角度
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        ..  code-block:: python

            from paddle import to_tensor
            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            angle = to_tensor([1], dtype='float64')
            cir.rz(angle, which_qubit)
            print(cir.get_circuit())

        ::

            [['rz_5', [0], Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True, [1.])]]
        """
        assert isinstance(theta, Tensor) and theta.shape == [1], "'theta' must be a 'Tensor' of shape [1]."
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['rz_5', [which_qubit], theta])

    def u(self, params, which_qubit):
        r"""添加单量子比特的任意酉门。

        Warning:
            与 ``UAnsatz`` 类中的 U3 的三个参数不同，这里的酉门采用 ``Rz Rx Rz`` 分解，

        其分解形式为：

        .. math::

            U(\alpha, \beta, \gamma) = Rz(\gamma) Rx(\beta) Rz(\alpha)

        Args:
            params (list): 单比特酉门的三个旋转角度
            which_qubit (int): 作用量子门的量子位编号

        代码示例：

        ..  code-block:: python

            from paddle import to_tensor
            from numpy import pi
            from paddle_quantum.mbqc.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            alpha = to_tensor([pi / 2], dtype='float64')
            beta = to_tensor([pi], dtype='float64')
            gamma = to_tensor([- pi / 2], dtype='float64')
            cir.u([alpha, beta, gamma], which_qubit)
            print(cir.get_circuit())

        ::

            [['u', [0], [Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True,
               [1.57079633]), Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True,
               [3.14159265]), Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True,
               [-1.57079633])]]]
       """
        assert isinstance(params, list) and len(params) == 3, "'params' must be a list of length 3."
        assert all([isinstance(par, Tensor) and par.shape == [1] for par in params]), \
            "item in 'params' must be 'Tensor' of shape [1]."
        assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
        assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
        assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
        self.__history.append(['u', [which_qubit], params])

    def cnot(self, which_qubits):
        r"""添加控制非门。

        当 ``which_qubits`` 为 ``[0, 1]`` 时，其矩阵形式为：

        .. math::

            \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

        Args:
            which_qubits (list): 作用量子门的量子位，其中列表第一个元素为控制位，第二个元素为受控位

        代码示例：

        ..  code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 2
            cir = Circuit(width)
            which_qubits = [0, 1]
            cir.cnot(which_qubits)
            print(cir.get_circuit())

        ::

            [['cnot', [0, 1], None]]
        """
        assert isinstance(which_qubits[0], int) and isinstance(which_qubits[1], int), \
            "items in 'which_qubits' must be of type 'int'."
        assert 0 <= which_qubits[0] < self.__width and 0 <= which_qubits[1] < self.__width, \
            "items in 'which_qubits' must be between zero and circuit width."
        assert which_qubits[0] != which_qubits[1], \
            "control qubit must not be the same as the target qubit."
        assert which_qubits[0] not in self.__measured_qubits and which_qubits[1] not in self.__measured_qubits, \
            "one of the qubits has already been measured."
        self.__history.append(['cnot', which_qubits, None])

    def cnot_15(self, which_qubits):
        r"""添加控制非门 （该门对应的测量模式由十五个量子比特构成）。

        当 ``which_qubits`` 为 ``[0, 1]`` 时，其矩阵形式为：

        .. math::

            \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

        Args:
            which_qubits (list): 作用量子门的量子位，其中列表第一个元素为控制位，第二个元素为受控位

        代码示例：

        ..  code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 2
            cir = Circuit(width)
            which_qubits = [0, 1]
            cir.cnot_15(which_qubits)
            print(cir.get_circuit())

        ::

            [['cnot_15', [0, 1], None]]
        """
        assert isinstance(which_qubits[0], int) and isinstance(which_qubits[1], int), \
            "items in 'which_qubits' must be of type 'int'."
        assert 0 <= which_qubits[0] < self.__width and 0 <= which_qubits[1] < self.__width, \
            "items in 'which_qubits' must be between zero and circuit width."
        assert which_qubits[0] != which_qubits[1], \
            "control qubit must not be the same as the target qubit."
        assert which_qubits[0] not in self.__measured_qubits and which_qubits[1] not in self.__measured_qubits, \
            "one of the qubits has already been measured."
        self.__history.append(['cnot_15', which_qubits, None])

    def cz(self, which_qubits):
        r"""添加控制 Z 门。

        当 ``which_qubits`` 为 ``[0, 1]`` 时，其矩阵形式为：

        .. math::

            \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{bmatrix}

        Args:
            which_qubits (list): 作用量子门的量子位，其中列表第一个元素为控制位，第二个元素为受控位

        代码示例：

        ..  code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            width = 2
            cir = Circuit(width)
            which_qubits = [0, 1]
            cir.cz(which_qubits)
            print(cir.get_circuit())

        ::

            [['cz', [0, 1], None]]
        """
        assert isinstance(which_qubits[0], int) and isinstance(which_qubits[1], int), \
            "items in 'which_qubits' must be of type 'int'."
        assert 0 <= which_qubits[0] < self.__width and 0 <= which_qubits[1] < self.__width, \
            "items in 'which_qubits' must be between zero and circuit width."
        assert which_qubits[0] != which_qubits[1], \
            "control qubit must not be the same as the target qubit."
        assert which_qubits[0] not in self.__measured_qubits and which_qubits[1] not in self.__measured_qubits, \
            "one of the qubits has already been measured."
        self.__history.append(['cz', which_qubits, None])

    def measure(self, which_qubit=None, basis_list=None):
        r"""对量子电路输出的量子态进行测量。

        Note:
            与 ``UAnsatz`` 类中的测量不同，除默认的 Z 测量外，此处的测量方式可以由用户自定义，但需要将测量方式与测量比特相对应。

        Warning:
            此方法只接受三种输入方式：
            1. 不输入任何参数，表示对所有的量子位进行 Z 测量；
            2. 输入量子位，但不输入测量基，表示对输入的量子位进行 Z 测量；
            3. 输入量子位和对应测量基，表示对输入量子位进行指定的测量。
            如果用户希望自定义测量基参数，需要注意输入格式为 ``[angle, plane, domain_s, domain_t]``，
            且当前版本的测量平面 ``plane`` 只能支持 ``XY`` 或 ``YZ``。

        Args:
            which_qubit (int, optional): 被测量的量子位
            basis_list (list, optional): 测量方式
        """
        # Measure all the qubits by Z measurement
        if which_qubit is None and basis_list is None:
            # Set Z measurement by default
            basis_list = [to_tensor([0], dtype='float64'), 'YZ', [], []]
            for which_qubit in range(self.__width):
                assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
                self.__measured_qubits.append(which_qubit)
                self.__history.append(['m', [which_qubit], basis_list])

        # Measure the referred qubit by Z measurement
        elif which_qubit is not None and basis_list is None:
            assert isinstance(which_qubit, int), "'which_qubit' must be a 'int'."
            assert 0 <= which_qubit < self.__width, "'which_qubit' must be a int between zero and circuit width."
            assert which_qubit not in self.__measured_qubits, "this qubit has already been measured."
            # Set Z measurement as default
            basis_list = [to_tensor([0], dtype='float64'), 'YZ', [], []]
            self.__measured_qubits.append(which_qubit)
            self.__history.append(['m', [which_qubit], basis_list])

        # Measure the referred qubit by the customized basis
        elif which_qubit is not None and basis_list is not None:
            assert isinstance(basis_list, list) and len(basis_list) == 4, \
                "'basis_list' must be a list of length 4."
            assert isinstance(basis_list[0], Tensor) and basis_list[0].shape == [1], \
                "measurement angle must be a 'Tensor' of shape [1]."
            assert basis_list[1] in ["XY", "YZ"], "measurement plane must be 'XY' or 'YZ'."
            self.__measured_qubits.append(which_qubit)
            self.__history.append(['m', [which_qubit], basis_list])
        else:
            raise ValueError("such a combination of input parameters is not supported. Please see our API for details.")

    def is_valid(self):
        r"""检查输入的量子电路是否符合规定。

        我们规定输入的量子电路中，每一个量子位上至少作用一个量子门。

        Returns:
            bool: 量子电路是否符合规定的布尔值
        """
        all_qubits = []
        for gate in self.__history:
            if gate[0] != 'm':
                all_qubits += gate[1]
        effective_qubits = list(set(all_qubits))

        return self.__width == len(effective_qubits)

    def get_width(self):
        r"""返回量子电路的宽度。

        Returns:
           int: 量子电路的宽度
        """
        return self.__width

    def get_circuit(self):
        r"""返回量子电路列表。

        Returns:
            list: 量子电路列表
        """
        return self.__history

    def get_measured_qubits(self):
        r"""返回量子电路中测量的比特位。

        Returns:
            list: 量子电路中测量的比特位列表
        """
        return self.__measured_qubits

    def print_circuit_list(self):
        r"""打印电路图的列表。

        Returns:
            string: 用来打印的字符串

        代码示例:

        .. code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            from paddle import to_tensor
            from numpy import pi

            n = 2
            theta = to_tensor([pi], dtype="float64")
            cir = Circuit(n)
            cir.h(0)
            cir.cnot([0, 1])
            cir.rx(theta, 1)
            cir.measure()
            cir.print_circuit_list()

        ::

            --------------------------------------------------
                             Current circuit
            --------------------------------------------------
            Gate Name       Qubit Index     Parameter
            --------------------------------------------------
            h               [0]             None
            cnot            [0, 1]          None
            rx              [1]             3.141592653589793
            m               [0]             [0.0, 'YZ', [], []]
            m               [1]             [0.0, 'YZ', [], []]
            --------------------------------------------------
        """
        print("--------------------------------------------------")
        print("                 Current circuit                  ")
        print("--------------------------------------------------")
        print("Gate Name".ljust(16) + "Qubit Index".ljust(16) + "Parameter".ljust(16))
        print("--------------------------------------------------")

        for gate in self.__history:
            name = gate[0]
            which_qubits = gate[1]
            parameters = gate[2]
            if isinstance(parameters, Tensor):
                par_show = parameters.numpy().item()
            elif name == 'm':
                par_show = [parameters[0].numpy().item()] + parameters[1:]
            else:
                par_show = parameters
            print(str(name).ljust(16) + str(which_qubits).ljust(16) + str(par_show).ljust(16))
        print("--------------------------------------------------")


class Pattern:
    r"""定义测量模式。

    该测量模式的结构依据文献 [The measurement calculus, arXiv: 0704.1263]。

    Attributes:
        name (str): 测量模式的名称
        space (list): 测量模式所有节点列表
        input_ (list): 测量模式的输入节点列表
        output_ (list): 测量模式的输出节点列表
        commands (list): 测量模式的命令列表
    """

    def __init__(self, name, space, input_, output_, commands):
        r"""构造函数，用于实例化一个 ``Pattern`` 对象。

        Args:
            name (str): 测量模式的名称
            space (list): 测量模式所有节点列表
            input_ (list): 测量模式的输入节点列表
            output_ (list): 测量模式的输出节点列表
            commands (list): 测量模式的命令列表
        """
        self.name = name
        self.space = space
        self.input_ = input_
        self.output_ = output_
        self.commands = commands

    class CommandE:
        r"""定义纠缠命令类。

        Note:
            此处纠缠命令对应作用控制 Z 门。

        Attributes:
            which_qubits (list): 作用纠缠命令的两个节点标签构成的列表
        """

        def __init__(self, which_qubits):
            r"""纠缠命令类构造函数，用于实例化一个 ``CommandE`` 对象。

            Args:
                which_qubits (list): 作用纠缠命令的两个节点标签构成的列表
            """
            self.name = "E"
            self.which_qubits = which_qubits

    class CommandM:
        r"""定义测量命令类。

        测量命令有五个属性，分别为测量比特的标签 ``which_qubit``，原始的测量角度 ``angle``，
        测量平面 ``plane``，域 s 对应的节点标签列表 ``domain_s``，域 t 对应的节点标签列表 ``domain_t``。
        设原始角度为 :math:`\alpha`，则考虑域中节点依赖关系后的测量角度 :math:`\theta` 为：

        .. math::

            \theta = (-1)^s \times \alpha + t \times \pi

        Note:
            域 s 和域 t 是 MBQC 模型中的概念，分别记录了 Pauli X 算符和 Pauli Z 算符对测量角度产生的影响，
            二者共同记录了该测量节点对其他节点的测量结果的依赖关系。

        Warning:
            该命令当前只支持 XY 和 YZ 平面的测量。

        Attributes:
            which_qubit (any): 作用测量命令的节点标签
            angle (Tensor): 原始的测量角度
            plane (str): 测量平面
            domain_s (list): 域 s 对应的节点标签列表
            domain_t (list): 域 t 对应的节点标签列表
        """

        def __init__(self, which_qubit, angle, plane, domain_s, domain_t):
            r"""构造函数，用于实例化一个 ``CommandM`` 对象。

            Args:
                which_qubit (any): 作用测量命令的节点标签
                angle (Tensor): 原始的测量角度
                plane (str): 测量平面
                domain_s (list): 域 s 对应的节点标签列表
                domain_t (list): 域 t 对应的节点标签列表
            """
            self.name = "M"
            self.which_qubit = which_qubit
            self.angle = angle
            self.plane = plane
            self.domain_s = domain_s
            self.domain_t = domain_t

    class CommandX:
        r"""定义 Pauli X 副产品修正命令类。

        Attributes:
            which_qubit (any): 作用修正算符的节点标签
            domain (list): 依赖关系列表
        """

        def __init__(self, which_qubit, domain):
            r"""构造函数，用于实例化一个 ``CommandX`` 对象。

            Args:
                which_qubit (any): 作用修正算符的节点标签
                domain (list): 依赖关系列表
            """
            self.name = "X"
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandZ:
        r"""定义 Pauli Z 副产品修正命令。

        Attributes:
            which_qubit (any): 作用修正命令的节点标签
            domain (list): 依赖关系列表
        """

        def __init__(self, which_qubit, domain):
            r"""构造函数，用于实例化一个 ``CommandZ`` 对象。

            Args:
                which_qubit (any): 作用修正命令的节点标签
                domain (list): 依赖关系列表
            """
            self.name = "Z"
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandS:
        r"""定义 "信号转移" 命令类。

        Note:
            "信号转移" 是一类特殊的操作，用于消除测量命令对域 t 中节点的依赖关系，在某些情况下对测量模式进行简化。

        Attributes:
            which_qubit (any): 消除依赖关系的测量命令作用的节点标签
            domain (list): 依赖关系列表
        """

        def __init__(self, which_qubit, domain):
            r"""构造函数，用于实例化一个 ``CommandS`` 对象。

            Args:
                which_qubit (any): 消除依赖关系的测量命令作用的节点标签
                domain (list): 依赖关系列表
            """
            self.name = "S"
            self.which_qubit = which_qubit
            self.domain = domain

    def print_command_list(self):
        r"""打印该 ``Pattern`` 类中的命令的信息，便于用户查看。

        代码示例:

        .. code-block:: python

            from paddle_quantum.mbqc.qobject import Circuit
            from paddle_quantum.mbqc.mcalculus import MCalculus

            n = 1
            cir = Circuit(n)
            cir.h(0)
            pat = MCalculus()
            pat.set_circuit(cir)
            pattern = pat.get_pattern()
            pattern.print_command_list()

        ::

            -----------------------------------------------------------
                                Current Command List
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [('0/1', '0/1'), ('0/1', '1/1')]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    ('0/1', '0/1')
            plane:          XY
            angle:          0.0
            domain_s:       []
            domain_t:       []
            -----------------------------------------------------------
            Command:        X
            which_qubit:    ('0/1', '1/1')
            domain:         [('0/1', '0/1')]
            -----------------------------------------------------------
        """
        print("-----------------------------------------------------------")
        print("                    Current Command List                   ")
        print("-----------------------------------------------------------")
        # Print commands list
        for cmd in self.commands:
            print('\033[91m' + "Command:".ljust(16) + cmd.name + '\033[0m')
            if cmd.name == "E":
                print("which_qubits:".ljust(15), cmd.which_qubits)
            elif cmd.name == "M":
                print("which_qubit:".ljust(15), cmd.which_qubit)
                print("plane:".ljust(15), cmd.plane)
                print("angle:".ljust(15), cmd.angle.numpy().item())
                print("domain_s:".ljust(15), cmd.domain_s)
                print("domain_t:".ljust(15), cmd.domain_t)
            else:
                print("which_qubit:".ljust(15), cmd.which_qubit)
                print("domain:".ljust(15), cmd.domain)
            print("-----------------------------------------------------------")
