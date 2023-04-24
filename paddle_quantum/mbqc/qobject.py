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
This module contains the commonly used class in quantum information, e.g. quantum state, quantum circuit and measurement
patterns.
"""

from numpy import log2, sqrt
from paddle import Tensor, to_tensor, t, conj, matmul

__all__ = [
    "State",
    "Circuit",
    "Pattern"
]


class State:
    r"""Define the quantum state.

    Attributes:
        vector (Tensor): the column vector of the quantum state.
        system (list): the list of system labels of the quantum state.
    """

    def __init__(self, vector=None, system=None):
        r""" the constructor for initialize an object of the class `` "State`` .

        Args:
            vector (Tensor, optional): the column vector of the quantum state.
            system (list, optional): the list of system labels of the quantum state.
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
        r"""print the information of this class
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
    r"""Define the quantum circuit.

    Note:
        This class is similar to ``UAnsatz``, one can imitate the use of ``UAnsatz`` to instantiate this class and
        construct the circuit diagram.

    Warning:
        The current version supports the quantum operations(gates and measurement) in ``H, X, Y, Z, S, T, Rx, Ry,
        Rz, Rz_5, U, CNOT, CNOT_15, CZ``.

    Attributes:
        width (int): The width of the circuit(number of qubits).
    """

    def __init__(self, width):
        r""" The constructor of the class ``Circuit`` used of instantiate an object.

        Args:
            width (int): The width of the circuit(number of qubits).
        """
        assert isinstance(width, int), "circuit 'width' must be a int."
        self.__history = []  # A list to record the circuit information
        self.__measured_qubits = []  # A list to record the measurement indices in the circuit
        self.__width = width  # The width of circuit

    def h(self, which_qubit):
        r"""Add a ``Hadamard`` gate.

        The matrix form:

        .. math::

            \frac{1}{\sqrt{2}}\begin{bmatrix} 1&1\\1&-1 \end{bmatrix}

        Args:
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a ``Pauli X`` gate.

        The matrix form:

        .. math::

            \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

        Args:
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a ``Pauli Y`` gate.

        The matrix form:

        .. math::

            \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}

        Args:
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a ``Pauli Z`` gate.

        The matrix form:

        .. math::

            \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

        Args:
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a ``S`` gate.

        The matrix form:

        .. math::

            \begin{bmatrix} 1&0\\0& i \end{bmatrix}

        Args:
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add ``T`` gate.

        The matrix form:

        .. math::

            \begin{bmatrix} 1&0\\0& e^{i\pi/ 4} \end{bmatrix}

        Args:
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a rotation gate in x direction.

        The matrix form:

        .. math::

            \begin{bmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Tensor): rotation angle
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a rotation gate in y direction.

        The matrix form:

        .. math::

            \begin{bmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

        Args:
            theta (Tensor): rotation angle
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a rotation gate in z direction.

        The matrix form:

        .. math::

            \begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}

        Args:
            theta (Tensor): rotation angle
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a rotation gate in the z direction(the measurement pattern corresponding to this gate
        is composed of five qubits).

        The matrix form:

        .. math::

            \begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}

        Args:
            theta (Tensor): rotation angle
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a general single qubit gate.

        Warning:
            Different from the 3 parameters of the U3 gate in ``UAnsatz``  class，
            the unitary here adopts a ``Rz Rx Rz`` decomposition.

        The decomposition takes the form:

        .. math::

            U(\alpha, \beta, \gamma) = Rz(\gamma) Rx(\beta) Rz(\alpha)

        Args:
            params (list): three rotation angles of the unitary gate
            which_qubit (int): The number(No.) of the qubit that the gate applies to.

        Code example:

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
        r"""Add a CNOT gate.

        When  ``which_qubits`` is ``[0, 1]`` ，the matrix form is:

        .. math::

            \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

        Args:
            which_qubits (list): A two element list contains the qubits that the CNOT gate applies to, the first
                element is the control qubit, the second is the target qubit.

        Code example:

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
        r"""Add a CNOT gate(the measurement pattern corresponding to this gate is composed of 15 qubits).

        When ``which_qubits`` is ``[0, 1]`` ，the matrix form is:

        .. math::

            \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

        Args:
            which_qubits (list): A two element list contains the qubits that the CNOT gate applies to, the first
                element is the control qubit, the second is the target qubit.

        Code example:

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
        r"""Add a controlled-Z gate.

        When ``which_qubits`` is ``[0, 1]`` ，the matrix form is:

        .. math::

            \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{bmatrix}

        Args:
            which_qubits (list): A two element list contains the qubits that the CZ gate applies to, the first
                element is the control qubit, the second is the target qubit.

        Code example:

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
        r"""Measure the output state of the quantum circuit.

        Note:
            Unlike the measurement operations in the ``UAnsatz`` class, besides the default measurement in Z basis,
            the users can define the measurement ways themselves by providing the measurement basis and the qubits
            to be measured.

        Warning:
            There are three kinds of inputs:
            1. which_qubit=None,basis_list=None(default): measure all the qubits in Z basis.
            2. which_qubit=input, basis_list=None: measure the input qubit in Z basis.
            3. which_qubit=input1, basis_list=input2: measure the input1 qubit by the basis in input2.
            If the users want to define the measurement basis themselves, the input format looks like:
            ``[angle,plane,domain_s,domain_t]``. By the way,in the current version paddle_quantum, the
            ``plane`` parameter can only take ``XY`` or ``YZ``.

        Args:
            which_qubit (int, optional): the qubit to be measured
            basis_list (list, optional): measurement basis
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
        r"""Check that if the circuit is valid.

        We require that for each qubit in the quantum circuit, at least one quantum gate should be applied to it.

        Returns:
            bool: the boolean values of whether the quantum circuit is valid.
        """
        all_qubits = []
        for gate in self.__history:
            if gate[0] != 'm':
                all_qubits += gate[1]
        effective_qubits = list(set(all_qubits))

        return self.__width == len(effective_qubits)

    def get_width(self):
        r"""Return the width of the quantum circuit.

        Returns:
           int: the width of the quantum circuit.
        """
        return self.__width

    def get_circuit(self):
        r"""Return the list of quantum circuits.

        Returns:
            list: the list of quantum circuits
        """
        return self.__history

    def get_measured_qubits(self):
        r"""Return the list of measured qubits in the quantum circuit.

        Returns:
            list: the list of measured qubits in the quantum circuit.
        """
        return self.__measured_qubits

    def print_circuit_list(self):
        r"""Print the list of the circuit.

        Returns:
            string: the strings to be printed

        Code example:

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
    r"""Define the measurement pattern.

    see more details of the measurement pattern in [The measurement calculus, arXiv: 0704.1263].

    Attributes:
        name (str): the name of the measurement pattern.
        space (list): the list contains all the nodes of the measurement pattern.
        input_ (list): the list contains the input nodes of the measurement pattern.
        output_ (list): the list contains the output nodes of the measurement pattern.
        commands (list): the list contains the commands of the measurement pattern.
    """

    def __init__(self, name, space, input_, output_, commands):
        r"""Constructor of the class ``Pattern`` used for instantiated an object.

        Args:
            name (str): the name of the measurement pattern.
            space (list): the list contains all the nodes of the measurement pattern.
            input_ (list): the list contains the input nodes of the measurement pattern.
            output_ (list): the list contains the output nodes of the measurement pattern.
            commands (list): the list contains the commands of the measurement pattern.
        """
        self.name = name
        self.space = space
        self.input_ = input_
        self.output_ = output_
        self.commands = commands

    class CommandE:
        r"""Define the class ``CommandE`` corresponding to some entanglement commands.

        Note:
            The entanglement command here corresponds to the controlled-Z gate.

        Attributes:
            which_qubits (list): A two-element list contains the labels of the node
            that the entanglement command applies to.
        """

        def __init__(self, which_qubits):
            r""" Constructor of the ``CommandE`` class used for instantiate an object.

            Args:
                which_qubits (list): A two-element list contains the labels of the node
            that the entanglement command applies to.
            """
            self.name = "E"
            self.which_qubits = which_qubits

    class CommandM:
        r"""Define the ``CommandM`` class corresponding to the measurement commands.

        ``CommandM`` has 5 attributes, including:
            1.which_qubit: the qubit to be measured.
            2.angle: the original measurement angle.
            3.plane: the measurement plane.
            4.domain_s: the node list corresponding to domain s.
            5.domain_t: the node list corresponding to domain t.

        The original angle :math:`\alpha` is transformed into :math:`\theta` as:
        .. math::

            \theta = (-1)^s \times \alpha + t \times \pi
        after considering the node dependence in the domain.

        Note:
            Domain s(domain t) is the concept in MBQC containing the influence on the measurement angles induced
            by Pauli X(Pauli Z) operator. Both of them record the dependence of the measurement node on the
            measurement results of other nodes.

        Warning:
            Only measurements in "XY" and "YZ" planes are allowed in this version.

        Attributes:
            which_qubit (any): the qubit to be measured.
            angle (Tensor): the original measurement angle.
            plane (str): the measurement plane.
            domain_s (list): the node list corresponding to domain s.
            domain_t (list): the node list corresponding to domain t.
        """

        def __init__(self, which_qubit, angle, plane, domain_s, domain_t):
            r"""The constructor of the ``CommandM`` class used for instantiated an object.

            Args:
                which_qubit (any): the qubit to be measured.
                angle (Tensor): the original measurement angle.
                plane (str): the measurement plane.
                domain_s (list): the node list corresponding to domain s.
                domain_t (list): the node list corresponding to domain t.
            """
            self.name = "M"
            self.which_qubit = which_qubit
            self.angle = angle
            self.plane = plane
            self.domain_s = domain_s
            self.domain_t = domain_t

    class CommandX:
        r"""Define the ``CommandX`` class used for correcting the byproduct induced by Pauli X.

        Attributes:
            which_qubit (any): the label of the qubit that the correcting operator applies to.
            domain (list): the list contains the dependence relation.
        """

        def __init__(self, which_qubit, domain):
            r"""The constructor of the ``CommandX`` class used for instantiated an object.

            Args:
                which_qubit (any): the label of the qubit that the correcting operator applies to.
                domain (list): the list contains the dependence relation.
            """
            self.name = "X"
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandZ:
        r"""Define the ``CommandZ`` class used for correcting the byproduct induced by Pauli Z.

        Attributes:
            which_qubit (any): the label of the qubit that the correcting operator applies to.
            domain (list): the list contains the dependence relation.
        """

        def __init__(self, which_qubit, domain):
            r"""The constructor of the ``CommandZ`` class used for instantiated an object.

            Args:
                which_qubit (any): the label of the qubit that the correcting operator applies to.
                domain (list): the list contains the dependence relation.
            """
            self.name = "Z"
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandS:
        r"""Define the ``CommandS`` class used for signal shifting.

        Note:
            Signal shifting is a class of special operations used for eliminating the measurement operations' dependence
            on the nodes in domain t and simplify the measurement patterns on some conditions.

        Attributes:
            which_qubit (any): the labels of the nodes applied by the signal shifting operations.
            domain (list): the list contains the dependence relation.
        """

        def __init__(self, which_qubit, domain):
            r"""The constructor of the ``CommandS`` class used for instantiated an object.

            Args:
                which_qubit (any): the labels of the nodes applied by the signal shifting operations.
                domain (list): the list contains the dependence relation.
            """
            self.name = "S"
            self.which_qubit = which_qubit
            self.domain = domain

    def print_command_list(self):
        r"""Print the information of commands in the ``Pattern`` class.

        Code example:

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
