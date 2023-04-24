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
This module contains various common classes and functions used for computation.
"""

from numpy import array, exp, pi, linalg
from numpy import sqrt as np_sqrt
from numpy import random as np_random
from paddle import Tensor, to_tensor, t, cos, eye, sin
from paddle import kron as pp_kron
from paddle import matmul, conj, real
from paddle import reshape, transpose
from paddle import sqrt as pp_sqrt
from paddle import multiply
from paddle_quantum.mbqc.qobject import State
import matplotlib.pyplot as plt

__all__ = ["plus_state",
           "minus_state",
           "zero_state",
           "one_state",
           "h_gate",
           "s_gate",
           "t_gate",
           "cz_gate",
           "cnot_gate",
           "swap_gate",
           "pauli_gate",
           "rotation_gate",
           "to_projector",
           "basis",
           "kron",
           "permute_to_front",
           "permute_systems",
           "compare_by_density",
           "compare_by_vector",
           "random_state_vector",
           "div_str_to_float",
           "int_to_div_str",
           "print_progress",
           "plot_results",
           "write_running_data",
           "read_running_data"
           ]


def plus_state():
    r"""Define plus state.

    The matrix form is:

    .. math::

        \frac{1}{\sqrt{2}}  \begin{bmatrix}  1 \\ 1 \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of plus state.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import plus_state
        print("State vector of plus state: \n", plus_state().numpy())

    ::

        State vector of plus state:
         [[0.70710678]
         [0.70710678]]
    """
    return to_tensor([[1 / np_sqrt(2)], [1 / np_sqrt(2)]], dtype='float64')


def minus_state():
    r"""Define minus state.

    The matrix form is:

    .. math::

        \frac{1}{\sqrt{2}}  \begin{bmatrix}  1 \\ -1 \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of minus state.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import minus_state
        print("State vector of minus state: \n", minus_state().numpy())

    ::

        State vector of minus state:
         [[ 0.70710678]
         [-0.70710678]]
    """
    return to_tensor([[1 / np_sqrt(2)], [-1 / np_sqrt(2)]], dtype='float64')


def zero_state():
    r"""Define zero state.

    The matrix form is:

    .. math::

        \begin{bmatrix}  1 \\ 0 \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of zero state.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import zero_state
        print("State vector of zero state: \n", zero_state().numpy())

    ::

        State vector of zero state:
         [[1.]
         [0.]]
    """
    return to_tensor([[1], [0]], dtype='float64')


def one_state():
    r"""Define one state.

    The matrix form is:

    .. math::

        \begin{bmatrix}  0 \\ 1 \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of one state.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import one_state
        print("State vector of one state: \n", one_state().numpy())

    ::

        State vector of one state:
         [[0.]
         [1.]]
    """
    return to_tensor([[0], [1]], dtype='float64')


def h_gate():
    r"""Define ``Hadamard`` gate.

    The matrix form is:

    .. math::

        \frac{1}{\sqrt{2}} \begin{bmatrix}  1 & 1 \\ 1 & -1 \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of ``Hadamard`` gate.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import h_gate
        print("Matrix of Hadamard gate: \n", h_gate().numpy())

    ::

        Matrix of Hadamard gate:
         [[ 0.70710678  0.70710678]
         [ 0.70710678 -0.70710678]]
    """
    return to_tensor((1 / np_sqrt(2)) * array([[1, 1], [1, -1]]), dtype='float64')


def s_gate():
    r"""Define ``S`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix}  1 & 0 \\ 0 & i \end{bmatrix}

    Returns:
        Tensor: ``S`` 门矩阵对应的 ``Tensor`` 形式

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import s_gate
        print("Matrix of S gate:\n", s_gate().numpy())

    ::

        Matrix of S gate:
         [[1.+0.j 0.+0.j]
         [0.+0.j 0.+1.j]]
    """
    return to_tensor([[1, 0], [0, 1j]], dtype='complex128')


def t_gate():
    r"""Define ``T`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix}  1 & 0 \\ 0 & e^{i \pi / 4} \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of ``T`` gate.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import t_gate
        print("Matrix of T gate: \n", t_gate().numpy())

    ::

        Matrix of T gate:
         [[1.        +0.j         0.        +0.j        ]
         [0.        +0.j         0.70710678+0.70710678j]]
    """
    return to_tensor([[1, 0], [0, exp(1j * pi / 4)]], dtype='complex128')


def cz_gate():
    r"""Define ``Controlled-Z`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of ``Controlled-Z`` gate.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import cz_gate
        print("Matrix of CZ gate: \n", cz_gate().numpy())

    ::

        Matrix of CZ gate:
         [[ 1.  0.  0.  0.]
         [ 0.  1.  0.  0.]
         [ 0.  0.  1.  0.]
         [ 0.  0.  0. -1.]]
    """
    return to_tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]], dtype='float64')


def cnot_gate():
    r"""Define ``Controlled-NOT (CNOT)`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of ``Controlled-NOT (CNOT)`` gate.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import cnot_gate
        print("Matrix of CNOT gate: \n", cnot_gate().numpy())

    ::

        Matrix of CNOT gate:
         [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]
         [0. 0. 1. 0.]]
    """
    return to_tensor([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]], dtype='float64')


def swap_gate():
    r"""Define ``SWAP`` gate.

    The matrix form is:

    .. math::

        \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}

    Returns:
        Tensor: The ``Tensor`` form of ``SWAP`` gate.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import swap_gate
        print("Matrix of Swap gate: \n", swap_gate().numpy())

    ::

        Matrix of Swap gate:
         [[1. 0. 0. 0.]
         [0. 0. 1. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]]
    """
    return to_tensor([[1, 0, 0, 0],
                      [0, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]], dtype='float64')


def pauli_gate(gate):
    r"""Define ``Pauli`` gate.

    The matrix form of Identity gate ``I`` is:

    .. math::

        \begin{bmatrix}  1 & 0 \\ 0 & 1 \end{bmatrix}

    The matrix form of Pauli gate ``X`` is:

    .. math::

        \begin{bmatrix}  0 & 1 \\ 1 & 0 \end{bmatrix}

    The matrix form of Pauli gate ``Y`` is:

    .. math::

        \begin{bmatrix}  0 & - i \\ i & 0 \end{bmatrix}

    The matrix form of Pauli gate ``Z`` is:

    .. math::

        \begin{bmatrix}  1 & 0 \\ 0 & - 1 \end{bmatrix}

    Args:
        gate (str): Index of Pauli gate. “I”, “X”, “Y”, or “Z” denotes the corresponding gate.

    Returns:
        Tensor: The matrix form of the Pauli gate.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import pauli_gate
        I = pauli_gate('I')
        X = pauli_gate('X')
        Y = pauli_gate('Y')
        Z = pauli_gate('Z')
        print("Matrix of Identity gate: \n", I.numpy())
        print("Matrix of Pauli X gate: \n", X.numpy())
        print("Matrix of Pauli Y gate: \n", Y.numpy())
        print("Matrix of Pauli Z gate: \n", Z.numpy())

    ::

        Matrix of Identity gate:
         [[1. 0.]
         [0. 1.]]
        Matrix of Pauli X gate:
         [[0. 1.]
         [1. 0.]]
        Matrix of Pauli Y gate:
         [[ 0.+0.j -0.-1.j]
         [ 0.+1.j  0.+0.j]]
        Matrix of Pauli Z gate:
         [[ 1.  0.]
         [ 0. -1.]]
    """
    if gate == 'I':  # Identity gate
        return to_tensor(eye(2, 2), dtype='float64')
    elif gate == 'X':  # Pauli X gate
        return to_tensor([[0, 1], [1, 0]], dtype='float64')
    elif gate == 'Y':  # Pauli Y gate
        return to_tensor([[0, -1j], [1j, 0]], dtype='complex128')
    elif gate == 'Z':  # Pauli Z gate
        return to_tensor([[1, 0], [0, -1]], dtype='float64')
    else:
        print("The Pauli gate must be 'I', 'X', 'Y' or 'Z'.")
        raise KeyError("invalid Pauli gate index: %s" % gate + ".")


def rotation_gate(axis, theta):
    r"""Define Rotation gate.

    .. math::

        R_{x}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) X

        R_{y}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Y

        R_{z}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Z

    Args:
        axis (str): Rotation axis. 'x', 'y' or 'z' denotes the corresponding axis.
        theta (Tensor): Rotation angle.

    Returns:
        Tensor: The matrix form of Rotation gate.

    Code example:

    .. code-block:: python

        from numpy import pi
        from paddle import to_tensor
        from paddle_quantum.mbqc.utils import rotation_gate

        theta = to_tensor([pi / 6], dtype='float64')
        Rx = rotation_gate('x', theta)
        Ry = rotation_gate('y', theta)
        Rz = rotation_gate('z', theta)
        print("Matrix of Rotation X gate with angle pi/6: \n", Rx.numpy())
        print("Matrix of Rotation Y gate with angle pi/6: \n", Ry.numpy())
        print("Matrix of Rotation Z gate with angle pi/6: \n", Rz.numpy())

    ::

        Matrix of Rotation X gate with angle pi/6:
         [[0.96592583+0.j         0.        -0.25881905j]
         [0.        -0.25881905j 0.96592583+0.j        ]]
        Matrix of Rotation Y gate with angle pi/6:
         [[ 0.96592583+0.j -0.25881905+0.j]
         [ 0.25881905+0.j  0.96592583+0.j]]
        Matrix of Rotation Z gate with angle pi/6:
         [[0.96592583-0.25881905j 0.        +0.j        ]
         [0.        +0.j         0.96592583+0.25881905j]]
    """
    # Check if the input theta is a Tensor and has 'float64' datatype
    float64_tensor = to_tensor([], dtype='float64')
    assert isinstance(theta, Tensor) and theta.dtype == float64_tensor.dtype, \
        "The rotation angle should be a Tensor and of type 'float64'."

    # Calculate half of the input theta
    half_theta = multiply(theta, to_tensor([0.5], dtype='float64'))

    if axis == 'x':  # Define the rotation - x gate matrix
        return multiply(pauli_gate('I'), cos(half_theta)) + \
               multiply(pauli_gate('X'), multiply(sin(half_theta), to_tensor([-1j], dtype='complex128')))
    elif axis == 'y':  # Define the rotation - y gate matrix
        return multiply(pauli_gate('I'), cos(half_theta)) + \
               multiply(pauli_gate('Y'), multiply(sin(half_theta), to_tensor([-1j], dtype='complex128')))
    elif axis == 'z':  # Define the rotation - z gate matrix
        return multiply(pauli_gate('I'), cos(half_theta)) + \
               multiply(pauli_gate('Z'), multiply(sin(half_theta), to_tensor([-1j], dtype='complex128')))
    else:
        raise KeyError("invalid rotation gate index: %s, the rotation axis must be 'x', 'y' or 'z'." % axis)


def to_projector(vector):
    r"""Transform a vector into its density matrix (or measurement projector).

    .. math::

        |\psi\rangle \to |\psi\rangle\langle\psi|

    Args:
        vector (Tensor): Vector of a quantum state or a measurement basis

    Returns:
        Tensor: Density matrix (or measurement projector)

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import zero_state, plus_state
        from paddle_quantum.mbqc.utils import to_projector

        zero_proj = to_projector(zero_state())
        plus_proj = to_projector(plus_state())
        print("The projector of zero state: \n", zero_proj.numpy())
        print("The projector of plus state: \n", plus_proj.numpy())

    ::

        The projector of zero state:
         [[1. 0.]
         [0. 0.]]
        The projector of plus state:
         [[0.5 0.5]
         [0.5 0.5]]
    """
    assert isinstance(vector, Tensor) and vector.shape[0] >= 1 and vector.shape[1] == 1, \
        "'vector' must be a Tensor of shape (x, 1) with x >= 1."
    return matmul(vector, t(conj(vector)))


def basis(label, theta=to_tensor([0], dtype='float64')):
    r"""Measurement basis.

    Note:
        Commonly used measurements are measurements in the XY and YZ planes, and Pauli X, Y, Z measurements.

    .. math::
        \begin{align*}
        & M^{XY}(\theta) = \{R_{z}(\theta)|+\rangle, R_{z}(\theta)|-\rangle\}\\
        & M^{YZ}(\theta) = \{R_{x}(\theta)|0\rangle, R_{x}(\theta)|1\rangle\}\\
        & X = M^{XY}(0)\\
        & Y = M^{YZ}(\pi / 2) = M^{XY}(-\pi / 2)\\
        & Z = M_{YZ}(0)
        \end{align*}

    Args:
        label (str): the labels of the measurement basis, "XY" denotes XY plane, "YZ" denotes YZ plane, 
            "X" denotes X measurement, "Y" denotes Y measurement, "Z" denotes Z measurement.
        theta (Tensor, optional): measurement angle, the parameter is needed when the measurement is in
            XY plane or YZ plane.

    Returns:
        list: the list composed of measurement basis, the elements are of type ``Tensor``.

    Code example:

    .. code-block:: python

        from numpy import pi
        from paddle import to_tensor
        from paddle_quantum.mbqc.utils import basis
        theta = to_tensor(pi / 6, dtype='float64')
        YZ_plane_basis = basis('YZ', theta)
        XY_plane_basis = basis('XY', theta)
        X_basis = basis('X')
        Y_basis = basis('Y')
        Z_basis = basis('Z')
        print("Measurement basis in YZ plane: \n", YZ_plane_basis)
        print("Measurement basis in XY plane: \n", XY_plane_basis)
        print("Measurement basis of X: \n", X_basis)
        print("Measurement basis of Y: \n", Y_basis)
        print("Measurement basis of Z: \n", Z_basis)

    ::

        Measurement basis in YZ plane:
         [Tensor(shape=[2, 1], dtype=complex128, place=CPUPlace, stop_gradient=True,
               [[(0.9659258262890683+0j)],
                [-0.25881904510252074j  ]]),
          Tensor(shape=[2, 1], dtype=complex128, place=CPUPlace, stop_gradient=True,
               [[-0.25881904510252074j  ],
                [(0.9659258262890683+0j)]])]
        Measurement basis in XY plane:
         [Tensor(shape=[2, 1], dtype=complex128, place=CPUPlace, stop_gradient=True,
               [[(0.6830127018922193-0.1830127018922193j)],
                [(0.6830127018922193+0.1830127018922193j)]]),
          Tensor(shape=[2, 1], dtype=complex128, place=CPUPlace, stop_gradient=True,
               [[ (0.6830127018922193-0.1830127018922193j)],
                [(-0.6830127018922193-0.1830127018922193j)]])]
        Measurement basis of X:
         [Tensor(shape=[2, 1], dtype=float64, place=CPUPlace, stop_gradient=True,
               [[0.70710678],
                [0.70710678]]),
          Tensor(shape=[2, 1], dtype=float64, place=CPUPlace, stop_gradient=True,
               [[ 0.70710678],
                [-0.70710678]])]
        Measurement basis of Y:
         [Tensor(shape=[2, 1], dtype=complex128, place=CPUPlace, stop_gradient=True,
               [[(0.5-0.5j)],
                [(0.5+0.5j)]]),
          Tensor(shape=[2, 1], dtype=complex128, place=CPUPlace, stop_gradient=True,
               [[ (0.5-0.5j)],
                [(-0.5-0.5j)]])]
        Measurement basis of Z:
         [Tensor(shape=[2, 1], dtype=float64, place=CPUPlace, stop_gradient=True,
               [[1.],
                [0.]]),
          Tensor(shape=[2, 1], dtype=float64, place=CPUPlace, stop_gradient=True,
               [[0.],
                [1.]])]
    """
    # Check the label and input angle
    assert label in ['XY', 'YZ', 'X', 'Y', 'Z'], "the basis label must be 'XY', 'YZ', 'X', 'Y' or 'Z'."
    float64_tensor = to_tensor([], dtype='float64')
    assert isinstance(theta, Tensor) and theta.dtype == float64_tensor.dtype, \
        "The input angle should be a tensor and of type 'float64'."

    if label == 'YZ':  # Define the YZ plane measurement basis
        return [matmul(rotation_gate('x', theta), zero_state()),
                matmul(rotation_gate('x', theta), one_state())]
    elif label == 'XY':  # Define the XY plane measurement basis
        return [matmul(rotation_gate('z', theta), plus_state()),
                matmul(rotation_gate('z', theta), minus_state())]
    elif label == 'X':  # Define the X-measurement basis
        return [plus_state(), minus_state()]
    elif label == 'Y':  # Define the Y-measurement basis
        return [matmul(rotation_gate('z', to_tensor([pi / 2], dtype='float64')), plus_state()),
                matmul(rotation_gate('z', to_tensor([pi / 2], dtype='float64')), minus_state())]
    elif label == 'Z':  # Define the Z-measurement basis
        return [zero_state(), one_state()]


def kron(tensor_list):
    r"""Take the tensor product of all the elements in the list.

    .. math::

        [A, B, C, \cdots] \to A \otimes B \otimes C \otimes \cdots

    Args:
        tensor_list (list): a list contains the element to taking tensor product.

    Returns:
        Tensor: the results of the tensor product are of type ``Tensor``. If there is only
        one ``Tensor`` in the list, return the ``Tensor``.

    Code example 1:

    .. code-block:: python

        from paddle import to_tensor
        from paddle_quantum.mbqc.utils import pauli_gate, kron
        tensor0 = pauli_gate('I')
        tensor1 = to_tensor([[1, 1], [1, 1]], dtype='float64')
        tensor2 = to_tensor([[1, 2], [3, 4]], dtype='float64')
        tensor_list = [tensor0, tensor1, tensor2]
        tensor_all = kron(tensor_list)
        print("The tensor product result: \n", tensor_all.numpy())

    ::

        The tensor product result:
        [[1. 2. 1. 2. 0. 0. 0. 0.]
         [3. 4. 3. 4. 0. 0. 0. 0.]
         [1. 2. 1. 2. 0. 0. 0. 0.]
         [3. 4. 3. 4. 0. 0. 0. 0.]
         [0. 0. 0. 0. 1. 2. 1. 2.]
         [0. 0. 0. 0. 3. 4. 3. 4.]
         [0. 0. 0. 0. 1. 2. 1. 2.]
         [0. 0. 0. 0. 3. 4. 3. 4.]]

    Code example 2:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import pauli_gate, kron
        tensor0 = pauli_gate('I')
        tensor_list = [tensor0]
        tensor_all = kron(tensor_list)
        print("The tensor product result: \n", tensor_all.numpy())

    ::

        The tensor product result:
        [[1. 0.]
        [0. 1.]]
    """
    assert isinstance(tensor_list, list), "'tensor_list' must be a `list`."
    assert all(isinstance(tensor, Tensor) for tensor in tensor_list), "each element in the list must be a `Tensor`."
    kron_all = tensor_list[0]
    if len(tensor_list) > 1:  # Kron together
        for i in range(1, len(tensor_list)):
            tensor = tensor_list[i]
            kron_all = pp_kron(kron_all, tensor)
    return kron_all


def permute_to_front(state, which_system):
    r"""Move a subsystem of a system to the first.

    Assume that a quantum state :math:`\psi\rangle` can be decomposed to tensor product form: 

    .. math::

        |\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes |\psi_3\rangle \otimes \cdots


    the labels of each :math:`|\psi_i\rangle` is :math:`i` , so the total labels of the current system are: 

    .. math::

        \text{label} = \{1, 2, 3, \cdots \}

    Assume that the label of the subsystem to be moved is: i

    The output new quantum state is: 

    .. math::

        |\psi_i\rangle \otimes |\psi_1\rangle \otimes \cdots |\psi_{i-1}\rangle \otimes |\psi_{i+1}\rangle \otimes \cdots

    Args:
        state (State): the quantum state to be processed
        which_system (str): the labels of the subsystem to be moved.

    Returns:
        State: the final state after the move operation.
    """
    assert which_system in state.system, 'the system to permute must be in the state systems.'
    system_idx = state.system.index(which_system)
    if system_idx == 0:  # system in the front
        return state
    elif system_idx == state.size - 1:  # system in the end
        new_shape = [2 ** (state.size - 1), 2]
        new_axis = [1, 0]
        new_system = [which_system] + state.system[: system_idx]
    else:  # system in the middle
        new_shape = [2 ** system_idx, 2, 2 ** (state.size - system_idx - 1)]
        new_axis = [1, 0, 2]
        new_system = [which_system] + state.system[: system_idx] + state.system[system_idx + 1:]
    new_vector = reshape(transpose(reshape(state.vector, new_shape), new_axis), [state.length, 1])

    return State(new_vector, new_system)


def permute_systems(state, new_system):
    r""" Permute the quantum system to given order

    Assume that a quantum state :math:`\psi\rangle` can be decomposed to tensor product form: 

    .. math::

        |\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes |\psi_3\rangle \otimes \cdots

    the labels of each :math:`|\psi_i\rangle` is :math:`i` , so the total labels of the current system are: 

    .. math::

        \text{label} = \{1, 2, 3, \cdots \}

    the order of labels of the given new system is: 

    .. math::

        \{i_1, i_2, i_3, \cdots \}

    The output new quantum state is: 

    .. math::

        |\psi_{i_1}\rangle \otimes |\psi_{i_2}\rangle \otimes |\psi_{i_3}\rangle \otimes \cdots

    Args:
        state (State): the quantum state to be processed
        new_system (list): target order of the system

    Returns:
        State: the quantum state after permutation.
    """
    for label in reversed(new_system):
        state = permute_to_front(state, label)
    return state


def compare_by_density(state1, state2):
    r"""Compare whether two quantum states are the same by their density operators.

    Args:
        state1 (State): the first quantum state
        state2 (State): the second quantum state
    """
    assert state1.size == state2.size, "two state vectors compared are not of the same length."

    # Permute the system order
    new_state1 = permute_systems(state1, state2.system)
    # Transform the vector to density
    density1 = to_projector(new_state1.vector)
    density2 = to_projector(state2.vector)

    error = linalg.norm(density1.numpy() - density2.numpy())

    print("Norm difference of the given states is: \r\n", error)
    eps = 1e-12  # Error criterion
    if error < eps:
        print("They are exactly the same states.")
    elif 1e-10 > error >= eps:
        print("They are probably the same states.")
    else:
        print("They are not the same states.")


def compare_by_vector(state1, state2):
    r"""Compare whether two quantum states are the same by their column vector form.

    Args:
        state1 (State): the first quantum state
        state2 (State): the second quantum state
    """
    assert state1.size == state2.size, "two state vectors compared are not of the same length."
    # Check if they are normalized quantum states
    eps = 1e-12  # Error criterion
    if state1.norm >= 1 + eps or state1.norm <= 1 - eps:
        raise ValueError("the first state is not normalized.")
    elif state2.norm >= 1 + eps or state2.norm <= 1 - eps:
        raise ValueError("the second state is not normalized.")
    else:
        new_state1 = permute_systems(state1, state2.system)
        vector1_list = list(new_state1.vector.numpy())
        idx = vector1_list.index(max(vector1_list, key=abs))
        if - eps <= state2.vector[idx].numpy() <= eps:
            print("They are not the same states.")
        else:
            # Calculate the phase and erase it
            phase = new_state1.vector[idx] / state2.vector[idx]
            vector1_phase = new_state1.vector / phase
            error = linalg.norm(vector1_phase - state2.vector)

            print("Norm difference of the given states is: \r\n", error)
            if error < eps:
                print("They are exactly the same states.")
            elif 1e-10 > error >= eps:
                print("They are probably the same states.")
            else:
                print("They are not the same states.")


def random_state_vector(n, is_real=False):
    r"""Generate a state vector randomly.

    Args:
        n (int): the number of qubits of the random state.
        is_real (int, optional): ``True`` denotes a state vector with real values, ``False`` denotes a quantum
        state with complex values, default to ``False``

    Returns:
        Tensor: the column vector of the random quantum state.

    Code example:

    .. code-block:: python

        from paddle_quantum.mbqc.utils import random_state_vector
        random_vec = random_state_vector(2)
        print(random_vec.numpy())
        random_vec = random_state_vector(1, is_real=True)
        print(random_vec.numpy())

    ::

        [[-0.06831946+0.04548425j]
         [ 0.60460088-0.16733175j]
         [ 0.39185213-0.24831266j]
         [ 0.45911355-0.41680807j]]
        [[0.77421121]
         [0.63292732]]
    """
    assert isinstance(n, int) and n >= 1, "the number of qubit must be a int larger than one."
    assert isinstance(is_real, bool), "'is_real' must be a bool."

    if is_real:
        psi = to_tensor(np_random.randn(2 ** n, 1), dtype='float64')
        inner_prod = matmul(t(conj(psi)), psi)
    else:
        psi = to_tensor(np_random.randn(2 ** n, 1) + 1j * np_random.randn(2 ** n, 1), dtype='complex128')
        inner_prod = real(matmul(t(conj(psi)), psi))

    psi = psi / pp_sqrt(inner_prod)  # Normalize the vector
    return psi


def div_str_to_float(div_str):
    r"""Converts the division string to the corresponding floating point number.

    For example, the string '3/2' to the float number 1.5.

    Args:
        div_str (str): division string

    Returns:
        float: the float number

    Code example:

    ..  code-block:: python

        from paddle_quantum.mbqc.utils import div_str_to_float
        division_str = "1/2"
        division_float = div_str_to_float(division_str)
        print("The corresponding float value is: ", division_float)

    ::

        The corresponding float value is:  0.5
    """
    div_str = div_str.split("/")
    return float(div_str[0]) / float(div_str[1])


def int_to_div_str(idx1, idx2=1):
    r"""Transform two integers to a division string.

    Args:
        idx1 (int): the first integer
        idx2 (int): the second integer

    Returns:
        str: the division string

    Code example:

    ..  code-block:: python

        from paddle_quantum.mbqc.utils import int_to_div_str
        one = 1
        two = 2
        division_string = int_to_div_str(one, two)
        print("The corresponding division string is: ", division_string)

    ::

        The corresponding division string is:  1/2
    """
    assert isinstance(idx1, int) and isinstance(idx2, int), "two input parameters must be int."
    return str(idx1) + "/" + str(idx2)


def print_progress(current_progress, progress_name, track=True):
    r"""Plot the progress bar.

    Args:
        current_progress (float / int): the percentage of the current progress.
        progress_name (str): the name of the current progress.
        track (bool): the boolean switch of whether plot.

    Code example:

    ..  code-block:: python

        from paddle_quantum.mbqc.utils import print_progress
        print_progress(14/100, "Current Progress")

    ::

       Current Progress              |■■■■■■■                                           |   14.00%
    """
    assert 0 <= current_progress <= 1, "'current_progress' must be between 0 and 1"
    assert isinstance(track, bool), "'track' must be a bool."
    if track:
        print(
            "\r"
            f"{progress_name.ljust(30)}"
            f"|{'■' * int(50 * current_progress):{50}s}| "
            f"\033[94m {'{:6.2f}'.format(100 * current_progress)}% \033[0m ", flush=True, end=""
        )
        if current_progress == 1:
            print(" (Done)")


def plot_results(dict_lst, bar_label, title, xlabel, ylabel, xticklabels=None):
    r"""Plot the histogram based on the key-value pair of the dict.
        The key is the abscissa, and the corresponding value is the ordinate

    Note:
        The function is mainly used for plotting the sampling statistics or histogram.

    Args:
        dict_lst (list): a list contains the data to be plotted
        bar_label (list): the name of different bars in the histogram
        title (str): the title of the figure
        xlabel (str): the label of the x axis.
        ylabel (str): the label of the y axis.
        xticklabels (list, optional): the label of each ticks of the x-axis.
    """
    assert isinstance(dict_lst, list), "please input a list with dictionaries."
    assert isinstance(bar_label, list), "please input a list with bar_labels."
    assert len(dict_lst) == len(bar_label), \
        "please check your input as the number of dictionaries and bar labels are not equal."
    bars_num = len(dict_lst)
    bar_width = 1 / (bars_num + 1)
    plt.ion()
    plt.figure()
    for i in range(bars_num):
        plot_dict = dict_lst[i]
        # Obtain the y label and xticks in order
        keys = list(plot_dict.keys())
        values = list(plot_dict.values())
        xlen = len(keys)
        xticks = [((i) / (bars_num + 1)) + j for j in range(xlen)]
        # Plot bars
        plt.bar(xticks, values, width=bar_width, align='edge', label=bar_label[i])
        plt.yticks()
    if xticklabels is None:
        plt.xticks(list(range(xlen)), keys, rotation=90)
    else:
        assert len(xticklabels) == xlen, "the 'xticklabels' should have the same length with 'x' length."
        plt.xticks(list(range(xlen)), xticklabels, rotation=90)
    plt.legend()
    plt.title(title, fontproperties='SimHei', fontsize='x-large')
    plt.xlabel(xlabel, fontproperties='SimHei')
    plt.ylabel(ylabel, fontproperties='SimHei')
    plt.ioff()
    plt.show()


def write_running_data(textfile, eg, width, mbqc_time, reference_time):
    r"""Write the running times of the quantum circuit.

    In many cases of circuit models, we need to compare the simulation time of our MBQC model with other methods
    such as qiskit or ``UAnsatz`` circuit model in paddle_quantum. So we define this function.

    Hint:
        this function is used with the ``read_running_data`` function.

    Warning:
        Before using this function, we need to open a textfile. After using this function,
        we need to close the textfile.

    Args:
        textfile (TextIOWrapper): the file to be written in.
        eg (str): the name of the current case
        width (float): the width of the circuit(number of qubits)
        mbqc_time (float): `the simulation time of the ``MBQC`` model.
        reference_time (float):  the simulation time of the circuit
        based model in qiskit or ``UAnsatz`` in paddle_quantum.
    """
    textfile.write("The current example is: " + eg + "\n")
    textfile.write("The qubit number is: " + str(width) + "\n")
    textfile.write("MBQC running time is: " + str(mbqc_time) + " s\n")
    textfile.write("Circuit model running time is: " + str(reference_time) + " s\n\n")


def read_running_data(file_name):
    r"""Read the running time of the quantum circuit.

    In many cases of circuit models, we need to compare the simulation time of our MBQC model with other methods
    such as qiskit or ``UAnsatz`` circuit model in paddle_quantum. So we define this function and save the running
    time to a list. There are two dicts in the list, the first dict contains the running time of qiskit or ``UAnsatz``,
    the second contains the simulation time of our MBQC model.

    Hint:
        This function is used with the ``write_running_data`` function.

    Args:
        file_name (str): the name of the file to be read.

    Returns:
        list: The list of running time.
    """
    bit_num_lst = []
    mbqc_list = []
    reference_list = []
    remainder = {2: bit_num_lst, 3: mbqc_list, 4: reference_list}
    # Read data
    with open(file_name, 'r') as file:
        counter = 0
        for line in file:
            counter += 1
            if counter % 5 in remainder.keys():
                remainder[counter % 5].append(float(line.strip("\n").split(":")[1].split(" ")[1]))

    # Transform the lists to dictionaries
    mbqc_dict = {i: mbqc_list[i] for i in range(len(bit_num_lst))}
    refer_dict = {i: reference_list[i] for i in range(len(bit_num_lst))}
    dict_lst = [mbqc_dict, refer_dict]
    return dict_lst
