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
此模块包含计算所需的各种常用类和函数。
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
    r"""定义加态。

    其矩阵形式为：

    .. math::

        \frac{1}{\sqrt{2}}  \begin{bmatrix}  1 \\ 1 \end{bmatrix}

    Returns:
        Tensor: 加态对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义减态。

    其矩阵形式为：

    .. math::

        \frac{1}{\sqrt{2}}  \begin{bmatrix}  1 \\ -1 \end{bmatrix}

    Returns:
        Tensor: 减态对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义零态。

    其矩阵形式为：

    .. math::

        \begin{bmatrix}  1 \\ 0 \end{bmatrix}

    Returns:
        Tensor: 零态对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义一态。

    其矩阵形式为：

    .. math::

        \begin{bmatrix}  0 \\ 1 \end{bmatrix}

    Returns:
        Tensor: 一态对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义 ``Hadamard`` 门。

    其矩阵形式为：

    .. math::

        \frac{1}{\sqrt{2}} \begin{bmatrix}  1 & 1 \\ 1 & -1 \end{bmatrix}

    Returns:
        Tensor: ``Hadamard`` 门对应矩阵的 ``Tensor`` 形式

    代码示例：

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
    r"""定义 ``S`` 门。

    其矩阵形式为：

    .. math::

        \begin{bmatrix}  1 & 0 \\ 0 & i \end{bmatrix}

    Returns:
        Tensor: ``S`` 门矩阵对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义 ``T`` 门。

    其矩阵形式为：

    .. math::

        \begin{bmatrix}  1 & 0 \\ 0 & e^{i \pi / 4} \end{bmatrix}

    Returns:
        Tensor: ``T`` 门矩阵对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义 ``Controlled-Z`` 门。

    其矩阵形式为：

    .. math::

        \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{bmatrix}

    Returns:
        Tensor: ``Controlled-Z`` 门矩阵对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义 ``Controlled-NOT (CNOT)`` 门。

    其矩阵形式为：

    .. math::

        \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

    Returns:
        Tensor: ``Controlled-NOT (CNOT)`` 门矩阵对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义 ``SWAP`` 门。

    其矩阵形式为：

    .. math::

        \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}

    Returns:
        Tensor: ``SWAP`` 门矩阵对应的 ``Tensor`` 形式

    代码示例：

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
    r"""定义 ``Pauli`` 门。

    单位阵 ``I`` 的矩阵形式为：

    .. math::

        \begin{bmatrix}  1 & 0 \\ 0 & 1 \end{bmatrix}

    ``Pauli X`` 门的矩阵形式为：

    .. math::

        \begin{bmatrix}  0 & 1 \\ 1 & 0 \end{bmatrix}

    ``Pauli Y`` 门的矩阵形式为：

    .. math::

        \begin{bmatrix}  0 & - i \\ i & 0 \end{bmatrix}

    ``Pauli Z`` 门的矩阵形式为：

    .. math::

        \begin{bmatrix}  1 & 0 \\ 0 & - 1 \end{bmatrix}

    Args:
        gate (str): Pauli 门的索引字符，"I", "X", "Y", "Z" 分别表示对应的门

    Returns:
        Tensor: Pauli 门对应的矩阵

    代码示例：

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
    r"""定义旋转门矩阵。

    .. math::

        R_{x}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) X

        R_{y}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Y

        R_{z}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Z

    Args:
        axis (str): 旋转轴，绕 ``X`` 轴旋转输入 'x'，绕 ``Y`` 轴旋转输入 'y'，绕 ``Z`` 轴旋转输入 'z'
        theta (Tensor): 旋转的角度

    Returns:
        Tensor: 旋转门对应的矩阵

    代码示例：

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
    r"""把列向量转化为密度矩阵（或测量基对应的投影算符）。

    .. math::

        |\psi\rangle \to |\psi\rangle\langle\psi|

    Args:
        vector (Tensor): 量子态列向量（或投影测量中的测量基向量）

    Returns:
        Tensor: 密度矩阵（或测量基对应的投影算符）

    代码示例：

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
    r"""测量基。

    Note:
        常用的测量方式有 XY-平面测量，YZ-平面测量，X 测量，Y 测量，Z 测量。

    .. math::
        \begin{align*}
        & M^{XY}(\theta) = \{R_{z}(\theta)|+\rangle, R_{z}(\theta)|-\rangle\}\\
        & M^{YZ}(\theta) = \{R_{x}(\theta)|0\rangle, R_{x}(\theta)|1\rangle\}\\
        & X = M^{XY}(0)\\
        & Y = M^{YZ}(\pi / 2) = M^{XY}(-\pi / 2)\\
        & Z = M_{YZ}(0)
        \end{align*}

    Args:
        label (str): 测量基索引字符，"XY" 表示 XY-平面测量，"YZ" 表示 YZ-平面测量，"X" 表示 X 测量，"Y" 表示 Y 测量，"Z" 表示 Z 测量
        theta (Tensor, optional): 测量角度，这里只有 XY-平面测量和 YZ-平面测量时需要

    Returns:
        list: 测量基向量构成的列表，列表元素为 ``Tensor`` 类型

    代码示例：

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
    r"""把列表中的所有元素做张量积。

    .. math::

        [A, B, C, \cdots] \to A \otimes B \otimes C \otimes \cdots

    Args:
        tensor_list (list): 需要做张量积的元素组成的列表

    Returns:
        Tensor: 所有元素做张量积运算得到的 ``Tensor``，当列表中只有一个 ``Tensor`` 时，返回该 ``Tensor`` 本身

    代码示例 1：

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

    代码示例 2：

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
    r"""将一个量子态中某个子系统的顺序变换到最前面。

    假设当前系统的量子态列向量 :math:`\psi\rangle` 可以分解成多个子系统列向量的 tensor product 形式：

    .. math::

        |\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes |\psi_3\rangle \otimes \cdots


    每个 :math:`|\psi_i\rangle` 的系统标签为 :math:`i` ，则当前总系统的标签为：

    .. math::

        \text{label} = \{1, 2, 3, \cdots \}

    假设需要操作的子系统的标签为：i

    输出新系统量子态的列向量为：

    .. math::

        |\psi_i\rangle \otimes |\psi_1\rangle \otimes \cdots |\psi_{i-1}\rangle \otimes |\psi_{i+1}\rangle \otimes \cdots

    Args:
        state (State): 需要操作的量子态
        which_system (str): 要变换到最前面的子系统标签

    Returns:
        State: 系统顺序变换后的量子态
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
    r"""变换量子态的系统到指定顺序。

    假设当前系统的量子态列向量 :math:`|\psi\rangle` 可以分解成多个子系统列向量的 tensor product 形式：

    .. math::

        |\psi\rangle = |\psi_1\rangle \otimes |\psi_2\rangle \otimes |\psi_3\rangle \otimes \cdots

    每个 :math:`\psi_i\rangle` 的系统标签为 :math:`i` ，则当前总系统的标签为：

    .. math::

        \text{label} = \{1, 2, 3, \cdots \}

    给定新系统的标签顺序为：

    .. math::

        \{i_1, i_2, i_3, \cdots \}

    输出新系统量子态的列向量为：

    .. math::

        |\psi_{i_1}\rangle \otimes |\psi_{i_2}\rangle \otimes |\psi_{i_3}\rangle \otimes \cdots

    Args:
        state (State): 需要操作的量子态
        new_system (list): 目标系统顺序

    Returns:
        State: 系统顺序变换后的量子态
    """
    for label in reversed(new_system):
        state = permute_to_front(state, label)
    return state


def compare_by_density(state1, state2):
    r"""通过密度矩阵形式比较两个量子态是否相同。

    Args:
        state1 (State): 第一个量子态
        state2 (State): 第二个量子态
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
    r"""通过列向量形式比较两个量子态是否相同。

    Args:
        state1 (State): 第一个量子态
        state2 (State): 第二个量子态
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
    r"""随机生成一个量子态列向量。

    Args:
        n (int): 随机生成的量子态的比特数
        is_real (int, optional): ``True`` 表示实数量子态，``False`` 表示复数量子态，默认为 ``False``

    Returns:
        Tensor: 随机生成量子态的列向量

    代码示例：

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
    r"""将除式字符串转化为对应的浮点数。

    例如将字符串 '3/2' 转化为 1.5。

    Args:
        div_str (str): 除式字符串

    Returns:
        float: 除式对应的浮点数结果

    代码示例：

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
    r"""将两个整数转化为除式字符串。

    Args:
        idx1 (int): 第一个整数
        idx2 (int): 第二个整数

    Returns:
        str: 对应的除式字符串

    代码示例：

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
    r"""画出当前步骤的进度条。

    Args:
        current_progress (float / int): 当前的进度百分比
        progress_name (str): 当前步骤的名称
        track (bool): 是否绘图的布尔开关

    代码示例：

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
    r"""根据字典的键值对，以键为横坐标，对应的值为纵坐标，画出柱状图。

    Note:
        该函数主要调用来画出采样分布或时间比较的柱状图。

    Args:
        dict_lst (list): 待画图的字典列表
        bar_label (list): 每种柱状图对应的名称
        title (str): 整个图的标题
        xlabel (str): 横坐标的名称
        ylabel (str): 纵坐标的名称
        xticklabels (list, optional): 柱状图中每个横坐标的名称
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
    r"""写入电路模拟运行的时间。

    由于在许多电路模型模拟案例中，需要比较我们的 ``MBQC`` 模拟思路与 ``Qiskit`` 或量桨平台的 ``UAnsatz`` 电路模型模拟思路的运行时间。
    因而单独定义了写入文件函数。

    Hint:
        该函数与 ``read_running_data`` 函数配套使用。

    Warning:
        在调用该函数之前，需要调用 ``open`` 打开 ``textfile``；在写入结束之后，需要调用 ``close`` 关闭 ``textfile``。

    Args:
        textfile (TextIOWrapper): 待写入的文件
        eg (str): 当前案例的名称
        width (float): 电路宽度（比特数）
        mbqc_time (float): ``MBQC`` 模拟电路运行时间
        reference_time (float):  ``Qiskit`` 或量桨平台的 ``UAnsatz`` 电路模型运行时间
    """
    textfile.write("The current example is: " + eg + "\n")
    textfile.write("The qubit number is: " + str(width) + "\n")
    textfile.write("MBQC running time is: " + str(mbqc_time) + " s\n")
    textfile.write("Circuit model running time is: " + str(reference_time) + " s\n\n")


def read_running_data(file_name):
    r"""读取电路模拟运行的时间。

    由于在许多电路模型模拟案例中，需要比较我们的 ``MBQC`` 模拟思路与 ``Qiskit`` 或量桨平台的 ``UAnsatz`` 电路模型模拟思路的运行时间。
    因而单独定义了读取文件函数读取运行时间，将其处理为一个列表，
    列表中的两个元素分别为 ``Qiskit`` 或 ``UAnsatz`` 电路模型模拟思路运行时间的字典和 ``MBQC`` 模拟思路运行时间的字典。

    Hint:
        该函数与 ``write_running_data`` 函数配套使用。

    Args:
        file_name (str): 待读取的文件名

    Returns:
        list: 运行时间列表
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
