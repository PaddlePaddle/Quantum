paddle\_quantum.mbqc.utils
==========================

此模块包含计算所需的各种常用类和函数。

.. py:function:: plus_state()

   定义加态。

   其矩阵形式为：

   .. math::

      \frac{1}{\sqrt{2}}  \begin{bmatrix}  1 \\ 1 \end{bmatrix}

   代码示例：

   .. code-block:: python

      from paddle_quantum.mbqc.utils import plus_state
      print("State vector of plus state: \n", plus_state().numpy())

   ::

      State vector of plus state:
      [[0.70710678]
       [0.70710678]]

   :return: 加态对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: minus_state()

   定义减态。

   其矩阵形式为：

   .. math::

      \frac{1}{\sqrt{2}}  \begin{bmatrix}  1 \\ -1 \end{bmatrix}

   代码示例：

   .. code-block:: python

      from paddle_quantum.mbqc.utils import minus_state
      print("State vector of minus state: \n", minus_state().numpy())

   ::

      State vector of minus state:
      [[ 0.70710678]
       [-0.70710678]]

   :return: 减态对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: zero_state()
   :noindex:

   定义零态。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}  1 \\ 0 \end{bmatrix}

   代码示例：

   .. code-block:: python

      from paddle_quantum.mbqc.utils import zero_state
      print("State vector of zero state: \n", zero_state().numpy())

   ::

      State vector of zero state:
      [[1.]
       [0.]]

   :return: 零态对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: one_state()

   定义一态。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}  0 \\ 1 \end{bmatrix}

   代码示例：

   .. code-block:: python

      from paddle_quantum.mbqc.utils import one_state
      print("State vector of one state: \n", one_state().numpy())

   ::

      State vector of one state:
      [[0.]
       [1.]]

   :return: 一态对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: h_gate()
   :noindex:

   定义 ``Hadamard`` 门。

   其矩阵形式为：

   .. math::

      \frac{1}{\sqrt{2}} \begin{bmatrix}  1 & 1 \\ 1 & -1 \end{bmatrix}

   代码示例：

   .. code-block:: python

      from paddle_quantum.mbqc.utils import h_gate
      print("Matrix of Hadamard gate: \n", h_gate().numpy())

   ::

      Matrix of Hadamard gate:
      [[ 0.70710678  0.70710678]
       [ 0.70710678 -0.70710678]]

   :return: ``Hadamard`` 门对应矩阵的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: s_gate()
   :noindex:

   定义 ``S`` 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}  1 & 0 \\ 0 & i \end{bmatrix}

   代码示例：

   .. code-block:: python

      from paddle_quantum.mbqc.utils import s_gate
      print("Matrix of S gate:\n", s_gate().numpy())

   ::

      Matrix of S gate:
      [[1.+0.j 0.+0.j]
       [0.+0.j 0.+1.j]]

   :return: ``S`` 门矩阵对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: t_gate()
   :noindex:

   定义 ``T`` 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}  1 & 0 \\ 0 & e^{i \pi / 4} \end{bmatrix}

   代码示例：

   .. code-block:: python

      from paddle_quantum.mbqc.utils import t_gate
      print("Matrix of T gate: \n", t_gate().numpy())

   ::

      Matrix of T gate:
      [[1.        +0.j         0.        +0.j        ]
       [0.        +0.j         0.70710678+0.70710678j]]

   :return: ``T`` 门矩阵对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: cz_gate()
   :noindex:

   定义 ``Controlled-Z`` 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{bmatrix}

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

   :return: ``Controlled-Z`` 门矩阵对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: cnot_gate()
   :noindex:

   定义 ``Controlled-NOT (CNOT)`` 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

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

   :return: ``Controlled-NOT (CNOT)`` 门矩阵对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: swap_gate()
   :noindex:

   定义 ``SWAP`` 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}  1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}

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

   :return: ``SWAP`` 门矩阵对应的 ``Tensor`` 形式
   :rtype: paddle.Tensor

.. py:function:: pauli_gate(gate)

   定义 ``Pauli`` 门。

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

   :param gate: Pauli 门的索引字符，"I", "X", "Y", "Z" 分别表示对应的门
   :type gate: str
   :return: Pauli 门对应的矩阵
   :rtype: paddle.Tensor

.. py:function:: rotation_gate(gate)

   定义旋转门矩阵。

   .. math::

      R_{x}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) X

      R_{y}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Y

      R_{z}(\theta) = \cos(\theta / 2) I - i\sin(\theta / 2) Z

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

   :param axis: 旋转轴，绕 ``X`` 轴旋转输入 'x'，绕 ``Y`` 轴旋转输入 'y'，绕 ``Z`` 轴旋转输入 'z'
   :type axis: str
   :param theta: 旋转的角度
   :type theta: paddle.Tensor
   :return: 旋转门对应的矩阵
   :rtype: paddle.Tensor

.. py:function:: to_projector(vector)

   把列向量转化为密度矩阵（或测量基对应的投影算符）。

   .. math::

      |\psi\rangle \to |\psi\rangle\langle\psi|

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

   :param vector: 量子态列向量（或投影测量中的测量基向量）
   :type vector: paddle.Tensor
   :return: 密度矩阵（或测量基对应的投影算符）
   :rtype: paddle.Tensor

.. py:function:: basis(label, theta)

   测量基。

   .. note::

      常用的测量方式有 XY-平面测量，YZ-平面测量，X 测量，Y 测量，Z 测量。

   .. math::

      \begin{align*}
      & M^{XY}(\theta) = \{R_{z}(\theta)|+\rangle, R_{z}(\theta)|-\rangle\}\\
      & M^{YZ}(\theta) = \{R_{x}(\theta)|0\rangle, R_{x}(\theta)|1\rangle\}\\
      & X = M^{XY}(0)\\
      & Y = M^{YZ}(\pi / 2) = M^{XY}(-\pi / 2)\\
      & Z = M_{YZ}(0)
      \end{align*}

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

   :param label: 测量基索引字符，"XY" 表示 XY-平面测量，"YZ" 表示 YZ-平面测量，"X" 表示 X 测量，"Y" 表示 Y 测量，"Z" 表示 Z 测量
   :type label: str
   :param theta: 测量角度，这里只有 XY-平面测量和 YZ-平面测量时需要
   :type theta: Optional[paddle.Tensor]
   :return: 测量基向量构成的列表，列表元素为 ``Tensor`` 类型
   :rtype: List[paddle.Tensor]

.. py:function:: kron(tensor_list)

   把列表中的所有元素做张量积。

   .. math::

      [A, B, C, \cdots] \to A \otimes B \otimes C \otimes \cdots

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

   :param tensor_list: 需要做张量积的元素组成的列表
   :type tensor_list: List[paddle.Tensor]
   :return: 所有元素做张量积运算得到的 ``Tensor``，当列表中只有一个 ``Tensor`` 时，返回该 ``Tensor`` 本身
   :rtype: paddle.Tensor

.. py:function:: permute_to_front(state, which_system)

   将一个量子态中某个子系统的顺序变换到最前面。

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

   :param state: 需要操作的量子态
   :type state: State
   :param which_system: 要变换到最前面的子系统标签
   :type which_system: str
   :return: 系统顺序变换后的量子态
   :rtype: State

.. py:function:: permute_systems(state, new_system)
   :noindex:

   变换量子态的系统到指定顺序。

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

   :param state: 需要操作的量子态
   :type state: State
   :param new_system: 目标系统顺序
   :type new_system: list
   :return: 系统顺序变换后的量子态
   :rtype: State

.. py:function:: compare_by_density(state1, state2)

   通过密度矩阵形式比较两个量子态是否相同。

   :param state1: 第一个量子态
   :type state1: State
   :param state2: 第二个量子态
   :type state2: State

.. py:function:: compare_by_vector(state1, state2)

   通过列向量形式比较两个量子态是否相同。

   :param state1: 第一个量子态
   :type state1: State
   :param state2: 第二个量子态
   :type state2: State

.. py:function:: random_state_vector(n, is_real)

   随机生成一个量子态列向量。

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

   :param n: 随机生成的量子态的比特数
   :type n: int
   :param is_real: ``True`` 表示实数量子态，``False`` 表示复数量子态，默认为 ``False``
   :type is_real: Optional[bool]
   :return: 随机生成量子态的列向量
   :rtype: paddle.Tensor

.. py:function:: div_str_to_float(div_str)

   将除式字符串转化为对应的浮点数。

   例如将字符串 '3/2' 转化为 1.5。

   代码示例：

   ..  code-block:: python

      from paddle_quantum.mbqc.utils import div_str_to_float
      division_str = "1/2"
      division_float = div_str_to_float(division_str)
      print("The corresponding float value is: ", division_float)

   ::

      The corresponding float value is:  0.5

   :param div_str: 除式字符串
   :type div_str: str
   :return: 除式对应的浮点数结果
   :rtype: float

.. py:function:: int_to_div_str(idx1, idx2)

   将两个整数转化为除式字符串。

   代码示例：

   ..  code-block:: python

      from paddle_quantum.mbqc.utils import int_to_div_str
      one = 1
      two = 2
      division_string = int_to_div_str(one, two)
      print("The corresponding division string is: ", division_string)

   ::

      The corresponding division string is:  1/2

   :param idx1: 第一个整数
   :type idx1: int
   :param idx2: 第二个整数
   :type idx2: int
   :return: 对应的除式字符串
   :rtype: str

.. py:function:: print_progress(current_progress, progress_name, track)

   画出当前步骤的进度条。

   代码示例：

   ..  code-block:: python

      from paddle_quantum.mbqc.utils import print_progress
      print_progress(14/100, "Current Progress")

   ::

      Current Progress              |■■■■■■■                                           |   14.00%

   :param current_progress: 当前的进度百分比
   :type current_progress: float
   :param progress_name: 当前步骤的名称
   :type progress_name: str
   :param track: 是否绘图的布尔开关
   :type track: bool
   :return: 对应的除式字符串
   :rtype: str

.. py:function:: plot_results(dict_lst, bar_label, title, xlabel, ylabel, xticklabels)

   根据字典的键值对，以键为横坐标，对应的值为纵坐标，画出柱状图。

   .. note::

      该函数主要调用来画出采样分布或时间比较的柱状图。

   :param dict_lst: 待画图的字典列表
   :type dict_lst: list
   :param bar_label: 每种柱状图对应的名称
   :type bar_label: list
   :param title: 整个图的标题
   :type title: str
   :param xlabel: 横坐标的名称
   :type xlabel: str
   :param ylabel: 纵坐标的名称
   :type ylabel: str
   :param xticklabels: 柱状图中每个横坐标的名称
   :type xticklabels: Optional[list]

.. py:function:: write_running_data(textfile, eg, width, mbqc_time, reference_time)

   写入电路模拟运行的时间。

   由于在许多电路模型模拟案例中，需要比较我们的 ``MBQC`` 模拟思路与 ``Qiskit`` 或量桨平台的电路模型模拟思路的运行时间。因而单独定义了写入文件函数。

   .. hint::

      该函数与 ``read_running_data`` 函数配套使用。

   .. warning::

      在调用该函数之前，需要调用 ``open`` 打开 ``textfile``；在写入结束之后，需要调用 ``close`` 关闭 ``textfile``。

   :param textfile: 待写入的文件
   :type textfile: TextIOWrapper
   :param eg: 当前案例的名称
   :type eg: str
   :param width: 电路宽度（比特数）
   :type width: float
   :param mbqc_time: ``MBQC`` 模拟电路运行时间
   :type mbqc_time: float
   :param reference_time: ``Qiskit`` 或量桨平台的 ``UAnsatz`` 电路模型运行时间
   :type reference_time: flaot

.. py:function:: write_running_data(file_name)
   :noindex:

   读取电路模拟运行的时间。

    由于在许多电路模型模拟案例中，需要比较我们的 ``MBQC`` 模拟思路与 ``Qiskit`` 或量桨平台的电路模型模拟思路的运行时间。因而单独定义了读取文件函数读取运行时间，将其处理为一个列表，列表中的两个元素分别为 ``Qiskit`` 或量桨平台的 ``UAnsatz`` 电路模型模拟思路的运行时间

   .. hint::

      该函数与 ``write_running_data`` 函数配套使用。

   :param file_name: 待读取的文件名
   :type file_name: str
   :return: 运行时间列表
   :rtype: list
