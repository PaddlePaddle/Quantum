paddle\_quantum.qinfo
============================

量子信息中的常用功能实现。

.. py:function:: partial_trace(state, dim1, dim2, A_or_B)

   计算量子态的偏迹。

   :param state: 输入的量子态。
   :type state: Union[np.ndarray, paddle.Tensor, State]
   :param dim1: 系统A的维数。
   :type dim1: int
   :param dim2: 系统B的维数。
   :type dim2: int
   :param A_or_B: 1或者2，1表示计算系统A上的偏迹，2表示计算系统B上的偏迹。
   :type A_or_B: int

   :return: 输入的量子态的偏迹。
   :rtype: Union[np.ndarray, paddle.Tensor, State]

.. py:function:: partial_trace_discontiguous(state, preserve_qubits=None)

   计算量子态的偏迹，可选取任意子系统。

   :param state: 输入的量子态。
   :type state: Union[np.ndarray, paddle.Tensor, State]
   :param preserve_qubits: 要保留的量子比特，默认为 None，表示全保留。
   :type preserve_qubits: list, optional
   
   :return: 所选子系统的量子态偏迹。
   :rtype: Union[np.ndarray, paddle.Tensor, State]

.. py:function:: trace_distance(rho, sigma)

   计算两个量子态的迹距离。

   .. math::

      D(\rho, \sigma) = 1 / 2 * \text{tr}|\rho-\sigma|

   :param rho: 量子态的密度矩阵形式。
   :type rho: Union[np.ndarray, paddle.Tensor, State]
   :param sigma: 量子态的密度矩阵形式。
   :type sigma: Union[np.ndarray, paddle.Tensor, State]

   :return: 输入的量子态之间的迹距离。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: state_fidelity(rho, sigma)

   计算两个量子态的保真度。

   .. math::

      F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})

   :param rho: 量子态的密度矩阵形式。
   :type rho: Union[np.ndarray, paddle.Tensor, State]
   :param sigma: 量子态的密度矩阵形式。
   :type sigma: Union[np.ndarray, paddle.Tensor, State]
   :return: 输入的量子态之间的保真度。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: gate_fidelity(U, V)

   计算两个量子门的保真度。

   .. math::

      F(U, V) = |\text{tr}(UV^\dagger)|/2^n

   :math:`U` 是一个 :math:`2^n\times 2^n` 的 Unitary 矩阵。

   :param U: 量子门 :math:`U` 的酉矩阵形式。
   :type U: Union[np.ndarray, paddle.Tensor]
   :param V: 量子门 :math:`V` 的酉矩阵形式。
   :type V: Union[np.ndarray, paddle.Tensor]

   :return: 输入的量子门之间的保真度。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: purity(rho)

   计算量子态的纯度。

   .. math::

      P = \text{tr}(\rho^2)

   :param rho: 量子态的密度矩阵形式。
   :type rho: Union[np.ndarray, paddle.Tensor, State]

   :return: 输入的量子态的纯度。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: von_neumann_entropy(rho, base)

    计算量子态的冯诺依曼熵。

   .. math::

      S = -\text{tr}(\rho \log(\rho))

   :param rho: 量子态的密度矩阵形式。
   :type rho: Union[np.ndarray, paddle.Tensor, State]
   :param base: 对数的底。默认为2。
   :type base: int, optional

   :return: 输入的量子态的冯诺依曼熵。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: relative_entropy(rho, sig, base)

   计算两个量子态的相对熵。

   .. math::

      S(\rho \| \sigma)=\text{tr} \rho(\log \rho-\log \sigma)

   :param rho: 量子态的密度矩阵形式。
   :type rho: Union[np.ndarray, paddle.Tensor, State]
   :param sig: 量子态的密度矩阵形式。
   :type sig: Union[np.ndarray, paddle.Tensor, State]
   :param base: 对数的底，默认为2。
   :type base: int, optional
   
   :return: 输入的量子态之间的相对熵。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: random_pauli_str_generator(n, terms=3)

   随机生成一个可观测量（observable）的列表（ ``list`` ）形式。

   一个可观测量 :math:`O=0.3X\otimes I\otimes I+0.5Y\otimes I\otimes Z` 的
   列表形式为 ``[[0.3, 'x0'], [0.5, 'y0,z2']]`` 。这样一个可观测量是由
   调用 ``random_pauli_str_generator(3, terms=2)`` 生成的。

   :param n: 量子比特数量。
   :type n: int
   :param terms: 可观测量的项数, 默认为 3。
   :type terms: int, optional

   :return: 随机生成的可观测量的列表形式。
   :rtype: List

.. py:function:: pauli_str_to_matrix(pauli_str, n)

   将输入的可观测量（observable）的列表（ ``list`` ）形式转换为其矩阵形式。

   如输入的 ``pauli_str`` 为 ``[[0.7, 'z0,x1'], [0.2, 'z1']]`` 且 ``n=3`` ,
   则此函数返回可观测量 :math:`0.7Z\otimes X\otimes I+0.2I\otimes Z\otimes I` 的
   矩阵形式。

   :param pauli_str: 一个可观测量的列表形式。
   :type pauli_str: list
   :param n: 量子比特数量。
   :type n: int

   :raises ValueError: 只有泡利算子 "I" 可以被接受，而不指定其位置。

   :return: 输入列表对应的可观测量的矩阵形式。
   :rtype: paddle.Tensor

.. py:function:: partial_transpose_2(density_op, sub_system=None)

   计算输入量子态的 partial transpose :math:`\rho^{T_A}`。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: Union[np.ndarray, paddle.Tensor, State]
   :param sub_system: 1或2，表示关于哪个子系统进行 partial transpose，默认为第二个。
   :type sub_system: int, optional

   :return: 输入的量子态的 partial transpose
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: partial_transpose(density_op, n)

   计算输入量子态的 partial transpose :math:`\rho^{T_A}`。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: Union[np.ndarray, paddle.Tensor, State]
   :param n: 需要转置系统的量子比特数量。
   :type n: int
   
   :return: 输入的量子态的 partial transpose。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: partial_transpose(mat, perm_list, dim_list)

   根据输入顺序组合量子系统。

   :param mat: 输入矩阵，通常为量子态。
   :type mat: Union[np.ndarray, paddle.Tensor, State]
   :param perm: 排列顺序，例如输入 ``[0,2,1,3]`` 将会交换第 2、3 个子系统的顺序。
   :type perm: List[int]
   :param dim: 每个子系统维度列表。
   :type dim: List[int]

   :return: 排序后的矩阵。
   :rtype: Union[np.ndarray, paddle.Tensor, State]

.. py:function:: negativity(density_op)

   计算输入量子态的 Negativity :math:`N = ||\frac{\rho^{T_A}-1}{2}||`。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: Union[np.ndarray, paddle.Tensor, State]

   :return: 输入的量子态的 Negativity。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: logarithmic_negativity(density_op)

   计算输入量子态的 Logarithmic Negativity :math:`E_N = ||\rho^{T_A}||`。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: Union[np.ndarray, paddle.Tensor, State]

   :return: 输入的量子态的 Logarithmic Negativity。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: is_ppt(density_op)

   计算输入量子态是否满足 PPT 条件。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: Union[np.ndarray, paddle.Tensor, State]
   
   :return: 输入的量子态是否满足 PPT 条件。
   :rtype: bool

.. py:function:: is_choi(op)

   判断输入算子是否为某个量子操作的 Choi 算子。

   :param op: 线性算子的矩阵形式。
   :type op: Union[np.ndarray, paddle.Tensor]
   
   :return: 输入算子是否为某个量子操作的 Choi 算子。
   :rtype: bool

   .. note::
      输入算子默认作用在第二个系统上。

.. py:function:: schmidt_decompose(psi, sys_A=None)

   计算输入量子态的施密特分解 :math:`\lvert\psi\rangle=\sum_ic_i\lvert i_A\rangle\otimes\lvert i_B \rangle`。

   :param psi: 量子态的向量形式，形状为（2**n）。
   :type psi: Union[np.ndarray, paddle.Tensor, State]
   :param sys_A: 包含在子系统 A 中的 qubit 下标（其余 qubit 包含在子系统B中），默认为量子态 :math:`\lvert \psi\rangle` 的前半数 qubit。
   :type sys_A: List[int], optional

   :return:
      包含如下元素：

      - 由施密特系数组成的一维数组，形状为 ``(k)``。
      - 由子系统A的基 :math:`\lvert i_A\rangle` 组成的高维数组，形状为 ``(k, 2**m, 1)``。
      - 由子系统B的基 :math:`\lvert i_B\rangle` 组成的高维数组，形状为 ``(k, 2**l, 1)``。

   :rtype: Union[Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor], Tuple[np.ndarray, np.ndarray, np.ndarray]]

.. py:function:: image_to_density_matrix(image_filepath)

   将图片编码为密度矩阵。

   :param image_filepath: 图片文件的路径。
   :type image_filepath: str

   :return: 编码得到的密度矩阵。
   :rtype: paddle_quantum.State

.. py:function:: shadow_trace(state, hamiltonian, sample_shots, method='CS')

   估计可观测量 :math:`H` 的期望值 :math:`\text{trace}(H\rho)`。

   :param state: 输入的量子态。
   :type state: paddle_quantum.State
   :param hamiltonian: 可观测量。
   :type hamiltonian: paddle_quantum.Hamiltonian
   :param sample_shots: 采样次数。
   :type sample_shots: int
   :param method: 使用 shadow 来进行估计的方法，可选 "CS"、"LBCS"、"APS" 三种方法，默认为 ``CS``。
   :type method: str, optional

   :raises ValueError: 输入的哈密顿量 (Hamiltonian) 形式不合法。

   :return: 估计可观测量 :math:`H` 的期望值。
   :rtype: float

.. py:function:: tensor_state(state_a, state_b, *args)

   计算输入的量子态(至少两个)的直积形式, 输出将自动返回 State 实例。

   :param state_a: 量子态 A。
   :type state_a: State
   :param state_b: 量子态 B。
   :type state_b: State
   :param args: 其他量子态。
   :type args: State

   .. note::

      需要注意输入态使用的 backend；
      若输入数据为 ``paddle.Tensor`` 或者 ``numpy.ndarray``，请使用 ``paddle_quantum.linalg.NKron`` 函数处理。

   :return: 输入量子态的直积。
   :rtype: State

.. py:function:: diamond_norm(channel_repr, dim_io, **kwargs)

   计算输入的菱形范数

   :param channel_repr: 信道对应的表示, ``ChoiRepr`` 或 ``KrausRepr`` 或 ``StinespringRepr`` 或 ``paddle.Tensor``。
   :type channel_repr: Union[ChoiRepr, KrausRepr, StinespringRepr, paddle.Tensor]
   :param dim_io: 输入和输出的维度。
   :type dim_io: Union[int, Tuple[int, int]], optional.
   :param kwargs: 使用cvx所需的参数。
   :type kwargs: Any

   :raises RuntimeError: ``channel_repr`` 必须是 ``ChoiRepr`` 或 ``KrausRepr`` 或 ``StinespringRepr`` 或 ``paddle.Tensor``。
   :raises TypeError: "dim_io" 必须是 "int" 或者 "tuple"。

   :warning: 输入的 ``channel_repr`` 不是choi表示，已被转换成 ``ChoiRepr``。

   :return: 返回菱形范数
   :rtype: float


.. py:function:: channel_repr_convert(representation, source, target, tol)

   将给定的信道转换成目标形式。

   :param representation: 输入信道的一种表示。
   :type representation: Union[paddle.Tensor, np.ndarray, List[paddle.Tensor], List[np.ndarray]]
   :param source: 输入信道的表示名称，应为 ``Choi``, ``Kraus`` 或 ``Stinespring``。
   :type source: str
   :param target: 可选 ``Choi``, ``Kraus`` 或 ``Stinespring``。
   :type target: str
   :param tol: 容错误差。
   :type tol: float, optional

   :raises ValueError: 不支持的信道表示形式，应为 ``Choi``，``Kraus`` 或 ``Stinespring``。

   .. note::

      Choi 变为 Kraus 目前因为 eigh 的精度会存在1e-6的误差。

   :raises NotImplementedError: 不支持输入数据类型的信道转换。

   :return: 返回目标形式的信道。
   :rtype: Union[paddle.Tensor, np.ndarray, List[paddle.Tensor], List[np.ndarray]]

.. py:function:: random_channel(num_qubits, rank, target)

   从 Stinespring 表示中随机生成一个量子信道。

   :param num_qubits: 量子比特数 :math:`n`。
   :type num_qubits: int
   :param rank: 信道的秩，默认从 :math:`[0, 2^n]` 中随机选择。
   :type rank: str
   :param target: 信道的表示，可选 ``Choi``，``Kraus`` 或 ``Stinespring``。
   :type target: str

   :return: 返回目标表示下的随机信道。
   :rtype: Union[paddle.Tensor, List[paddle.Tensor]]

.. py:function:: kraus_unitary_random(num_qubits, num_oper)

   随机输出一组描述量子信道的 Kraus 算符。

   :param num_qubits: 信道对应的量子比特数量。
   :type num_qubits: int
   :param num_oper: Kraus算符的数量。
   :type num_oper: int

   :return: 一组 Kraus 算符。
   :rtype: list

.. py:function:: grover_generation(oracle)

   Grover 算子生成函数。

   :param oracle: 给定酉算子。
   :type oracle: Union[np.ndarray, paddle.Tensor]

   :return: 根据 ``oracle`` 搭建的 Grover 算子。
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: qft_generation(num_qubits)

   量子傅里叶变换算子生成函数。其矩阵形式为

   .. math::

      \begin{align}
         QFT = \frac{1}{\sqrt{N}}
         \begin{bmatrix}
               1 & 1 & .. & 1 \\
               1 & \omega_N & .. & \omega_N^{N-1} \\
               .. & .. & .. & .. \\
               1 & \omega_N^{N-1} & .. & \omega_N^{(N-1)^2}
         \end{bmatrix}
      \end{align}

   :param num_qubits: 算子作用的系统比特数。
   :type num_qubits: int

   :return: 量子傅里叶变换算子。
   :rtype: paddle.Tensor

