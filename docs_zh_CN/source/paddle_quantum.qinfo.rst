paddle\_quantum.qinfo
============================

量子信息中的常用功能实现。

.. py:function:: partial_trace(rho_AB, dim1, dim2, A_or_B)

   计算量子态的偏迹。

   :param rho_AB: 输入的量子态。
   :type rho_AB: paddle_quantum.State
   :param dim1: 系统A的维数。
   :type dim1: int
   :param dim2: 系统B的维数。
   :type dim2: int
   :param A_or_B: 1或者2，1表示计算系统A上的偏迹，2表示计算系统B上的偏迹。
   :type A_or_B: int

   :return: 输入的量子态的偏迹。
   :rtype: paddle.Tensor

.. py:function:: partial_trace_discontiguous(rho, preserve_qubits = None)

   计算量子态的偏迹，可选取任意子系统。

   :param rho: 输入的量子态。
   :type rho: paddle_quantum.State
   :param preserve_qubits: 要保留的量子比特，默认为 None，表示全保留。
   :type preserve_qubits: list, optional
   
   :return: 所选子系统的量子态偏迹。
   :rtype: paddle.Tensor

.. py:function:: trace_distance(rho, sigma)

   计算两个量子态的迹距离。

   .. math::

      D(\rho, \sigma) = 1 / 2 * \text{tr}|\rho-\sigma|

   :param rho: 量子态的密度矩阵形式。
   :type rho: paddle_quantum.State
   :param sigma: 量子态的密度矩阵形式。
   :type sigma: paddle_quantum.State

   :return: 输入的量子态之间的迹距离。
   :rtype: paddle.Tensor

.. py:function:: state_fidelity(rho, sigma)

   计算两个量子态的保真度。

   .. math::

      F(\rho, \sigma) = \text{tr}(\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})

   :param rho: 量子态的密度矩阵形式。
   :type rho: paddle_quantum.State
   :param sigma: 量子态的密度矩阵形式。
   :type sigma: paddle_quantum.State
   :return: 输入的量子态之间的保真度。
   :rtype: paddle.Tensor

.. py:function:: gate_fidelity(U, V)

   计算两个量子门的保真度。

   .. math::

      F(U, V) = |\text{tr}(UV^\dagger)|/2^n

   :math:`U` 是一个 :math:`2^n\times 2^n` 的 Unitary 矩阵。

   :param U: 量子门 :math:`U` 的酉矩阵形式
   :type U: paddle.Tensor
   :param V: 量子门 :math:`V` 的酉矩阵形式
   :type V: paddle.Tensor

   :return: 输入的量子门之间的保真度
   :rtype: paddle.Tensor

.. py:function:: purity(rho)

   计算量子态的纯度。

   .. math::

      P = \text{tr}(\rho^2)

   :param rho: 量子态的密度矩阵形式。
   :type rho: paddle_quantum.State

   :return: 输入的量子态的纯度。
   :rtype: paddle.Tensor

.. py:function:: von_neumann_entropy(rho)

    计算量子态的冯诺依曼熵。

   .. math::

      S = -\text{tr}(\rho \log(\rho))

   :param rho: 量子态的密度矩阵形式。
   :type rho: paddle_quantum.State

   :return: 输入的量子态的冯诺依曼熵。
   :rtype: paddle.Tensor

.. py:function:: relative_entropy(rho, sig)

   计算两个量子态的相对熵。

   .. math::

      S(\rho \| \sigma)=\text{tr} \rho(\log \rho-\log \sigma)

   :param rho: 量子态的密度矩阵形式
   :type rho: paddle_quantum.State
   :param sig: 量子态的密度矩阵形式
   :type sig: paddle_quantum.State
   
   :return: 输入的量子态之间的相对熵
   :rtype: paddle.Tensor

.. py:function:: random_pauli_str_generator(n, terms = 3)

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

.. py:function:: partial_transpose_2(density_op, sub_system = None)

   计算输入量子态的 partial transpose :math:`\rho^{T_A}`。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: paddle_quantum.State
   :param sub_system: 1或2，表示关于哪个子系统进行 partial transpose，默认为第二个。
   :type sub_system: int, optional

   :return: 输入的量子态的 partial transpose
   :rtype: paddle.Tensor

.. py:function:: partial_transpose(density_op, n)

   计算输入量子态的 partial transpose :math:`\rho^{T_A}`。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: paddle_quantum.State
   :param n: 需要转置系统的量子比特数量。
   :type n: int
   
   :return: 输入的量子态的 partial transpose。
   :rtype: paddle.Tensor

.. py:function:: negativity(density_op)

   计算输入量子态的 Negativity :math:`N = ||\frac{\rho^{T_A}-1}{2}||`。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: paddle_quantum.State

   :return: 输入的量子态的 Negativity。
   :rtype: paddle.Tensor

.. py:function:: logarithmic_negativity(density_op)

   计算输入量子态的 Logarithmic Negativity :math:`E_N = ||\rho^{T_A}||`。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: paddle_quantum.State

   :return: 输入的量子态的 Logarithmic Negativity。
   :rtype: paddle.Tensor

.. py:function:: is_ppt(density_op: paddle_quantum.State)

   计算输入量子态是否满足 PPT 条件。

   :param density_op: 量子态的密度矩阵形式。
   :type density_op: paddle_quantum.State
   
   :return: 输入的量子态是否满足 PPT 条件。
   :rtype: bool

.. py:function:: schmidt_decompose(psi, sys_A = None)

   计算输入量子态的施密特分解 :math:`\lvert\psi\rangle=\sum_ic_i\lvert i_A\rangle\otimes\lvert i_B \rangle`。

   :param psi: 量子态的向量形式，形状为（2**n）。
   :type psi: paddle_quantum.State
   :param sys_A: 包含在子系统 A 中的 qubit 下标（其余 qubit 包含在子系统B中），默认为量子态 :math:`\lvert \psi\rangle` 的前半数 qubit。
   :type sys_A: List[int], optional

   :return:
      包含如下元素：

      - 由施密特系数组成的一维数组，形状为 ``(k)``。
      - 由子系统A的基 :math:`\lvert i_A\rangle` 组成的高维数组，形状为 ``(k, 2**m, 1)``。
      - 由子系统B的基 :math:`\lvert i_B\rangle` 组成的高维数组，形状为 ``(k, 2**l, 1)``。

   :rtype: Tuple[paddle.Tensor]

.. py:function:: image_to_density_matrix(image_filepath)

   将图片编码为密度矩阵。

   :param image_filepath: 图片文件的路径。
   :type image_filepath: str

   :return: 编码得到的密度矩阵。
   :rtype: paddle_quantum.State

.. py:function:: shadow_trace(state, hamiltonian, sample_shots, method = 'CS')

   估计可观测量 :math:`H` 的期望值 :math:`\text{trace}(H\rho)`。

   :param state: 输入的量子态。
   :type state: paddle_quantum.State
   :param hamiltonian: 可观测量。
   :type hamiltonian: paddle_quantum.Hamiltonian
   :param sample_shots: 采样次数。
   :type sample_shots: int
   :param method: 使用 shadow 来进行估计的方法，可选 "CS"、"LBCS"、"APS" 三种方法，默认为 ``CS``。
   :type method: str, optional

   :return: 估计可观测量 :math:`H` 的期望值。
   :rtype: float
