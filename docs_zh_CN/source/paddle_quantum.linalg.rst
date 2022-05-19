paddle\_quantum.linalg
=============================

量桨中的线性代数的功能实现。

.. py:function:: abs_norm(mat)

   计算矩阵范数

   :param mat: 矩阵
   :type mat: paddle.Tensor

   :return: 范数
   :rtype: float

.. py:function:: dagger(mat)

   计算矩阵的转置共轭

   :param mat: 矩阵
   :type mat: paddle.Tensor

   :return: 矩阵的转置共轭
   :rtype: paddle.Tensor

.. py:function:: is_hermitian(mat, eps=1e-6)

   验证矩阵 ``P`` 是否为厄密矩阵

   :param mat: 矩阵
   :type mat: paddle.Tensor
   :param eps: 容错率
   :type eps: float, optional

   :return: 决定是否 :math:`P - P^\dagger = 0`
   :rtype: bool

.. py:function:: is_projector(mat, eps=1e-6)

   验证矩阵 ``P`` 是否为映射算子

   :param mat: 矩阵
   :type mat: paddle.Tensor
   :param eps: 容错率
   :type eps: float, optional

   :return: 决定是否 :math:`PP - P = 0`
   :rtype: bool

.. py:function:: is_unitary(mat, eps = 1e-5)

   验证矩阵 ``P`` 是否为酉矩阵

   :param mat: 矩阵
   :type mat: paddle.Tensor
   :param eps: 容错率
   :type eps: float, optional

   :return: 决定是否 :math:`PP^\dagger - I = 0`
   :rtype: bool

.. py:function:: hermitian_random(num_qubits)

   随机生成一个厄密矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 厄密矩阵
   :rtype: paddle.Tensor

.. py:function:: orthogonal_projection_random(num_qubits)

   随机生成一个秩是 1 的正交投影算子

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 正交投影算子
   :rtype: paddle.Tensor

.. py:function:: unitary_hermitian_random(num_qubits)

   随机生成一个厄密酉矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 厄密共轭酉矩阵
   :rtype: paddle.Tensor

.. py:function:: unitary_random_with_hermitian_block(num_qubits)

   随机生成一个左上半部分为厄密矩阵的酉矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return:  一个左上半部分为厄密矩阵的 :math:`2^n \times 2^n` 酉矩阵
   :rtype: paddle.Tensor

.. py:function:: unitary_random(num_qubits)

   随机生成一个酉矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 酉矩阵
   :rtype: paddle.Tensor

.. py:function:: haar_orthogonal(num_qubits)

   生成一个服从 Haar random 的正交矩阵。采样算法参考文献: arXiv:math-ph/0609050v2

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return:  一个 :math:`2^n \times 2^n` 正交矩阵
   :rtype: paddle.Tensor

.. py:function::  haar_unitary(num_qubits)

   生成一个服从 Haar random 的酉矩阵。采样算法参考文献: arXiv:math-ph/0609050v2

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return:  一个 :math:`2^n \times 2^n` 酉矩阵
   :rtype: paddle.Tensor

.. py:function::  haar_state_vector(num_qubits, is_real=False)

   生成一个服从 Haar random 的态矢量。采样算法参考文献: arXiv:math-ph/0609050v2

   :param num_qubits: 量子比特数 n
   :type num_qubits: int
   :param is_real: 生成的态矢量是否为实数
   :type is_real: bool, optional

   :return:  一个 :math:`2^n \times 1` 态矢量
   :rtype: paddle.Tensor

.. py:function::  haar_density_operator(num_qubits, rank=None, is_real=False)

   生成一个服从 Haar random 的密度矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int
   :param rank: 矩阵秩, 默认满秩
   :type rank: int, optional
   :param is_real: 生成的态矢量是否为实数
   :type is_real: bool, optional

   :return:  一个 :math:`2^n x 2^n` 密度矩阵
   :rtype: paddle.Tensor

.. py:function::  NKron(matrix_A, matrix_B, *args)

   计算两个及以上的矩阵的克罗内克乘积

   :param matrix_A: 矩阵
   :type num_qubits: np.ndarray
   :param matrix_B: 矩阵
   :type matrix_B: np.ndarray
   :param \*args: 更多矩阵
   :type \*args: np.ndarray
   
   :return:  克罗内克乘积
   :rtype: np.ndarray
