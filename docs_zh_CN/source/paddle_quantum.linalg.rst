paddle\_quantum.linalg
=============================

量桨中的线性代数的功能实现。

.. py:function:: abs_norm(mat)

   计算矩阵范数

   :param mat: 矩阵
   :type mat: Union[np.ndarray, paddle.Tensor, State]

   :return: 范数
   :rtype: float

.. py:function:: dagger(mat)

   计算矩阵的转置共轭

   :param mat: 矩阵
   :type mat: Union[np.ndarray, paddle.Tensor]

   :return: 矩阵的转置共轭
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: is_hermitian(mat, eps=1e-6)

   验证矩阵 ``P`` 是否为厄密矩阵

   :param mat: 厄密矩阵
   :type mat: Union[np.ndarray, paddle.Tensor]
   :param eps: 容错率
   :type eps: float, optional

   :return: 决定是否 :math:`P - P^\dagger = 0`
   :rtype: bool

.. py:function:: is_projector(mat, eps=1e-6)

   验证矩阵 ``P`` 是否为映射算子

   :param mat: 映射算子
   :type mat: Union[np.ndarray, paddle.Tensor]
   :param eps: 容错率
   :type eps: float, optional

   :return: 决定是否 :math:`PP - P = 0`
   :rtype: bool

.. py:function:: is_unitary(mat, eps=1e-5)

   验证矩阵 ``P`` 是否为酉矩阵

   :param mat: 酉矩阵
   :type mat: Union[np.ndarray, paddle.Tensor]
   :param eps: 容错率
   :type eps: float, optional

   :return: 决定是否 :math:`PP^\dagger - I = 0`
   :rtype: bool

.. py:function:: hermitian_random(num_qubits)

   随机生成一个厄密矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 厄密矩阵 (n 为量子比特数) 
   :rtype: paddle.Tensor

.. py:function:: orthogonal_projection_random(num_qubits)

   随机生成一个秩是 1 的正交投影算子

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 正交投影算子 (n 为量子比特数)
   :rtype: paddle.Tensor

.. py:function:: density_matrix_random(num_qubits)

   随机生成一个密度矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 密度矩阵 (n 为量子比特数)
   :rtype: paddle.Tensor

.. py:function:: unitary_random(num_qubits)

   随机生成一个酉矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 酉矩阵 (n 为量子比特数)
   :rtype: paddle.Tensor

.. py:function:: unitary_hermitian_random(num_qubits)

   随机生成一个厄密酉矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return: 一个 :math:`2^n \times 2^n` 厄密共轭酉矩阵 (n 为量子比特数)
   :rtype: paddle.Tensor

.. py:function:: unitary_random_with_hermitian_block(num_qubits, is_unitary)

   随机生成一个左上半部分为厄密矩阵的酉矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int
   :param is_unitary: 厄密矩阵块是否是酉矩阵的 1/2
   :type is_unitary: bool, optional

   :return:  一个左上半部分为厄密矩阵的 :math:`2^n \times 2^n` 酉矩阵 (n 为量子比特数)
   :rtype: paddle.Tensor

.. py:function:: block_enc_herm(mat, num_block_qubits)
   
   生成厄密矩阵的分组编码

   :param mat: 用于分组编码的矩阵
   :type mat: Union[np.ndarray, paddle.Tensor]
   :param num_block_qubits: 用于分组编码的辅助量子比特数
   :type num_block_qubits: int, optional

   :return: 返回分组编码后的酉矩阵
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: haar_orthogonal(num_qubits)

   生成一个服从 Haar random 的正交矩阵。采样算法参考文献: arXiv:math-ph/0609050v2

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return:  一个 :math:`2^n \times 2^n` 正交矩阵 (n 为量子比特数)
   :rtype: paddle.Tensor

.. py:function::  haar_unitary(num_qubits)

   生成一个服从 Haar random 的酉矩阵。采样算法参考文献: arXiv:math-ph/0609050v2

   :param num_qubits: 量子比特数 n
   :type num_qubits: int

   :return:  一个 :math:`2^n \times 2^n` 酉矩阵 (n 为量子比特数)
   :rtype: paddle.Tensor

.. py:function::  haar_state_vector(num_qubits, is_real=False)

   生成一个服从 Haar random 的态矢量。采样算法参考文献: arXiv:math-ph/0609050v2

   :param num_qubits: 量子比特数 n
   :type num_qubits: int
   :param is_real: 生成的态矢量是否为实数
   :type is_real: bool, optional

   :return:  一个 :math:`2^n \times 1` 态矢量 (n 为量子比特数)
   :rtype: paddle.Tensor

.. py:function::  haar_density_operator(num_qubits, rank=None, is_real=False)

   生成一个服从 Haar random 的密度矩阵

   :param num_qubits: 量子比特数 n
   :type num_qubits: int
   :param rank: 矩阵秩, 默认满秩
   :type rank: int, optional
   :param is_real: 生成的态矢量是否为实数
   :type is_real: bool, optional

   :return:  一个 :math:`2^n \times 2^n` 密度矩阵 (n 为量子比特数)
   :rtype: paddle.Tensor


.. py:function:: direct_sum(A,B)
   
   计算A和B的直和

   :param A: :math:`m \times n` 的矩阵
   :type A: Union[np.ndarray, paddle.Tensor]
   :param B: :math:`p \times q` 的矩阵
   :type B: Union[np.ndarray, paddle.Tensor]

   :return: A和B的直和，维度为 :math:`(m + p) \times (n + q)`
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function::  NKron(matrix_A, matrix_B, *args)

   计算两个及以上的矩阵的克罗内克乘积

   :param matrix_A: 矩阵
   :type matrix_A: Union[np.ndarray, paddle.Tensor]
   :param matrix_B: 矩阵
   :type matrix_B: Union[np.ndarray, paddle.Tensor]
   :param \*args: 更多矩阵
   :type \*args: Union[np.ndarray, paddle.Tensor]

   .. code-block:: python

      from paddle_quantum.linalg import density_op_random, NKron
      A = density_op_random(2)
      B = density_op_random(2)
      C = density_op_random(2)
      result = NKron(A, B, C)

   .. note::
      上述代码块的 ``result`` 应为 :math:`A \otimes B \otimes C`
   
   :return:  克罗内克乘积
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: herm_transform(fcn, mat, ignore_zero)
   
   厄密矩阵的函数变换

   :param fcn: 可以展开成泰勒级数的函数 `f`
   :type fcn: Callable[[float], float]
   :param mat: 厄密矩阵 :math:`H`
   :type mat: Union[paddle.Tensor, np.ndarray, State]
   :param ignore_zero: 是否忽略特征值0所在的特征空间，默认为 ``False`` 
   :type ignore_zero: bool, optional

   :return: :math:`f(H)`
   :rtype: paddle.Tensor