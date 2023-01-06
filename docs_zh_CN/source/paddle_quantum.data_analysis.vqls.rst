paddle\_quantum.data_analysis.vqls
=============================================
VQLS模型

.. py:function:: hadamard_test(phi, U, num_qubits)

   给定酉算子 U 和量子态 :math:`|\phi\rangle`，计算 U 关于 :math:`|\phi\rangle` 的期望，即 :math:`\langle\phi|U|\phi\rangle`。

   :param phi: 期望值里的量子态。
   :type phi: State
   :param U: 期望值里的酉算子。
   :type U: paddle.Tensor
   :param num_qubits: 量子态的比特数。
   :type num_qubits: int

   :return: 返回计算的期望值的实数和虚数部分。
   :rtype: Tuple[paddle.Tensor, paddle.Tensor]

.. py:function:: hadamard_overlap_test(phi, b, An, Am, num_qubits)

   给定酉算子 Am， An 和量子态 :math:`|\phi\rangle`， b， 计算 :math:`\langle{b}| An |\phi\rangle\langle\phi| Am^\dagger |b\rangle` 的值。

   :param phi: 计算里的量子态。
   :type phi: State
   :param b: 计算里的量子态。
   :type b: State
   :param Am: 计算里的酉算子。
   :type Am: paddle.Tensor
   :param An: 计算里的酉算子。
   :type An: paddle.Tensor
   :param num_qubits: 量子态的比特数。
   :type num_qubits: int

   :return: 返回计算的实数和虚数部分。
   :rtype: Tuple[paddle.Tensor, paddle.Tensor]

.. py:class:: VQLS(num_qubits, A, coefficients_real, coefficients_img, b, depth)

   基类：:py:class:`paddle.nn.Layer`

   变分量子线性求解器（variational quantum linear solver, VQLS）模型的实现。

   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param A: 分解输入矩阵所需要的酉矩阵列表。
   :type A: List[paddle.Tensor]
   :param coefficients_real: 对应酉矩阵系数的实数部分。
   :type coefficients_real: List[float]
   :param coefficients_img: 对应酉矩阵系数的虚数部分。
   :type coefficients_img: List[float]
   :param b: 输入答案被编码成的量子态。
   :type b: State
   :param depth: 模拟电路的深度。
   :type depth: int

   .. py:method:: forward()

      :return: 返回模型的输出。
      :rtype: paddle.Tensor

.. py:function:: compute(A, b, depth, iterations, LR, gamma)

   求解线性方程组 Ax=b。

   :param A: 输入矩阵。
   :type A: numpy.ndarray
   :param b: 输入向量。
   :type b: numpy.ndarray
   :param depth: 模拟电路的深度。
   :type depth: int
   :param iterations: 优化的迭代次数。
   :type iterations: int
   :param LR: 优化器的学习率。
   :type LR: float
   :param gamma: 如果损失函数低于此值，则可以提前结束优化。 默认值为 ``0``。
   :type gamma: Optional[float]