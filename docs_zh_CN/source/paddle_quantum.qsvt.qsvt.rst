paddle\_quantum.qsvt.qsvt
============================

量子奇异值变换

.. py:function:: block_encoding_projector(num_qubits, num_projected_qubits)

   生成块编码的投影算子

   :param num_qubits: 量子比特数量
   :type num_qubits: int
   :param num_projected_qubits: 被投影的量子比特数量，默认为 `num_qubits - 1`
   :type num_projected_qubits: int

   :return: 投影算子 :math:`|0\rangle\langle0| \otimes I`
   :rtype: paddle.Tensor


.. py:function:: qubitization(proj, phi)

   单比特化操作，生成等同于 :math:`e^{i \phi (2P - I)}` 的电路

   :param proj: 正交投影算子 :math:`P`
   :type proj: paddle.Tensor
   :param phi: 角度 :math:`\phi`
   :type phi: paddle.Tensor

   :return: :math:`e^{i \phi (2P - I)}` 的电路
   :rtype: Circuit


.. py:class:: QSVT

   基类: :py:class:`object`

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param oracle: 酉算子 :math:`U`，为一个厄米特矩阵 :math:`X` 的块编码
   :type oracle: paddle.Tensor
   :param m: 厄米特矩阵 :math:`X` 的系统量子比特数量，默认为酉算子 :math:`U` 量子比特数量 - 1

   .. py:method:: block_encoding_matrix()

      构造一个对于厄米特矩阵 :math:`X` 的量子奇异值变换矩阵，即实现多项式 :math:`P(X)` 的块编码矩阵

      :return: 量子奇异值变换矩阵
      :rtype: paddle.Tensor


   .. py:method:: block_encoding_circuit()

      构造一个对于厄米特矩阵 :math:`X` 的量子奇异值变换电路，即实现多项式 :math:`P(X)` 的块编码电路

      :return: 量子奇异值变换电路
      :rtype: Circuit

   .. py:method:: block_encoding_unitary()

      返回一个对于厄米特矩阵 :math:`X` 的量子奇异值变换电路的酉矩阵形式，用于验证正确性

      :return: 量子奇异值变换电路的酉矩阵
      :rtype: paddle.Tensor