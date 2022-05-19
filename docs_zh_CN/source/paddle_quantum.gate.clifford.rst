paddle\_quantum.gate.clifford
====================================

随机生成 Clifford 算子的功能实现。

.. py:function:: compose_clifford_circuit(clifd1, clifd2)

   计算两个指定的 Clifford 的复合，得到复合后的电路。

   :param clifd1: 需要复合的第 1 个 Clifford。
   :type clifd1: Clifford
   :param clifd2: 需要复合的第 2 个 Clifford。
   :type clifd2: Clifford
   :return: 复合后的 Clifford 所对应的电路，作用的顺序为 clif1、clif2。
   :rtype: paddle_quantum.ansatz.Circuit

.. py:class:: Clifford(num_qubits)

   用户可以通过实例化该 ``class`` 来随机生成一个 Clifford operator。

   :param num_qubits: 该 Clifford operator 作用的量子比特数目。
   :type num_qubits: int

   :参考文献:
      1. Bravyi, Sergey, and Dmitri Maslov. "Hadamard-free circuits expose the structure of the Clifford group."
      IEEE Transactions on Information Theory 67.7 (2021): 4546-4563.

   .. py:method:: print_clifford()

      输出该 Clifford 在 Pauli 基上的作用关系来描述这个 Clifford。

   .. py:method:: sym()

      获取该 Clifford operator 所对应的辛矩阵。

      :return: 该 Clifford 对应的辛矩阵。
      :rtype: np.ndarray

   .. py:method:: tableau()

      获取该 Clifford operator 所对应的 table。

      对 ``num_qubits`` 个 qubits 的情况，前 ``num_qubits`` 行对应 :math:`X_i` 的结果，后 ``num_qubits`` 行对应 :math:`Z_i` 的结果。

      :return: 该 Clifford 的 table。
      :rtype: np.ndarray

   .. py:method:: circuit()

      获取该 Clifford operator 所对应的电路。

      :return: 该 Clifford 对应的电路。
      :rtype: paddle_quantum.ansatz.Circuit
