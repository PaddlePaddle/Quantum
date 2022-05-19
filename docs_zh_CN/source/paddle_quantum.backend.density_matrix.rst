paddle\_quantum.backend.density\_matrix
==============================================

密度矩阵后端的功能实现。

.. py:function:: paddle_quantum.backend.density_matrix.unitary_transformation(state, gate, qubit_idx, num_qubits)

   在密度矩阵模式下实现酉变换的函数。

   :param state: 输入的量子态。
   :type state: paddle.Tensor
   :param gate: 输入量子门，表示要进行的酉变换。
   :type gate: paddle.Tensor
   :param qubit_idx: 量子门要作用到的量子比特的下标。
   :type qubit_idx: Union[List[int], int]
   :param num_qubits: 输入的量子态所拥有的量子比特的数量。
   :type num_qubits: int
   :return: 变换后的量子态。
   :rtype: paddle.Tensor
