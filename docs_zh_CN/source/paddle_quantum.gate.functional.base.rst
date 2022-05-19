paddle\_quantum.gate.functional.base
===========================================

量子门的基础函数的功能实现。

.. py:function:: simulation(state, gate, qubit_idx, num_qubits, backend)

   在输入态上作用一个量子门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param gate: 要执行的门。
   :type gate: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: Union[int, List[int]]
   :param num_qubits: 总的量子比特个数。
   :type num_qubits: int
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle.Tensor
