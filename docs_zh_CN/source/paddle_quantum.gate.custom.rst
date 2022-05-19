paddle\_quantum.gate.custom
==================================

自定义量子门和受控量子门的功能实现。

.. py:class:: Oracle(oracle, qubits_idx, num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   一个 oracle 门。

   :param oracle: 要实现的 oracle。
   :type oracle: paddle.Tensor
   :param qubits_idx: 作用在的量子比特的编号。
   :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: ControlOracle(oracle, qubits_idx, num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   一个受控 oracle 门。

   :param oracle: 要实现的 oracle。
   :type oracle: paddle.Tensor
   :param qubits_idx: 作用在的量子比特的编号。
   :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
