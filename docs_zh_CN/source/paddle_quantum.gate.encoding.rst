paddle\_quantum.gate.encoding
====================================

量子编码的功能实现。

.. py:class:: BasisEncoding(qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.base.Operator`

   将输入的经典数据编码成量子态的基态编码门。

   在基态编码中，输入的经典数据只能包括 0 和 1。如输入数据为 1101，则编码后的量子态为 :math:`|1101\rangle`。
   这里假设量子态在编码前为全 0 的态，即 :math:`|00\ldots 0\rangle`。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: AmplitudeEncoding(qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.base.Operator`

   将输入的经典数据编码成量子态的振幅编码门。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: AngleEncoding(feature, qubits_idx='full', num_qubits=None, encoding_gate=None)

   基类：:py:class:`paddle_quantum.base.Operator`

   将输入的经典数据编码成量子态的角度编码门。

   :param feature: 待编码的向量。
   :type feature: paddle.Tensor
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param encoding_gate: 编码用的量子门，应是 ``"rx"``、``"ry"``，和 ``"rz"`` 中的一种。默认为 ``None``。
   :type encoding_gate: str, optional

.. py:class:: IQPEncoding(feature, qubits_idx=None, num_qubits=None, num_repeat=1)

   基类：:py:class:`paddle_quantum.base.Operator`

   将输入的经典数据编码成量子态的 IQP 编码门。

   :param feature: 待编码的向量。
   :type feature: paddle.Tensor
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``None``。
   :type qubits_idx: Iterable[Iterable[int]], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param num_repeat: 编码层的层数，默认为 ``1``。
   :type num_repeat: int, optional
