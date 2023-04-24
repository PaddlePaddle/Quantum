paddle\_quantum.channel.custom
=====================================

自定义量子信道的类的功能实现。

.. py:class:: ChoiRepr(choi_oper, qubits_idx=None, num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   Choi 表示的自定义量子信道。

   :param choi_oper: 该信道的 Choi 算符。
   :type choi_oper: paddle.Tensor
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``None``。
   :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :raises NotImplementedError: 噪声信道只能在密度矩阵模式下运行。

.. py:class:: KrausRepr(kraus_oper, qubits_idx=None, num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   Kraus 表示的自定义量子信道。

   :param kraus_oper: 该信道的 Kraus 算符。
   :type kraus_oper: Iterable[paddle.Tensor]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``None``。
   :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: StinespringRepr(stinespring_mat, qubits_idx=None, num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   Stinespring 表示的自定义量子信道。

   :param stinespring_mat: 一个用来表示该信道的 Stinespring 矩阵。
   :type stinespring_mat: paddle.Tensor
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``None``。
   :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional