paddle\_quantum.gate.layer
=================================

量子电路模板的功能实现。

.. py:function:: qubits_idx_filter(qubits_idx, num_qubits)

   检查 ``qubits_idx`` 与 ``num_qubits`` 是否合法。

   :param qubits_idx: 量子比特的编号。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 总的量子比特个数。
   :type num_qubits: int
   :raises RuntimeError: 须声明 ``qubits_idx`` 或 ``num_qubits`` 以实例化类。
   :raises ValueError: ``qubits_idx`` 须为 ``Iterable`` 或 ``'full'``。
   :return: 检查过的量子比特的编号。
   :rtype: List[Iterable[int]]

.. py:class:: SuperpositionLayer(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   Hadamard 门组成的层。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: WeakSuperpositionLayer(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   转角度为 :math:`\pi/4` 的 Ry 门组成的层。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: LinearEntangledLayer(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   包含 Ry 门、Rz 门，和 CNOT 门的线性纠缠层。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: RealEntangledLayer(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   包含 Ry 门和 CNOT 门的强纠缠层。

   .. note::

      这一层量子门的数学表示形式为实数酉矩阵。电路模板来自论文：https://arxiv.org/pdf/1905.10876.pdf。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: ComplexEntangledLayer(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   包含 U3 门和 CNOT 门的强纠缠层。

   .. note::

      这一层量子门的数学表示形式为复数酉矩阵。电路模板来自论文：https://arxiv.org/abs/1804.00633。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: RealBlockLayer(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   包含 Ry 门和 CNOT 门的弱纠缠层。

   .. note::

      这一层量子门的数学表示形式为实数酉矩阵。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: ComplexBlockLayer(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   包含 U3 门和 CNOT 门的弱纠缠层。

   .. note::

      这一层量子门的数学表示形式为复数酉矩阵。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: QAOALayer(Gate)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   QAOA 驱动层

   .. note::

      仅支持 MaxCut 问题

   :param edges: 图的边
   :type edges: Iterable
   :param nodes: 图的节点
   :type nodes: Iterable
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: QAOALayer(Gate)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   带权重的 QAOA 驱动层

   :param edges: 带权重的图的边
   :type edges: Dict[Tuple[int, int], float]
   :param nodes: 带权重的图的节点
   :type nodes: Dict[int, float]
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
