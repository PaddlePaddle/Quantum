paddle\_quantum.ansatz.layer
==================================

量子电路模板的功能实现。

.. py:class:: Layer

   基类: :py:class:`paddle_quantum.ansatz.container.Sequential`

   量子电路层。

   :param qubits_idx: 作用的量子比特的编号。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 量子比特的总数。
   :type num_qubits: int
   :param depth: 电路层的数量，默认为 ``1``。
   :type depth: int

   .. note::

      一个 Circuit 实例需要拓展为一个 Layer 实例以应用于电路中。

   .. py:property:: gate_history()

      量子门的插入信息。

      :return: 量子门的插入历史。
      :rtype: List[Dict[str, Union[str, List[int], paddle.Tensor]]]

.. py:class:: SuperpositionLayer

   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   由 Hadamard 门组成的电路层。

   :param qubits_idx: 作用的量子比特的编号，默认为 ``None``，即作用在所有量子比特上。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 量子比特的总数，默认为 ``None``。
   :type num_qubits: int
   :param depth: 电路层的深度，默认为 ``1``。
   :type depth: int

.. py:class:: WeakSuperpositionLayer
   
   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   由旋转角为 :math:`\pi/4` 的 Ry 门组成的电路层。

   :param qubits_idx: 作用的量子比特的编号，默认为 ``None``，即作用在所有量子比特上。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 量子比特的总数，默认为 ``None``。
   :type num_qubits: int
   :param depth: 电路层的深度，默认为 ``1``。
   :type depth: int

.. py:class:: LinearEntangledLayer
   
   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   由 Ry 门， Rz 门，和 CNOT 门组成的线性纠缠电路层。

   :param qubits_idx: 作用的量子比特的编号，默认为 ``None``，即作用在所有量子比特上。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 量子比特的总数，默认为 ``None``。
   :type num_qubits: int
   :param depth: 电路层的深度，默认为 ``1``。
   :type depth: int

.. py:class:: RealEntangledLayer
   
   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   由 Ry 门和 CNOT 门组成的强纠缠电路层。

   .. note::

      本电路层的数学表示是一个实值的酉矩阵。此电路层来源于论文 https://arxiv.org/pdf/1905.10876.pdf。

   :param qubits_idx: 作用的量子比特的编号，默认为 ``None``，即作用在所有量子比特上。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 量子比特的总数，默认为 ``None``。
   :type num_qubits: int
   :param depth: 电路层的深度，默认为 ``1``。
   :type depth: int

.. py:class:: ComplexEntangledLayer
   
   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   由单量子比特旋转门和 CNOT 门组成的强纠缠电路层。

   .. note::

      本电路层的数学表示是一个复值的酉矩阵。此电路层来源于论文 https://arxiv.org/abs/1804.00633。

   :param qubits_idx: 作用的量子比特的编号，默认为 ``None``，即作用在所有量子比特上。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 量子比特的总数，默认为 ``None``。
   :type num_qubits: int
   :param depth: 电路层的深度，默认为 ``1``。
   :type depth: int

.. py:class:: RealBlockLayer
   
   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   由 Ry 门和 CNOT 门组成的弱纠缠电路层。

   .. note::

      本电路层的数学表示是一个实值的酉矩阵。

   :param qubits_idx: 作用的量子比特的编号，默认为 ``None``，即作用在所有量子比特上。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 量子比特的总数，默认为 ``None``。
   :type num_qubits: int
   :param depth: 电路层的深度，默认为 ``1``。
   :type depth: int

.. py:class:: ComplexBlockLayer
   
   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   由单量子比特旋转门和 CNOT 门组成的弱纠缠电路层。

   .. note::

      本电路层的数学表示是一个复值的酉矩阵。

   :param qubits_idx: 作用的量子比特的编号，默认为 ``None``，即作用在所有量子比特上。
   :type qubits_idx: Union[Iterable[int], str]
   :param num_qubits: 量子比特的总数，默认为 ``None``。
   :type num_qubits: int
   :param depth: 电路层的深度，默认为 ``1``。
   :type depth: int

.. py:class:: QAOALayer

   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   QAOA 驱动层。

   .. note::

      仅支持MAXCUT问题。

   :param edges: 图的边。
   :type edges: Iterable
   :param nodes: 图的节点。
   :type nodes: Iterable
   :param depth: 层数，默认为 ``1``。
   :type depth: int

.. py:class:: QAOALayerWeighted

   基类: :py:class:`paddle_quantum.ansatz.layer.Layer`

   带权重的 QAOA 驱动层。

   :param edges: 带权重的图的边。
   :type edges: Dict[Tuple[int, int], float]
   :param nodes: 带权重的图的节点。
   :type nodes: Dict[int, float]
   :param depth: 层数，默认为 ``1``。
   :type depth: int