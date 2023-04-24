paddle\_quantum.gate.custom
==================================

自定义量子门和受控量子门的功能实现。

.. py:class:: Oracle(oracle, qubits_idx, num_qubits=None, depth=1, gate_info=None)

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
   :param gate_info: 量子门的信息，用于信息追溯或者画图。
   :type gate_info: dict, optional

.. py:class:: ControlOracle(oracle, qubits_idx, num_qubits=None, depth=1, gate_info=None)

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
   :param gate_info: 量子门的信息，用于信息追溯或者画图。
   :type gate_info: dict, optional

.. py:class:: ParamOracle(generator, param=None, depth=1, num_acted_param=1, param_sharing=False, qubits_idx=None, gate_info=None, num_qubits=None)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   一个参数化的 oracle 门。

   :param oracle: 用于产生 oracle 的函数。
   :type oracle: Callable[[Tensor],Tensor]
   :param param: 输入参数，默认为 ``None`` i.e. 随机。
   :type param: Union[Tensor,float,List[float]]
   :param qubits_idx: 作用在的量子比特的编号。
   :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param num_acted_param: 单次操作需要的参数数量。
   :type num_acted_param: int, optional
   :param param_sharing: 所有操作是否共享相同的一组参数。
   :type param_sharing: bool
   :param gate_info: 量子门的信息，用于信息追溯或者画图。
   :type gate_info: dict, optional
   :param num_qubits: 量子比特总数，默认为 ``None``
   :type num_qubits: int
