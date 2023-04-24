paddle\_quantum.gate.base
================================

量子门的基类的功能实现。

.. py:class:: Gate(matrix=None, qubits_idx=None, depth=1, gate_info=None, num_qubits=None, check_legality=True, num_acted_qubits=None, backend=None, dtype=None, name_scope=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   量子门的基类。

   :param matrix: 量子门的矩阵
   :type matrix: paddle.Tensor
   :param qubits_idx: 作用在的量子比特的编号。
   :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param gate_info: 量子门的信息，用于信息追溯或者画图。
   :type gate_info: dict, optional
   :param check_legality: 表示的完整性校验，默认为 ``True``。
   :type check_legality: bool
   :param num_acted_qubits: 量子门作用的量子比特的数量，默认为 ``None``。
   :type num_acted_qubits: int
   :param backend: 执行门的后端，默认为 ``None``。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 数据的类型，默认为 ``None``。
   :type dtype: str, optional
   :param name_scope: 为 Layer 内部参数命名而采用的名称前缀。如果前缀为 "mylayer"，在一个类名为MyLayer的Layer中，参数名为"mylayer_0.w_n"，其中 "w" 是参数的名称，"n" 为自动生成的具有唯一性的后缀。如果为 ``None``，前缀名将为小写的类名。默认为 ``None``。
   :type name_scope: str, optional

   .. py:property:: matrix()

      此门的酉矩阵

      :raises ValueError: 需要在门的实例给出矩阵。

   .. py:method:: gate_history_generation()

      生成量子门的历史记录

   .. py:method:: set_gate_info(**kwargs)

      设置 `gate_info` 的接口

      :param kwargs: 用于设置 `gate_info` 的参数。
      :type kwargs: Any

   .. py:method:: display_in_circuit(ax, x)

      画出量子电路图，在 `Circuit` 类中被调用。

      :param ax: ``matplotlib.axes.Axes`` 的实例.
      :type ax: matplotlib.axes.Axes
      :param x: 开始的水平位置。
      :type x: float

      :return: 占用的总宽度。
      :rtype: float

   .. note::

      使用者可以覆写此函数，从而自定义显示方式。

.. py:class:: ParamGate(generator, param=None, depth=1, num_acted_param=1, param_sharing=False, qubits_idx=None, gate_info=None, num_qubits=None, check_legality=True, num_acted_qubits=None, backend=None, dtype=None, name_scope=None)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   可参数化量子门的基类。

   :param generator: 用于产生量子门酉矩阵的函数。
   :type generator: Callable[[Tensor],Tensor]
   :param param: 输入参数，默认为 ``None`` i.e. 随机。
   :type param: Union[Tensor,float,List[float]]
   :param qubits_idx: 作用在的量子比特的编号。默认为 ``None`` i.e. list(range(num_acted_qubits))。
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
   :param check_legality: 表示的完整性校验，默认为 ``True``。
   :type check_legality: bool
   :param num_acted_qubits: 量子门作用的量子比特的数量，默认为 ``None``。
   :type num_acted_qubits: int
   :param backend: 执行门的后端，默认为 ``None``。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 数据的类型，默认为 ``None``。
   :type dtype: str, optional
   :param name_scope: 为 Layer 内部参数命名而采用的名称前缀。如果前缀为 "mylayer"，在一个类名为MyLayer的Layer中，参数名为"mylayer_0.w_n"，其中 "w" 是参数的名称，"n" 为自动生成的具有唯一性的后缀。如果为 ``None``，前缀名将为小写的类名。默认为 ``None``。
   :type name_scope: str, optional

   .. py:method:: theta_generation(param, param_shape)

      规范可参数化量子门的输入，并根据输入决定是否要管理或者生成参数

      :param param: 可参数化量子门的输入
      :type param: Union[paddle.Tensor, float, List[float]]
      :param param_shape: 输入的形状
      :type param_shape: List[int]

   .. note::

      在以下情况 ``param`` 会被转为一个参数
         - ``param`` 是 ``None``
      在以下情况 ``param`` 会被记录为一个参数
         - ``param`` 是 `ParamBase`
      在以下情况 ``param`` 会保持不变
         - ``param`` 是一个 `paddle.Tensor` 但不是 `ParamBase`
         - ``param`` 是一个 `float` 或者 `List[float]`

   .. py:method:: gate_history_generation()

      生成量子门的历史记录

   .. py:method:: display_in_circuit(ax, x)

      画出量子电路图，在 `Circuit` 类中被调用。

      :param ax: ``matplotlib.axes.Axes`` 的实例.
      :type ax: matplotlib.axes.Axes
      :param x: 开始的水平位置。
      :type x: float

      :return: 占用的总宽度。
      :rtype: float

   
