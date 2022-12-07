paddle\_quantum.gate.base
================================

量子门的基类的功能实现。

.. py:class:: Gate(depth=1, backend=None, dtype=None, name_scope=None)

   基类：:py:class:`paddle_quantum.base.Operator`

   量子门的基类。

   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param backend: 执行门的后端，默认为 ``None``。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 数据的类型，默认为 ``None``。
   :type dtype: str, optional
   :param name_scope: 为 Layer 内部参数命名而采用的名称前缀。如果前缀为 "mylayer"，在一个类名为MyLayer的Layer中，
      参数名为"mylayer_0.w_n"，其中 "w" 是参数的名称，"n" 为自动生成的具有唯一性的后缀。如果为 ``None``，
      前缀名将为小写的类名。默认为 ``None``。
   :type name_scope: str, optional

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

.. py:class:: ParamGate

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   可参数化量子门的基类。

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

      生成可参数化量子门的历史记录

   
