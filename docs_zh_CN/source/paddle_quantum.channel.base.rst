paddle\_quantum.channel.base
===================================

量子信道基类的功能实现。

.. py:class:: Channel(backend=None, dtype=None, name_scope=None)

   基类：:py:class:`paddle_quantum.base.Operator`

   量子信道的基类。


   :param type_repr: 表示的类型。合法的取值包括 ``'Choi'``， ``'Kraus'``， ``'Stinespring'``。
   :type type_repr: str
   :param representation: 信道的表示，默认为 ``None``，即未指定。
   :type representation: Union[paddle.Tensor, List[paddle.Tensor]]
   :param qubits_idx: 作用的量子比特的编号，默认为 ``None``，表示 list(range(num_acted_qubits))。
   :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
   :param num_qubits: 量子比特的总数，默认为 ``None``。
   :type num_qubits: int
   :param check_legality: 表示的完整性校验，默认为 ``True``。
   :type check_legality: bool
   :param num_acted_qubits: 信道作用的量子比特的数量，默认为 ``None``。
   :type num_acted_qubits: int
   :param backend: 执行信道的后端，默认为 ``None``。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 数据的类型, 默认为 ``None``。
   :type dtype: str, optional
   :param name_scope: 为 Layer 内部参数命名而采用的名称前缀。如果前缀为 "my_layer"，在一个类名为MyLayer的Layer中，
      参数名为 "my_layer_0.w_n"，其中 "w" 是参数的名称，"n" 为自动生成的具有唯一性的后缀。如果为 ``None``，
      前缀名将为小写的类名。默认为 ``None``。
   :type name_scope: str, optional

   .. note::

      当 ``representation`` 给定时，不管 ``num_acted_qubits`` 是否为 ``None``， ``num_acted_qubits`` 将由 ``representation`` 自动确定。

   .. py:property:: choi_repr()

      该信道的 Choi 表达式。

      :raises ValueError: 需要指定此 Channel 实例的 Choi 表示。

      :return: 一个形状为 :math:`[d_\text{out}^2, d_\text{in}^2]` 的 Tensor，这里 :math:`d_\text{in/out}` 为信道的输入/出维度。
      :rtype: paddle.Tensor

   .. py:property:: kraus_repr()

      该信道的 Kraus 表达式。

      :raises ValueError: 需要指定此 Channel 实例的 Kraus 表示。

      :return: 一个形状为 :math:`[d_\text{out}, d_\text{in}]` 的 Tensor，这里 :math:`d_\text{in/out}` 为信道的输入/出维度。
      :rtype: paddle.Tensor
   
   .. py:property:: stinespring_repr()

      该信道的 Stinespring 表达式。

      :raises ValueError: 需要指定此 Channel 实例的 Stinespring 表示。

      :return: 一个形状为 :math:`[r * d_\text{out}, d_\text{in}]` 的 Tensor，这里 :math:`r` 为信道的秩，且 :math:`d_\text{in/out}` 为信道的输入/出维度。
      :rtype: paddle.Tensor