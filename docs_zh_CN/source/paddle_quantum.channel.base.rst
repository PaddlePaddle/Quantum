paddle\_quantum.channel.base
===================================

量子信道基类的功能实现。

.. py:class:: Channel(backend=None, dtype=None, name_scope=None)

   基类：:py:class:`paddle_quantum.base.Operator`

   量子信道的基类。

   :param backend: 执行信道的后端，默认为 ``None``。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 数据的类型, 默认为 ``None``。
   :type dtype: str, optional
   :param name_scope: 为 Layer 内部参数命名而采用的名称前缀。如果前缀为 "my_layer"，在一个类名为MyLayer的Layer中，
      参数名为"my_layer_0.w_n"，其中 "w" 是参数的名称，"n" 为自动生成的具有唯一性的后缀。如果为 ``None``，
      前缀名将为小写的类名。默认为 ``None``。
   :type name_scope: str, optional
