paddle\_quantum.base
===========================

量桨的基础函数的功能实现。

.. py:function:: set_device(device)

   设置量桨的 tensor 的存储设备。

   :param device: 要设置的设备名称。
   :type backend: str

.. py:function:: get_device()

   得到当前量桨的 tensor 的存储设备名称。

   :return: 当前的设备名称。
   :rtype: str

.. py:function:: set_backend(backend)

   设置量桨的后端实现方式。

   :param backend: 要设置的后端实现方式。
   :type backend: Union[str, paddle_quantum.Backend]

.. py:function:: get_backend()

   得到量桨的当前后端实现方式。

   :return: 量桨当前的后端名称。
   :rtype: paddle_quantum.Backend

.. py:function:: set_dtype(dtype)

   设置量桨中变量的数据类型。

   :param dtype: 你想要设置的数据类型，可以是 ``complex64`` 或 ``complex128``。
   :type dtype: str

.. py:function:: get_dtype()

   得到当前量桨中变量的数据类型。

   :return: 当前的数据类型。
   :rtype: str

.. py:class:: paddle_quantum.base.Operator(backend=None, dtype=None, name_scope=None)

   基类：:py:class:`paddle.fluid.dygraph.layers.Layer`
   
   用于实现量子操作的基类

   :param backend: 量子操作的后端实现方式。默认为 ``None``，使用默认方式。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 量子操作中的数据类型。默认为 ``None``，使用默认类型。
   :type dtype: str, optional
   :param name_scope: 为 Operator 内部参数命名而采用的名称前缀。默认为 ``None``，没有前缀。
   :type name_scope: str, optional
