paddle\_quantum.ansatz.container
=======================================

顺序电路类的功能实现。

.. py:class:: Sequential(*operators)

   基类：:py:class:`paddle_quantum.base.Operator`

   顺序容器。

   :param \*operators: 准备组建 Sequential 的 Operator类
   :type \*operators: Operator

   .. Note::

      子 Layer 将按构造函数参数的顺序添加到此容器中。传递给构造函数的参数可以 Layers 或可迭代的 (name, Layer) 元组。

   .. py:method:: append(operator)
      
      增加一个 Operator 类

      :param operator: 一个（附带名字的）Operator 类
      :type operator: Union[Iterable, Operator]

   .. py:method:: extend(operator)
      
      增加一组 Operator 类

      :param operators: 一组 Operator 类
      :type operators: List[Operator]
   
   .. py:method:: insert(index, operator)
      
      在指定位置插入一个 Operator 类

      :param index: 插入的位置
      :type index: int
      :param operator: 一个 Operator
      :type operator: Operator

   .. py:method:: pop(index, operator)
      
      在指定位置或者指定 Operator 下删除一个 Operator 类

      :param index: 指定删除的 Operator 的索引位置
      :type index: int
      :param operator: 指定删除的 Operator
      :type operator: Operator, optional
   
   .. py:method:: forward(state)

      前向传播输入数据

      :param state: 输入数据
      :type state: Any

      :return: 输出数据
      :rtype: Any
