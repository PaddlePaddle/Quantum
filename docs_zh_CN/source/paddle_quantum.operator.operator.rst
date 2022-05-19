paddle\_quantum.operator.operator
========================================

特殊量子操作的功能实现。

.. py:class:: ResetState()

   基类：:py:class:`paddle_quantum.Operator`

   重置量子态。该类目前还没有实现。

   .. py:method:: forward()

      前向函数，目前还没有实现。

      :return: 还没有实现。
      :rtype: NotImplementedType

.. py:class:: PartialState()

   基类：:py:class:`paddle_quantum.Operator`

   得到部分量子态。该类目前还没有实现。

   .. py:method:: forward()

      前向函数，目前还没有实现。

      :return: 还没有实现。
      :rtype: NotImplementedType

.. py:class:: Collapse(measure_basis)

   基类：:py:class:`paddle_quantum.Operator`

   该类可以让你使用对量子态进行坍缩，坍缩到某一本征态。

   :param measure_basis: 测量基底。量子态会坍缩到对应的本征态上。
   :type measure_basis: Union[Iterable[paddle.Tensor], str]
   :raises NotImplementedError: 所输入的测量基底还没有实现。

   .. py:method:: forward(state, desired_result)

      计算输入的量子态的坍缩。

      :param state: 输入的量子态，其将会被坍缩。
      :type state: paddle_quantum.State
      :param desired_result: 想要坍缩到的特定结果。
      :type desired_result: Union[int, str]
      :raises NotImplementedError: 当前仅支持 z 基底。
      :return: 坍缩后的量子态。
      :rtype: paddle_quantum.State
