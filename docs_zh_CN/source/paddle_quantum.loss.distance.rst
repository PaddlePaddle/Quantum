paddle\_quantum.loss.distance
====================================

距离度量的损失函数的功能实现。

.. py:class:: TraceDistance(target_state)

   基类：:py:class:`paddle_quantum.Operator`

   该类用于实现迹距离的损失函数。

   该类允许用户使用迹距离作为损失函数来训练量子神经网络。

   :param target_state: 用于计算迹距离的目标量子态。
   :type target_state: paddle_quantum.State

   .. py:method:: forward(state)

      计算输入量子态和目标量子态的迹距离。

      该函数计算的值可以作为损失函数进行优化。

      :param state: 输入量子态，它将会和目标量子态计算迹距离。
      :type state: paddle_quantum.State
      :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
      :return: 输入量子态和目标量子态之间的迹距离。
      :rtype: paddle.Tensor

.. py:class:: StateFidelity(target_state)

   基类：:py:class:`paddle_quantum.Operator`

   该类用于实现量子态保真度的损失函数。

   该类允许用户使保真度作为损失函数来训练量子神经网络。

   :param target_state: 用于计算保真度的目标量子态。
   :type target_state: paddle_quantum.State

   .. py:method:: forward(state)

      计算输入量子态和目标量子态的保真度。

      该函数计算的值可以作为损失函数进行优化。

      :param state: 输入量子态，它将会和目标量子态计算保真度。
      :type state: paddle_quantum.State
      :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
      :return: 输入量子态和目标量子态之间的保真度。
      :rtype: paddle.Tensor
