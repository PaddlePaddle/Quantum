paddle\_quantum.loss.measure
===================================

测量的损失函数的功能实现。

.. py:class:: ExpecVal(hamiltonian, shots=0)

   基类：:py:class:`paddle_quantum.Operator`

   该类用来计算可观测量的期望值。

   该类可以让你使用对可观测量的期望值作为损失函数。

   :param hamiltonian: 输入的可观测量的信息。
   :type hamiltonian: paddle_quantum.Hamiltonian
   :param shots: 测量的次数。默认是 ``0``，使用解析解。只有当后端为 QuLeaf 时，才需要指定该参数。
   :type shots: int, optional

   .. py:method:: forward(state, decompose=False)

      计算可观测量对于输入的量子态的期望值。

      该函数计算的值可以作为损失函数进行优化。

      :param state: 输入量子态，它将被用来计算期望值。
      :type state: paddle_quantum.State
      :param decompose: 输出每项可观测量的期望值。默认为 ``False``，表示返回输入量子态的期望值之和。
      :type decompose: bool, optional
      :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
      :return: 计算得到的期望值，如果后端是 QuLeaf，则返回根据采样计算的到的结果。
      :rtype: paddle.Tensor

.. py:class:: Measure(measure_basis='z')

   基类：:py:class:`paddle_quantum.Operator`

   该类用来计算可观测量的期望值。

   该类可以让你使用对可观测量的期望值作为损失函数。

   :param measure_basis: 要观测的测量基底。默认为 ``'z'``，在 Z 方向上测量。
   :type measure_basis: Union[Iterable[paddle.Tensor], str]

   .. py:method:: forward(state, qubits_idx='full', desired_result=None)

      计算对输入量子态进行测量得到的概率值。

      :param state: 需要测量的量子态。
      :type state: paddle_quantum.State
      :param qubits_idx: 要测量的量子比特的下标，默认为 ``'full'``，表示全都测量。
      :type qubits_idx: Union[Iterable[int], int, str], optional
      :param desired_result: 指定要返回的测量结果的概率值。默认为 ``None``，返回所有测量结果的概率值。
      :type desired_result: Union[Iterable[str], str], optional
      :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端
      :raises NotImplementedError: ``qubits_idx`` 须为 ``Iterable`` 或 ``'full'``。
      :raises NotImplementedError: 目前我们只支持在Z方向上测量。
      :return: 测量结果所对应的概率值。
      :rtype: paddle.Tensor
