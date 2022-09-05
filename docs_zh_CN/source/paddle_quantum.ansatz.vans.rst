paddle\_quantum.ansatz.vans
==================================

可变结构电路的功能实现。

.. py:class:: Inserter

   基类: :py:class:`object`

   用于向电路中加入模块的插入器类。

   .. py:classmethod:: insert_identities(cir, insert_rate, epsilon)

      根据插入比例，向当前量子电路中添加电路模块。

      :param cir: 输入量子电路。
      :type cir: Circuit
      :param insert_rate: 添加速率。
      :type insert_rate: float
      :param epsilon: 添加模块初始化参数的浮动范围。
      :type epsilon: float
      :return: 插入后的电路。
      :rtype: Circuit

.. py:class:: Simplifier

   基类: :py:class:`object`

   用于电路简化的简化器类。

   .. py:classmethod:: simplify_circuit(cir, zero_init_state=True)

      根据简化规则删除电路中的量子门。

      :param cir: 待简化电路。
      :type cir: Circuit
      :param zero_init_state: 量子电路作用的初始态是否为 :math:`|0\rangle`，默认为 ``True``。
      :type zero_init_state: bool, optional
      :return: 简化后的电路。
      :rtype: Circuit

.. py:function:: cir_decompose(cir)

   将电路中的 Layer 分解成量子门, 如果需要的话可以把所有参数门的输入转为可训练参数

   :param cir: 待分解电路
   :type cir: Circuit
   :param trainable: 是否将分解后的参数量子门输入转为参数
   :type trainable: bool, optional
   :return: 分解后的电路
   :rtype: Circuit

   .. note::

      该量子电路稳定支持原生门，不支持 oracle 等其他自定义量子门。

.. py:class:: VAns(n, loss_func, *loss_func_args, epsilon=0.1, insert_rate=2, iter=100, iter_out=10, LR =0.1, threshold=0.002, accept_wall=100, zero_init_state=True)

   基类: :py:class:`object`

   自动优化电路结构的 VAns 类。

   .. note::

      输入的损失函数的第一个参数必须为量子电路。

   :param n: 量子比特数量。
   :type n: int
   :param loss_func: 损失函数。
   :type loss_func: Callable[[Circuit, Any], paddle.Tensor]
   :param \*loss_func_args: 损失函数除了电路以外的所有参数。
   :type \*loss_func_args: Any
   :param epsilon: 添加模块的初始化参数浮动范围，默认为 ``0.1``。
   :type epsilon: float, optional
   :param insert_rate: 添加率，控制一次添加模块的数量，默认为 ``2``。
   :type insert_rate: float, optional
   :param iter: 优化参数迭代次数，默认为 ``100``。
   :type iter: int, optional
   :param iter_out: 优化结构的迭代次数，默认为 ``10``。
   :type iter_out: int, optional
   :param LR: 学习率，默认为 ``0.1``。
   :type LR: float, optional
   :param threshold: 删除量子门时允许损失上升的阈值，默认为 ``0.002``。
   :type threshold: float, optional
   :param accept_wall: 完成一轮结构优化后的电路采纳率，默认为 ``100``。
   :type accept_wall: float, optional
   :param zero_init_state: 电路作用的初始态是否为 :math:`|0\rangle`，默认为 ``True``。
   :type zero_init_state: bool, optional

   .. py:method:: train()

      使用 VAns 方法进行训练。

      :return: 优化过程中损失最低的电路。
      :rtype: Circuit

   .. py:method:: optimization(cir)

      对电路参数进行优化。

      :param cir: 当前电路。
      :type cir: Circuit
      :return: 优化后的损失值。
      :rtype: float

   .. py:method:: delete_gates(cir, loss)

      在损失增加小于一定阈值的情况下，删除电路中的参数化量子门以进一步简化电路。

      :param cir: 目标量子电路。
      :type cir: Circuit
      :param loss: 当前损失值。
      :type loss: float
      :return: 删除多余量子门后的电路。
      :rtype: Circuit
