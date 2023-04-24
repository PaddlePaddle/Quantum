paddle\_quantum.data_analysis.power_flow
=============================================
Power Flow模型

.. py:class:: Bus(data, name)
    
    电力系统节点

    :param data: 节点数据。
    :type data: List[float]
    :param name: 节点名称。
    :type name: str

.. py:class:: Branch(branchIndex, fromBus, toBus, data)
    
    电力系统链路

    :param branchIndex: 链路的编号.
    :type branchIndex: int
    :param fromBus: 链路起点节点的编号。
    :type fromBus: int
    :param toBus: 链路终点节点的编号。
    :type toBus: int
    :param data: 链路数据。
    :type data: List[float]

.. py:class:: Grid(buses, branches, Mva_base)
    
    电力系统网络

    :param buses: 电力系统中的节点.
    :type buses: List[Bus]
    :param branches: 电力系统中的链路。
    :type branches: List[Branch]
    :param Mva_base: 功率基准。
    :type Mva_base: float

    .. py:method:: get_bus_by_number(number)
        
        :param number: 指定节点编号。
        :type number: int
        :return: 返回给定编号的节点。
        :rtype: Bus

        :raises NameError: 不存在给定编号的节点。
    
    .. py:method:: get_branch_by_number(number)

        :param number: 指定链路编号。
        :type number: int
        :return: 返回给定编号的链路。
        :rtype: Branch

        :raises NameError: 不存在给定编号的链路。

    .. py:method:: get_branch_by_bus(busNumber)

        :param busNumber: 指定节点编号。
        :type busNumber: int
        :return: 返回给定编号节点对应的链路。
        :rtype: List[Branch]
    
    .. py:property:: pq_buses()

        返回PQ节点

    .. py:property:: pv_buses()

        返回PV节点

    .. py:method:: powerflow(threshold, minIter, maxIter, depth, iteration, LR, gamma)

        :param threshold: 结束潮流计算的误差阈值。
        :type threshold: double
        :param minIter: 潮流计算中最小迭代次数。
        :type minIter: int
        :param maxIter: 潮流计算中最大迭代次数。
        :type maxIter: int
        :param depth: 模拟电路的深度。
        :type depth: int
        :param iterations: 优化的迭代次数。
        :type iterations: int
        :param LR: 优化器的学习率。
        :type LR: float
        :param gamma: 如果损失函数低于此值，则可以提前结束优化。 默认值为 ``0``。
        :type gamma: Optional[float]

    .. py:method:: printResults()

        返回潮流计算结果

    .. py:method:: saveResults()
        
        保存潮流计算结果
    

.. py:function:: compute(A, b, depth, iterations, LR, gamma)

   求解线性方程组 Ax=b。

   :param A: 输入矩阵。
   :type A: numpy.ndarray
   :param b: 输入向量。
   :type b: numpy.ndarray
   :param depth: 模拟电路的深度。
   :type depth: int
   :param iterations: 优化的迭代次数。
   :type iterations: int
   :param LR: 优化器的学习率。
   :type LR: float
   :param gamma: 如果损失函数低于此值，则可以提前结束优化。 默认值为 ``0``。
   :type gamma: Optional[float]

   :return: 返回线性方程组的解
   :rtype: np.ndarray

   :raises ValueError: A不是一个方阵。
   :raises ValueError: A和b的维度不一致。
   :raises ValueError: A是一个奇异矩阵，因此不存在唯一解。

.. py:function:: data_to_Grid(file_name)
    
    将数据文件转化为电网模型。

    :param file_name: 数据文件名称。
    :type file_name: str
    :return: 返回电力系统网络。
    :rtype: Grid
    