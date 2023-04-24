paddle\_quantum.mbqc.mcalculus
===================================

此模块包含处理 MBQC 测量模式的相关操作。

.. py:class:: MCalculus()

   基类：``object``

   定义测量模式类。

   跟据文献 [The measurement calculus, arXiv: 0704.1263] 的测量语言，该类提供处理测量模式的各种基本操作。

   .. py:method:: track_progress(track)

      显示测量模式处理过程的进度条开关。

      :param track: ``True`` 为打开进度条显示，``False`` 为关闭进度条显示，默认为 ``True``
      :type track: Optional[bool]

   .. py:method:: set_circuit(circuit)

      对 ``MCalculus`` 类设置量子电路。

      :param circuit: 量子电路
      :type circuit: Circuit

   .. py:method:: standardize()

      对测量模式进行标准化。

      该方法对测量模式进行标准化操作，转化成等价的 EMC 模型。即将所有的 ``CommandE`` 交换到最前面，其次是 ``CommandM``， ``CommandX`` 和 ``CommandZ``。为了简化测量模式，该方法在标准化各类命令之后还对 ``CommandM`` 进行 Pauli 简化。

   .. py:method:: shift_signals()

      信号转移操作。

      .. note::

         这是用户选择性调用的方法之一。

   .. py:method:: get_pattern()

      返回测量模式。

      :return: 处理后的测量模式
      :rtype: Pattern

   .. py:method:: optimize_by_row()

      按照行序优先的原则对测量模式中的测量顺序进行优化。

      .. warning::

         这是一种启发式的优化算法，对于特定的测量模式可以起到优化测量顺序的作用，不排除存在更优的测量顺序。例如，对于浅层量子电路，
         按照行序优先原则，测量完同一量子位上的量子门、测量对应的节点后，该量子位不再起作用，进而减少后续计算时可能涉及到的节点数目。
