paddle\_quantum.mbqc.simulator
==============================

此模块包含构造 MBQC 模型的常用类和配套的运算模拟工具。

.. py:class:: MBQC()

   基类：``object``

   定义基于测量的量子计算模型 ``MBQC`` 类。

   用户可以通过实例化该类来定义自己的 MBQC 模型。

   .. py:method:: set_graph(graph)

      设置 MBQC 模型中的图。

      该函数用于将用户自己构造的图传递给 ``MBQC`` 实例。

      :param graph: MBQC 模型中的图，由列表 ``[V, E]`` 给出， 其中 ``V`` 为节点列表，``E`` 为边列表
      :type graph: list

   .. py:method:: get_graph()

      获取图的信息。

      该函数用于将用户自己构造的图传递给 ``MBQC`` 实例。

      :return: 图
      :rtype: networkx.Graph

   .. py:method:: set_pattern(pattern)

      设置 MBQC 模型的测量模式。

      该函数用于将用户由电路图翻译得到或自己构造的测量模式传递给 ``MBQC`` 实例。

      :param pattern: MBQC 算法对应的测量模式
      :type pattern: Pattern

   .. py:method:: get_pattern()

      获取测量模式的信息。

      :return: 测量模式
      :rtype: Pattern

   .. py:method:: set_input_state(state)

      设置需要替换的输入量子态。

      .. warning::

         与电路模型不同，MBQC 模型通常默认初始态为加态。如果用户不调用此方法设置初始量子态，则默认为加态。
         如果用户以测量模式运行 MBQC，则此处输入量子态的系统标签会被限制为从零开始的自然数，类型为整型。

      :param state: 需要替换的量子态，默认为加态
      :type state: Optional[State]

   .. py:method:: draw_process(draw, pos, pause_time)

      动态过程图绘制，用以实时展示 MBQC 模型的模拟计算过程。

      .. warning::

         与电路模型不同，MBQC 模型通常默认初始态为加态。如果用户不调用此方法设置初始量子态，则默认为加态。
         如果用户以测量模式运行 MBQC，则此处输入量子态的系统标签会被限制为从零开始的自然数，类型为整型。

      :param draw: 是否绘制动态过程图的布尔开关
      :type draw: Optional[bool]
      :param pos: 节点坐标的字典数据或者内置的坐标选择，内置的坐标选择有： ``True`` 为测量模式自带的坐标，``False`` 为 ``spring_layout`` 坐标
      :type pos: Optional[bool]
      :param pause_time: 需要替换的量子态，默认为加态
      :type pause_time: Optional[float]

   .. py:method:: track_process(track)

      显示 MBQC 模型运行进度的开关。

      :param track: ``True`` 打开进度条显示功能， ``False`` 关闭进度条显示功能
      :type track: Optional[bool]

   .. py:method:: measure(which_qubit, basis_list)

      显示 MBQC 模型运行进度的开关。

      .. note::

         这是用户在实例化 MBQC 类之后最常调用的方法之一，此处我们对单比特测量模拟进行了最大程度的优化，随着用户对该函数的调用，MBQC 类将自动完成激活相关节点、生成所需的图态以及对特定比特进行测量的全过程，并记录测量结果和对应测量后的量子态。用户每调用一次该函数，就完成一次对单比特的测量操作。

      .. warning::

         当且仅当用户调用 ``measure`` 类方法时，MBQC 模型才真正进行运算。

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.simulator import MBQC
         from paddle_quantum.mbqc.qobject import State
         from paddle_quantum.mbqc.utils import zero_state, basis

         G = [['1', '2', '3'], [('1', '2'), ('2', '3')]]
         mbqc = MBQC()
         mbqc.set_graph(G)
         state = State(zero_state(), ['1'])
         mbqc.set_input_state(state)
         mbqc.measure('1', basis('X'))
         mbqc.measure('2', basis('X'))
         print("Measurement outcomes: ", mbqc.get_classical_output())

      ::

         Measurement outcomes:  {'1': 0, '2': 1}

      :param which_qubit: 待测量量子比特的系统标签，可以是 ``str``, ``tuple`` 等任意数据类型，但需要和 MBQC 模型的图上标签匹配
      :type which_qubit: Any
      :param basis_list: 测量基向量构成的列表，列表元素为 ``Tensor`` 类型的列向量
      :type basis_list: list

   .. py:method:: sum_outcomes(which_qubits, start)

      根据输入的量子系统标签，在存储测量结果的字典中找到对应的测量结果，并进行求和。

      .. note::

         在进行副产品纠正操作和定义适应性测量角度时，用户可以调用该方法对特定比特的测量结果求和。

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.simulator import MBQC
         from paddle_quantum.mbqc.qobject import State
         from paddle_quantum.mbqc.utils import zero_state, basis

         G = [['1', '2', '3'], [('1', '2'), ('2', '3')]]
         mbqc = MBQC()
         mbqc.set_graph(G)
         input_state = State(zero_state(), ['1'])
         mbqc.set_input_state(input_state)
         mbqc.measure('1', basis('X'))
         mbqc.measure('2', basis('X'))
         mbqc.measure('3', basis('X'))
         print("All measurement outcomes: ", mbqc.get_classical_output())
         print("Sum of outcomes of qubits '1' and '2': ", mbqc.sum_outcomes(['1', '2']))
         print("Sum of outcomes of qubits '1', '2' and '3' with an extra 1: ", mbqc.sum_outcomes(['1', '2', '3'], 1))

      ::

         All measurement outcomes:  {'1': 0, '2': 0, '3': 1}
         Sum of outcomes of qubits '1' and '2':  0
         Sum of outcomes of qubits '1', '2' and '3' with an extra 1:  2

      :param which_qubits: 需要查找测量结果并求和的比特的系统标签列表
      :type which_qubits: list
      :param start: 对结果进行求和后需要额外相加的整数
      :type start: int
      :return: 指定比特的测量结果的和
      :rtype: int

   .. py:method:: correct_byproduct(gate, which_qubit, power)

      对测量后的量子态进行副产品纠正。

      .. note::

         这是用户在实例化 MBQC 类并完成测量后，经常需要调用的一个方法。

      代码示例：

      此处展示的是 MBQC 模型下实现隐形传态的一个例子。

      .. code-block:: python

         from paddle_quantum.mbqc.simulator import MBQC
         from paddle_quantum.mbqc.qobject import State
         from paddle_quantum.mbqc.utils import random_state_vector, basis, compare_by_vector

         G = [['1', '2', '3'], [('1', '2'), ('2', '3')]]
         state = State(random_state_vector(1), ['1'])
         mbqc = MBQC()
         mbqc.set_graph(G)
         mbqc.set_input_state(state)
         mbqc.measure('1', basis('X'))
         mbqc.measure('2', basis('X'))
         outcome = mbqc.get_classical_output()
         mbqc.correct_byproduct('Z', '3', outcome['1'])
         mbqc.correct_byproduct('X', '3', outcome['2'])
         state_out = mbqc.get_quantum_output()
         state_std = State(state.vector, ['3'])
         compare_by_vector(state_out, state_std)

      ::

         Norm difference of the given states is: 0.0
         They are exactly the same states.

      :param gate: ``'X'`` 或者 ``'Z'``，分别表示 Pauli X 或 Z 门修正
      :type gate: str
      :param which_qubit: 待操作的量子比特的系统标签，可以是 ``str``, ``tuple`` 等任意数据类
      :type which_qubit: list
      :param power: 副产品纠正算符的指数
      :type power: int

   .. py:method:: run_pattern()

      按照设置的测量模式对 MBQC 模型进行模拟。

      .. warning::

         该方法必须在 ``set_pattern`` 调用后调用。

   .. py:method:: get_classical_output()

      获取 MBQC 模型运行后的经典输出结果。

      :return: 如果用户输入是测量模式，则返回测量输出节点得到的比特串，与原电路的测量结果相一致，没有被测量的比特位填充 "?"，如果用户输入是图，则返回所有节点的测量结果
      :rtype: Union[str, dict]

   .. py:method:: get_history()

      获取 MBQC 计算模拟时的中间步骤信息。

      :return: 生成图态、进行测量、纠正副产品后运算结果构成的列表
      :rtype: list

   .. py:method:: get_quantum_output()

      获取 MBQC 模型运行后的量子态输出结果。

      :return: MBQC 模型运行后的量子态
      :rtype: State

.. py:function:: simulate_by_mbqc(circuit, input_state)

   使用等价的 MBQC 模型模拟量子电路。

   该函数通过将量子电路转化为等价的 MBQC 模型并运行，从而获得等价于原始量子电路的输出结果。

   .. warning::

      此处输入的 ``circuit`` 参数包含了测量操作。
      另，MBQC 模型默认初始态为加态，因此，如果用户不输入参数 ``input_state`` 设置初始量子态，则默认为加态。

   :param circuit: 量子电路图
   :type circuit: Circuit
   :param input_state: 量子电路的初始量子态，默认为 :math:`|+\rangle` 态
   :type input_state: Optional[State]
   :return: 
      包含如下两个元素：

      - str: 经典输出
      - State: 量子输出
   :rtype: Tuple[str, State]

.. py:function:: sample_by_mbqc(circuit, input_state, plot, shots, print_or_not)

   将 MBQC 模型重复运行多次，获得经典结果的统计分布。

   该函数通过将量子电路转化为等价的 MBQC 模型并运行，从而获得等价于原始量子电路的输出结果。

   .. warning::

      此处输入的 circuit 参数包含了测量操作。
      另，MBQC 模型默认初始态为加态，因此，如果用户不输入参数 `input_state` 设置初始量子态，则默认为加态。

   :param circuit: 量子电路图
   :type circuit: Circuit
   :param input_state: 量子电路的初始量子态，默认为加态
   :type input_state: Optional[State]
   :param plot: 绘制经典采样结果的柱状图开关，默认为关闭状态
   :type plot: Optional[bool]
   :param shots: 采样次数，默认为 1024 次
   :type shots: Optional[int]
   :param print_or_not: 是否打印采样结果和绘制采样进度，默认为开启状态
   :type print_or_not: Optional[bool]
   :return: 
      包含如下两个元素：

      - 经典结果构成的频率字典
      - 经典测量结果和所有采样结果（包括经典输出和量子输出）的列表
   :rtype: Tuple[dict, list]
