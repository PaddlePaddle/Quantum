paddle\_quantum.mbqc.qobject
============================

此模块包含量子信息处理的常用对象，如量子态、量子电路、测量模式等。

.. py:class:: State(vector, system)
   :noindex:

   基类：``object``

   定义量子态。

   :param vector: 量子态的列向量
   :type vector: Optional[paddle.Tensor]
   :param system: 量子态的系统标签列表
   :type system: Optional[list]

.. py:class:: Circuit(width)
   :noindex:

   基类：``object``

   定义量子电路。

   :param width: 电路的宽度（比特数）
   :type width: int

   .. warning::

      当前版本仅支持 ``H, X, Y, Z, S, T, Rx, Ry, Rz, Rz_5, U, CNOT, CNOT_15, CZ`` 中的量子门以及测量操作。

   .. py:method:: h(which_qubit)
      :noindex:

      添加 ``Hadamard`` 门。

      其矩阵形式为：

      .. math::

         \frac{1}{\sqrt{2}}\begin{bmatrix} 1&1\\1&-1 \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         cir.h(which_qubit)
         print(cir.get_circuit())

      ::

         [['h', [0], None]]

      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: x(which_qubit)
      :noindex:

      添加 ``Pauli X`` 门。

      其矩阵形式为：

        .. math::

            \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         cir.x(which_qubit)
         print(cir.get_circuit())

      ::

         [['x', [0], None]]

      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: y(which_qubit)
      :noindex:

      添加 ``Pauli Y`` 门。

      其矩阵形式为：

      .. math::

         \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         cir.y(which_qubit)
         print(cir.get_circuit())

      ::

         [['y', [0], None]]

      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: z(which_qubit)
      :noindex:

      添加 ``Pauli Z`` 门。

      其矩阵形式为：

      .. math::

         \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         cir.z(which_qubit)
         print(cir.get_circuit())

      ::

         [['z', [0], None]]

      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: s(which_qubit)
      :noindex:

      添加 ``S`` 门。

      其矩阵形式为：

      .. math::

         \begin{bmatrix} 1&0\\0& i \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         cir.s(which_qubit)
         print(cir.get_circuit())

      ::

         [['s', [0], None]]

      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: t(which_qubit)
      :noindex:

      添加 ``T`` 门。

      其矩阵形式为：

      .. math::

         \begin{bmatrix} 1&0\\0& e^{i\pi/ 4} \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         cir.t(which_qubit)
         print(cir.get_circuit())

      ::

         [['t', [0], None]]

      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: rx(theta, which_qubit)
      :noindex:

      添加关于 x 轴的旋转门。

      其矩阵形式为：

      .. math::

         \begin{bmatrix} \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\ -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

      代码示例：

      ..  code-block:: python

         from paddle import to_tensor
         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         angle = to_tensor([1], dtype='float64')
         cir.rx(angle, which_qubit)
         print(cir.get_circuit())

      ::

         [['rx', [0], Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True, [1.])]]

      :param theta: 旋转角度
      :type theta: paddle.Tensor
      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: ry(theta, which_qubit)
      :noindex:

      添加关于 y 轴的旋转门。

      其矩阵形式为：

      .. math::

         \begin{bmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{bmatrix}

      代码示例：

      ..  code-block:: python

         from paddle import to_tensor
         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         angle = to_tensor([1], dtype='float64')
         cir.ry(angle, which_qubit)
         print(cir.get_circuit())

      ::

         [['ry', [0], Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True, [1.])]]

      :param theta: 旋转角度
      :type theta: paddle.Tensor
      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: rz(theta, which_qubit)
      :noindex:

      添加关于 z 轴的旋转门。

      其矩阵形式为：

      .. math::

         \begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}


      代码示例：

      ..  code-block:: python

         from paddle import to_tensor
         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         angle = to_tensor([1], dtype='float64')
         cir.rz(angle, which_qubit)
         print(cir.get_circuit())

      ::

         [['rz', [0], Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True, [1.])]]

      :param theta: 旋转角度
      :type theta: paddle.Tensor
      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: rz_5(theta, which_qubit)
      :noindex:

      添加关于 z 轴的旋转门（该旋转门对应的测量模式由五个量子比特构成）。

      其矩阵形式为：

      .. math::

         \begin{bmatrix} 1 & 0 \\ 0 & e^{i\theta} \end{bmatrix}


      代码示例：

      ..  code-block:: python

         from paddle import to_tensor
         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         angle = to_tensor([1], dtype='float64')
         cir.rz(angle, which_qubit)
         print(cir.get_circuit())

      ::

         [['rz_5', [0], Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True, [1.])]]

      :param theta: 旋转角度
      :type theta: paddle.Tensor
      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: u(params, which_qubit)

      添加单量子比特的任意酉门。

      .. warning::

         这里的酉门采用 ``Rz Rx Rz`` 分解，

      其分解形式为：

      .. math::

         U(\alpha, \beta, \gamma) = Rz(\gamma) Rx(\beta) Rz(\alpha)

      代码示例：

      ..  code-block:: python

         from paddle import to_tensor
         from numpy import pi
         from paddle_quantum.mbqc.qobject import Circuit
         width = 1
         cir = Circuit(width)
         which_qubit = 0
         alpha = to_tensor([pi / 2], dtype='float64')
         beta = to_tensor([pi], dtype='float64')
         gamma = to_tensor([- pi / 2], dtype='float64')
         cir.u([alpha, beta, gamma], which_qubit)
         print(cir.get_circuit())

      ::

         [['u', [0], [Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True,
            [1.57079633]), Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True,
            [3.14159265]), Tensor(shape=[1], dtype=float64, place=CPUPlace, stop_gradient=True,
            [-1.57079633])]]]

      :param params: 单比特酉门的三个旋转角度
      :type params: List[paddle.Tensor]
      :param which_qubit: 作用量子门的量子位编号
      :type which_qubit: int

   .. py:method:: cnot(which_qubits)
      :noindex:

      添加控制非门。

      当 ``which_qubits`` 为 ``[0, 1]`` 时，其矩阵形式为：

      .. math::

         \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 2
         cir = Circuit(width)
         which_qubits = [0, 1]
         cir.cnot(which_qubits)
         print(cir.get_circuit())
      ::

         [['cnot', [0, 1], None]]

      :param which_qubits: 作用量子门的量子位，其中列表第一个元素为控制位，第二个元素为受控位
      :type which_qubits: List[int]

   .. py:method:: cnot_15(which_qubits)

      添加控制非门（该门对应的测量模式由十五个量子比特构成）。

      当 ``which_qubits`` 为 ``[0, 1]`` 时，其矩阵形式为：

      .. math::

         \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 2
         cir = Circuit(width)
         which_qubits = [0, 1]
         cir.cnot_15(which_qubits)
         print(cir.get_circuit())
      ::

         [['cnot_15', [0, 1], None]]

      :param which_qubits: 作用量子门的量子位，其中列表第一个元素为控制位，第二个元素为受控位
      :type which_qubits: List[int]

   .. py:method:: cz(which_qubits)
      :noindex:

      添加控制 Z 门。

      当 ``which_qubits`` 为 ``[0, 1]`` 时，其矩阵形式为：

      .. math::

         \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{bmatrix}

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         width = 2
         cir = Circuit(width)
         which_qubits = [0, 1]
         cir.cz(which_qubits)
         print(cir.get_circuit())
      ::

         [['cz', [0, 1], None]]

      :param which_qubits: 作用量子门的量子位，其中列表第一个元素为控制位，第二个元素为受控位
      :type which_qubits: List[int]

   .. py:method:: measure(which_qubit, basis_list)

      对量子电路输出的量子态进行测量。

      .. note::

         除默认的 Z 测量外，此处的测量方式可以由用户自定义，但需要将测量方式与测量比特相对应。

      .. warning::

         此方法只接受三种输入方式：
         1. 不输入任何参数，表示对所有的量子位进行 Z 测量；
         2. 输入量子位，但不输入测量基，表示对输入的量子位进行 Z 测量；
         3. 输入量子位和对应测量基，表示对输入量子位进行指定的测量。
         如果用户希望自定义测量基参数，需要注意输入格式为 ``[angle, plane, domain_s, domain_t]``，
         且当前版本的测量平面 ``plane`` 只能支持 ``XY`` 或 ``YZ``。

      :param which_qubit: 被测量的量子位
      :type which_qubit: Optional[int]
      :param basis_list: 测量方式
      :type basis_list: Optional[list]

   .. py:method:: is_valid()

      检查输入的量子电路是否符合规定。

      我们规定输入的量子电路中，每一个量子位上至少作用一个量子门。

      :return: 量子电路是否符合规定的布尔值
      :rtype: bool

   .. py:method:: get_width()

      返回量子电路的宽度。

      :return: 量子电路的宽度
      :rtype: int

   .. py:method:: get_circuit()

      返回量子电路列表。

      :return: 量子电路列表
      :rtype: list

   .. py:method:: get_measured_qubits()

      返回量子电路中测量的比特位。

      :return: 量子电路中测量的比特位列表
      :rtype: list

   .. py:method:: print_circuit_list()

      打印电路图的列表。

      代码示例：

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         from paddle import to_tensor
         from numpy import pi

         n = 2
         theta = to_tensor([pi], dtype="float64")
         cir = Circuit(n)
         cir.h(0)
         cir.cnot([0, 1])
         cir.rx(theta, 1)
         cir.measure()
         cir.print_circuit_list()

      ::

         --------------------------------------------------
                          Current circuit
         --------------------------------------------------
         Gate Name       Qubit Index     Parameter
         --------------------------------------------------
         h               [0]             None
         cnot            [0, 1]          None
         rx              [1]             3.141592653589793
         m               [0]             [0.0, 'YZ', [], []]
         m               [1]             [0.0, 'YZ', [], []]
         --------------------------------------------------

      :return: 用来打印的字符串
      :rtype: string

.. py:class:: Pattern(name, space, input_, output_, commands)

   基类：``object``

   定义测量模式。

   该测量模式的结构依据文献 [The measurement calculus, arXiv: 0704.1263]。

   :param name: 测量模式的名称
   :type name: str
   :param space: 测量模式所有节点列表
   :type space: list
   :param input_: 测量模式的输入节点列表
   :type input_: list
   :param output_: 测量模式的输出节点列表
   :type output_: list
   :param commands: 测量模式的命令列表
   :type commands: list

   .. py:class:: CommandE

      基类：``object``

      定义纠缠命令类。

      .. note::

         此处纠缠命令对应作用控制 Z 门。

      :param which_qubits: 作用纠缠命令的两个节点标签构成的列表
      :type which_qubits: list

   .. py:class:: CommandM

      基类：``object``

      定义测量命令类。

      测量命令有五个属性，分别为测量比特的标签 ``which_qubit``，原始的测量角度 ``angle``，测量平面 ``plane``，域 s 对应的节点标签列表 ``domain_s``，域 t 对应的节点标签列表 ``domain_t``。设原始角度为 :math:`\alpha`，则考虑域中节点依赖关系后的测量角度 :math:`\theta` 为：

      .. math::

         \theta = (-1)^s \times \alpha + t \times \pi

      .. note::

         域 s 和域 t 是 MBQC 模型中的概念，分别记录了 Pauli X 算符和 Pauli Z 算符对测量角度产生的影响，二者共同记录了该测量节点对其他节点的测量结果的依赖关系。

      .. warning::

         该命令当前只支持 XY 和 YZ 平面的测量。

      :param which_qubit: 作用测量命令的节点标签
      :type which_qubit: Any
      :param angle: 原始的测量角度
      :type angle: paddle.Tensor
      :param plane: 测量平面
      :type plane: str
      :param domain_s: 域 s 对应的节点标签列表
      :type domain_s: list
      :param domain_t: 域 t 对应的节点标签列表
      :type domain_t: list

   .. py:class:: CommandX

      基类：``object``

      定义 Pauli X 副产品修正命令类。

      :param which_qubit: 作用修正算符的节点标签
      :type which_qubit: Any
      :param domain: 依赖关系列表
      :type domain: list

   .. py:class:: CommandZ

      基类：``object``

      定义 Pauli Z 副产品修正命令。

      .. note::

         此处纠缠命令对应作用控制 Z 门。

      :param which_qubit: 作用修正算符的节点标签
      :type which_qubit: Any
      :param domain: 依赖关系列表
      :type domain: list

   .. py:class:: CommandS

      基类：``object``

      定义“信号转移”命令类。

      .. note::

         “信号转移”是一类特殊的操作，用于消除测量命令对域 t 中节点的依赖关系，在某些情况下对测量模式进行简化。

      :param which_qubit: 消除依赖关系的测量命令作用的节点标签
      :type which_qubit: Any
      :param domain: 依赖关系列表
      :type domain: list

   .. py:method:: print_command_list()

      打印该 ``Pattern`` 类中的命令的信息，便于用户查看。

      代码示例:

      .. code-block:: python

         from paddle_quantum.mbqc.qobject import Circuit
         from paddle_quantum.mbqc.mcalculus import MCalculus

         n = 1
         cir = Circuit(n)
         cir.h(0)
         pat = MCalculus()
         pat.set_circuit(cir)
         pattern = pat.get_pattern()
         pattern.print_command_list()

      ::

         -----------------------------------------------------------
                              Current Command List
         -----------------------------------------------------------
         Command:        E
         which_qubits:   [('0/1', '0/1'), ('0/1', '1/1')]
         -----------------------------------------------------------
         Command:        M
         which_qubit:    ('0/1', '0/1')
         plane:          XY
         angle:          0.0
         domain_s:       []
         domain_t:       []
         -----------------------------------------------------------
         Command:        X
         which_qubit:    ('0/1', '1/1')
         domain:         [('0/1', '0/1')]
         -----------------------------------------------------------
