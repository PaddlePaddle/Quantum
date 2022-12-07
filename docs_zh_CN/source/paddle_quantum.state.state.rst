paddle\_quantum.state.state
==================================

量子态类的功能实现。

.. py:class:: State(data, num_qubits=None, backend=None, dtype=None)

   基类：:py:class:`object`

   用于实现量子态的类。

   :param data: 量子态的数学解析形式。
   :type data: Union[paddle.Tensor, np.ndarray, QCompute.QEnv]
   :param num_qubits: 量子态所包含的量子比特数。默认为 None，会自动从 data 中推导出来。
   :type num_qubits: int, optional
   :param backend: 指定量子态的后端实现形式。默认为 None，使用全局的默认后端。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 量子态的数据类型。默认为 None，使用全局的默认数据类型。
   :type dtype: str, optional
   :raises ValueError: 无法识别后端


   .. py:property:: ket()

      得到量子态的列向量形式。

   .. py:property:: bra()

      得到量子态的行向量形式。

   .. py:method:: normalize()

      得到归一化后量子态

      :raises NotImplementedError: 当前后端不支持归一化
      
   .. py:method:: evolve(H,t)

      得到经过给定哈密顿量演化后的量子态

      :param H: 系统哈密顿量
      :type H: Union[np.ndarray, paddle.Tensor, Hamiltonian]
      :param t: 演化时间
      :type t: float
      :raises NotImplementedError: 当前后端不支持量子态演化

   .. py:method:: kron(other)

      得到与给定量子态之间的张量积

      :param other: 给定量子态
      :type other: State
      
      :return: 返回张量积
      :rtype: State

   .. py:method:: __matmul__(other)

      得到与量子态或张量之间的乘积

      :param other: 给定量子态
      :type other: State
      :raises NotImplementedError: 不支持与该量子态进行乘积计算
      :raises ValueError: 无法对两个态向量相乘，请检查使用的后端

      :return: 返回量子态的乘积
      :rtype: paddle.Tensor

   .. py:method:: __rmatmul__(other)

      得到与量子态或张量之间的乘积

      :param other: 给定量子态
      :type other: State
      :raises NotImplementedError: 不支持与该量子态进行乘积计算
      :raises ValueError: 无法对两个态向量相乘，请检查使用的后端

      :return: 返回量子态的乘积
      :rtype: paddle.Tensor

   .. py:method:: numpy()

      得到量子态的数据的 numpy 形式。

      :return: 量子态的数据的 numpy.ndarray 形式。
      :rtype: np.ndarray

   .. py:method:: to(backend, dtype, device, blocking)

      改变量子态的属性。

      :param backend: 指定量子态的新的后端实现形式。
      :type backend: str
      :param dtype: 指定量子态的新的数据类型。
      :type dtype: str
      :param device: 指定量子态的新的存储设备。
      :type device: str
      :param blocking: 如果为 False 并且当前 Tensor 处于固定内存上，将会发生主机到设备端的异步拷贝。否则会发生同步拷贝。如果为 None，blocking 会被设置为 True，默认 为False。
      :type blocking: str
      :raises NotImplementedError: 仅支持在态向量与密度矩阵之间进行转换
      :raises NotImplementedError: 不支持在该设备或blocking上进行转换

   .. py:method:: clone()

      返回当前量子态的副本。

      :return: 一个内容和当前量子态都相同的新的量子态。
      :rtype: paddle_quantum.State

   .. py:method:: expec_val(hamiltonian, shots: int)

      量子态关于输入的可观测量的期望值。

      :param hamiltonian: 输入的可观测量。
      :type hamiltonian: paddle_quantum.Hamiltonian
      :param shots: 测量次数。
      :type shots: int
      :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
      :return: 该量子态关于可观测量的期望值。
      :rtype: float
        

   .. py:method:: measure(shots=0, qubits_idx=None, plot=False)

      对量子态进行测量。

      :param shots: 测量次数。默认为 0，即计算解析解。
      :type shots: int, optional
      :param qubits_idx: 要测量的量子态下标。默认为 None，表示全部测量。
      :type qubits_idx: Union[Iterable[int], int], optional
      :param plot: 是否画图。默认为 Flase，表示不画图。
      :type plot: bool, optional
      :raises Exception: 测量的次数必须大于0。
      :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
      :raises NotImplementedError: 输入的量子比特下标有误。
      :return: 测量结果。
      :rtype: dict

.. py:function:: _type_fetch(data)

   获取数据的类型

   :param data: 输入数据
   :type data: Union[np.ndarray, paddle.Tensor, State]

   :raises ValueError: 输入量子态不支持所选后端
   :raises TypeError: 无法识别输入量子态的数据类型

   :return: 返回输入量子态的数据类型
   :rtype: str

.. py:function:: _density_to_vector(rho)

   将密度矩阵转换为态向量

   :param rho: 输入的密度矩阵
   :type rho: Union[np.ndarray, paddle.Tensor]

   :raises ValueError: 输出量子态可能不为纯态

   :return: 返回态向量
   :rtype: Union[np.ndarray, paddle.Tensor]

.. py:function:: _type_transform(data, output_type)

   将输入量子态转换成目标类型

   :param data: 需要转换的数据
   :type data: Union[np.ndarray, paddle.Tensor, State]
   :param output_type: 目标数据类型
   :type output_type: str

   :raises ValueError: 输入态不支持转换为目标数据类型

   :return: 返回目标数据类型的量子态
   :rtype: Union[np.ndarray, paddle.Tensor, State]

.. py:function:: is_state_vector(vec, eps)

   检查输入态是否为量子态向量

   :param vec: 输入的数据 :math:`x`
   :type vec: Union[np.ndarray, paddle.Tensor]
   :param eps: 容错率
   :type eps: float, optional

   :return: 返回是否满足 :math:`x^\dagger x = 1` ，以及量子比特数目或错误信息
   :rtype: Tuple[bool, int]

   .. note::
      错误信息为:
        * ``-1`` 如果上述公式不成立
        * ``-2`` 如果输入数据维度不为2的幂
        * ``-3`` 如果输入数据不为向量

.. py:function:: is_density_matrix(rho, eps)

   检查输入数据是否为量子态的密度矩阵

   :param rho: 输入的数据 ``rho`` 
   :type rho: Union[np.ndarray, paddle.Tensor]
   :param eps: 容错率
   :type eps: float, optional

   :return: 返回输入数据 ``rho`` 是否为迹为1的PSD矩阵，以及量子比特数目或错误信息
   :rtype: Tuple[bool, int]

   .. note::
      错误信息为:
        * ``-1`` 如果 ``rho`` 不为PSD矩阵
        * ``-2`` 如果 ``rho`` 的迹不为1
        * ``-3`` 如果 ``rho`` 的维度不为2的幂
        * ``-4`` 如果 ``rho`` 不为一个方阵
