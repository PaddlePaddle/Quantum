paddle\_quantum.state.state
==================================

量子态类的功能实现。

.. py:class:: State(data, num_qubits=None, backend=None, dtype=None, override=False)

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
   :param override: 是否跳过输入校验，仅限于内部开发使用。默认为 ``False``。
   :type override: bool, optional
   :raises ValueError: 无法识别后端


   .. py:property:: ket()

      得到量子态的列向量形式。

      :raises ValueError: 后端必须为 StateVector。
      :return: 量子态的列向量形式

   .. py:property:: bra()

      得到量子态的行向量形式。

      :raises ValueError: 后端必须为 StateVector。
      :return: 量子态的行向量形式

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

   .. py:property:: oper_history()

      储存在QPU后端的算子历史信息

      :return: 算子的历史信息
      :rtype: List[Dict[str, Union[str, List[int], paddle.Tensor]]]
      :raises NotImplementedError: 此属性应仅适用于 QuLeaf 后端。
      :raises ValueError: 无法获取算子历史信息，请先运行电路

   .. py:method:: expec_val(hamiltonian, shots: int)

      量子态关于输入的可观测量的期望值。

      :param hamiltonian: 输入的可观测量。
      :type hamiltonian: paddle_quantum.Hamiltonian
      :param shots: 测量次数。
      :type shots: int
      :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
      :return: 该量子态关于可观测量的期望值。
      :rtype: float
        

   .. py:method:: measure(shots=0, qubits_idx=None, plot=False, record=False)

      对量子态进行测量。

      :param shots: 测量次数。默认为 0，即计算解析解。
      :type shots: int, optional
      :param qubits_idx: 要测量的量子态下标。默认为 None，表示全部测量。
      :type qubits_idx: Union[Iterable[int], int], optional
      :param plot: 是否画图。默认为 Flase，表示不画图。
      :type plot: bool, optional
      :param record: 是否返回原始的测量结果记录。默认为 Flase，表示不返回。
      :type record: bool, optional
      :raises ValueError: 测量的次数必须大于0。
      :raises NotImplementedError: Quleaf后端暂不支持record功能。
      :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
      :raises NotImplementedError: 输入的量子比特下标有误。
      :raises ValueError: 使用record功能要求测量次数必须大于0。
      :return: 测量结果。
      :rtype: dict

   .. py:method:: reset_sequence(target_sequence=None)

      根据输入顺序重置量子比特顺序

      :param target_sequence: 目标顺序，默认为 ``None``。
      :type target_sequence: Union[List[int],None]
      :return: 在输入比特顺序下的量子态
      :rtype: paddle_quantum.state.state.State
