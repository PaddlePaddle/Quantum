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
   :raises Exception: 所输入的量子态维度不正确。
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。


   .. py:property:: ket()

      得到量子态的列向量形式。

   .. py:property:: bra()

      得到量子态的行向量形式。

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
      :return: 返回 NotImplementedError，该函数会在后续实现。
      :rtype: Error

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
