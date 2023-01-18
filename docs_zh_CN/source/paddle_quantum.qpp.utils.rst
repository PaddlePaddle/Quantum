paddle\_quantum.qpp.utils
============================

QPP 线路变换和其他相关工具。更多细节参考论文 https://arxiv.org/abs/2209.14278 中的 Theorem 6。

.. py:function:: qpp_cir(list_theta, list_phi, U) -> Circuit

    根据 ``list_theta`` 和 ``list_phi`` 创建量子相位处理器

   :param list_theta: :math:`R_Y` 门角度输入。
   :type list_theta: Union[np.ndarray, paddle.Tensor]
   :param list_phi: :math:`R_Z` 门角度输入。
   :type list_phi: Union[np.ndarray, paddle.Tensor]
   :param U: 酉矩阵或标量输入。
   :type U: Union[np.ndarray, paddle.Tensor, float]

   :return: 三角 QSP 的多比特一般化线路。
   :rtype: Circuit

.. py:function:: simulation_cir(fn, U, approx=True, deg=50, length=np.pi, step_size=0.00001*np.pi, tol=1e-30)

    返回一个模拟 ``fn`` 的 QPP 线路，见论文 https://arxiv.org/abs/2209.14278 中的 Theorem 6。

   :param fn: 要模拟的函数。
   :type fn: Callable[[np.ndarray], np.ndarray]
   :param U: 酉矩阵。
   :type U: Union[np.ndarray, paddle.Tensor, float]
   :param approx: 是否估算线路角度。默认为 ``True``。
   :type approx: : Optional[bool] 
   :param deg: 模拟的级数。默认为 ``50``。
   :type deg: : Optional[int]
   :param length: 模拟宽度的一半。默认为 :math:`\pi`。
   :type length: Optional[float]
   :param step_size: 采样点的频率。默认为 :math:`0.00001 \pi`。
   :type step_size: Optional[float]
   :param tol: 误差容忍度。默认为 :math:`10^{-30}`，即机械误差。
   :type tol: Optional[float]

   :return: 模拟 ``fn`` 的 QPP 线路。
   :rtype: Circuit

.. py:function:: qps(U, initial_state)

    量子相位搜索算法，见论文 https://arxiv.org/abs/2209.14278 中的算法 1 和 2

   :param U: 目标酉矩阵。
   :type U: Union[np.ndarray, paddle.Tensor]
   :param initial_state: 输入量子态。
   :type initial_state: Union[np.ndarray, paddle.Tensor, State]

   :return:
      包含如下元素的 tuple:

      - 一个 ``U`` 的本征相位；
      - 其相位对应的，存在和 ``initial_state`` 内积不为零的本征态。
   :rtype: Tuple[float, State]

.. py:function:: qubitize(block_enc, num_block_qubits)

    使用一个额外辅助比特来比特化块编码，来保证子空间不变。更多细节见论文 http://arxiv.org/abs/1610.06546。

   :param block_enc: 目标块编码。
   :type block_enc: Union[np.ndarray, paddle.Tensor]
   :param num_block_qubits: 块编码自身所使用的辅助比特数。
   :type num_block_qubits: int

   :return: 比特化的 ``block_enc``
   :rtype: paddle.Tensor

.. py:function:: purification_block_enc(num_qubits, num_block_qubits):

    随机生成一个 :math:`n`-比特密度矩阵的 :math:`(n + m)`-比特的比特化块编码。

   :param num_qubits: 量子态比特数 :math:`n`。
   :type num_qubits: int
   :param num_block_qubits: 块编码的辅助比特数 :math:`m > n`。
   :type num_block_qubits: int

   :return: 一个 :math:`2^{n + m} \times 2^{n + m}` 的左上角为密度矩阵的酉矩阵
   :rtype: paddle.Tensor
