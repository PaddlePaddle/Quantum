paddle\_quantum.qsvt.qsp
============================

量子信号处理相关类与函数，具体参考论文 https://arxiv.org/abs/1806.01838

.. py:function:: signal_unitary(signal_x)

   实现论文中的信号矩阵 :math:`W(x)`

   :param signal_x: 输入信号，区间为[-1, 1]
   :type signal_x: float

   :return: matrix :math:`W(x=\text{signal_x})`
   :rtype: ndarray

.. py:function:: poly_parity_verification(poly_p, k, error)

   对输入多项式进行奇偶校验，判断 :math:`P` 奇偶性是否为 (k mod 2)，详见论文定理 3 中的条件 2.

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param k: 项数 k
   :type k: int
   :param error: 误差阈值，默认为 `1e-6`.
   :type error: float

   :return: poly_p 奇偶性是否为 (k mod 2)
   :rtype: bool

.. py:function:: normalization_verification(poly_p, poly_q, trials, error)

   归一化验证，判断多项式 :math:`P(x)` 和 :math:`Q(x)` 是否满足归一化条件，详见论文定理 3 中的条件 3

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param poly_q: 多项式 :math:`Q(x)`
   :type poly_q: Polynomial
   :param trials: 验证次数，默认为 `10`
   :type trials: int
   :param error: 误差阈值，默认为 `1e-2`.
   :type error: float

   :return: 多项式是否满足归一化条件 :math:`|P|^2 + (1 - x^2)|Q|^2 = 1`
   :rtype: bool

.. py:function:: angle_phi_verification(phi, poly_p, poly_p_hat, poly_q_hat, trials, error)

   验证角度 :math:`\phi` 是否满足论文中的等式 6

   :param phi: 旋转角 :math:`\phi`
   :type phi: float
   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param poly_q: 多项式 :math:`Q(x)`
   :type poly_q: Polynomial
   :param poly_p_hat: 多项式 :math:`\tilde{P}(x)`
   :type poly_p: Polynomial
   :param poly_q_hat: 多项式 :math:`\tilde{Q}(x)`
   :type poly_q: Polynomial
   :param trials: 验证次数，默认为 `10`
   :type trials: int
   :param error: 误差阈值，默认为 `1e-2`.
   :type error: float

   :return: 角度 :math:`\phi` 是否满足论文中的等式 6.
   :rtype: bool

.. py:function:: processing_unitary(list_matrices, signal_x)

   构造量子信号处理矩阵 :math:`W_\Phi(x)`，详见论文中的等式 1

   :param list_matrices: 一个包含信号处理矩阵的数组
   :type list_matrices: List[ndarray]
   :param signal_x: 输入信号 x，范围为 [-1, 1]
   :type signal_x: float

   :return: 量子信号处理矩阵 :math:`W_\Phi(x)`
   :rtype: ndarray

.. py:function:: Phi_verification(list_phi, poly_p, trials, error)

   验证完整的角度 :math:`\Phi`

   :param list_phi: 包含所有角度 :math:`\phi` 的数组
   :type list_phi: ndarray
   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param trials: 验证次数，默认为 `100`
   :type trials: trials
   :param error: 误差阈值，默认为 `1e-6`
   :type error: float

   :return: 角度 :math:`\Phi` 是否使得 :math:`W_\Phi(x)` 为 :math:`P(x)` 的块编码
   :rtype: bool

.. py:function:: update_polynomial(poly_p, poly_q, phi)

   计算 :math:`P, Q` 经过一层量子信号处理后的多项式 :math:`\tilde{P}, \tilde{Q}`

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param poly_q: 多项式 :math:`Q(x)`
   :type poly_q: Polynomial
   :param phi: 量子信号处理的旋转角 :math:`\phi`
   :type phi: float

   :return: 更新之后的多项式 :math:`\tilde{P}(x), \tilde{Q}(x)`
   :rtype: Tuple[Polynomial, Polynomial]


.. py:function:: alg_find_Phi(poly_p, poly_q, length)

   计算角度 :math:`\Phi` 的算法

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param poly_q: 多项式 :math:`Q(x)`
   :type poly_q: Polynomial
   :param length: 返回角度的个数，即量子信号处理的层数
   :type length: int

   :return: 包含角度的数组 :math:`\Phi`
   :rtype: ndarray


.. py:function:: poly_A_hat_generation(poly_p)

   计算多项式 :math:`\hat{A}(y) = 1 - P(x)P^*(x)`，其中 :math:`y = x^2`

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial

   :return: 多项式 :math:`\hat{A}(y)`
   :rtype: Polynomial

.. py:function:: poly_A_hat_decomposition(A_hat, error)

   通过求根的方式分解多项式 :math:`\hat{A}(y)`
   
   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param error: 误差阈值，默认为 `0.001`
   :type error: float

   :return: 多项式 :math:`\hat{A}(y)` 的最高项系数以及根
   :rtype: Tuple[float, List[float]]

.. py:function:: poly_Q_generation(leading_coef, roots, parity)

   根据多项式 :math:`\hat{A}(y)` 的分解，构造多项式 :math:`Q(x)`
   
   :param leading_coef: 多项式 :math:`\hat{A}(y)` 的最高项系数
   :type leading_coef: float
   :param roots: 多项式 :math:`\hat{A}(y)` 的根
   :type roots: List[float]
   :param parity: 多项式 :math:`Q(x)` 的奇偶性
   :type parity: int

   :return: 多项式 :math:`Q(x)`
   :rtype: Polynomial

.. py:function:: alg_find_Q(poly_p, k)

   根据多项式 :math:`P(x)` 构造多项式 :math:`Q(x)` 的算法

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param k: 多项式 :math:`Q(x)` 的项数
   :type k: int

   :return: 多项式 :math:`Q(x)`
   :rtype: Polynomial

.. py:function:: quantum_signal_processing(poly_p, length)

   量子信号处理函数，找到一组角度 :math:`\Phi` 使得量子信号处理算子 :math:`W_\Phi(x)` 是一个多项式 :math:`P(x)` 的块编码

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param length: 角度的个数，即量子信号处理的层数，默认 `None` 为多项式 :math:`P(x)` 的度
   :type length: int

   :return: 角度 :math:`\Phi`
   :rtype: ndarray

.. py:function:: reflection_based_quantum_signal_processing(P)

   基于反射的量子信号处理函数，找到一组角度 :math:`\Phi` 使得量子信号处理算子 :math:`W_\Phi(x)` 是一个多项式 :math:`P(x)` 的块编码，详见论文引理 8

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial

   :return: 角度 :math:`\Phi`
   :rtype: ndarray

.. py:class:: ScalarQSP

   基类: :py:class:`object`

   基于量子信号处理的类

   :param poly_p: 多项式 :math:`P(x)`
   :type poly_p: Polynomial
   :param length: 角度的个数，即量子信号处理的层数，默认 `None` 为多项式 :math:`P(x)` 的度
   :type length: int

   .. py:method:: block_encoding(signal_x)

      构造一个量子信号处理的电路，即实现多项式 :math:`P(x)` 的块编码电路

      :param signal_x: 输入的信号 x
      :type signal_x: float

      :return: 量子信号处理的电路
      :rtype: Circuit

   .. py:method:: block_encoding_matrix(signal_x)

      构造一个量子信号处理的矩阵，即实现多项式 :math:`P(x)` 的块编码矩阵

      :param signal_x: 输入的信号 x
      :type signal_x: float

      :return: 量子信号处理的矩阵
      :rtype: paddle.Tensor
