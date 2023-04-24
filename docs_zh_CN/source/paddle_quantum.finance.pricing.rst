paddle\_quantum.finance.pricing
=================================

量子蒙特卡洛及期权定价相关工具。

.. py:function:: qae_cir(oracle, num_ancilla)

   根据给定酉算子搭建一条量子振幅估计电路。

   :param oracle: 给定酉算子。
   :type oracle: paddle.Tensor
   :param num_ancilla: 辅助比特使用数。
   :type num_ancilla: int

   :return: 一条用于量子振幅估计的量子电路
   :rtype: Circuit

.. py:function:: qae_alg(oracle, num_ancilla)

   量子振幅估计算法。

   :param oracle: 一个 :math:`n`-比特酉算子 :math:`\mathcal{A}`。
   :type oracle: paddle.Tensor
   :param num_ancilla: 辅助比特使用数。
   :type num_ancilla: int

   :return:
      包含如下元素的 tuple:

      - 用于量子振幅估计的量子电路。
      - 振幅估计结果，即 :math:`|\sin(2\pi\theta)|`。
   :rtype: Tuple[Circuit, paddle.Tensor]

   .. note::

      :math:`\mathcal{A}` 满足 :math:`\mathcal{A}|0^{\otimes n}\rangle=\cos(2\pi\theta)|0\rangle|\psi\rangle+\sin(2\pi\theta)|1\rangle|\phi\rangle.`

.. py:function:: qmc_alg(fcn, list_prob, num_ancilla=6)

   量子蒙特卡洛算法。

   :param fcn: 应用于随机变量 :math:`X` 的实函数 :math:`f`。
   :type fcn: Callable[[float], float]
   :param list_prob: 随机变量 :math:`X` 的概率分布，其中第 j 个元素对应第 j 个事件的发生几率。
   :type list_prob: List[float]
   :param num_ancilla: 辅助比特使用数。默认为 ``6``。
   :type num_ancilla: int

   :return:
      包含如下元素的 tuple:
      - 用于量子蒙特卡洛的量子电路。
      - 期望值估计结果，即 :math:`\mathbb{E}[f(X)]`。
   :rtype: Tuple[Circuit, paddle.Tensor]

.. py:class:: EuroOptionEstimator(initial_price, strike_price, interest_rate, volatility, maturity_date, degree_of_estimation=5)

   基类: :py:class:`object`

   欧式期权定价估算器

   :param initial_price: 初始价格。
   :type initial_price: float
   :param strike_price: 成交价。
   :type strike_price: float
   :param interest_rate: 无风险利率。
   :type interest_rate: float
   :param volatility: 市场波动性。
   :type volatility: float
   :param maturity_date: 期权到期日（以年为单位）。
   :type maturity_date: float
   :param degree_of_estimation: 估计精度指数。
   :type degree_of_estimation: int

   .. note::

      假设欧式期权定价处于 `Black-Scholes-Merton 模型 <https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model>`_ 中。

   .. py:method:: estimate()

      使用量子蒙特卡洛算法估算欧式期权定价。

   :return: 给定资产的期权定价。
   :rtype: float

   .. py:method:: plot()

      画出在该方案中使用的量子电路。

.. py:function:: cra_oracle(prob, param, lgd, threshold)

    构建用于信贷风险估计的 Oracle。

    :param prob: 隐变量的概率分布。
    :type prob: numpy.ndarray
    :param param: 资产对应的伯努利变量参数。
    :type param: numpy.ndarray
    :param lgd: 资产违约损失。
    :type lgd: List[int]
    :param threshold: VaR 的猜测值。
    :type threshold: float
   
    :return: 即将用于 QAE 算法的 Grover 算子。
    :rtype: paddle.Tensor

.. py:class:: CreditRiskAnalyzer(num_assets, base_default_prob, sensitivity, lgd, confidence_level, degree_of_simulation=4, even_sample=True)

   基类: :py:class:`object`

   使用量子算法加速的信贷风险估计模拟器。

   :param num_assets: 资产数量。
   :type num_assets: int
   :param base_default_prob: 基础违约概率。
   :type base_default_prob: numpy.ndarray
   :param sensitivity: 敏感度。
   :type sensitivity: numpy.ndarray
   :param lgd: 违约损失。
   :type lgd: numpy.ndarray
   :param confidence_level: 置信度。
   :type confidence_level: float
   :param degree_of_simulation: 模拟精度系数，默认为 ``4``。
   :type degree_of_simulation: int
   :param even_sample: 是否按照概率分布均匀采样，默认为 ``True``。
   :type even_sample: bool

   .. note::

        信贷风险估计的数学模型根据 `Tarca 和 Silvio 提供的模型 <https://arxiv.org/abs/1412.1183>`_ 构建，即金融系统风险默认遵从标准正态分布。

   .. py:method:: estimate_var()

      使用量子振幅估计算法分析配置好的风险资产。

    :return: 资产组合的在险价值。
    :rtype: float
