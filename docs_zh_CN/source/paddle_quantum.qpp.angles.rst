paddle\_quantum.qpp.angles
===============================

量子相位处理相关工具函数包

.. py:function:: qpp_angle_finder(P, Q)

    对一个劳伦对 ``P``, ``Q``,找到相应的角度集合

   :param P: 一个劳伦多项式
   :type P: Laurent
   :param Q: 一个劳伦多项式
   :type Q: Laurent

   :return: 
        包含如下元素：

        -list_theta: :math:`R_Y` 门对应的角度
        -list_phi: :math:`R_Z` 门对应的角度
   :rtype: Tuple[List[float], List[float]]

.. py:function:: qpp_angle_approximator(P, Q)

    对一个劳伦对 ``P``, ``Q``,估计相应的角度集合

   :param P: 一个劳伦多项式
   :type P: Laurent
   :param Q: 一个劳伦多项式
   :type Q: Laurent

   :return: 
        包含如下元素：

        -list_theta: :math:`R_Y` 门对应的角度
        -list_phi: :math:`R_Z` 门对应的角度
   :rtype: Tuple[List[float], List[float]]

   .. note::
        与 `yzzyz_angle_finder` 不同的是， `yzzyz_angle_approximator` 假定唯一的误差来源是精度误差（而这一般来讲是不正确的）。

.. py:function:: update_angle(coef)

    通过 `coef` 从 ``P`` 和 ``Q`` 中计算角度

   :param coef: ``P`` 和 ``Q`` 中的第一项和最后一项
   :type coef: List[complex]

   :return: 角度 `theta` 和 `phi`
   :rtype: Tuple[float, float]

   :raises ValueError: 参数错误：检查这四个参数{[p_d, p_nd, q_d, q_nd]}

.. py:function:: update_polynomial(P, Q, theta, phi, verify)

    通过 `theta` , `phi` 更新 ``P`` , ``Q``

   :param P: 一个劳伦多项式
   :type P: Laurent
   :param Q: 一个劳伦多项式
   :type Q: Laurent
   :param theta: 一个参数
   :type theta: float
   :param phi: 一个参数
   :type phi: float
   :param verify: 验证计算是否正确，默认值为True
   :type verify: Optional[bool] = True

   :return: 更新后的 ``P``, ``Q``
   :rtype: Tuple[List[float], List[float]]

.. py:function:: condition_test(P, Q)

    检查 ``P``, ``Q`` 是否满足：
        - deg(`P`) = deg(`Q`)
        - ``P``, ``Q`` 是否具有相同宇称
        - :math:`PP^* + QQ^* = 1`

   :param P: 一个劳伦多项式
   :type P: Laurent
   :param Q: 一个劳伦多项式
   :type Q: Laurent

   :raises ValueError: PP* + QQ* != 1: 检查你的代码

.. py:function:: yz_decomposition(U)

    返回U的yz分解

   :param U: 单比特幺正变换
   :type U: np.ndarray

   :return: `alpha`, `theta`, `phi` 使得 :math:`U[0, 0] = \alpha R_Y(\theta) R_Z(\phi) [0, 0]`
   :rtype: Tuple[complex, float, float]