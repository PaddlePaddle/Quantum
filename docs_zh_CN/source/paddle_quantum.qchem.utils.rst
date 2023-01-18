paddle\_quantum.qchem.utils
=========================================

utility 函数。

.. py:function:: orb2spinorb(num_modes, single_ex_amps, double_ex_amps)

   将分子轨道积分转变为自旋轨道积分。

   :param num_modes: 希尔伯特空间的维度。
   :type num_modes: int
   :param single_ex_amps: 单粒子激发矩阵元。
   :type single_ex_amps: np.ndarray
   :param double_ex_amps: 双粒子激发张量。
   :type double_ex_amps: np.ndarray

   :return: 自旋轨道基下的分子单体双体积分。
   :rtype: Tuple[np.ndarray]
