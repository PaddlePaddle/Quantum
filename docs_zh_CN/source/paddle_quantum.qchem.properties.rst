paddle\_quantum.qchem.properties
=========================================

计算分子性质。

.. py:function:: energy(psi, mol, shots, use_shadow, **shadow_kwargs)

   计算哈密顿量在给定量子态下的能量。

   :param psi: 量子态。
   :type psi: paddle_quantum.state.State
   :param mol: 分子类型。
   :type mol: paddle_quantum.qchem.Molecule
   :param shots: 测量次数。
   :type shots: Optional[int]
   :param use_shadow: 是否使用经典影子方法。
   :type use_shadow: Optional[bool]
   :param \*\*shadow_kwargs: 经典影子方法的配置。
   :type \*\*shadow_kwargs: Dict

   :return: 哈密顿量的能量。
   :rtype: float

.. py:function:: symmetric_rdm1e(psi, shots, use_shadow, **shadow_kwargs)

   对称化的单电子约化密度矩阵。

   :param psi: 量子态。
   :type psi: paddle_quantum.state.State
   :param shots: 测量次数。
   :type shots: int
   :param use_shadow: 是否使用经典影子方法。
   :type use_shadow: bool
   :param \*\*shadow_kwargs: 经典影子方法的配置。
   :type \*\*shadow_kwargs: Dict

   :return: 单电子密度矩阵。
   :rtype: np.ndarray

.. py:function:: dipole_moment(psi, mol, shots, use_shadow, **shadow_kwargs)

   利用给定的量子态计算分子偶极矩。

   :param psi: 量子态。
   :type psi: paddle_quantum.state.State
   :param mol: 分子类型。
   :type mol: paddle_quantum.qchem.Molecule
   :param shots: 测量次数。
   :type shots: int
   :param use_shadow: 是否使用经典影子方法。
   :type use_shadow: bool
   :param \*\*shadow_kwargs: 经典影子方法的配置。
   :type \*\*shadow_kwargs: Dict

   :return: 分子偶极矩。
   :rtype: np.ndarray
