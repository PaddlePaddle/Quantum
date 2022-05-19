paddle\_quantum.qchem.loss
=================================

量子化学中的损失函数。

.. py:class:: MolEnergyLoss(geometry, basis, multiplicity = 1, charge = 0)

   基类：:py:class:`paddle_quantum.loss.ExpecVal`

   分子基态计算的损失函数。

   :param geometry: 表示分子位置几何信息，例如 ``H 0.0 0.0 0.0; H 0.0 0.0 0.74``。
   :type geometry: str
   :param basis: 化学基选取，例如 ``sto-3g``。
   :type basis: str
   :param multiplicity: 自旋多重度, 默认值为 ``1``。
   :type multiplicity: int, optional
   :param charge: 分子电荷量， 默认值为 ``0``。
   :type charge: int, optional

.. py:class:: RHFEnergyLoss(geometry, basis, multiplicity = 1, charge = 0)

   基类: :py:class:`paddle_quantum.Operator`

   Restricted Hartree Fock 计算的损失函数。

   :param geometry:  表示分子位置几何信息，例如 ``H 0.0 0.0 0.0; H 0.0 0.0 0.74``。
   :type geometry: str
   :param basis: 化学基选取，例如 ``sto-3g``。
   :type basis: str
   :param multiplicity: 自旋多重度, 默认值为 ``1``。
   :type multiplicity: int, optional
   :param charge: 分子电荷量， 默认值为 ``0``。
   :type charge: int, optional

   :raises ModuleNotFoundError: Hartree Fock 方法需要安装 ``pyscf`` 包。安装请运行 ``pip install -U pyscf``。