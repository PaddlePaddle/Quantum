paddle\_quantum.qchem.molecule
=========================================

量子化学中的分子类型。

.. py:class:: Molecule(geometry, basis, multiplicity, charge, mol_expr, use_angstrom, driver)

   量子化学分子类型。

   :param geometry: 分子中原子符号与原子位置坐标。
   :type geometry: Optional[List[Tuple[str,list]]]
   :param basis: 量子化学基组。
   :type basis: Optional[str]
   :param multiplicity: 分子的自旋多重度。
   :type multiplicity: Optional[int]
   :param charge: 分子中的总电荷数。
   :type charge: Optional[int]
   :param mol_expr: 分子表达式。
   :type mol_expr: Optional[str]
   :param use_angstrom: 是否用埃作为分子中的长度单位。
   :type use_angstrom: bool
   :param driver: 经典量子化学计算工具（计算分子积分）。
   :type driver: paddle_quantum.qchem.Driver

   .. py:method:: build()

      利用经典量子化学工具完成相关计算。

   .. py:property:: atom_charges()

      分子中每个原子的核电荷数，例如，氢分子为 [1, 1]
    
   .. py:property:: atom_coords()

      分子中每个原子的位置坐标，返回一个 ``numpy ndarray`` 。

   .. py:property:: unit()

      分子中原子间距离的长度单位。

   .. py:method:: get_mo_integral(integral_type)

      计算分子积分。

      :param integral_type: 分子积分的类型，如动能积分 "int1e_kin"。
      :type integral_type: str
    
      :return: 分子积分。
      :rtype: numpy.ndarray

   .. py:method:: get_molecular_hamiltonian()

      分子的哈密顿量。

      :return: 分子哈密顿量。
      :rtype: paddle_quantum.Hamiltonian