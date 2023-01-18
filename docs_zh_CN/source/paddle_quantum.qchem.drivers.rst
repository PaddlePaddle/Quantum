paddle\_quantum.qchem.drivers
==================================

经典量子化学求解器。

.. py:class:: Driver

   求解器基类。

   .. py:method:: run_scf()

      SCF计算方法

   .. py:property:: num_modes()

      希尔伯特空间模式数量。

   .. py:property:: energy_nuc()

      原子核之间相互作用能量。

   .. py:property:: mo_coeff()

      原子轨道与分子轨道之间的转换矩阵。

   .. py:method:: load_molecule()

      加载分子。
    
   .. py:method:: get_onebody_tensor()

      哈密顿量中的单体算符张量。

   .. py:method:: get_twobody_tensor()

      哈密顿量中的双体算符张量。

.. py:class:: PySCFDriver

   基类：:py:class:`Driver`

   基于PySCF的经典求解器。

   .. py:method:: load_molecule(atom, basis, multiplicity, charge, unit)

      根据提供的信息构建量子化学分子类型。

      :param atom: 分子中原子标记及坐标。
      :type atom: List[Tuple[str,List[float]]]
      :param basis: 量子化学基组。
      :type basis: str
      :param multiplicity: 分子的自旋多重度。
      :type multiplicity: int
      :param charge: 分子中的总电荷量。
      :type charge: int
      :param unit: 构建分子使用的长度单位。
      :type unit: str

   .. py:method:: run_scf()

      SCF计算方法
      
   .. py:property:: energy_nuc()

      原子核之间相互作用能量。

   .. py:property:: mo_coeff()

      原子轨道与分子轨道之间的转换矩阵。
    
   .. py:method:: get_onebody_tensor(integral_type)

      :param integral_type: 单体积分类型。
      :type integral_type: str

      :return: 哈密顿量中的单体算符张量。
      :rtype: np.ndarray

   .. py:method:: get_twobody_tensor()

      :return: 哈密顿量中的双体算符张量。
      :rtype: np.ndarray