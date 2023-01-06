paddle\_quantum.qchem.qchem
==================================

量子化学中的功能函数。

.. py:function:: qubitOperator_to_Hamiltonian(spin_h,tol)

   将openfermion形式转化为量桨的哈密顿量形式。

   :param spin_h: openfermion形式的哈密顿量。
   :type spin_h: openfermion.ops.operators.qubit_operator.QubitOperator
   :param tol: 阈值
   :type tol: float, optional

   :return: 返回转换成量桨形式的哈密顿量
   :rtype: Hamiltonian

.. py:function:: geometry(structure, file)

   读取分子几何信息。

   :param structure: 分子几何信息的字符串形式， 以 H2 分子为例 ``[['H', [-1.68666, 1.79811, 0.0]], ['H', [-1.12017, 1.37343, 0.0]]]``。
   :type structure: string, optional
   :param file: xyz 文件的路径。
   :type file: str, optional
   
   :raises AssertionError: 两个输入参数不可以同时为 ``None``。

   :return: 分子的几何信息。
   :rtype: str

.. py:function:: get_molecular_data(geometry, charge, multiplicity, basis, method, if_save, if_print, name, file_path)

   计算分子的必要信息，包括单体积分（one-body integrations）和双体积分（two-body integrations，以及用选定的方法计算基态的能量。

   :param geometry: 分子的几何信息。
   :type geometry: str
   :param charge: 分子电荷, 默认值为 ``0``。
   :type charge: int, optional
   :param multiplicity: 分子的多重度, 默认值为 ``1``。
   :type multiplicity: int, optional
   :param basis: 常用的基组是 ``sto-3g、6-31g`` 等, 默认的基组是 ``sto-3g``，更多的基组选择可以参考网站 
                  https://psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement。
   :type basis: str, optional
   :param method: 用于计算基态能量的方法, 包括 ``scf`` 和 ``fci``，默认的方法为 ``scf``。
   :type method: str, optional
   :param if_save: 是否需要将分子信息存储成 .hdf5 文件，默认为 ``True``。
   :type if_save: bool, optional
   :param if_print: 是否需要打印出选定方法 (method) 计算出的分子基态能量，默认为 ``True``。
   :type if_print: bool, optional
   :param name: 命名储存的文件, 默认为 ``""``。
   :type name: str, optional
   :param file_path: 文件的储存路径, 默认为 ``"."``。
   :type file_path: str, optional

   :return: 包含分子所有信息的类。
   :rtype: MolecularData

.. py:function:: active_space(electrons, orbitals, multiplicity, active_electrons, active_orbitals)

   对于给定的活跃电子和活跃轨道计算相应的活跃空间（active space）。

   :param electrons: 电子数。
   :type electrons: int
   :param orbitals: 轨道数。
   :type orbitals: int
   :param multiplicity: 自旋多重度, 默认值为 ``1``。
   :type multiplicity: int, optional
   :param active_electrons: 活跃 (active) 电子数，默认情况为所有电子均为活跃电子。
   :type active_electrons: int, optional
   :param active_orbitals: 活跃 (active) 轨道数，默认情况为所有轨道均为活跃轨道。
   :type active_orbitals: int, optional

   :return: 核心轨道和活跃轨道的索引。
   :rtype: tuple

.. py:function:: fermionic_hamiltonian(molecule, filename, multiplicity, active_electrons, active_orbitals)

   计算给定分子的费米哈密顿量。

   :param molecule: 包含分子所有信息的类。
   :type molecule: MolecularData
   :param filename: 分子的 .hdf5 文件的路径。
   :type filename: str, optional
   :param multiplicity: 自旋多重度, 默认值为 ``1``。
   :type multiplicity: int, optional
   :param active_electrons: 活跃 (active) 电子数，默认情况为所有电子均为活跃电子。
   :type active_electrons: int, optional
   :param active_orbitals: 活跃 (active) 轨道数，默认情况为所有轨道均为活跃轨道。
   :type active_orbitals: int, optional

   :return: openfermion 格式的哈密顿量。
   :rtype: openfermion.ops.operators.qubit_operator.QubitOperator

.. py:function:: spin_hamiltonian(molecule, filename, multiplicity, mapping_method, active_electrons, active_orbitals)

   生成 Paddle Quantum 格式的哈密顿量。

   :param molecule: openfermion 格式的哈密顿量。
   :type molecule: openfermion.ops.operators.qubit_operator.QubitOperator
   :param filename: 分子的 .hdf5 文件的路径。
   :type filename: str, optional
   :param multiplicity: 自旋多重度, 默认值为 ``1``。
   :type multiplicity: int, optional
   :param mapping_method: 映射方法，这里默认为 ``jordan_wigner``，此外还提供 ``bravyi_kitaev`` 方法。
   :type mapping_method: str, optional
   :param active_electrons: 活跃 (active) 电子数，默认情况为所有电子均为活跃电子。
   :type active_electrons: int, optional
   :param active_orbitals:  活跃 (active) 轨道数默认情况为所有轨道均为活跃轨道。
   :type active_orbitals: int, optional

   :return: Paddle Quantum 格式的哈密顿量。
   :rtype: Hamiltonian