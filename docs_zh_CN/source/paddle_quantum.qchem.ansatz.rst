paddle\_quantum.qchem.ansatz
=================================

量子化学常用变分量子线路模版

.. py:class:: HardwareEfficient(num_qubits, depth, use_cz, angles, rot_type)

   基类：:py:class:`paddle_quantum.ansatz.Circuit`

   Hardware Efficient量子线路模版。

   :param num_qubits: 量子比特数。
   :type num_qubits: int
   :param depth: 量子线路深度（以重复单元数量计数）。
   :type depth: int   
   :param use_cz: 是否使用CZ门作为两比特门。
   :type use_cz: bool
   :param angles: 线路中的可变分的角度。
   :type angles: Optional[np.ndarray]
   :param rot_type: 线路中旋转门类型。
   :type rot_type: Optional[str]

   .. py:property:: rot_type
      
      旋转门类型。

   .. py:property:: entangle_type

      纠缠门类型。

.. py:class:: UCC(num_qubits, ucc_order, single_ex_amps, double_ex_amps, **trotter_kwargs)

   基类：:py:class:`paddle_quantum.ansatz.Circuit`

   Unitary Coupled Cluster线路模版。

   :param num_qubits: 量子比特数量。
   :type num_qubits: int
   :param ucc_order: 耦合簇阶数。
   :type ucc_order: Optional[str]
   :param single_ex_amps: 单粒子激发矩阵。
   :type single_ex_amps: Optional[np.ndarray]
   :param double_ex_amps: 双粒子激发张量。
   :type double_ex_amps: Optional[np.ndarray]
   :param \*\*trotter_kwargs: trotter分解方法配置参数。
   :type \*\*trotter_kwargs: Dict

   .. py:property:: onebody_tensor

      单体算符张量。

   .. py:property:: twobody_tensor

      双体算符张量。

.. py:class:: HartreeFock(num_qubits, angles)

   基类：:py:class:`paddle_quantum.ansatz.Circuit`

   哈特利-福克量子线路。

   :param num_qubits: 量子比特数量。
   :type num_qubits: int
   :param angles: 吉文斯旋转角度。
   :type angles: Optional[np.ndarray]
