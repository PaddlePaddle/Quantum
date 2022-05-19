paddle\_quantum.qchem.hardware\_efficient
================================================

Hardware Efficient 电路模板。

.. py:class:: HardwareEfficientModel(n_qubits, depth, theta = None)

   基类： :py:class:`paddle_quantum.gate.base.Gate`

   :param n_qubits: 量子态所包含的量子比特数。
   :type n_qubits: int
   :param depth: 量子电话深度，单层 Hardware Efficient 量子电路包含 [Ry, Rz, CNOT] 门。
   :type depth: int
   :param theta: 线路中 Ry, Rz 量子门的参数，默认值为 ``None``。
   :type theta: paddle.Tensor, optional
