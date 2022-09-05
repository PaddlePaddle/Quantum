paddle\_quantum.shadow
=============================

量子态的经典影子的功能实现。

.. py:function:: shadow_sample(state, num_qubits, sample_shots, mode, hamiltonian=None, method='CS')

   对给定的量子态进行随机的泡利测量并返回测量结果。

   :param state: 输入量子态，支持态矢量和密度矩阵形式。
   :type state: paddle_quantum.State
   :param num_qubits: 量子比特数量。
   :type num_qubits: int
   :param sample_shots: 随机采样的次数。
   :type sample_shots: int
   :param mode: 输入量子态的表示方式, ``'state_vector'`` 表示态矢量形式， ``'density_matrix'`` 表示密度矩阵形式。
   :type mode: paddle_quantum.Backend
   :param hamiltonian: 可观测量的相关信息，输入形式为 ``Hamiltonian`` 类，默认为 ``None``。
   :type hamiltonian: paddle_quantum.Hamiltonian, optional
   :param method: 进行随机采样的方法，有 ``'CS'`` 、 ``'LBCS'`` 、 ``'APS'`` 三种方法，默认为 ``'CS'``。
   :type method: str, optional

   :raises ValueError: 输入的哈密顿量 (Hamiltonian) 形式不合法
   :raises NotImplementedError: 输入 ``state`` 的 ``backend`` 必须是 ``StateVector`` 或 ``DensityMatrix``

   :return: 随机选择的泡利测量基和测量结果，形状为 ``(sample_shots, 2)`` 的list。
   :rtype: list
