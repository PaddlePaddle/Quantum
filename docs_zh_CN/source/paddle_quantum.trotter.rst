paddle\_quantum.trotter
==============================

Trotter 哈密顿量时间演化的功能实现。


.. py:function:: construct_trotter_circuit(circuit, hamiltonian, tau, steps, method='suzuki', order=1, grouping=None, coefficient=None, permutation=None)

   向 circuit 的后面添加 trotter 时间演化电路，即给定一个系统的哈密顿量 H，该电路可以模拟系统的时间演化 :math:`U_{cir} e^{-iHt}`。

   :param circuit: 需要添加时间演化电路的 Circuit 对象。
   :type circuit: Circuit
   :param hamiltonian:  需要模拟时间演化的系统的哈密顿量 H。
   :type hamiltonian: Hamiltonian
   :param tau: 每个 trotter 块的演化时间长度。
   :type tau: float
   :param steps: 添加多少个 trotter 块。（提示： ``steps * tau`` 即演化的时间总长度 t）
   :type steps: int
   :param method: 搭建时间演化电路的方法，默认为 ``'suzuki'`` ，即使用 Trotter-Suzuki 分解。可以设置为 ``'custom'`` 来使用自定义的演化策略。（需要用 permutation 和 coefficient 来定义）。
   :type method: str, optional
   :param order: Trotter-Suzuki decomposition 的阶数，默认为 ``1``，仅在使用 ``method='suzuki'`` 时有效。
   :type order: int, optional
   :param grouping: 是否对哈密顿量进行指定策略的重新排列，默认为 ``None`` ，支持 ``'xyz'`` 和 ``'even_odd'`` 两种方法。
   :type grouping: str, optional
   :param coefficient: 自定义时间演化电路的系数，对应哈密顿量中的各项，默认为 ``None`` ，仅在 ``method='custom'`` 时有效。
   :type coefficient: np.ndarrayorpaddle.Tensor, optional
   :param permutation: 自定义哈密顿量的排列方式，默认为 ``None``，仅在 ``method='custom'`` 时有效。
   :type permutation: np.ndarray, optional

   :raises ValueError: Trotter-Suzuki 分解的阶数 ``order`` 必须为 ``1``, ``2``, 或 ``2k``, 其中 ``k`` 是一个整数
   :raises ValueError: ``permutation`` 和 ``coefficient`` 的形状不一致
   :raises ValueError: 重排策略 ``grouping`` 的方法不支持, 仅支持 ``'xyz'``, ``'even_odd'``
   :raises ValueError: 搭建时间演化电路的方法 ``method`` 不支持, 仅支持 ``'suzuki'``, ``'custom'``

   .. Hint::

      想知道该函数是如何模拟的？更多信息请移步至量桨官网教程: https://qml.baidu.com/tutorials/overview.html.

.. py:function:: optimal_circuit(circuit, theta, which_qubits)

   添加一个优化电路，哈密顿量为’XXYYZZ’。

   :param circuit: 需要添加门的电路。
   :type circuit: paddle_quantum.ansatz.Circuit
   :param theta: 旋转角度需要传入三个参数。
   :type theta: Union[paddle.Tensor, float]
   :param which_qubits: ``pauli_word`` 中的每个算符所作用的量子比特编号。
   :type which_qubits: Iterable

.. py:function:: paddle_quantum.trotter.add_n_pauli_gate(circuit, theta, pauli_word, which_qubits)

   添加一个对应着 N 个泡利算符张量积的旋转门，例如 :math:`e^{-\theta/2 \cdot X\otimes I\otimes X\otimes Y}`。

   :param circuit: 需要添加门的电路。
   :type circuit: paddle_quantum.ansatz.Circuit
   :param theta: 旋转角度。
   :type theta: Union[paddle.Tensor, float]
   :param pauli_word: 泡利算符组成的字符串，例如 ``"XXZ"``。
   :type pauli_word: str
   :param which_qubits: ``pauli_word`` 中的每个算符所作用的量子比特编号。
   :type which_qubits: Iterable

   :raises ValueError: The ``which_qubits`` 需要为 ``list``, ``tuple``, 或者 ``np.ndarray`` 格式。

.. py:function:: get_suzuki_permutation(length, order)

   计算 Suzuki 分解对应的置换数组。

   :param length: 对应哈密顿量中的项数，即需要置换的项数。
   :type length: int
   :param order: Suzuki 分解的阶数。
   :type order: int

   :return: 置换数组。
   :rtype: np.ndarray

.. py:function:: get_suzuki_p_values(k)

   计算 Suzuki 分解中递推关系中的因数 p(k)。

   :param k: Suzuki 分解的阶数。
   :type k: int

   :return: 一个长度为 5 的列表，其形式为 [p, p, (1 - 4 * p), p, p]。
   :rtype: list

.. py:function:: get_suzuki_coefficients(length, order)

   计算 Suzuki 分解对应的系数数组。

   :param length: 对应哈密顿量中的项数，即需要置换的项数。
   :type length: int
   :param order: Suzuki 分解的阶数。
   :type order: int

   :return: 系数数组。
   :rtype: np.ndarray

.. py:function:: get_1d_heisenberg_hamiltonian(length, j_x=1.0, j_y=1.0, j_z=1.0, h_z=0.0, periodic_boundary_condition=True)

   生成一个一维海森堡链的哈密顿量。

   :param length: 链长。
   :type length: int
   :param j_x: x 方向的自旋耦合强度 Jx，默认为 ``1``。
   :type j_x: float, optional
   :param j_y: y 方向的自旋耦合强度 Jy，默认为 ``1``。
   :type j_y: float, optional
   :param j_z: z 方向的自旋耦合强度 Jz，默认为 ``1``。
   :type j_z: float, optional
   :param h_z: z 方向的磁场，默认为 ``0``，若输入为单个 float 则认为是均匀磁场。（施加在每一个格点上）
   :type h_z: floatornp.ndarray, optional
   :param periodic_boundary_condition: 是否考虑周期性边界条件，即 l + 1 = 0，默认为 ``True``。
   :type periodic_boundary_condition: bool, optional

   :return: 该海森堡链的哈密顿量。
   :rtype: Hamiltonian