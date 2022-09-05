paddle\_quantum.state.common
===================================

常见的量子态的实现。

.. py:function:: to_state(data, num_qubits=None, backend=None, dtype=None)

   根据给定的输入，生成对应的量子态。

   :param data: 量子态的数学解析形式。
   :type data: Union[paddle.Tensor, np.ndarray, QCompute.QEnv]
   :param num_qubits: 量子态所包含的量子比特数。默认为 ``None``，会自动从 data 中推导出来。
   :type num_qubits: int, optional
   :param backend: 指定量子态的后端实现形式。默认为 ``None``，使用全局的默认后端。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 量子态的数据类型。默认为 ``None``，使用全局的默认数据类型。
   :type dtype: str, optional
   :return: 生成的量子态。
   :rtype: paddle_quantum.State

.. py:function:: zero_state(num_qubits, backend=None, dtype=None)

   生成零态。

   :param num_qubits: 量子态所包含的量子比特数。
   :type num_qubits: int
   :param backend: 指定量子态的后端实现形式。默认为 None，使用全局的默认后端。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 量子态的数据类型。默认为 None，使用全局的默认数据类型。
   :type dtype: str, optional
   :raises NotImplementedError: 所指定的后端必须为量桨已实现的后端。
   :return: 所生成的零态。
   :rtype: paddle_quantum.State

.. py:function:: computational_basis(num_qubits, index, backend, dtype=None)

   生成计算基态 :math:`|e_{i}\rangle` ，其中 :math:`|e_{i}\rangle` 的第 :math:`i` 个元素为 1，其余元素为 0。

   :param num_qubits: 量子态所包含的量子比特数。
   :type num_qubits: int
   :param index: 计算基态 :math:`|e_{i}\rangle` 的下标 :math:`i` 。
   :type index: int
   :param backend: 指定量子态的后端实现形式。默认为 ``None``，使用全局的默认后端。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 量子态的数据类型。默认为 ``None``，使用全局的默认数据类型。
   :type dtype: str, optional
   :raises NotImplementedError: 所指定的后端必须为量桨已实现的后端。
   :return: 所生成的计算基态。
   :rtype: paddle_quantum.State
    
.. py:function:: bell_state(num_qubits, backend=None)

   生成贝尔态。

   其数学表达形式为：

   .. math::

      |\Phi_{D}\rangle=\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}

   :param num_qubits: 量子态所包含的量子比特数。
   :type num_qubits: int
   :param backend: 指定量子态的后端实现形式。默认为 ``None``，使用全局的默认后端。
   :type backend: paddle_quantum.Backend, optional
   :raises NotImplementedError: 所指定的后端必须为量桨已实现的后端。
   :return: 生成的贝尔态。
   :rtype: paddle_quantum.State

.. py:function:: bell_diagonal_state(prob)

   生成对角贝尔态。

   其数学表达形式为：

   .. math::

      p_{1}|\Phi^{+}\rangle\langle\Phi^{+}|+p_{2}| \Psi^{+}\rangle\langle\Psi^{+}|+p_{3}| \Phi^{-}\rangle\langle\Phi^{-}| +
      p_{4}|\Psi^{-}\rangle\langle\Psi^{-}|

   :param prob: 各个贝尔态的概率。
   :type: List[float]
   :raises Exception: 当后端为态矢量时，所输入量子态应该为纯态。
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。

   :returns: 生成的量子态。

.. py:function:: random_state(num_qubits, is_real=False, rank=None)

   生成一个随机的量子态。

   :param num_qubits: 量子态所包含的量子比特数。
   :type num_qubits: int
   :param is_real: 是否为实数。默认为 ``False``，表示为复数。
   :type is_real: bool, optional
   :param rank: 密度矩阵的秩。默认为 ``None``，表示使用满秩。
   :type rank: int, optional
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
   :return: 随机生成的一个量子态。
   :rtype: paddle_quantum.State

.. py:function:: w_state(num_qubits)

   生成一个 W-state。

   :param num_qubits: 量子态所包含的量子比特数。
   :type num_qubits: int
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
   :returns: 生成的 W-state。
   :rtype: paddle_quantum.State

.. py:function:: ghz_state(num_qubits)

   生成一个 GHZ-state。

   :param num_qubits: 量子态所包含的量子比特数。
   :type num_qubits: int
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
   :returns: 生成的 GHZ-state。
   :rtype: paddle_quantum.State

.. py:function:: completely_mixed_computational(num_qubits)

   生成一个完全混合态。


   :param num_qubits: 量子态所包含的量子比特数。
   :type num_qubits: int
   :raises Exception: 当后端为态矢量时，所输入量子态应该为纯态。
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
   :returns: 生成的 GHZ-state。
   :rtype: paddle_quantum.State

.. py:function:: r_state(prob)

   生成一个 R-state。

   其数学表达形式为：

   .. math::

      p|\Psi^{+}\rangle\langle\Psi^{+}| + (1 - p)|11\rangle\langle11|

   :param prob: 控制生成 R-state 的参数，它应该在 :math:`[0, 1]` 区间内。
   :type prob: float
   :raises Exception: 当后端为态矢量时，所输入量子态应该为纯态。
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
   :returns: 生成的 R-state。
   :rtype: paddle_quantum.State

.. py:function:: s_state(prob)

   生成一个 S-state。

   其数学表达形式为：

   .. math::

      p|\Phi^{+}\rangle\langle\Phi^{+}| + (1 - p)|00\rangle\langle00|

   :param prob: 控制生成 S-state 的参数，它应该在 :math:`[0, 1]` 区间内。
   :type prob: float
   :raises Exception: 当后端为态矢量时，所输入量子态应该为纯态。
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
   :returns: 生成的 S-state。
   :rtype: paddle_quantum.State

.. py:function:: isotropic_state(num_qubits, prob)

   生成 isotropic state。

   其数学表达形式为：

   .. math::

      p(\frac{1}{\sqrt{D}} \sum_{j=0}^{D-1}|j\rangle_{A}|j\rangle_{B}) + (1 - p)\frac{I}{2^n}

   :param num_qubits: 量子态所包含的量子比特数。
   :type num_qubits: int
   :param prob: 控制生成 isotropic state 的参数，它应该在 :math:`[0, 1]` 区间内。
   :type prob: float
   :raises Exception: 当后端为态矢量时，所输入量子态应该为纯态。
   :raises NotImplementedError: 所指定的后端必须为量桨已经实现的后端。
   :returns: 生成的 isotropic state。
   :rtype: paddle_quantum.State
