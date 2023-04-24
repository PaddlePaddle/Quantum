paddle\_quantum.channel.common
=====================================

常用的量子信道的功能实现。

.. py:class:: BitFlip(prob, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   比特反转信道。

   其 Kraus 算符为：

   .. math::

      E_0 = \sqrt{1-p} I,
      E_1 = \sqrt{p} X.

   :param prob: 发生比特反转的概率，其值应该在 :math:`[0, 1]` 区间内。
   :type prob: Union[paddle.Tensor, float]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: PhaseFlip(prob, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   相位反转信道。

   其 Kraus 算符为：

   .. math::

      E_0 = \sqrt{1 - p} I,
      E_1 = \sqrt{p} Z.

   :param prob: 发生相位反转的概率，其值应该在 :math:`[0, 1]` 区间内。
   :type prob: Union[paddle.Tensor, float]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: BitPhaseFlip(prob, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   比特相位反转信道。

   其 Kraus 算符为：

   .. math::

      E_0 = \sqrt{1 - p} I,
      E_1 = \sqrt{p} Y.

   :param prob: 发生比特相位反转的概率，其值应该在 :math:`[0, 1]` 区间内。
   :type prob: Union[paddle.Tensor, float]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: AmplitudeDamping(gamma, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   振幅阻尼信道。

   其 Kraus 算符为：

   .. math::

      E_0 =
      \begin{bmatrix}
         1 & 0 \\
         0 & \sqrt{1-\gamma}
      \end{bmatrix},
      E_1 =
      \begin{bmatrix}
         0 & \sqrt{\gamma} \\
         0 & 0
      \end{bmatrix}.

   :param gamma: 减振概率，其值应该在 :math:`[0, 1]` 区间内。
   :type gamma: Union[paddle.Tensor, float]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: GeneralizedAmplitudeDamping(gamma, prob, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   广义振幅阻尼信道。

   其 Kraus 算符为：

   .. math::

      E_0 = \sqrt{p}
      \begin{bmatrix}
         1 & 0 \\
         0 & \sqrt{1-\gamma}
      \end{bmatrix},
      E_1 = \sqrt{p} \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix},\\
      E_2 = \sqrt{1-p} \begin{bmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{bmatrix},
      E_3 = \sqrt{1-p} \begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}.

   :param gamma: 减振概率，其值应该在 :math:`[0, 1]` 区间内。
   :type gamma: Union[paddle.Tensor, float]
   :param prob: 激发概率，其值应该在 :math:`[0, 1]` 区间内。
   :type prob: Union[paddle.Tensor, float]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: PhaseDamping(gamma, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   相位阻尼信道。

   其 Kraus 算符为：

   .. math::

      E_0 =
      \begin{bmatrix}
         1 & 0 \\
         0 & \sqrt{1-\gamma}
      \end{bmatrix},
      E_1 =
      \begin{bmatrix}
         0 & 0 \\
         0 & \sqrt{\gamma}
      \end{bmatrix}.

   :param gamma: 该信道的参数，其值应该在 :math:`[0, 1]` 区间内。
   :type gamma: Union[paddle.Tensor, float]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: Depolarizing(prob, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   去极化信道。

   其 Kraus 算符为：

   .. math::

      E_0 = \sqrt{1-3p/4} I,
      E_1 = \sqrt{p/4} X,
      E_2 = \sqrt{p/4} Y,
      E_3 = \sqrt{p/4} Z.

   :param prob: 该信道的参数，其值应该在 :math:`[0, 1]` 区间内。
   :type prob: Union[paddle.Tensor, float]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

   .. note::
      该功能的实现逻辑已更新。
      当前版本请参考 M.A.Nielsen and I.L.Chuang 所著 Quantum Computation and Quantum Information 第10版中的 (8.102) 式。
      参考文献: Nielsen, M., & Chuang, I. (2010). Quantum Computation and Quantum Information: 10th Anniversary Edition. Cambridge: Cambridge University Press. doi:10.1017/CBO9780511976667

.. py:class:: GeneralizedDepolarizing(prob, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   广义去极化信道。

   其 Kraus 算符为：

   .. math::

      E_0 = \sqrt{1-(D - 1)p/D} I, \text{ where } D = 4^n, \\
      E_k = \sqrt{p/D} \sigma_k, \text{ for } 0 < k < D.

   :param prob: 该信道的参数 :math:`p`，其值应该在 :math:`[0, 1]` 区间内。
   :type prob: Union[paddle.Tensor, float]
   :param qubits_idx: 长度为 :math:`n` 的作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: PauliChannel(prob, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   泡利信道。

   :param prob: 泡利算符 X、Y、Z 对应的概率，各值均应在 :math:`[0, 1]` 区间内。
   :type prob: Union[paddle.Tensor, Iterable[float]]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

   .. note::

      三个输入的概率加起来需要小于等于 1。

.. py:class:: ResetChannel(prob, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   重置信道。

   该信道以 p 的概率将量子态重置为 :math:`|0\rangle`，并以 q 的概率重置为 :math:`|1\rangle`。其 Kraus 算符为：

   .. math::

      E_0 =
      \begin{bmatrix}
         \sqrt{p} & 0 \\
         0 & 0
      \end{bmatrix},
      E_1 =
      \begin{bmatrix}
         0 & \sqrt{p} \\
         0 & 0
      \end{bmatrix},\\
      E_2 =
      \begin{bmatrix}
         0 & 0 \\
         \sqrt{q} & 0
      \end{bmatrix},
      E_3 =
      \begin{bmatrix}
         0 & 0 \\
         0 & \sqrt{q}
      \end{bmatrix},\\
      E_4 = \sqrt{1-p-q} I.

   :param prob: 重置为 :math:`|0\rangle` 和重置为 :math:`|1\rangle` 的概率，各值均应在 :math:`[0, 1]` 区间内。
   :type prob: Union[paddle.Tensor, Iterable[float]]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

   .. note::

      两个输入的概率加起来需要小于等于 1。

.. py:class:: ThermalRelaxation(const_t, exec_time, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   热弛豫信道。
   
   该信道模拟超导硬件上的 T1 和 T2 混合过程。

   :param const_t: :math:`T_1` 和 :math:`T_2` 过程的弛豫时间常数，单位是微秒。
   :type const_t: Union[paddle.Tensor, Iterable[float]]
   :param exec_time: 弛豫过程中量子门的执行时间，单位是纳秒。
   :type exec_time: Union[paddle.Tensor, float]
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

   .. note::

      时间常数必须满足 :math:`T_2 \le T_1`，见参考文献 https://arxiv.org/abs/2101.02109。

.. py:class:: MixedUnitaryChannel(num_unitary, qubits_idx='full', num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   混合酉矩阵信道。

   :param num_unitary: 用于构成信道的酉矩阵数量。
   :type num_unitary: int
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional

.. py:class:: ReplacementChannel(prob, qubits_idx=None, num_qubits=None)

   基类：:py:class:`paddle_quantum.channel.base.Channel`

   量子置换信道的合集。

   对于一个量子态 :math:`\sigma`，其对应的替换信道 :math:`R` 定义为

   .. math::

      R(\rho) = \text{tr}(\rho)\sigma

   :param sigma: 输入的量子态 :math:`\sigma`。
   :type sigma: State
   :param qubits_idx: 作用在的量子比特的编号，默认为 ``None``。
   :type qubits_idx: Union[Iterable[int], int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int