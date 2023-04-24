paddle\_quantum.channel.representation
==========================================

量桨量子信道的表达式库。

.. py:function:: bit_flip_kraus(prob, dtype=None)

   比特反转信道的Kraus表达式，其形式为

   .. math::

      E_0 = \sqrt{1-p} I,
      E_1 = \sqrt{p} X.

   :param prob: 概率 :math:`p`。
   :type prob: Union[float, np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: phase_flip_kraus(prob, dtype=None)

   相位反转信道的Kraus表达式，其形式为

   .. math::

      E_0 = \sqrt{1-p} I,
      E_1 = \sqrt{p} Z.

   :param prob: 概率 :math:`p`。
   :type prob: Union[float, np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: bit_phase_flip_kraus(prob, dtype=None)

   比特相位反转信道的Kraus表达式，其形式为

   .. math::

      E_0 = \sqrt{1-p} I,
      E_1 = \sqrt{p} Y.

   :param prob: 概率 :math:`p`。
   :type prob: Union[float, np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: amplitude_damping_kraus(gamma, dtype=None)

   振幅阻尼信道的Kraus表达式，其形式为

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

   :param gamma: 系数 :math:`\gamma`。
   :type gamma: Union[float, np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: generalized_amplitude_damping_kraus(gamma, prob, dtype=None)

   广义振幅阻尼信道的Kraus表达式，其形式为

    .. math::

       E_0 = \sqrt{p} \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{bmatrix},
       E_1 = \sqrt{p} \begin{bmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{bmatrix},\\
       E_2 = \sqrt{1-p} \begin{bmatrix} \sqrt{1-\gamma} & 0 \\ 0 & 1 \end{bmatrix},
       E_3 = \sqrt{1-p} \begin{bmatrix} 0 & 0 \\ \sqrt{\gamma} & 0 \end{bmatrix}.

   :param gamma: 系数 :math:`\gamma`。
   :type gamma: Union[float, np.ndarray, paddle.Tensor]
   :param prob: 概率 :math:`p`。
   :type prob: Union[float, np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: phase_damping_kraus(gamma, dtype=None)

   相位阻尼信道的Kraus表达式，其形式为

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

   :param gamma: 系数 :math:`\gamma`。
   :type gamma: Union[float, np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: depolarizing_kraus(prob, dtype=None)

   去极化信道的Kraus表达式，其形式为

    .. math::

        E_0 = \sqrt{1-3p/4} I,
        E_1 = \sqrt{p/4} X,
        E_2 = \sqrt{p/4} Y,
        E_3 = \sqrt{p/4} Z.

   :param prob: 概率 :math:`p`。
   :type prob: Union[float, np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: generalized_depolarizing_kraus(prob, num_qubits, dtype=None)

   广义去极化信道的Kraus表达式，其形式为

    .. math::

        E_0 = \sqrt{1-(D - 1)p/D} I, \text{ where } D = 4^n, \\
        E_k = \sqrt{p/D} \sigma_k, \text{ for } 0 < k < D.

   :param prob: 概率 :math:`p`。
   :type prob: float
   :param num_qubits: 信道的比特数 :math:`n`。
   :type num_qubits: int
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: pauli_kraus(prob, dtype=None)

   泡利信道的Kraus表达式。

   :param prob: 泡利算符 X、Y、Z 对应的概率。
   :type prob: Union[List[float], np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: reset_kraus(prob, dtype=None)

   重置信道的Kraus表达式，其形式为

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

   :param prob: 重置为 :math:`|0\rangle` 和重置为 :math:`|1\rangle` 的概率。
   :type prob: Union[List[float], np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: thermal_relaxation_kraus(const_t, exec_time, dtype=None)

   热弛豫信道的Kraus表达式。

   :param const_t: :math:`T_1` 和 :math:`T_2` 过程的弛豫时间常数，单位是微秒。
   :type const_t: Union[List[float], np.ndarray, paddle.Tensor]
   :param exec_time: 弛豫过程中量子门的执行时间，单位是纳秒。
   :type exec_time: Union[List[float], np.ndarray, paddle.Tensor]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str, optional

   :return: 返回对应的 Kraus 算符
   :rtype: List[paddle.Tensor]

.. py:function:: replacement_choi(sigma, dtype=None)

   置换信道的Choi表达式。

   :param sigma: 这个信道的输出态。
   :type sigma: Union[np.ndarray, paddle.Tensor, State]
   :param dtype: 数据类型。默认为 ``None``。
   :type dtype: str

   :return: 返回对应的 Choi 算符
   :rtype: paddle.Tensor