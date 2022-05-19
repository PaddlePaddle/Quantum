paddle\_quantum.channel.functional.common
================================================

常见量子信道的函数的功能实现。

.. py:function:: bit_flip(state, prob, qubit_idx, dtype, backend)

   在输入态上作用一个比特反转信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param prob: 发生比特反转的概率。
   :type prob: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: phase_flip(state, prob, qubit_idx, dtype, backend)

   在输入态上作用一个相位反转信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param prob: 发生相位反转的概率。
   :type prob: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: bit_phase_flip(state, prob, qubit_idx, dtype, backend)

   在输入态上作用一个比特相位反转信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param prob: 发生比特相位反转的概率。
   :type prob: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: amplitude_damping(state, gamma, qubit_idx, dtype, backend)

   在输入态上作用一个振幅阻尼信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param gamma: 减振概率。
   :type gamma: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: generalized_amplitude_damping(state, gamma, prob, qubit_idx, dtype, backend)

   在输入态上作用一个广义振幅阻尼信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param gamma: 减振概率。
   :type gamma: paddle.Tensor
   :param prob: 激发概率。
   :type prob: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: phase_damping(state, gamma, qubit_idx, dtype, backend)

   在输入态上作用一个相位阻尼信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param gamma: 该信道的参数。
   :type gamma: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: depolarizing(state, prob, qubit_idx, dtype, backend)

   在输入态上作用一个去极化信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param prob: 该信道的参数。
   :type prob: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: pauli_channel(state, prob, qubit_idx, dtype, backend)

   在输入态上作用一个泡利信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param prob: 泡利算符 X、Y、Z 对应的概率。
   :type prob: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: reset_channel(state, prob, qubit_idx, dtype, backend)

   在输入态上作用一个重置信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param prob: 重置为 :math:`|0\rangle` 和重置为 :math:`|1\rangle` 的概率。
   :type prob: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: thermal_relaxation(state, const_t, exec_time, qubit_idx, dtype, backend)

   在输入态上作用一个热弛豫信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param const_t: :math:`T_1` 和 :math:`T_2` 过程的弛豫时间常数，单位是微秒。
   :type const_t: paddle.Tensor
   :param exec_time: 弛豫过程中量子门的执行时间，单位是纳秒。
   :type exec_time: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: kraus_repr(state, kraus_oper, qubit_idx:, dtype, backend)

   在输入态上作用一个 Kraus 表示的自定义量子信道。

   :param state:  输入态。
   :type state: paddle_quantum.State
   :param kraus_oper: 该信道的 Kraus 算符。
   :type kraus_oper: Iterable[paddle.Tensor]
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: choi_repr()
