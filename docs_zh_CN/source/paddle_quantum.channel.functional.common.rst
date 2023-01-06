paddle\_quantum.channel.functional.common
================================================

量子信道的底层逻辑。

.. py:function:: kraus_repr(state, kraus_oper, qubit_idx, dtype, backend)

   在输入态上作用一个 Kraus 表示的自定义量子信道。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param kraus_oper: 该信道的 Kraus 算符。
   :type kraus_oper: Iterable[paddle.Tensor]
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。

   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: choi_repr(state, choi_oper, qubit_idx:, dtype, backend)

   在输入态上作用一个 Choi 表示的自定义量子信道。Choi 表示的数学形式为

   .. math::

      \sum_{i, j} |i\rangle\langle j| \otimes N(|i\rangle\langle j|)

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param choi_oper: 该信道 :math:`N` 的 Choi 算符。
   :type choi_oper: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。

   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: stinespring_repr(state, stinespring_mat, qubit_idx:, dtype, backend)

   在输入态上作用一个 Stinespring 表示的自定义量子信道。 ``stinespring_mat`` 是一个 :math:`(d_1 * d_2) \times d_1` 的长方矩阵。
   其中 :math:`d_1` 为 ``qubit_idx`` 所在系统维度；:math:`d_2` 为辅助系统维度。通过 Dirac 符号我们可以将 ``stinespring_mat`` 表示为

   .. math::
   
      \text{stinespring_mat.reshape}([d_1, d_2, d_1])[i, j, k] = \langle i, j| A |k \rangle

   这里 :math:`A` 为 Stinespring 表示，且该信道可定义为 :math:`\rho \mapsto \text{tr}_2 (A \rho A^dagger)`。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param stinespring_mat: 该信道的 Stinespring 表示所组成的矩阵。
   :type stinespring_mat: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :raises RuntimeError: 噪声信道只能在密度矩阵模式下运行。

   :return: 输出态。
   :rtype: paddle_quantum.State
