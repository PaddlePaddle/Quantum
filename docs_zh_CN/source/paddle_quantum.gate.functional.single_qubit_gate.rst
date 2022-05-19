paddle\_quantum.gate.functional.single\_qubit\_gate
==========================================================

单量子比特量子门的函数的功能实现。

.. py:function:: h(state, qubit_idx, dtype, backend)

   在输入态上作用一个 Hadamard 门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: s(state, qubit_idx, dtype, backend)

   在输入态上作用一个 S 门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: t(state, qubit_idx, dtype, backend)

   在输入态上作用一个 T 门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: x(state, qubit_idx, dtype, backend)

   在输入态上作用一个 X 门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: y(state, qubit_idx, dtype, backend)

   在输入态上作用一个 Y 门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: z(state, qubit_idx, dtype, backend)

   在输入态上作用一个 Z 门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: p(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个 P 门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param theta: 量子门参数。
   :type theta: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: rx(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个关于 x 轴的单量子比特旋转门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param theta: 量子门参数。
   :type theta: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: ry(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个关于 y 轴的单量子比特旋转门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param theta: 量子门参数。
   :type theta: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: rz(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个关于 z 轴的单量子比特旋转门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param theta: 量子门参数。
   :type theta: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State

.. py:function:: u3(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个单量子比特旋转门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param theta: 量子门参数。
   :type theta: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: int
   :param dtype: 数据的类型。
   :type dtype: str
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State
