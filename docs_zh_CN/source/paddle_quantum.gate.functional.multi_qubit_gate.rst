paddle\_quantum.gate.functional.multi\_qubit\_gate
=========================================================

多量子比特量子门的函数的功能实现。

.. py:function:: cnot(state, qubit_idx, dtype, backend)

   在输入态上作用一个 CNOT 门。

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

.. py:function:: cx(state, qubit_idx, dtype, backend)

   在输入态上作用一个 CNOT 门。

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

.. py:function:: cy(state, qubit_idx, dtype, backend)

   在输入态上作用一个受控 Y 门。

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

.. py:function:: cz(state, qubit_idx, dtype, backend)

   在输入态上作用一个受控 Z 门。

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

.. py:function:: swap(state, qubit_idx, dtype, backend)

   在输入态上作用一个 SWAP 门。

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

.. py:function:: cp(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个受控 P 门。

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

.. py:function:: crx(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个关于 x 轴的受控单量子比特旋转门。

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

.. py:function:: cry(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个关于 y 轴的受控单量子比特旋转门。

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

.. py:function:: crz(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个关于 z 轴的受控单量子比特旋转门。

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

.. py:function:: cu(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个受控单量子比特旋转门。

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

.. py:function:: rxx(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个 RXX 门。

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

.. py:function:: ryy(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个 RYY 门。

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

.. py:function:: rzz(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个 RZZ 门。

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

.. py:function:: ms(state, qubit_idx, dtype, backend)

   在输入态上作用一个 Mølmer-Sørensen (MS) 门。

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

.. py:function:: cswap(state, qubit_idx, dtype, backend)

   在输入态上作用一个 CSWAP (Fredkin) 门。

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

.. py:function:: toffoli(state, qubit_idx, dtype, backend)

   在输入态上作用一个 Toffoli 门。

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

.. py:function:: universal_two_qubits(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个两量子比特通用门。

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

.. py:function:: universal_three_qubits(state, theta, qubit_idx, dtype, backend)

   在输入态上作用一个三量子比特通用门。

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

.. py:function:: oracle(state, oracle, qubit_idx, backend)

   在输入态上作用一个 oracle 门。

   :param state: 输入态。
   :type state: paddle_quantum.State
   :param oracle: 要执行的 oracle。
   :type oracle: paddle.Tensor
   :param qubit_idx: 作用在的量子比特的编号。
   :type qubit_idx: list
   :param backend: 运行模拟的后端。
   :type backend: paddle_quantum.Backend
   :return: 输出态。
   :rtype: paddle_quantum.State
