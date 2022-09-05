paddle\_quantum.gate.multi\_qubit\_gate
==============================================

多量子比特门的类的功能实现。

.. py:class:: CNOT(qubits_idx='cycle', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   CNOT 门。

   对于两量子比特的量子电路，当 ``qubits_idx`` 为 ``[0, 1]`` 时，其矩阵形式为：

   .. math::

      \begin{align}
         CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
         &=
         \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & 0 & 1 \\
               0 & 0 & 1 & 0
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: CX(qubits_idx='cycle', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   同 CNOT。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: CY(qubits_idx='cycle', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   受控 Y 门。

   对于两量子比特的量子电路，当 ``qubits_idx`` 为 ``[0, 1]`` 时，其矩阵形式为：

   .. math::

      \begin{align}
         CY &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
         &=
         \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & 0 & -1j \\
               0 & 0 & 1j & 0
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: CZ(qubits_idx='cycle', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   受控 Z 门。

   对于两量子比特的量子电路，当 ``qubits_idx`` 为 ``[0, 1]`` 时，其矩阵形式为：

   .. math::

      \begin{align}
         CZ &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Z\\
         &=
         \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & 1 & 0 \\
               0 & 0 & 0 & -1
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: SWAP(qubits_idx='cycle', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   SWAP 门。

   其矩阵形式为：

   .. math::

      \begin{align}
         SWAP =
         \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 0 & 1 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & 0 & 1
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: CP(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   受控 P 门。

   对于两量子比特的量子电路，当 ``qubits_idx`` 为 ``[0, 1]`` 时，其矩阵形式为：

   .. math::

      \begin{bmatrix}
         1 & 0 & 0 & 0\\
         0 & 1 & 0 & 0\\
         0 & 0 & 1 & 0\\
         0 & 0 & 0 & e^{i\theta}
      \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: CRX(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   关于 x 轴的受控单量子比特旋转门。

   对于两量子比特的量子电路，当 ``qubits_idx`` 为 ``[0, 1]`` 时，其矩阵形式为：

   .. math::

      \begin{align}
         CRx &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Rx\\
         &=
         \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
               0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: CRY(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   关于 y 轴的受控单量子比特旋转门。

   对于两量子比特的量子电路，当 ``qubits_idx`` 为 ``[0, 1]`` 时，其矩阵形式为：

   .. math::

      \begin{align}
         CRy &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Ry\\
         &=
         \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
               0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: CRZ(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   关于 z 轴的受控单量子比特旋转门。

   对于两量子比特的量子电路，当 ``qubits_idx`` 为 ``[0, 1]`` 时，其矩阵形式为：

   .. math::

      \begin{align}
         CRz &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Rz\\
         &=
         \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & 1 & 0 \\
               0 & 0 & 0 & e^{i\theta}
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: CU(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   受控单量子比特旋转门。

   对于两量子比特的量子电路，当 ``qubits_idx`` 为 ``[0, 1]`` 时，其矩阵形式为：

   .. math::

      \begin{align}
         CU
         &=
         \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & \cos\frac\theta2 &-e^{i\lambda}\sin\frac\theta2 \\
               0 & 0 & e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: RXX(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   RXX 门。

   其矩阵形式为：

   .. math::

      \begin{align}
         RXX(\theta) =
               \begin{bmatrix}
                  \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                  0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                  0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                  -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
               \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: RYY(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   RYY 门。

   其矩阵形式为：

   .. math::

      \begin{align}
         RYY(\theta) =
               \begin{bmatrix}
                  \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                  0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                  0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                  i\sin\frac{\theta}{2} & 0 & 0 & cos\frac{\theta}{2}
               \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: RZZ(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   RZZ 门。

   其矩阵形式为：

   .. math::

      \begin{align}
         RZZ(\theta) =
               \begin{bmatrix}
                  e^{-i\frac{\theta}{2}} & 0 & 0 & 0 \\
                  0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
                  0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
                  0 & 0 & 0 & e^{-i\frac{\theta}{2}}
               \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: MS(qubits_idx='cycle', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   用于离子阱设备的 Mølmer-Sørensen (MS) 门。

   其矩阵形式为：

   .. math::

      \begin{align}
         MS = RXX(-\frac{\pi}{2}) = \frac{1}{\sqrt{2}}
               \begin{bmatrix}
                  1 & 0 & 0 & i \\
                  0 & 1 & i & 0 \\
                  0 & i & 1 & 0 \\
                  i & 0 & 0 & 1
               \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: CSWAP(qubits_idx='cycle', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   CSWAP (Fredkin) 门。

   其矩阵形式为：

   .. math::

      \begin{align}
         CSWAP =
         \begin{bmatrix}
               1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
               0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
               0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
               0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
               0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
               0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
               0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: Toffoli(qubits_idx='cycle', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   Toffoli 门。

   其矩阵形式为：

   .. math::

      \begin{align}
         \begin{bmatrix}
               1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
               0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
               0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
               0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
               0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
               0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
               0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
         \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: UniversalTwoQubits(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   两量子比特通用门，该通用门需要 15 个参数。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: UniversalThreeQubits(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.ParamGate`

   三量子比特通用门，该通用门需要 81 个参数。

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
   :type qubits_idx: Union[Iterable, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。
