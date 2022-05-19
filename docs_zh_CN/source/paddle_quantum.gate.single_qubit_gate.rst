paddle\_quantum.gate.single\_qubit\_gate
===============================================

单量子比特门的类的功能实现。

.. py:class:: H(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   单量子比特 Hadamard 门。

   其矩阵形式为：

   .. math::

      H = \frac{1}{\sqrt{2}}
         \begin{bmatrix}
               1&1\\
               1&-1
         \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: S(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   单量子比特 S 门。

   其矩阵形式为：

   .. math::

      S =
         \begin{bmatrix}
               1&0\\
               0&i
         \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: T(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   单量子比特 T 门。

   其矩阵形式为：

   .. math::

      T =
         \begin{bmatrix}
               1&0\\
               0&e^\frac{i\pi}{4}
         \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: X(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   单量子比特 X 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}
         0 & 1 \\
         1 & 0
      \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: Y(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   单量子比特 Y 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}
         0 & -i \\
         i & 0
      \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: Z(qubits_idx='full', num_qubits=None, depth=1)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   单量子比特 Z 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}
         1 & 0 \\
         0 & -1
      \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional

.. py:class:: P(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   单量子比特 P 门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}
         1 & 0 \\
         0 & e^{i\theta}
      \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: RX(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   关于 x 轴的单量子比特旋转门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}
         \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
         -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
      \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: RY(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   关于 y 轴的单量子比特旋转门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}
         \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
         \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
      \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: RZ(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   关于 z 轴的单量子比特旋转门。

   其矩阵形式为：

   .. math::

      \begin{bmatrix}
         e^{-i\frac{\theta}{2}} & 0 \\
         0 & e^{i\frac{\theta}{2}}
      \end{bmatrix}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。

.. py:class:: U3(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

   基类：:py:class:`paddle_quantum.gate.base.Gate`

   单量子比特旋转门。

   其矩阵形式为：

   .. math::

      \begin{align}
         U3(\theta, \phi, \lambda) =
               \begin{bmatrix}
                  \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                  e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
               \end{bmatrix}
      \end{align}

   :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
   :type qubits_idx: Union[Iterable, int, str], optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param depth: 层数，默认为 ``1``。
   :type depth: int, optional
   :param param: 量子门参数，默认为 ``None``。
   :type param: Union[paddle.Tensor, float], optional
   :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
   :type param_sharing: bool, optional
   :raises ValueError: ``param`` 须为 ``paddle.Tensor`` 或 ``float``。
