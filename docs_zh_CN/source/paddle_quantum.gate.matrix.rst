paddle\_quantum.gate.matrix
===============================================

门的矩阵实现。

以下是单比特量子门的矩阵。

.. py:function:: h_gate(dtype=None)
    
    生成矩阵

    .. math::

        \begin{align}
            H = \frac{1}{\sqrt{2}}
            \begin{bmatrix}
                    1&1\\
                    1&-1
            \end{bmatrix}
        \end{align}

        

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: H 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: s_gate(dtype=None)

    生成矩阵

    .. math::

        \begin{align}
            S =
            \begin{bmatrix}
                    1&0\\
                    0&i
            \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: S 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: sdg_gate(dtype=None)

    生成矩阵

    .. math::

        \begin{align}
            S^\dagger =
            \begin{bmatrix}
                    1&0\\
                    0&-i
            \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: :math:`S^\dagger` 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: t_gate(dtype=None)

    生成矩阵

    .. math::

        \begin{align}
            T =
            \begin{bmatrix}
                    1&0\\
                    0&e^\frac{i\pi}{4}
            \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: T 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: tdg_gate(dtype=None)

    生成矩阵

    .. math::

        \begin{align}
            T^\dagger =
            \begin{bmatrix}
                    1&0\\
                    0&e^{-\frac{i\pi}{4}}
            \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: :math:`T^\dagger` 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: x_gate(dtype=None)

    生成矩阵

    .. math::

        \begin{align}
            X =
            \begin{bmatrix}
                    0&1\\
                    1&0
            \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: X 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: y_gate(dtype=None)

    生成矩阵

    .. math::

        \begin{align}
            Y =
            \begin{bmatrix}
                    0&-i\\
                    i&0
            \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: Y 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: z_gate(dtype=None)

    生成矩阵

    .. math::

        \begin{align}
            Z =
            \begin{bmatrix}
                    1&0\\
                    0&-1
            \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: Z 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: p_gate(theta)

    生成矩阵

    .. math::

        \begin{align}
            P =
            \begin{bmatrix}
                    1 & 0 \\
                    0 & e^{i\theta}
            \end{bmatrix}
        \end{align}

    :param theta: P 门的参数。
    :type theta: paddle.Tensor
    :return: P 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: rx_gate(theta)

    生成矩阵

    .. math::

        \begin{align}
            R_X =
            \begin{bmatrix}
                    \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                    -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    :param theta: :math:`R_X` 门的参数。
    :type theta: paddle.Tensor
    :return: :math:`R_X` 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: ry_gate(theta)

    生成矩阵

    .. math::

        \begin{align}
            R_Y =
            \begin{bmatrix}
                    \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                    \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    :param theta: :math:`R_Y` 门的参数。
    :type theta: paddle.Tensor
    :return: :math:`R_Y` 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: rz_gate(theta)

    生成矩阵

    .. math::

        \begin{align}
            R_Z =
            \begin{bmatrix}
                    e^{-i\frac{\theta}{2}} & 0 \\
                    0 & e^{i\frac{\theta}{2}}
            \end{bmatrix}
        \end{align}

    :param theta: :math:`R_Z` 门的参数。
    :type theta: paddle.Tensor
    :return: :math:`R_Z` 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: u3_gate(theta)

    生成矩阵

    .. math::

        \begin{align}
            U_3(\theta, \phi, \lambda) =
               \begin{bmatrix}
                    \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                    e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
               \end{bmatrix}
        \end{align}

    :param theta: :math:`U_3` 门的参数。
    :type theta: paddle.Tensor
    :return: :math:`U_3` 门的矩阵。
    :rtype: paddle.Tensor

以下是多量子比特门的矩阵

.. py:function:: cnot_gate(dtype)
    
    生成矩阵

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

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: CNOT 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: cy_gate(dtype)
    
    生成矩阵

    .. math::

        \begin{align}
            CY &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
            &=
            \begin{bmatrix}
               1 & 0 & 0 & 0 \\
               0 & 1 & 0 & 0 \\
               0 & 0 & 0 & -i \\
               0 & 0 & i & 0
            \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: CY 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: cz_gate(dtype)
    
    生成矩阵

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

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: CZ 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: swap_gate(dtype)
    
    生成矩阵

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

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: SWAP 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: cp_gate(theta)
    
    生成矩阵

    .. math::

        \begin{align}
            CP =
            \begin{bmatrix}
                1 & 0 & 0 & 0\\
                0 & 1 & 0 & 0\\
                0 & 0 & 1 & 0\\
                0 & 0 & 0 & e^{i\theta}
            \end{bmatrix}
        \end{align}

    :param theta: CP 门的参数。
    :type theta: paddle.Tensor
    :return: CP 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: crx_gate(theta)
    
    生成矩阵

    .. math::

        \begin{align}
            CR_X &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_X\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    :param theta: :math:`CR_X` 门的参数。
    :type theta: paddle.Tensor
    :return: :math:`CR_X` 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: cry_gate(theta)
    
    生成矩阵

    .. math::

        \begin{align}
            CR_Y &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Y\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    :param theta: :math:`CR_Y` 门的参数。
    :type theta: paddle.Tensor
    :return: :math:`CR_Y` 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: crz_gate(theta)
    
    生成矩阵

    .. math::

        \begin{align}
            CR_Z &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Z\\
            &=
            \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}
        \end{align}

    :param theta: :math:`CR_Z` 门的参数。
    :type theta: paddle.Tensor
    :return: :math:`CR_Z` 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: cu_gate(theta)
    
    生成矩阵

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

    :param theta: CU 门的参数。
    :type theta: paddle.Tensor
    :return: CU 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: rxx_gate(theta)
    
    生成矩阵

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

    :param theta: RXX 门的参数。
    :type theta: paddle.Tensor
    :return: RXX 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: ryy_gate(theta)
    
    生成矩阵

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

    :param theta: RYY 门的参数。
    :type theta: paddle.Tensor
    :return: RYY 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: rzz_gate(theta)
    
    生成矩阵

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

    :param theta: RZZ 门的参数。
    :type theta: paddle.Tensor
    :return: RZZ 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: ms_gate(dtype)
    
    生成矩阵

    .. math::

        \begin{align}
            \mathit{MS} = \mathit{R_{XX}}(-\frac{\pi}{2}) = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                    1 & 0 & 0 & i \\
                    0 & 1 & i & 0 \\
                    0 & i & 1 & 0 \\
                    i & 0 & 0 & 1
                \end{bmatrix}
        \end{align}

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: MS 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: cswap_gate(dtype)
    
    生成矩阵

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

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: CSWAP 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: toffoli_gate(dtype)
    
    生成矩阵

    .. math::

        \begin{align}
            Toffoli =
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

    :param dtype: 此矩阵的类型，默认值为 ``'None'``。
    :type dtype: str, optional
    :return: Toffoli 门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: universal2_gate(theta)

    :param theta: 两量子比特通用门参数。
    :type theta: paddle.Tensor
    :return: 两量子比特通用门的矩阵。
    :rtype: paddle.Tensor

.. py:function:: universal3_gate(theta)

    :param theta: 三量子比特通用门参数。
    :type theta: paddle.Tensor
    :return: 三量子比特通用门的矩阵。
    :rtype: paddle.Tensor
