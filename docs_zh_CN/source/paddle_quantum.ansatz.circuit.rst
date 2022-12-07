paddle\_quantum.ansatz.circuit
=====================================

量子电路类的功能实现。

.. py:class:: Circuit(num_qubits=None)
      
      基类: :py:class:`paddle_quantum.ansatz.container.Sequential`
      
      量子电路。

      :param num_qubits: 量子比特数量, 默认为 ``None``。
      :type num_qubits: int, optional

      .. py:property:: num_qubits()

         该电路的量子比特数量。

      .. py:property:: isdynamic()

         是否电路为动态电路。
      
      .. py:property:: param()

         展平后的电路参数。
      
      .. py:property:: grad()

         展平后的电路参数梯度。

      .. py:method:: update_param(theta, idx=None)

         替换单层或所有的电路参数。

         :param theta: 新的参数。
         :type theta: Union[paddle.Tensor, np.ndarray, float]
         :param idx: 量子层的索引, 默认为替换所有。
         :type idx: int, optional

         :raises ValueError: 索引必须是整数或者 None。

      .. py:method:: transfer_static()

         将该线路的所有参数的 ``stop_grdient`` 设为 ``True``

      .. py:method:: randomize_param(low=0, high=2 * pi)

         在 ``[low, high)`` 的范围内随机化电路参数

         :param low: 随机参数的下界, 默认为 ``0``。
         :type low: float, optional
         :param high: 随机参数的上界, 默认为 ``2*pi``。
         :type high: float, optional

      .. py:method:: h(qubits_idx='full', num_qubits=None, depth=1)

         添加一个单量子比特的 Hadamard 门。

         其矩阵形式为：

         .. math::
            
            H = \frac{1}{\sqrt{2}}
                \begin{bmatrix}
                     1&1\\
                     1&-1
                \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: s(qubits_idx='full', num_qubits=None, depth=1)

         添加单量子比特 S 门。

         其矩阵形式为：

         .. math::

            S =
               \begin{bmatrix}
                  1&0\\
                  0&i
               \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: sdg(qubits_idx='full', num_qubits=None, depth=1)

         添加单量子比特 S dagger (逆S)门。

         其矩阵形式为：

         .. math::

           S ^\dagger =
              \begin{bmatrix}
                  1 & 0\ \
                  0 & -i
              \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: t(qubits_idx='full', num_qubits=None, depth=1)

         添加单量子比特 T 门。

         其矩阵形式为：

         .. math::

            T =
               \begin{bmatrix}
                  1&0\\
                  0&e^\frac{i\pi}{4}
               \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: tdg(qubits_idx='full', num_qubits=None, depth=1)

         添加单量子比特 T dagger (逆T)门。

         其矩阵形式为：

         .. math::

           T ^\dagger =
              \begin{bmatrix}
                  1 & 0\ \
                  0 & e^\frac{i\pi}{4}
              \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: x(qubits_idx='full', num_qubits=None, depth=1)

         添加单量子比特 X 门。

         其矩阵形式为：

         .. math::

            X = \begin{bmatrix}
                     0 & 1 \\
                     1 & 0
                \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable, int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: y(qubits_idx='full', num_qubits=None, depth=1)

         添加单量子比特 Y 门。

         其矩阵形式为：      

         .. math::

            Y = \begin{bmatrix}
                0 & -i \\
                i & 0
            \end{bmatrix}        

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable, int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: z(qubits_idx='full', num_qubits=None, depth=1)

         添加单量子比特 Z 门。

         其矩阵形式为：   

         .. math::

            Z = \begin{bmatrix}
                1 & 0 \\
                0 & -1
            \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable, int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: p(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加单量子比特 P 门。

         其矩阵形式为：

         .. math::

            P(\theta) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{bmatrix}


         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable, int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: rx(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加关于 x 轴的单量子比特旋转门。

         其矩阵形式为：
         
         .. math::

            R_X(\theta) = \begin{bmatrix}
                \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable, int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: ry(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加关于 y 轴的单量子比特旋转门。
         
         其矩阵形式为：

         .. math::

            R_Y(\theta) = \begin{bmatrix}
                \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable, int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: rz(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加关于 z 轴的单量子比特旋转门。

         其矩阵形式为：

         .. math::

            R_Z(\theta) = \begin{bmatrix}
                e^{-i\frac{\theta}{2}} & 0 \\
                0 & e^{i\frac{\theta}{2}}
            \end{bmatrix}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable, int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: u3(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加单量子比特旋转门。

         其矩阵形式为：

         .. math::

            \begin{align}
                U_3(\theta, \phi, \lambda) =
                    \begin{bmatrix}
                        \cos\frac\theta2&-e^{i\lambda}\sin\frac\theta2\\
                        e^{i\phi}\sin\frac\theta2&e^{i(\phi+\lambda)}\cos\frac\theta2
                    \end{bmatrix}
            \end{align}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable, int, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: cnot(qubits_idx='cycle', num_qubits=None, depth=1)

         添加 CNOT 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{CNOT} &= |0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: cx(qubits_idx='cycle', num_qubits=None, depth=1)

         与 cnot 相同。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: cy(qubits_idx='cycle', num_qubits=None, depth=1)

         添加受控 Y 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{CY} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Y\\
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: cz(qubits_idx='cycle', num_qubits=None, depth=1)

         添加受控 Z 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{CZ} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Z\\
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: swap(qubits_idx='cycle', num_qubits=None, depth=1)

         添加 SWAP 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{SWAP} =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}
            \end{align}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: cp(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加受控 P 门。

         其矩阵形式为：
         
         .. math::

            \begin{align}
                \mathit{CP}(\theta) =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 0 & 0 & e^{i\theta}
                \end{bmatrix}
            \end{align}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: crx(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加关于 x 轴的受控单量子比特旋转门。

         其矩阵形式为：

          .. math::

            \begin{align}
                \mathit{CR_X} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_X\\
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: cry(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加关于 y 轴的受控单量子比特旋转门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{CR_Y} &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Y\\
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: crz(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加关于 z 轴的受控单量子比特旋转门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{CR_Z} &= |0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes R_Z\\
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional


      .. py:method:: cu(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加受控单量子比特旋转门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{CU}
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: rxx(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加 RXX 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{R_{XX}}(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & -i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        -i\sin\frac{\theta}{2} & 0 & 0 & \cos\frac{\theta}{2}
                    \end{bmatrix}
            \end{align}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: ryy(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加 RYY 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{R_{YY}}(\theta) =
                    \begin{bmatrix}
                        \cos\frac{\theta}{2} & 0 & 0 & i\sin\frac{\theta}{2} \\
                        0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} & 0 \\
                        0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2} & 0 \\
                        i\sin\frac{\theta}{2} & 0 & 0 & cos\frac{\theta}{2}
                    \end{bmatrix}
            \end{align}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional

      .. py:method:: rzz(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加 RZZ 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{R_{ZZ}}(\theta) =
                    \begin{bmatrix}
                        e^{-i\frac{\theta}{2}} & 0 & 0 & 0 \\
                        0 & e^{i\frac{\theta}{2}} & 0 & 0 \\
                        0 & 0 & e^{i\frac{\theta}{2}} & 0 \\
                        0 & 0 & 0 & e^{-i\frac{\theta}{2}}
                    \end{bmatrix}
            \end{align}

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: ms(qubits_idx='cycle', num_qubits=None, depth=1)

         添加 Mølmer-Sørensen (MS) 门。

         其矩阵形式为：

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

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: cswap(qubits_idx='cycle', num_qubits=None, depth=1)

         添加 CSWAP (Fredkin) 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                \mathit{CSWAP} =
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: ccx(qubits_idx='cycle', num_qubits=None, depth=1)

         添加 CCX 门。

         其矩阵形式为：

         .. math::

            \begin{align}
                    \mathit{CCX} = \begin{bmatrix}
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
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: universal_two_qubits(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加两量子比特通用门，该通用门需要 15 个参数。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: universal_three_qubits(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

         添加三量子比特通用门，该通用门需要 81 个参数。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
         :type qubits_idx: Union[Iterable, str], optional
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param param: 量子门参数，默认为 ``None``。
         :type param: Union[paddle.Tensor, float], optional
         :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
         :type param_sharing: bool, optional
      
      .. py:method:: oracle(oracle, qubits_idx, num_qubits=None, depth=1, gate_name='0', latex_name=None, plot_width=None)

         添加一个 oracle 门。

         :param oracle: 要实现的 oracle。
         :type oracle: paddle.tensor
         :param qubits_idx: 作用在的量子比特的编号。
         :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param gate_name: oracle 的名字，默认为 ``O``。
         :type gate_name: str, optional
         :param latex_name: oracle 的Latex名字，默认为 None, 此时用 gate_name。
         :type latex_name: str, optional
         :param plot_width: 电路图中此门的宽度，默认为None，此时与门名称成比例。
         :type gate_name: float, optional
      
      .. py:method:: control_oracle(oracle, qubits_idx, num_qubits=None, depth=1, gate_name='0', latex_name=None, plot_width=None)

         添加一个受控 oracle 门。

         :param oracle: 要实现的 oracle。
         :type oracle: paddle.tensor
         :param qubits_idx: 作用在的量子比特的编号。
         :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
         :param gate_name: oracle 的名字，默认为 ``cO``。
         :type gate_name: str, optional
         :param latex_name: oracle 的Latex名字，默认为 None, 此时用 gate_name。
         :type latex_name: str, optional
         :param plot_width: 电路图中此门的宽度，默认为None，此时与门名称成比例。
         :type gate_name: float, optional

      .. py:method:: collapse(qubits_idx='full', num_qubits=None, desired_result=None, if_print=False, measure_basis='z')

         添加一个坍缩算子

         :param qubits_idx: 作用的量子比特的编号。
         :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
         :param num_qubits: 总共的量子比特数量，默认为 ``None``。
         :type num_qubits: int, optional
         :param desired_result: 期望的坍缩态（现只支持输入计算基），默认为 ``None`` （随机坍缩）。
         :type desired_result: Union[int, str]
         :param if_print: 是否要打印坍缩的信息，默认为 ``True``。
         :type if_print: bool, optional
         :param measure_basis: 要观测的测量基底，默认为 ``z``。
         :type measure_basis: Union[Iterable[paddle.Tensor], str]

         :raises NotImplementdError: 要观测的测量基底只能为 ``z``，其他测量基底会在之后推出。
         :raises TypeError: 当 ``backend`` 为 ``unitary_matrix`` 时，无法获取输入态的概率。
      
      .. py:method:: superposition_layer(qubits_idx='full', num_qubits=None, depth=1)

         添加一个 Hadamard 门组成的层。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional

      .. py:method:: weak_superposition_layer(qubits_idx='full', num_qubits=None, depth=1)

         转角度为 :math:`\pi/4` 的 Ry 门组成的层。
      
         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: linear_entangled_layer(qubits_idx='full', num_qubits=None, depth=1)
         
         包含 Ry 门、Rz 门，和 CNOT 门的线性纠缠层。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: real_entangled_layer(qubits_idx='full', num_qubits=None, depth=1)

         包含 Ry 门和 CNOT 门的强纠缠层。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: complex_entangled_layer(qubits_idx='full', num_qubits=None, depth=1)

         包含 U3 门和 CNOT 门的强纠缠层。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: real_block_layer(qubits_idx='full', num_qubits=None, depth=1)

         包含 Ry 门和 CNOT 门的弱纠缠层。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: complex_block_layer(qubits_idx='full', num_qubits=None, depth=1)

         包含 U3 门和 CNOT 门的弱纠缠层。

         :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
         :param depth: 层数，默认为 ``1``。
         :type depth: int, optional
      
      .. py:method:: bit_flip(prob, qubits_idx='full', num_qubits=None)

         添加比特反转信道。

         :param prob: 发生比特反转的概率。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
      
      .. py:method:: phase_flip(prob, qubits_idx='full', num_qubits=None)

         添加相位反转信道。

         :param prob: 发生相位反转的概率。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional

      .. py:method:: bit_phase_flip(prob, qubits_idx='full', num_qubits=None)

         添加比特相位反转信道。

         :param prob: 发生比特相位反转的概率。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
      
      .. py:method:: amplitude_damping(gamma, qubits_idx='full', num_qubits=None)

         添加振幅阻尼信道。

         :param gamma: 减振概率。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
      
      .. py:method:: generalized_amplitude_damping(gamma, prob, qubits_idx='full', num_qubits=None)

         添加广义振幅阻尼信道。

         :param gamma: 减振概率，其值应该在 :math:`[0, 1]` 区间内。
         :type prob: Union[paddle.Tensor, float]
         :param prob: 激发概率，其值应该在 :math:`[0, 1]` 区间内。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
      
      .. py:method:: phase_damping(gamma, qubits_idx='full', num_qubits=None)

         添加相位阻尼信道。

         :param gamma: 该信道的参数。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional

      .. py:method:: depolarizing(prob, qubits_idx='full', num_qubits=None)

         添加去极化信道。

         :param prob: 该信道的参数。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
      
      .. py:method:: pauli_channel(prob, qubits_idx='full', num_qubits=None)

         添加泡利信道。

         :param prob: 该信道的参数。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
      
      .. py:method:: reset_channel(prob, qubits_idx='full', num_qubits=None)

         添加重置信道。

         :param prob: 重置为 :math:`|0\rangle` 和重置为 :math:`|1\rangle` 的概率。
         :type prob: Union[paddle.Tensor, float]
         :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
         :type qubits_idx: Union[Iterable[int], int, str], optional
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional
      
      .. py:method:: thermal_relaxation(const_t, exec_time, qubits_idx='full', num_qubits=None)

         添加热弛豫信道。

        :param const_t: :math:`T_1` 和 :math:`T_2` 过程的弛豫时间常数，单位是微秒。
        :type const_t: Union[paddle.Tensor, Iterable[float]]
        :param exec_time: 弛豫过程中量子门的执行时间，单位是纳秒。
        :type exec_time: Union[paddle.Tensor, float]
        :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
        :type qubits_idx: Union[Iterable[int], int, str], optional
        :param num_qubits: 总的量子比特个数，默认为 ``None``。
        :type num_qubits: int, optional

      .. py:method:: mixed_unitary_channel(num_unitary, qubits_idx='full', num_qubits=None)

         添加混合酉矩阵信道

        :param num_unitary: 用于构成信道的酉矩阵的数量。
        :type num_unitary: Union[paddle.Tensor, Iterable[int]]
        :param qubits_idx: 作用在的量子比特的编号, 默认为 ``'full'``。
        :type qubits_idx: Union[Iterable[int], int, str], optional
        :param num_qubits: 总的量子比特个数，默认为 ``None``。
        :type num_qubits: int, optional      

      .. py:method:: kraus_repr(kraus_oper, qubits_idx, num_qubits=None)

         添加一个 Kraus 表示的自定义量子信道。

         :param kraus_oper: 该信道的 Kraus 算符。
         :type kraus_oper: Iterable[paddle.Tensor]
         :param qubits_idx: 作用在的量子比特的编号。
         :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional

      .. py:method:: unitary_matrix(num_qubits=None)

         电路的酉矩阵形式

         :param num_qubits: 总的量子比特个数，默认为 ``None``。
         :type num_qubits: int, optional

         :return: 返回电路的酉矩阵形式
         :rtype: paddle.Tensor

      .. py:property:: gate_history()
      
         量子门的插入信息

         :return: 量子门的插入历史
         :rtype: List[Dict[str, Union[str, List[int], paddle.Tensor]]]

      .. py:property:: qubit_history()
      
         每个比特上的量子门的插入信息

         :return: 每个比特上的量子门的插入历史
         :rtype: List[List[Tuple[Dict[str, Union[str, List[int], paddle.Tensor]], int]]]

      .. py:method:: plot(save_path, dpi=100, show=True, output=False, scale=1.0, tex=False)

         画出量子电路图

         :param save_path: 图像保存的路径，默认为 ``None``。
         :type save_path: str, optional
         :param dpi: 每英寸像素数，这里指分辨率, 默认为 `100`。
         :type dpi: int, optional
         :param show: 是否执行 ``plt.show()``, 默认为 ``True``。
         :type show: bool, optional
         :param output: 是否返回 ``matplotlib.figure.Figure`` 实例，默认为 ``False``。
         :type output: bool, optional
         :param scale: ``figure`` 的 ``scale`` 系数，默认为 `1.0`。
         :type scale: float, optional
         :param tex: 一个布尔变量，用于控制是否使用 TeX 字体，默认为 ``False``。
         :type tex: bool, optional

         :return: 根据 ``output`` 参数返回 ``matplotlib.figure.Figure`` 实例或 ``None``。
         :rtype: Union[None, matplotlib.figure.Figure]

      .. note:: 
         
         使用 ``plt.show()`` 或许会导致一些问题，但是在保存图片时不会发生。如果电路太深可能会有一些图形无法显示。如果设置 ``tex = True`` 则需要在你的系统上安装 TeX 及其相关的依赖包。更多细节参考 https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html

      .. py:method:: extend(cir)

         量子电路扩展

         :param cir: 量子电路。
         :type cir: Circuit