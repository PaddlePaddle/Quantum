paddle\_quantum.qchem.fermionic\_state
=========================================

费米子波函数。

.. py:class:: WaveFunction(data, convention, backend, dtype, override)

   基类：:py:class:`paddle_quantum.state.State`

   费米子波函数。

   :param data: 费米子波函数在计算基下的展开系数。
   :type data: Union[paddle.Tensor, np.ndarray]
   :param convention: 计算基中自旋轨道排列方式，自旋轨道分开排列（`separated`） 或混合排列（`mixed`）。
   :type convention: Optional[str]
   :param backend: 后端使用的计算方式。
   :type backend: Optional[paddle_quantum.backend.Backend]
   :param dtype: 存储费米子波函数所用的数据类型，32位或64位。
   :type dtype: str
   :param override: 是否在copy时覆盖原有数据。
   :type override: bool

   .. py:method:: clone()

      :return: 原费米子量子态的复制。
      :rtype: WaveFunction

   .. py:method:: swap(p, q)
      
      交换两个费米子。

      :param p: 被交换的费米子序号。
      :type p: int 
      :param q: 被交换的费米子序号。
      :type q: int

      :return: 交换之后的费米子波函数。
      :rtype: WaveFunction

   .. py:method:: to_spin_mixed()
      
      将自旋轨道的排列顺序变为“上下上下......”混合的形式。

      :return: 原费米子波函数在新排列下的表示。
      :rtype: WaveFunction

   .. py:method:: to_spin_separated()

      将自旋轨道的排列顺序变为“上上...下下下...”的分离形式。

      :return: 原费米子波函数在新排列下的表示。
      :rtype: WaveFunction

   .. py:method:: slater_determinant_state(num_qubits, num_elec, mz, backend, dtype)

      根据给定的量子比特数、电子数和磁量子数生成Slater行列式态。

      :param num_qubits: 量子比特数量。
      :type num_qubits: int
      :param num_elec: 电子数。
      :type num_elec: int
      :param mz: 磁量子数。
      :type mz: int
      :param backend: 后端使用的计算方式。
      :type backend: Optional[paddle_quantum.backend.Backend]
      :param dtype: 数据类型，32位或64位。
      :type dtype: str

      :return: 存储Slater行列式的费米子波函数。
      :rtype: WaveFunction

   .. py:method:: zero_state(num_qubits, backend, dtype)

      根据给定的量子比特数生成零态。

      :param num_qubits: 量子比特数。
      :type num_qubits: int
      :param backend: 后端使用的计算方式。
      :type backend: Optional[paddle_quantum.backend.Backend]
      :param dtype: 存储数据类型，32位或64位。
      :type dtype: Optional[str]

      :return: 存储零态的费米子波函数。
      :rtype: WaveFunction

   .. py:method:: num_elec(shots)

      多电子费米子波函数中包含的电子数。

      :param shots: 测量次数。
      :type shots: Optional[int]

      :return: 电子数。
      :rtype: int

   .. py:method:: total_SpinZ(shots)

      总自旋的z分量。

      :param shots: 测量次数。
      :type shots: Optional[int]

      :return: 总自旋的z分量。
      :rtype: float

   .. py:method:: total_Spin2(shots)

      总自旋的平方。

      :param shots: 测量次数。
      :type shots: Optional[int]

      :return: 总自旋的平方
      :rtype: float