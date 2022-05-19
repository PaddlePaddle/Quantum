paddle\_quantum.qchem.uccsd
==================================

UCCSD 电路模板。

.. py:class:: UCCSDModel(n_qubits, n_electrons, n_trotter_steps, single_excitation_amplitude = None, double_excitation_amplitude = None)

   基类：:py:class:`paddle_quantum.gate.Gate`

   量子化学计算中的酉耦合簇拟设 (UCCSD)。 
   
   .. note::
     UCCSD 模型一般需要建立非常深的量子电路。因此，对于 H2 元素序列之后的分子结构，训练 UCCSD 拟设需要更好的设备及大量的运算时间。

   .. math::

     \begin{align}
          U(\theta)&=e^{\hat{T}-\hat{T}^{\dagger}}\\
          \hat{T}&=\hat{T}_1+\hat{T}_2\\
          \hat{T}_1&=\sum_{a\in{\text{virt}}}\sum_{i\in\text{occ}}t_{ai}\sum_{\sigma}\hat{c}^{\dagger}_{a\sigma}\hat{c}_{i\sigma}-h.c.\\
          \hat{T}_2&=\frac{1}{2}\sum_{a,b\in\text{virt}}\sum_{i,j\in\text{occ}}t_{aibj}\sum_{\sigma\tau}\hat{c}^{\dagger}_{a\sigma}\hat{c}^{\dagger}_{b\tau}\hat{c}_{j\tau}\hat{c}_{i\sigma}-h.c. 
     \end{align}

   :param n_qubits: 量子态所包含的量子比特数。
   :type n_qubits: int
   :param n_electrons: 分子中所包含的电子数。
   :type n_electrons: int
   :param n_trotter_steps: 建立UCCSD电路所需的特罗特分解步数。
   :type n_trotter_steps: int
   :param single_excitation_amplitude: :math:`\hat{T}_1` 定义中的 :math:`t_{ai}`, 默认值为 ``None``。
   :type single_excitation_amplitude: Union[np.array, None], optional
   :param double_excitation_amplitude: :math:`\hat{T}_2` 定义中的 :math:`t_{aibj}`, 默认值为 ``None``。
   :type double_excitation_amplitude: Union[np.array, None], optional



