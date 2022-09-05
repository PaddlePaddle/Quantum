paddle\_quantum.qchem.slater\_determinant
================================================

受限 Hartree Fock 下的 Slater 行列式电路模板。

.. py:class:: GivensRotationBlock(pindex, qindex, theta)

   基类：:py:class:`paddle_quantum.gate.Gate`

   双量子门，用于实现给定旋转。

   .. math:: 
      
      \begin{align}
            U(\theta)=e^{-i\frac{\theta}{2}(Y\otimes X-X\otimes Y)}
      \end{align}

   :param pindex: 第一个qubit的编号。
   :type pindex: int
   :param qindex: 第二个qubit的编号。
   :type qindex: int
   :param theta: 给定旋转角度。
   :type theta: float

.. py:class:: RHFSlaterDeterminantModel(n_qubits, n_electrons, mo_coeff=None)

   基类：:py:class:`paddle_quantum.gate.Gate`

   在 Restricted Hartree Fock 计算中使用的 Slater 方阵拟设。

   :param n_qubits: 量子态所包含的量子比特数。
   :type n_qubits: int
   :param n_electrons: 分子中所包含的电子数。
   :type n_electrons: int
   :param mo_coeff: 初始化Slater 方阵态的参数, 默认值为 ``None``。
   :type mo_coeff: Union[np.array, None], optional