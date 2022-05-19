paddle\_quantum.qchem.density\_matrix
============================================

对量子态测量单体密度矩阵。

.. py:class:: OneBodyDensityMatrix()

   基类： :py:class:`paddle.autograd.PyLayer`

   用于测量给定量子态的单体密度矩阵。

.. py:function:: get_spinorb_onebody_dm(n_qubits, state)

   获取给定量子态的单体密度矩阵，并以自旋轨道数标注量子比特。

   :param n_qubits: 量子态所包含的量子比特数。
   :type n_qubits: int
   :param state: 给定量子态。
   :type state: paddle.Tensor

   :return: 自旋轨道数标注的单体密度矩阵。
   :rtype: Tuple[paddle.Tensor]