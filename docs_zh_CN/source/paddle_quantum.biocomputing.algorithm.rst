paddle\_quantum.biocomputing.biocomputing.algorithm
=======================================================
蛋白质折叠模型的VQE算法


.. py:function:: cvar_expectation(psi, h, alpha)

   计算条件风险价值(CVaR)。

   :param psi: 输入量子态。
   :type psi: paddle_quantum.state.State
   :param h: 哈密顿量。
   :type h: paddle_quantum.Hamiltonian
   :param alpha: 控制计算CVaR期望值中包括的basis数量的参数。

   :return: 返回CVaR期望值。
   :rtype: paddle.Tensor

.. py:class:: ProteinFoldingSolver(penalty_factors, alpha, optimizer, num_iterations, tol, save_every)

   基类：:py:class:`paddle_quantum.qchem.VQESolver`

   求解蛋白质折叠模型的VQE求解器。

   :param penalty_factors: 蛋白质哈密顿量中的正则化因子。
   :type penalty_factors: List[float]
   :param alpha: 控制计算CVaR期望值中包括的basis数量的参数。
   :type alpha: float
   :param optimizer: 量桨优化器。
   :type optimizer: paddle.optimizer.Optimizer
   :param num_iterations: VQE迭代次数。
   :type num_iterations: int
   :param tol: VQE算法收敛的判据。
   :type tol: Optional[float]
   :param save_every: 控制优化过程中每隔多少步记录优化结果。
   :type save_every: Optional[int]

   .. py:method:: solve(protein, ansatz, optimizer_kwargs)

      求解函数。

      :param protein: 输入的待优化蛋白质结构。
      :type protein: paddle_quantum.biocomputing.Protein
      :param ansatz: VQE中的参数化量子线路。
      :type ansatz: paddle_quantum.ansatz.Circuit
      :param optimizer_kwargs: 优化器配置参数。
      :type optimizer_kwargs: Optional[Dict]

      :return: 返回最终的损失函数值及其对应的计算基态。
      :rtype: Tuple[float, str]
   
       
