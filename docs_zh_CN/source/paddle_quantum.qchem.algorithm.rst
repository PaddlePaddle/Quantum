paddle\_quantum.qchem.algorithm
=========================================

变分量子求解器

.. py:class:: VQESolver(optimizer, num_iterations, tol, save_every)

   变分量子算法基类

   :param optimizer: 优化器。
   :type optimizer: paddle.optimizer.Optimizer
   :param num_iterations: 优化迭代次数。
   :type num_iterations: int
   :param tol: 优化器收敛精度。
   :type tol: float 
   :param save_every: 日志记录设置。
   :type save_every: int

   .. py:method:: solve

      求解方法。

.. py:class:: GroundStateSolver(optimizer, num_iterations, tol, save_every)

   基类：:py:class:`VQESolver`

   基态能量求解器。

   :param optimizer: 优化器。
   :type optimizer: paddle.optimize.Optimizer
   :param num_iterations: 优化迭代次数。
   :type num_iterations: int
   :param tol: 优化器收敛精度。
   :type tol: float 
   :param save_every: 日志记录设置。
   :type save_every: int

   .. py:method:: solve(mol, ansatz, init_state, **optimizer_kwargs)

      运行VQE方法计算分子基态能量。

      :param mol: 给定需要计算的分子类型。
      :type mol: paddle_quantum.qchem.Molecule
      :param ansatz: 变分量子线路。
      :type ansatz: paddle_quantum.ansatz.Circuit
      :param init_state: 给定的初态。
      :type init_state: Optional[paddle_quantum.state.State]
      :param optimizer_kwargs: 优化器配置参数。
      :type optimizer_kwargs: Optional[Dict]

      :return: 最终的损失函数值和优化后的量子态。
      :rtype: Tuple[float, paddle.Tensor]