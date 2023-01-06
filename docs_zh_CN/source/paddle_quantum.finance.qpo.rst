paddle\_quantum.finance.qpo
===============================

量子金融优化模型库的封装函数

.. py:function:: portfolio_combination_optimization(num_asset, data, iter, lr, risk, budget, penalty, circuit, init_state, optimizer, measure_shots, logger, compare)

   用于解决金融组合优化问题的高度封装的函数

   :param num_asset: 可投资项目的数目。
   :type num_asset: int
   :param data: 股票数据。
   :type data: Union[pq.finance.DataSimulator, Tuple[paddle.Tensor, paddle.Tensor]]
   :param iter: 循环迭代次数。
   :type iter: int
   :param lr: 梯度下降学习速率。
   :type lr: Optional[float] = None
   :param risk: 投资的风险系数。
   :type risk: float
   :param budget: 投资红利。
   :type budget: int
   :param penalty: 投资惩罚。
   :type penalty: float
   :param circuit: 量子电路的种类，若输入整数则搭建该整数层complex_entangled_layer。
   :type circuit: Union[pq.ansatz.Circuit, int] = 2
   :param init_state: 输入到变分量子电路的初态，默认为零态的直积。
   :type init_state: Optional[pq.state.State] = None
   :param optimizer: 优化器类型，默认为 `paddle.optimizer.Adam`
   :type optimizer: Optional[paddle.optimizer.Optimizer] = None
   :param measure_shots: 对末态做测量的次数，默认为2048。
   :type measure_shots: int
   :param logger: 开启日志记录。
   :type logger: Optional[logging.Logger] = None
   :param compare: 是否把梯度下降优化得到的损失最小值与真实损失最小值相比。
   :type compare: bool = False

   :return: 列表形式的最优的投资组合
   :rtype: List[int]

   .. note:: 
    
      此函数只用于解决一个特定问题，见：https://qml.baidu.com/tutorials/combinatorial-optimization/quantum-finance-application-on-portfolio-optimization.html
