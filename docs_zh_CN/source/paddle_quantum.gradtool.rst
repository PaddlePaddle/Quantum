paddle\_quantum.gradtool
===============================

梯度分析工具的功能实现。

.. py:function:: show_gradient(circuit, loss_func, ITR, LR, *args)

   计算量子神经网络中各可变参数的梯度值和损失函数值。

   :param circuit: 要训练的量子神经网络。
   :type circuit: Circuit
   :param loss_func: 计算该量子神经网络损失值的函数。
   :type loss_func: Callable[[Circuit, Any], paddle.Tensor]
   :param ITR: 训练的次数。
   :type ITR: int
   :param LR: 学习训练的速率。
   :type LR: float
   :param \*args: 用于损失函数计算的额外参数列表。
   :type \*args: Any

   :return: 包含如下两个元素：
               - loss_list: 损失函数值随训练次数变化的列表。
               - grad_list: 各参数梯度随训练次变化的列表。
   :rtype: Tuple[List[float], List[float]]

.. py:function:: plot_distribution(grad)
   
   根据输入的梯度的列表，画出梯度的分布图。

   :param grad: 量子神经网络某参数的梯度列表。
   :type grad: np.ndarray

.. py:function:: random_sample(circuit, loss_func, sample_num, *args, mode, if_plot, param)

   对模型进行随机采样，根据不同的计算模式，获得对应的平均值和方差。

   :param circuit: 要训练的量子神经网络。
   :type circuit: Circuit
   :param loss_func: 计算该量子神经网络损失值的函数。
   :type loss_func: Callable[[Circuit, Any], paddle.Tensor]
   :param sample_num: 随机采样的次数
   :type sample_num: int
   :param mode: 随机采样后的计算模式，默认为 ``'single'``。
   :type mode: string
   :param if_plot: 是否对梯度进行画图，默认为 ``True``。
   :type if_plot: boolean
   :param param: ``'single'`` 模式中对第几个参数进行画图，默认为 ``0``，即第一个参数。
   :type param: int
   :param \*args: 用于损失函数计算的额外参数列表。
   :type \*args: Any

   .. note::
      在本函数中提供了三种计算模式，``mode`` 分别可以选择 ``'single'``, ``'max'``, 以及 ``'random'``。
         - mode='single': 表示计算电路中的每个可变参数梯度的平均值和方差。
         - mode='max': 表示对电路中每轮采样的所有参数梯度的最大值求平均值和方差。
         - mode='random': 表示对电路中每轮采样的所有参数随机取一个梯度，求平均值和方差。

   :return: 包含如下两个元素：
               - loss_list: 损失函数值随训练次数变化的列表。
               - grad_list: 各参数梯度随训练次变化的列表。
   :rtype: Tuple[List[float], List[float]]

.. py:function:: plot_loss_grad(circuit, loss_func, ITR, LR, *args)

   绘制损失值和梯度随训练次数变化的图。

   :param circuit: 传入的参数化量子电路，即要训练的量子神经网络。
   :type circuit: Circuit
   :param loss_func: 计算该量子神经网络损失值的函数。
   :type loss_func: Callable[[Circuit, Any], paddle.Tensor]
   :param ITR: 训练的次数。
   :type ITR: int
   :param LR: 学习训练的速率。
   :type LR: float
   :param \*args: 用于损失函数计算的额外参数列表。
   :type \*args: Any

.. py:function:: plot_supervised_loss_grad(circuit, loss_func, N, EPOCH, LR, BATCH, TRAIN_X, TRAIN_Y, *args)

   绘制监督学习中损失值和梯度随训练次数变化的图。

   :param circuit: 要训练的量子神经网络。
   :type circuit: Circuit
   :param loss_func: 计算该量子神经网络损失值的函数。
   :type loss_func: Callable[[Circuit, Any], paddle.Tensor]
   :param N: 量子比特的数量。
   :type N: int
   :param EPOCH: 训练的轮数。
   :type EPOCH: int
   :param LR: 学习训练的速率。
   :type LR: float
   :param BATCH: 训练时 batch 的大小。
   :type BATCH: int
   :param TRAIN_X: 训练数据集。
   :type TRAIN_X: paddle.Tensor
   :param TRAIN_Y: 训练数据集的标签。
   :type TRAIN_Y: list
   :param \*args: 用于损失函数计算的额外参数列表。
   :type \*args: Any

   :raise Exception: 训练数据必须是 ``paddle.Tensor`` 类型

   :return: 包含如下两个元素：
               - loss_list: 损失函数值随训练次数变化的列表。
               - grad_list: 各参数梯度随训练次变化的列表。
   :rtype: Tuple[List[float], List[float]]

.. py:function:: random_sample_supervised(circuit, loss_func, N, sample_num, BATCH, TRAIN_X, TRAIN_Y, *args: Any, mode:='single', if_plot:=True, param:=0)

   对监督学习模型进行随机采样，根据不同的计算模式，获得对应的平均值和方差。

   :param circuit: 要训练的量子神经网络。
   :type circuit: Circuit
   :param loss_func: 计算该量子神经网络损失值的函数。
   :type loss_func: Callable[[Circuit, Any], paddle.Tensor]
   :param N: 量子比特的数量。
   :type N: int
   :param sample_num: 随机采样的次数。
   :type sample_num: int
   :param BATCH: 训练时 batch 的大小。
   :type BATCH: int
   :param TRAIN_X: 训练数据集。
   :type TRAIN_X: paddle.Tensor
   :param TRAIN_Y: 训练数据集的标签。
   :type TRAIN_Y: list
   :param mode: 随机采样后的计算模式，默认为 ``'single'``。
   :type mode: string
   :param if_plot: 是否对梯度进行画图，默认为 ``True``。
   :type if_plot: boolean
   :param param: ``Single`` 模式中对第几个参数进行画图，默认为 ``0``，即第一个参数。
   :type param: int
   :param \*args: 用于损失函数计算的额外参数列表。
   :type \*args: Any

   :raise Exception: 训练数据必须是 ``paddle.Tensor`` 类型

   .. note::
      在本函数中提供了三种计算模式，``mode`` 分别可以选择 ``'single'``, ``'max'``, 以及 ``'random'``。
         - mode='single': 表示计算电路中的每个可变参数梯度的平均值和方差。
         - mode='max': 表示对电路中每轮采样的所有参数梯度的最大值求平均值和方差。
         - mode='random': 表示对电路中每轮采样的所有参数随机取一个梯度，求平均值和方差。
   
   :return: 包含如下两个元素：
               - loss_list: 损失函数值随训练次数变化的列表。
               - grad_list: 各参数梯度随训练次变化的列表。
   :rtype: Tuple[List[float], List[float]]
