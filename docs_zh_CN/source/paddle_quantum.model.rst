paddle\_quantum.model
==================================

量子神经网络的通用模型模板

.. py:function:: reset_settings()

   重置 ``rcParams`` 为默认设定。

.. py:function:: random_batch(data, label, batch_size)

   从数据集和标签集中随机返回一个数据批次。

   :param data: 数据集。
   :type data: Iterable
   :param label: 标签集。
   :type label: Iterable
   :param batch_size: 数据批次的大小。
   :type batch_size: int

   :return: 一批随机的数据。
   :rtype: Tuple[list, list]

.. py:class:: NullScheduler()

   基类：:py:class:`paddle.optimizer.lr.LRScheduler`

   用于不希望使用学习率策略的用户，可以被以下代码激活

   .. code-block:: python

      from paddle_quantum.model import rcParams
      rcParams['scheduler'] = None

.. py:class:: Model(network, name="Model")

   基类：:py:class:`object`

   量子神经网络模型的通用模板。

   :param network: 一个量子神经网络。
   :type network: paddle.nn.Layer
   :param name: 模型的名字。默认为 ``"Model"``。
   :type name: str

   .. py:method:: parameters()

   返回神经网络的参数。

   :return: 神经网络的参数。
   :rtype: List[paddle.fluid.framework.ParamBase]

   .. py:method:: prepare(loss_fcn, metric_fcn=None, metric_name=None)

      量子神经网络的常规功能设置。

      :param loss_fcn: 量子神经网络的损失函数。
      :type loss_fcn: Callable[[Union[State, Circuit, Sequential], Any], Any]
      :param metric_fcn: 量子神经网络的度量函数，不会干扰训练过程。默认为 ``None``。
      :type metric_fcn: Callable[[Union[Circuit, Sequential]], float]
      :param metric_name: 度量函数的名字。默认为 ``None``。
      :type metric_name: str
      :raises ValueError: 度量函数的输出必须为 float。

   .. note::

      该函数同时会引入 ``rcParams`` 里的参数。

   .. py:method:: check_prepared()

      检测模型是否准备好训练。

   .. py:method:: train(loss_generator)

      量子神经网络单批次训练的通用模板

      :param loss_generator: 计算量子神经网络，以 ``Model.network`` 为输入的损失函数。默认为 ``None`` ，即使用在 ``Model.prepare`` 里定义的损失函数。
      :type loss_generator: Callable[[Any], Any]
      
      :return: 包含以下元素：

         - 一组损失数值。
         - 若给定了度量函数，同时返回一组度量值。
      
      :rtype: Union[List[float], Tuple[List[float], List[float]]]

   .. py:method:: evaluate(loss_generator)

      量子神经网络评估的通用模板

      :param loss_generator: 计算量子神经网络，以 ``Model.network`` 为输入的损失函数。默认为 ``None`` ，即使用在 ``Model.prepare`` 里定义的损失函数。
      
      :return: 
         包含以下元素：

         - 损失数值；
         - 若给定了度量函数，同时返回度量值。
      
      :rtype: Union[float, Tuple[float, float]]

   .. py:method:: fit(train_data, train_label, test_data, test_label)

      量子神经网络训练的通用模板

      :param train_data: 训练集的数据。
      :type train_data: Iterable
      :param train_label: 训练集的标签。
      :type train_label: Iterable
      :param test_data: 测试集的数据。
      :type test_data: Iterable
      :param test_label: 测试集的标签。
      :type test_label: Iterable

   .. py:method:: plot(include_metric, apply_log, has_epoch)

      画图展示训练数据

      :param include_metric: 是否包含度量值。
      :type include_metric: bool
      :param apply_log: 是否对数据施加 log。
      :type apply_log: bool
      :param has_epoch: 是否数据分批次训练
      :type has_epoch: bool

.. py:class:: OptModel(circuit, name="OptModel")

   基类：:py:class:`paddle_quantum.model.Model`

   用于实现优化类量子神经网络的类。

   :param circuit: 被优化的 Circuit 的实例。
   :type data: Circuit
   :param name: 模型的名字。默认为 ``"OptModel"``。
   :type name: str

   .. py:method:: prepare(loss_fcn, metric_fcn=None, metric_name=None, *loss_args)

      准备及检查优化类量子神经网络的功能设置。

      :param loss_fcn: 量子神经网络的损失函数。
      :type loss_fcn: Callable[[Circuit, Any], paddle.Tensor]
      :param metric_fcn: 量子神经网络的度量函数，不会干扰训练过程。默认为 ``None``。
      :type metric_fcn: Callable[[Union[Circuit, Sequential]], float]
      :param metric_name: 度量函数的名字。默认为 ``None``。
      :type metric_name: str
      :param loss_args: loss_fcn 除了量子神经网络输入以外的的参数。
      :type loss_args: any
      :raises ValueError: 损失函数的输出必须为 paddle.Tensor。

   .. note::

      该函数同时会引入 ``rcParams`` 里的参数。

   .. py:method:: optimize()

      根据损失函数优化电路。


      :return: 
         包含以下元素：

         - 一组损失数值；
         - 若给定了度量函数，同时返回一组度量值。

      :rtype: Union[List[float], Tuple[List[float], List[float]]]

   .. py:method:: evaluate()

      计算当前量子神经网络的损失值和度量值。
      
      :return: 
         包含以下元素：

         - 损失数值；
         - 若给定了度量函数，同时返回度量值。
      
      :rtype: Union[float, Tuple[float, float]]

   .. py:method:: fit()

      :raises NotImplementedError: 优化模型不支持 fit 功能：请直接使用 OptModel.optimize。

   .. py:method:: plot(include_metric=True, apply_log=False)

      画图展示训练数据

      :param include_metric: 是否包含度量值。 默认为 ``True``。
      :type include_metric: bool
      :param apply_log: 是否对数据施加 log。 默认为 ``False``。
      :type apply_log: bool

.. py:class:: LearningModel(circuit, name="LearningModel")

   基类：:py:class:`paddle_quantum.model.Model`

   用于实现学习类量子神经网络的类。

   :param circuit: 被优化的 Circuit 的实例。
   :type data: Circuit
   :param name: 模型的名字。默认为 ``"LearningModel"``。
   :type name: str

   .. py:method:: prepare(loss_fcn, metric_fcn=None, metric_name=None, *loss_args)

      准备及检查学习类量子神经网络的功能设置。

      :param loss_fcn: 量子神经网络输出的损失函数。
      :type loss_fcn: Callable[[State, Any, Any], Any]
      :param metric_fcn: 量子神经网络的度量函数，不会干扰训练过程。默认为 ``None``。
      :type metric_fcn: Callable[[Union[Circuit, Sequential]], float]
      :param metric_name: 度量函数的名字。默认为 ``None``。
      :type metric_name: str
      :param loss_args: loss_fcn 除了量子神经网络输入和标签以外的的参数。
      :type loss_args: any

   .. note::

      -  此类模型的数据输入必须为 ``paddle_quantum.State``。
         若数据需要编码到量子电路中，请使用 ``paddle_quantum.model.EncodingModel``。
      -  该函数同时会引入 ``rcParams`` 里的参数。

   .. py:method:: train_batch(data, label)

      用单批次数据训练电路。

      :param data: 一组输入量子态
      :type param: List[State]
      :param data: 预期标签
      :type param: List[Any]
      
      :return: 
         包含以下元素：

         - 一组损失数值；
         - 若给定了度量函数，同时返回一组度量值。
      
      :rtype: Union[List[float], Tuple[List[float], List[float]]]

   .. py:method:: eval_batch(data, label)

      用单批次数据评估电路。

      :param data: 一组输入量子态
      :type data: List[State]
      :param label: 预期标签
      :type label: List[Any]
      
      :return: 
         包含以下元素：

         - 损失数值；
         - 若给定了度量函数，同时返回度量值。
      
      :rtype: Union[float, Tuple[float, float]]

   .. py:method:: fit(train_data, train_label, test_data, test_label)

      使用输入数据训练电路。

      :param train_data: 训练集的数据。
      :type train_data: List[State]
      :param train_label: 训练集的标签。
      :type train_label: Iterable
      :param test_data: 测试集的数据。
      :type test_data: List[State]
      :param test_label: 测试集的标签。
      :type test_label: Iterable

   .. py:method:: plot(include_metric=True, apply_log=False)

      画图展示训练数据

      :param include_metric: 是否包含度量值。 默认为 ``True``。
      :type include_metric: bool
      :param apply_log: 是否对数据施加 log。 默认为 ``False``。
      :type apply_log: bool

.. py:class:: EncodingNetwork(encoding_func, param_shape, initial_state=None)

   基类：:py:class:`paddle.nn.Layer`

   编码模型的量子神经网络。

   :param encoding_func: 决定如何构建量子电路的编码函数。
   :type encoding_func: Callable[[Any, paddle.Tensor], Circuit]
   :param param_shape: 输入参数的 shape。
   :type param_shape: Iterable[int]
   :param initial_state: 电路的初始态。
   :type initial_state: State

   .. note::

      仅用于 ``paddle_quantum.model.EncodingModel``。

   .. py:method:: forward(input_data)

      计算输入对应的输出。

      :param input_data: 用于编码电路的输入数据。
      :type input_data: List[Any]
      :return: 电路的输出态。
      :rtype: List[State]

.. py:class:: EncodingModel(encoding_fcn, param_shape, initial_state=None, name="EncodingModel")

   基类：:py:class:`Model`

   用于实现编码类量子神经网络的类。

   :param encoding_fcn: 编码函数，用编码数据和参数决定如何构建量子电路。
   :type encoding_fcn: Callable[[Any, paddle.Tensor], Circuit]
   :param param_shape: encoding_fcn 参数的 shape。
   :type param_shape: Iterable[int]
   :param initial_state: 电路的初始态。默认为 ``None``，即零态。
   :type initial_state: State
   :param name: 模型的名字。默认为 ``"EncodingModel"``。
   :type name: str

   .. note::

      与 ``paddle_quantum.model.LearningModel`` 不同的是，该模型的数据需要编码至量子电路而不是量子态。
      因此该模型需要知道输入数据是如何编入至量子电路的。该模型会根据 ``param_shape`` 自动生成所需的训练参数。

   .. py:method:: prepare(loss_fcn, metric_fcn=None, metric_name=None, *loss_args)

      准备及检查编码类量子神经网络的功能设置。

      :param loss_fcn: 量子神经网络输出的损失函数。
      :type loss_fcn: Callable[[State, Any, Any], Any]
      :param metric_fcn: 量子神经网络的度量函数，不会干扰训练过程。默认为 ``None``。
      :type metric_fcn: Callable[[Union[Circuit, Sequential]], float]
      :param metric_name: 度量函数的名字。默认为 ``None``。
      :type metric_name: str
      :param loss_args: loss_fcn 除了量子神经网络输入和标签以外的的参数。
      :type loss_args: any

   .. note::

      该函数同时会引入 ``rcParams`` 里的参数。

   .. py:method:: train_batch(data, label)

      用单批次数据训练电路。

      :param data: 一组数据
      :type param: Iterable
      :param data: 预期标签
      :type param: Iterable
      
      :return: 
         包含以下元素：

         - 一组损失数值；
         - 若给定了度量函数，同时返回一组度量值。
      
      :rtype: Union[List[float], Tuple[List[float], List[float]]]

   .. py:method:: eval_batch(data, label)

      用单批次数据评估电路。

      :param data: 一组数据
      :type data: Iterable
      :param label: 预期标签
      :type label: Iterable
      
      :return: 
         包含以下元素：

         - 损失数值；
         - 若给定了度量函数，同时返回度量值。
      
      :rtype: Union[float, Tuple[float, float]]

   .. py:method:: fit(train_data, train_label, test_data, test_label)

      使用输入数据训练电路。

      :param train_data: 训练集的数据。
      :type train_data: Iterable
      :param train_label: 训练集的标签。
      :type train_label: Iterable
      :param test_data: 测试集的数据。
      :type test_data: Iterable
      :param test_label: 测试集的标签。
      :type test_label: Iterable

   .. py:method:: plot(include_metric=True, apply_log=False)

      画图展示训练数据

      :param include_metric: 是否包含度量值。 默认为 ``True``。
      :type include_metric: bool
      :param apply_log: 是否对数据施加 log。 默认为 ``False``。
      :type apply_log: bool
