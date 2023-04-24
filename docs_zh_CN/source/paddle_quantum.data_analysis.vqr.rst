paddle\_quantum.data_analysis.vqr
=============================================

量子回归分析的相关函数和模拟器类。

.. py:function:: load_dataset(data_file, model_name)

   加载需要分析的数据集 .csv 文件。

   :param data_file: 数据集文件所在目录。
   :type data_file: str
   :param model_name: 目前支持的所有模型类型，包括 ``linear`` 和 ``poly`` 两种模式。
   :type model_name: str

   :return: 返回计算所需的 pandas 解码文件。

.. py:function:: IPEstimator(circuit, input_state: State, measure_idx)

   基于量桨模拟器运行的电路来实现量子态内积的估计器。

   :param circuit: 运行电路。
   :type circuit: Circuit
   :param input_state: 电路输入的量子态。
   :type input_state: State
   :param measure_idx: 需要测量的比特序号。默认为 ``[0]``。
   :type measure_idx: List[int]

   :return: 返回计算的内积值（支持梯度分析）。
   :rtype: paddle.Tensor

.. py:class:: QRegressionModel(data_file, model_name, x_feature, y_feature, num_variable, init_params, num_qubits, learning_rate, iteration, language)

   基类：:py:class:`object`

   变分量子回归分析器（variational quantum regression, VQR）模型实现。

   :param data_file: 需要分析的数据集路径。
   :type data_file: str
   :param model_name: 所需使用的模型类型。目前只支持 ``linear`` 和 ``poly``。
   :type model_name: str 
   :param x_feature: 自变量名称。
   :type x_feature: List[str]
   :param y_feature: 因变量名称。
   :type y_feature: List[str]
   :param num_variable: 需要调用的模型参数量，即所有回归模型中的系数量。
   :type num_variable: int
   :param init_params: 调用参数的初始化数值。
   :type init_params: List[float]
   :param num_qubits: 所需使用到的量子比特数量。默认值为 ``6``。
   :type num_qubits: int
   :param learning_rate: 学习率。默认值为 ``0.1``。
   :type learning_rate: float
   :param iteration: 学习迭代次数。默认值为 ``100``。
   :type iteration: int
   :param language: 结果显示的语言。默认值为 ``CN``。
   :type iteration: str

   .. py:method:: regression_analyse()

      对输入的数据进行回归分析。

      :return: 返回可继续用来预测的模型。
      :rtype: Union[LinearRegression, PolyRegression]

.. py:class:: LinearRegression(num_qubits, num_x)

   基类：:py:class:`paddle.nn.Layer`

   量子线性回归分析器。

   :param num_qubits: 需要的量子比特数。
   :type num_qubits: int
   :param num_x: 线性自变量个数。默认为 ``1``。
   :type num_x: int

   .. py:method:: reg_param()

      输出当前回归分析器中的参数值。

      :return: 返回当前模型中的参数值。
      :rtype: paddle.Tensor
   
   .. py:method:: set_params(new_params)

      设定回归分析器中的参数。

      :param new_params: 输入的新参数值。
      :type new_params: Union[paddle.Tensor, np.ndarray]

   .. py:method:: fit(X, y, learning_rate, iteration, saved_dir, print_score, model_name)

      输入训练集数据用来训练回归模型。

      :param X: 自变量训练集数据。
      :type X: Union[paddle.Tensor, np.ndarray]
      :param y: 因变量训练集数据。
      :type y: Union[paddle.Tensor, np.ndarray]

   .. py:method:: predict(X)

      根据现有模型预测测试集数据。

      :param X: 自变量测试集数据。
      :type X: Union[paddle.Tensor, np.ndarray]

      :return: 返回当前模型的预测值。
      :rtype: Union[paddle.Tensor, np.ndarray]

   .. py:method:: score(X, y, metric)

      计算模型对测试集数据的回归拟合度。

      :param X: 自变量测试集数据。
      :type X: Union[paddle.Tensor, np.ndarray]
      :param y: 自变量测试集数据。
      :type y: Union[paddle.Tensor, np.ndarray]
      :param metric: 用于计算的度量类型。默认为 ``R2``。
      :type metric: str

      :return: 返回当前模型的拟合度。
      :rtype: float

