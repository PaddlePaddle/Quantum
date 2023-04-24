paddle\_quantum.qml.bert\_qtc
==============================================

BERT-量子时序卷积（Quantum Temporal Convolution, QTC）模型

.. py:class:: QuantumTemporalConvolution(num_filter, kernel_size, circuit_depth, padding)

   基类：:py:class:`paddle.nn.Layer`

   量子时序卷积（Quantum Temporal Convolution, QTC）模型的实现。具体细节可以参考：https://arxiv.org/abs/2203.03550 。

   :param num_filter: 量子时序卷积核的数量。
   :type num_filter: int
   :param kernel_size: 量子时序卷积的卷积核大小，它也是对应的量子神经网络所包含的量子比特数量。
   :type kernel_size: int
   :param circuit_depth: 量子神经网络中的电路模板的层数。
   :type circuit_depth: int
   :param padding: 序列的填充的长度。如果它是 int 型数据，表示左右两端各填充对应长度的值。如果是list类型，则应该是 [pad_left, pad_right]，表示左边填充多少、右边填充多少。
   :type padding: int

   .. py:method:: forward(bert_feature)

      模型的前向执行函数。

      :param bert_feature: 模型的输入，它是由 BERT 模型对输入文本提取得到的特征。
      :type bert_feature: paddle.Tensor

      :return: 量子时序卷积网络提取得到的特征。
      :rtype: paddle.Tensor

.. py:class:: Decoder(num_filter, kernel_size, circuit_depth, padding, num_classes, hidden_size)

   基类：:py:class:`paddle.nn.Layer`

   BERT-QTC 模型的解码器模块。它包含量子时序卷积层、全局最大池化层和线性层。

   :param num_filter: 量子时序卷积核的数量。
   :type num_filter: int
   :param kernel_size: 量子时序卷积的卷积核大小，它也是对应的量子神经网络所包含的量子比特数量。
   :type kernel_size: int
   :param circuit_depth: 量子神经网络中的电路模板的层数。
   :type circuit_depth: int
   :param padding: 序列的填充的长度。如果它是 int 型数据，表示左右两端各填充对应长度的值。如果是list类型，则应该是 [pad_left, pad_right]，表示左边填充多少、右边填充多少。
   :type padding: int
   :param num_classes: 模型所要分类的类别的数量。
   :type num_classes: int
   :param hidden_size: BERT 模型的隐层状态的向量维数。
   :type hidden_size: int

   .. py:method:: forward(bert_feature)

      模型的前向执行函数。

      :param bert_feature: 模型的输入，它是由 BERT 模型对输入文本提取得到的特征。
      :type bert_feature: paddle.Tensor

      :return: 模型的预测结果，即对各个类别的概率分数。
      :rtype: paddle.Tensor

.. py:class:: BERTQTC(bert_model, decoder)

   基类：:py:class:`paddle.nn.Layer`

   BERT-QTC 模型。它包含 BERT 预训练模型和解码器模块。

   具体细节可以参考 https://arxiv.org/abs/2203.03550 。

   :param bert_model: 预训练的 BERT 模型。
   :type bert_model: str
   :param decoder: 解码器模块。
   :type decoder: paddle.nn.Layer

   .. py:method:: forward(token_ids)

      模型的前向执行函数。

      :param token_ids: 模型的输入，它是文本令牌化后的数字表示。
      :type token_ids: paddle.Tensor

      :return: 模型的预测结果，即对各个类别的概率分数。
      :rtype: paddle.Tensor

.. py:class:: TextDataset(file_path, bert_model)
   :noindex:

   基类：:py:class:`paddle.io.Dataset`

   实现文本数据集的类。

   :param file_path: 数据集的文件路径。其里面应该由多行组成。每一行包含文本标签，由制表符或空格分开。
   :type file_path: str
   :param bert_model: 预训练的 BERT 模型，用于构建其对应的令牌器（tokenizer）。
   :type bert_model: str

.. py:function:: train(num_filter, kernel_size, circuit_depth, padding, num_classes, model_name, dataset, batch_size, num_epochs, bert_model, hidden_size, learning_rate, saved_dir, using_validation, early_stopping)
   :noindex:
   
   训练 BERT-QTC 模型的函数。

   :param num_filter: 量子时序卷积核的数量。
   :type num_filter: int
   :param kernel_size: 量子时序卷积的卷积核大小，它也是对应的量子神经网络所包含的量子比特数量。
   :type kernel_size: int
   :param circuit_depth: 量子神经网络中的电路模板的层数。
   :type circuit_depth: int
   :param padding: 序列的填充的长度。如果它是 int 型数据，表示左右两端各填充对应长度的值。如果是list类型，则应该是 [pad_left, pad_right]，表示左边填充多少、右边填充多少。
   :type padding: int
   :param num_classes: 模型所要分类的类别的数量。
   :type num_classes: int
   :param model_name: 模型的名字，用于作为保存的模型参数的文件名。
   :type model_name: str
   :param dataset: 模型的名字，用于作为保存的模型参数的文件名。
   :type dataset: str
   :param batch_size: 数据的批大小。
   :type batch_size: int
   :param num_epochs: 训练的轮数。
   :type num_epochs: int
   :param bert_model: 预训练的 BERT 模型，默认为 ``bert-base-chinese`` ，即官方的 BERT 中文预训练模型。
   :type bert_model: str
   :param hidden_size: BERT 模型的隐层状态的向量维数，默认为 ``768`` 。
   :type hidden_size: int
   :param learning_rate: 更新参数的学习率，默认为 ``0.01`` 。
   :type learning_rate: float
   :param saved_dir: 训练得到的模型文件的保存路径，默认使用当前目录。
   :type saved_dir: str
   :param using_validation: 是否使用验证集。默认为 ``False`` ，即不包含验证集。
   :type using_validation: bool
   :param early_stopping: 默认为 ``1000`` ，即如果模型在 1000 次迭代中，在验证集上的 loss 没有提升，则会自动停止训练。
   :type early_stopping: int

.. py:function:: evaluate(model, data_loader)
   :noindex:

   对模型进行评估。

   :param model: 训练得到的模型，用于被评估。
   :type model: paddle.nn.Layer
   :param data_loader: 用于评估模型的数据加载器。
   :type data_loader: DataLoader

   :return: 返回模型在输入数据上的平均的损失值和平均准确率。
   :rtype: Tuple[float, float]

.. py:function:: test(model, model_path, test_loader)
   :noindex:
   
   使用测试集对模型进行测试。

   :param model: 训练得到的模型，用于被评估。
   :type model: paddle.nn.Layer
   :param model_path: 保存的模型参数的文件路径。
   :type model_path: str
   :param test_loader: 测试集的数据加载器。
   :type test_loader: DataLoader

.. py:function:: inference(text, model_path, classes, num_filter, kernel_size, circuit_depth, padding, bert_model, hidden_size)
   :noindex:

   推理函数。使用训练好的模型对输入的图片进行预测。

   :param text: 要预测的文本内容。
   :type text: str
   :param model_path: 保存的模型参数的文件路径。
   :type model_path: str
   :param classes: 要预测的文本的类别。
   :type classes: List[str]
   :param num_filter: 量子时序卷积核的数量。
   :type num_filter: int
   :param kernel_size: 量子时序卷积的卷积核大小，它也是对应的量子神经网络所包含的量子比特数量。
   :type kernel_size: int
   :param circuit_depth: 量子神经网络中的电路模板的层数。
   :type circuit_depth: int
   :param padding: 序列的填充的长度。如果它是 int 型数据，表示左右两端各填充对应长度的值。如果是list类型，则应该是 [pad_left, pad_right]，表示左边填充多少、右边填充多少。
   :type padding: int
   :param bert_model: 预训练的 BERT 模型，默认为 ``bert-base-chinese`` ，即官方的 BERT 中文预训练模型。
   :type bert_model: str
   :param hidden_size: BERT 模型的隐层状态的向量维数，默认为 ``768`` 。

   :return: 返回模型预测的类别。
   :rtype: str
