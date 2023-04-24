paddle\_quantum.qml.qsann
==============================================

量子自注意力神经网络（Quantum Self-Attention Neural Network, QSANN）模型

.. py:function:: generate_observable(num_qubits, num_terms)
   :noindex:

   生成测量量子态所需要的可观测量。

   :param num_qubits: 量子比特的数量。
   :type num_qubits: int
   :param num_terms: 生成的可观测量的项数。
   :type num_terms: int

   :return: 返回生成的可观测量。
   :rtype: paddle_quantum.Hamiltonian

.. py:class:: QSANN(num_qubits, len_vocab, num_layers, depth_ebd, depth_query, depth_key, depth_value)

   基类：:py:class:`paddle.nn.Layer`

   量子自注意力神经网络（Quantum Self-Attention Neural Network, QSANN）模型的实现。具体细节可以参考：https://arxiv.org/abs/2205.05625 。

   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param len_vocab: 数据集的词表的长度。
   :type len_vocab: int
   :param num_layers: 自注意力层的层数。
   :type num_layers: int
   :param depth_ebd: embedding 电路的深度。
   :type depth_ebd: int
   :param depth_query: query 电路的深度。
   :type depth_query: int
   :param depth_key: key 电路的深度。
   :type depth_key: int
   :param depth_value: value 电路的深度。
   :type depth_value: int

   .. py:method:: forward(batch_text)

      模型的前向执行函数。

      :param batch_input: 模型的输入，它是一个列表，每一项都是一个由整数组成的列表。
      :type batch_input: List[List[int]]

      :return: 返回一个列表，其每一项都是对输入文本的预测结果。
      :rtype: List[paddle.Tensor]

.. py:function:: deal_vocab(vocab_path)

   根据输入的词汇表文件，得到从词到索引的映射。

   :param vocab_path: 词表文件的路径。
   :type vocab_path: str

   :return: 返回从词到对应的索引的映射。
   :rtype: Dict[str, int]

.. py:class:: TextDataset(file_path, word_idx, pad_size)
   :noindex:

   基类：:py:class:`paddle.io.Dataset`

   实现文本数据集的类。

   :param file_path: 数据集的文件路径。其里面应该由多行组成。每一行包含文本标签，由制表符或空格分开。
   :type file_path: str
   :param word2idx: 数据集的数据量大小。默认为 ``0`` ，表示使用所有数据。
   :type word2idx: dict
   :param pad_size: 要将文本序列填充到的长度。默认为 ``0`` ，即不进行填充。
   :type pad_size: int

.. py:function:: build_iter(dataset, batch_size, shuffle)

   建立批数据的可迭代类型。

   :param dataset: 输入的数据集，对其进行构建批数据的可迭代类型。
   :type dataset: paddle.io.Dataset
   :param batch_size: 批数据的大小。
   :type batch_size: int
   :param shuffle: 是否要随机打乱数据。默认为 ``Flase`` ，即不随机打乱。
   :type shuffle: bool

   :return: 构建的可迭代类型，其中包含生成的批数据。
   :rtype: list

.. py:function:: train(model_name, dataset, num_qubits, num_layers, depth_ebd, depth_query, depth_key, depth_value, batch_size, num_epochs, learning_rate, saved_dir, using_validation, early_stopping)
   :noindex:
   
   训练 VSQL 模型的函数。

   :param model_name: 模型的名字，用于作为保存的模型参数的文件名。
   :type model_name: str
   :param dataset: 模型的名字，用于作为保存的模型参数的文件名。
   :type dataset: str
   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param num_layers: 自注意力层的层数。
   :type num_layers: int
   :param depth_ebd: embedding 电路的深度。
   :type depth_ebd: int
   :param depth_query: query 电路的深度。
   :type depth_query: int
   :param depth_key: key 电路的深度。
   :type depth_key: int
   :param depth_value: value 电路的深度。
   :type depth_value: int
   :param batch_size: 数据的批大小。
   :type batch_size: int
   :param num_epochs: 训练的轮数。
   :type num_epochs: int
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
   :type data_loader: list

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
   :type test_loader: list

.. py:function:: inference(text, model_path, vocab_path, classes, num_qubits, num_layers, depth_ebd, depth_query, depth_key, depth_value)
   :noindex:

   推理函数。使用训练好的模型对输入的图片进行预测。

   :param text: 要预测的图片的路径。
   :type text: str
   :param model_path: 保存的模型参数的文件路径。
   :type model_path: str
   :param vocab_path: 词表文件的路径。
   :type vocab_path: str
   :param classes: 要预测的文本的类别。
   :type classes: List[str]
   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param num_layers: 自注意力层的层数。
   :type num_layers: int
   :param depth_ebd: embedding 电路的深度。
   :type depth_ebd: int
   :param depth_query: query 电路的深度。
   :type depth_query: int
   :param depth_key: key 电路的深度。
   :type depth_key: int
   :param depth_value: value 电路的深度。
   :type depth_value: int

   :return: 返回模型预测的类别。
   :rtype: str
