paddle\_quantum.qml.vsql
==============================================

变分影子量子学习（variational shadow quantum learning, VSQL）模型。

.. py:function:: image_process(images, num_qubits)

   对输入的图片进行归一化。先将图片展开为向量，再进行归一化。

   :param images: 输入的图片。
   :type images: numpy.ndarray
   :param num_qubits: 量子比特的数量，决定了归一化向量的维度。
   :type num_qubits: int

   :return: 返回归一化之后的向量，它是由 ``paddle.Tensor`` 组成的列表。
   :rtype: numpy.ndarray

.. py:class:: ImageDataset(file_path, num_samples, transform)

   基类：:py:class:`paddle.io.Dataset`

   实现图片数据集的类。

   :param file_path: 数据集的文件路径。其里面应该由多行组成。每一行包含图片的文件路径和标签，由制表符分开。
   :type file_path: str
   :param num_samples: 数据集的数据量大小。默认为 ``0`` ，表示使用所有数据。
   :type num_samples: int
   :param transform: 对图片进行预处理的方法。默认为 ``None`` ，即不进行任何预处理。
   :type transform: Optional[Callable]

.. py:function:: generate_observable(start_idx, num_shadow)

   生成测量量子态所需要的可观测量。

   :param start_idx: 要测量的量子比特的起始索引。
   :type start_idx: int
   :param num_shadow: 影子电路所包含的量子比特的数量。
   :type num_shadow: int

   :return: 返回生成的可观测量。
   :rtype: paddle_quantum.Hamiltonian

.. py:class:: VSQL(num_qubits, num_shadow, num_classes, depth)

   基类：:py:class:`paddle.nn.Layer`

   变分影子量子学习（variational shadow quantum learning, VSQL）模型的实现。具体细节可以参考：https://ojs.aaai.org/index.php/AAAI/article/view/17016 。

   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param num_shadow: 影子电路所包含的量子比特的数量。
   :type num_shadow: int
   :param num_classes: 模型所要分类的类别的数量。
   :type num_classes: int
   :param depth: 量子电路的深度，默认为 ``1`` 。
   :type depth: Optional[int]

   .. py:method:: forward(batch_input)

      模型的前向执行函数。

      :param batch_input: 模型的输入，其形状为 :math:`(\text{batch_size}, 2^{\text{num_qubits}})` 。
      :type batch_input: List[paddle.Tensor]

      :return: 返回模型的输出，其形状为 :math:`(\text{batch_size}, \text{num_classes})` 。
      :rtype: paddle.Tensor

.. py:function:: train(model_name, num_qubits, num_shadow, classes, batch_size, num_epochs, depth, datasets, saved_dir, learning_rate, using_validation, num_workers, early_stopping, num_train, num_dev, num_test)

   训练 VSQL 模型的函数。

   :param model_name: 模型的名字，用于作为保存的模型参数的文件名。
   :type model_name: str
   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param num_shadow: 影子电路所包含的量子比特的数量。
   :type num_shadow: int
   :param classes: 要预测的图片的类别。
   :type classes: list
   :param batch_size: 数据的批大小。
   :type batch_size: int
   :param num_epochs: 训练的轮数。
   :type num_epochs: int
   :param depth: 量子电路的深度，默认为 ``1`` 。
   :type depth: int
   :param datasets: 训练所使用的数据集文件夹路径。默认为 ``MNIST``，即使用内置的 MNIST 数据集。
   :type datasets: str
   :param saved_dir: 训练得到的模型文件的保存路径，默认使用当前目录。
   :type saved_dir: str
   :param learning_rate: 更新参数的学习率，默认为 ``0.01`` 。
   :type learning_rate: float
   :param using_validation: 是否使用验证集。默认为 ``False`` ，即不包含验证集。
   :type using_validation: bool
   :param num_workers: 构建数据集加载器的线程数，默认为 ``0`` ，即不使用额外线程。
   :type num_workers: int
   :param early_stopping: 默认为 ``1000`` ，即如果模型在 1000 次迭代中，在验证集上的 loss 没有提升，则会自动停止训练。
   :type early_stopping: int
   :param num_train: 训练集的数据量。默认为 ``0`` ，即使用所有的训练数据。
   :type num_train: int
   :param num_dev: 验证集的数据量。默认为 ``0`` ，即使用所有的训练数据。
   :type num_dev: int
   :param num_test: 测试集的数据量。默认为 ``0`` ，即使用所有的训练数据。
   :type num_test: int

.. py:function:: evaluate(model, data_loader)

   对模型进行评估。

   :param model: 训练得到的模型，用于被评估。
   :type model: paddle.nn.Layer
   :param data_loader: 用于评估模型的数据集的 dataloader。
   :type data_loader: paddle.io.DataLoader

   :return: 返回模型在输入数据上的平均的损失值和平均准确率。
   :rtype: Tuple[float, float]

.. py:function:: test(model, model_path, test_loader)

   使用测试集对模型进行测试。

   :param model: 训练得到的模型，用于被评估。
   :type model: paddle.nn.Layer
   :param model_path: 保存的模型参数的文件路径。
   :type model_path: str
   :param test_loader: 测试集的 dataloader。
   :type test_loader: paddle.io.DataLoader

.. py:function:: inference(image_path, is_dir, model_path, num_qubits, num_shadow, classes, depth)

   推理函数。使用训练好的模型对输入的图片进行预测。

   :param image_path: 要预测的图片的路径。
   :type image_path: str
   :param is_dir: 所输入的 ``image_path`` 是否为文件夹路径。如果是文件夹路径，则会对文件夹下的所有图片都进行预测。
   :type is_dir: bool
   :param model_path: 保存的模型参数的文件路径。
   :type model_path: str
   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param num_shadow: 影子电路所包含的量子比特的数量。
   :type num_shadow: int
   :param classes: 要预测的图片的类别。
   :type classes: list
   :param depth: 量子电路的深度，默认为 ``1`` 。
   :type depth: int

   :return: 返回模型预测的类别，以及模型对每个类别的置信度。
   :rtype: Tuple[int, list]
