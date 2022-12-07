paddle\_quantum.qml.vsql
==============================================
VSQL 模型。

.. py:function:: norm_image(images, num_qubits)

   对输入的图片进行归一化。先将图片展开为向量，再进行归一化。

   :param images: 输入的图片。
   :type images: List[np.ndarray]
   :param num_qubits: 量子比特的数量，决定了归一化向量的维度。
   :type num_qubits: int

   :return: 返回归一化之后的向量，它是由 ``paddle.Tensor`` 组成的列表。
   :rtype: List[paddle.Tensor]

.. py:function:: data_loading(num_qubits, mode, classes, num_data)

   加载 MNIST 数据集，其中只包含指定的数据。

   :param num_qubits: 量子比特的数量，决定了归一化向量的维度。
   :type num_qubits: int
   :param mode: 指定要加载的数据集，为 ``'train'`` 或 ``'test'`` 。

      - ``'train'`` ：表示加载训练集。
      - ``'test'`` ：表示加载测试集。

   :type mode: str
   :param classes: 要加载的数据的标签。对应标签的数据会被加载。
   :type classes: list
   :param num_data: 要加载的数据的数量。默认为 ``None`` ，加载所有数据。
   :type num_data: Optional[int]

   :return: 返回加载的数据集，其组成为 ``(images, labels)`` 。
   :rtype: Tuple[List[np.ndarray], List[int]]

.. py:function:: observable(start_idx, num_shadow)

   生成测量量子态所需要的哈密顿量。

   :param start_idx: 要测量的量子比特的起始索引。
   :type start_idx: int
   :param num_shadow: 影子电路所包含的量子比特的数量。
   :type num_shadow: int

   :return: 返回生成的哈密顿量。
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

      :param batch_input: 模型的输入，其形状为 :math:`(\text{batch_size}, 2^{\text{num_qubits}})` 。
      :type batch_input: List[paddle.Tensor]

      :return: 返回模型的输出，其形状为 :math:`(\text{batch_size}, \text{num_classes})` 。
      :rtype: paddle.Tensor

.. py:function:: train(num_qubits, num_shadow, depth, batch_size, epoch, learning_rate, classes, num_train, num_test)

   训练 VSQL 模型。

   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param num_shadow: 影子电路所包含的量子比特的数量。
   :type num_shadow: int
   :param depth: 量子电路的深度，默认为 ``1`` 。
   :type depth: Optional[int]
   :param batch_size: 数据的批大小，默认为 ``16`` 。
   :type batch_size: Optional[int]
   :param epoch: 训练的轮数，默认为 ``10`` 。
   :type epoch: Optional[int]
   :param learning_rate: 更新参数的学习率，默认为 ``0.01`` 。
   :type learning_rate: Optional[float]
   :param classes: 要预测的手写数字的类别。默认为 ``None`` ，即预测所有的类别。
   :type classes: Optional[list]
   :param num_train: 训练集的数据量。默认为 ``None`` ，即使用所有的训练数据。
   :type num_train: Optional[int]
   :param num_test: 测试集的数据量。默认为 ``None`` ，即使用所有的训练数据。
   :type num_test: Optional[int]
