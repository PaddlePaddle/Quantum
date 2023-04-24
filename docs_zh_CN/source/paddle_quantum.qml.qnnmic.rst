paddle\_quantum.qml.qnnmic
==============================================
QNNMIC 模型。


.. py:class:: QNNMIC(num_qubits, num_depths, observables)

   基类：:py:class:`paddle.nn.Layer`

   基于量子神经网络进行医学图片分类。

   :param num_qubits: 每层量子电路的比特数。
   :type num_qubits: List[int]
   :param num_depths: 每层参数化部分的电路深度。
   :type num_depths: List[int]
   :param observables: 每层电路的测量算子。
   :type observables: List

   .. py:method:: forward(batch_input)

      :param batch_input: 模型的输入，其形状为 :math:`(\text{batch_size}, -1)` 。
      :type batch_input: List[paddle.Tensor]

      :return: 返回模型的输出，其形状为 :math:`(\text{batch_size}, \text{num_classes})` 。
      :rtype: paddle.Tensor

.. py:function:: train(model_name, num_qubits, num_depths, observables, batch_size: int=20, num_epochs: int=4, learning_rate: float=0.1, dataset: str='SurfaceCrack', saved_dir: str='./', using_validation: bool=False, num_train: int=-1, num_val: int=-1, num_test: int=-1)
   :noindex:

   训练 QNNMIC 模型。

   :param model_name: 模型的名字，用于保存模型。
   :type model_name: str
   :param num_qubits: 每层量子电路的比特数。
   :type num_qubits: List[int]
   :param num_depths: 每层参数化部分的电路深度。
   :type num_depths: List[int]
   :param observables: 每层电路的测量算子。
   :type observables: List
   :param batch_size: 数据的批大小，默认为 ``20`` 。
   :type batch_size: Optional[int]
   :param num_epochs: 训练的轮数，默认为 ``4`` 。
   :type epoch: Optional[int]
   :param learning_rate: 更新参数的学习率，默认为 ``0.1`` 。
   :type learning_rate: Optional[float]
   :param dataset: 需要使用的数据集，默认为 ``SurfaceCrack``。
   :type dataset: str
   :param saved_dir: 日志文件保存的路径，默认为 ``./``。
   :type saved_dir: str
   :param using_validation: 是否使用验证集，默认为 ``False``。
   :type using_validation: bool
   :param num_train: 训练集的数据量。默认为 ``-1`` ，即使用所有的训练数据。
   :type num_train: Optional[int]
   :param num_val: 验证集的数据量。默认为 ``-1`` ，即使用所有的训练数据。
   :type num_val: Optional[int]
   :param num_test: 测试集的数据量。默认为 ``-1`` ，即使用所有的训练数据。
   :type num_test: Optional[int]

.. py:function:: inference(image_path: str, num_samples: int, model_path: str, num_qubits: List, num_depths: List, observables: List)
   :noindex:
   
    使用 QNNMIC 模型进行推理。

   :param image_path: 需要推理的数据集。
   :type image_path: str
   :param num_samples: 需要推理数据集中图片的数量。
   :type num_samples: Optional[int]
   :param model_path: 推理使用的模型。
   :type model_path: str
   :param num_qubits: 每层量子电路的比特数。
   :type num_qubits: List[int]
   :param num_depths: 每层参数化部分的电路深度。
   :type num_depths: List[int]
   :param observables: 每层电路的测量算子。
   :type observables: List