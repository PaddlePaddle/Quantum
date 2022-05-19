paddle\_quantum.dataset
==============================

数据集的功能实现。

.. py:class:: Dataset()

      基类: :py:class:`object`

      所有数据集的基类，集成了多种量子编码方法。

   .. py:method:: data2circuit(classical_data, encoding, num_qubits, can_describe_dimension, split_circuit,return_state: bool, is_image=False)

      将输入的经典数据 ``classical_data`` 用编码方式 ``encoding`` 编码成量子态，这里的经典数据经过了截断或者补零，因而可以正好被编码。

      :param classical_data: 待编码的向量，经过了截断或者补零，刚好可以被编码。
      :type classical_data: list
      :param encoding: 编码方式，参见 MNIST 编码注释。
      :type encoding: str
      :param num_qubits: 量子比特数目。
      :type num_qubits: int
      :param can_describe_dimension:  ``encoding`` 编码方式可以编码的数目。
      :type can_describe_dimension: int
      :param split_circuit: 是否切分电路。
      :type split_circuit: bool
      :param return_state: 是否返回量子态。
      :type return_state: bool
      :param is_image: 是否是图片，如果是图片，归一化方法不同。
      :type is_image: bool, optional

      :raises Exception: 暂不支持返回 amplitude encoding 的电路。

      :return: 如果 ``return_state == True`` ，返回编码后的量子态，否则返回编码的电路。
      :rtype: list
   
   .. py:method:: filter_class(x, y, classes, data_num, need_relabel, seed=0)

      将输入的 ``x`` , ``y`` 按照 ``classes`` 给出的类别进行筛选，数目为 ``data_num``。

      :param x: 样本的特征。
      :type x: Union[list, np.ndarray]
      :param y: 样本标签。
      :type y: Union[list, np.ndarray]
      :param classes: 需要筛选的类别。
      :type classes: list
      :param data_num: 筛选出来的样本数目。
      :type data_num: int
      :param need_relabel: 将原有类别按照顺序重新标记为 0、1、2 等新的名字，比如传入 ``[1,2]`` ， 重新标记之后变为 ``[0,1]`` 主要用于二分类。
      :type need_relabel: bool
      :param seed: 随机种子，默认为 ``0``。
      :type seed: int, optional

      :return: 
         包含如下元素:

         - new_x: 筛选出的特征
         - new_y: 对应于 new_x 的标签
      :rtype: Tuple[list]


.. py:class:: VisionDataset(figure_size)
      
   基类: :py:class:`paddle_quantum.dataset.Dataset`

   图片数据集类，通过继承 VisionDataset 类，用户可以快速生成自己的图片量子数据。

   :param figure_size: 图片大小，也就是长和高的数值。
   :type figure_size: int

   .. py:method:: encode(feature, encoding, num_qubits, split_circuit, downscaling_method='resize', target_dimension=-1, return_state=True, full_return=False)

      将 ``feature`` 编码到 ``num_qubits`` 量子比特中，再降维到 ``target_dimension`` 后使用 ``encoding``。 ``feature`` 是一维的图像向量。

      :param feature: 一维图片向量组成的list/ndarray。
      :type feature: Union[list, np.ndarray]
      :param encoding: ``angle_encoding`` 表示角度编码，一个量子比特编码一个旋转门； ``amplitude_encoding`` 表示振幅编码； ``pauli_rotation_encoding`` 表示SU(3)的角度编码; 还有 ``linear_entangled_encoding`` , ``real_entangled_encoding`` , ``complex_entangled_encoding`` 和 ``IQP_encoding`` 编码。
      :type encoding: str
      :param num_qubits: 编码后的量子比特数目。
      :type num_qubits: int
      :param split_circuit: 是否需要切分电路。除了振幅之外的所有电路都会存在堆叠的情况，如果选择 ``True`` 就将块与块分开，默认为 ``False``。
      :type split_circuit: bool, optional
      :param downscaling_method: 包括 ``PCA`` 和 ``resize``, 默认为 ``resize``。
      :type downscaling_method: str, optional
      :param target_dimension: 降维之后的尺度大小，如果是 ``PCA`` ，不能超过图片大小；如果是 ``resize`` ，不能超过原图大小，默认为 ``-1``。
      :type target_dimension: int, optional
      :param return_state: 是否返回量子态，如果是 ``False`` 返回量子电路，默认为 ``True``。
      :type return_state: bool, optional
      :param full_return: 是否返回原始图像, 经典图像矢量, 量子态和量子电路, 默认为 ``False``。
      :type full_return: bool, optional

      :raises Exception: PCA维度应小于图片大小。
      :raises Exception: 调整大小的尺寸应该是一个平方数。
      :raises Exception: 缩小尺寸的方法只能是 resize 和 PCA。
      :raises Exception: 无效的编码方法。

      :return: 
         包含如下元素:

         - quantum_image_states: 量子态，只有 ``full_return==True`` 或者 ``return_state==True`` 的时候会返回。
         - quantum_image_circuits: 所有特征编码的电路，只有 ``full_return==False`` 或者 ``return_state==True`` 的时候会返回。
         - 图片经过类别过滤，但是还没有降维、补零的特征，是一个一维向量（可以 reshape 成图片），只有 ``return_state==True`` 的时候会返回。
         - 经过类别过滤和降维、补零等操作之后的特征，并未编码为量子态，只有 ``return_state==True`` 的时候会返回。
      :rtype: Tuple[paddle.Tensor, list, np.ndarray, np.ndarray]

.. py:class:: MNIST(mode, encoding, num_qubits, classes, data_num=-1, split_circuit=False, downscaling_method='resize', target_dimension=-1, need_cropping=True, need_relabel=True, return_state=True, seed=0)
      
   基类: :py:class:`paddle_quantum.dataset.VisionDataset`

   MNIST 数据集，它继承了 VisionDataset 图片数据集类。

   :param mode: 数据模式，包括 ``train`` 和 ``test``。
   :type mode: str
   :param encoding: ``angle_encoding`` 表示角度编码，一个量子比特编码一个旋转门； ``amplitude_encoding`` 表示振幅编码； ``pauli_rotation_encoding`` 表示SU(3)的角度编码; 还有 ``linear_entangled_encoding`` , ``real_entangled_encoding`` , ``complex_entangled_encoding`` 和 ``IQP_encoding`` 编码。
   :type encoding: str
   :param num_qubits: 编码后的量子比特数目。
   :type num_qubits: int
   :param classes: 用列表给出需要的类别，类别用数字标签表示。
   :type classes: list
   :param data_num: 使用的数据量大小, 默认为 ``-1``。
   :type data_num: int, optional
   :param split_circuit: 是否需要切分电路。除了振幅之外的所有电路都会存在堆叠的情况，如果选择 ``True`` 就将块与块分开。
   :type split_circuit: bool, optional
   :param downscaling_method: 包括 ``PCA`` 和 ``resize``。默认为 ``resize``。
   :type downscaling_method: str, optional
   :param target_dimension: 降维之后的尺度大小，如果是 ``PCA`` ，不能超过图片大小；如果是 ``resize`` ，不能超过原图大小。
   :type target_dimension: int, optional
   :param need_cropping: 是否需要裁边，如果为 ``True`` ，则从 ``image[0:27][0:27]`` 裁剪为 ``image[4:24][4:24]``。
   :type need_cropping: bool, optional
   :param need_relabel: 将原有类别按照顺序重新标记为 0, 1, 2 等新的名字，比如传入 ``[1,2]`` ，重新标记之后变为 ``[0,1]`` ，主要用于二分类。
   :type need_relabel: bool, optional
   :param return_state: 是否返回量子态，如果是 ``False`` 返回量子电路。
   :type return_state: bool, optional
   :param seed: 筛选样本的随机种子，默认为 ``0``。
   :type seed: int, optional

   :raises Exception: 数据模式只能为训练和测试。

.. py:class:: FashionMNIST(mode, encoding, num_qubits, classes, data_num=-1, split_circuit=False, downscaling_method, target_dimension=-1, need_relabel=True, return_state=True, seed=0)

   基类: :py:class:`paddle_quantum.dataset.VisionDataset`

   FashionMNIST 数据集，它继承了 ``VisionDataset`` 图片数据集类。

   :param mode: 数据模式，包括 ``train`` 和 ``test``。
   :type mode: str
   :param encoding: ``angle_encoding`` 表示角度编码，一个量子比特编码一个旋转门； ``amplitude_encoding`` 表示振幅编码； ``pauli_rotation_encoding`` 表示SU(3)的角度编码; 还有 ``linear_entangled_encoding`` , ``real_entangled_encoding`` , ``complex_entangled_encoding`` 和 ``IQP_encoding`` 编码。
   :type encoding: str
   :param num_qubits: 编码后的量子比特数目。
   :type num_qubits: int
   :param classes: 用列表给出需要的类别，类别用数字标签表示。
   :type classes: list
   :param data_num: 使用的数据量大小，默认为 ``-1``。
   :type data_num: int, optional
   :param split_circuit: 是否需要切分电路。除了振幅之外的所有电路都会存在堆叠的情况，如果选择 ``True`` 就将块与块分开， 默认为 ``False``。
   :type split_circuit: bool, optional
   :param downscaling_method: 包括 ``PCA`` 和 ``resize``，默认为 ``resize``。
   :type downscaling_method: str, optional
   :param target_dimension: 降维之后的尺度大小，如果是 ``PCA`` ，不能超过图片大小；如果是 ``resize`` ，不能超过原图大小， 默认为 ``-1``。
   :type target_dimension: int, optional
   :param need_relabel: 将原有类别按照顺序重新标记为 0, 1, 2 等新的名字，比如传入 ``[1,2]`` ，重新标记之后变为 ``[0,1]`` ，主要用于二分类， 默认为 ``True``。
   :type need_relabel: bool, optional
   :param return_state: 是否返回量子态，如果是 ``False`` 返回量子电路， 默认为 ``True``。
   :type return_state: bool, optional
   :param seed: 筛选样本的随机种子，默认为 ``0``。
   :type seed: int, optional

   :raises Exception: 数据模式只能为训练和测试。

.. py:class:: SimpleDataset(dimension)
      
   基类: :py:class:`paddle_quantum.dataset.Dataset`

   用于不需要降维的简单分类数据。用户可以通过继承 ``SimpleDataset`` ，将自己的分类数据变为量子态。下面的几个属性也会被继承。

   :param dimension: 编码数据的维度。
   :type dimension: int

   .. py:method:: encode(feature, encoding, num_qubits, return_state = True, full_return = False)

      用 ``num_qubits`` 的量子比特对 ``feature`` 进行编码 ``encoding``。

      :param feature: 编码的特征，每一个分量都是一个 ndarray 的特征向量。
      :type feature: Union[list, np.ndarray]
      :param encoding: 编码方法。
      :type encoding: str
      :param num_qubits: 编码的量子比特数目。
      :type num_qubits: int
      :param return_state: 是否返回量子态，默认为 ``True``。
      :type return_state: bool, optional
      :param full_return: 是否返回原始图像, 经典图像矢量, 量子态和量子电路, 默认为 ``False``。
      :type full_return: bool, optional
         
      :raises Exception: 无效特征类型。
      :raises Exception: 无效编码方式。
      :raises Exception: 量子比特数不足。

      :return: 
         包含如下元素:

         - quantum_states: 量子态，只有 ``full_return==True`` 或者 ``return_state==True`` 的时候会返回。
         - quantum_circuits: 所有特征编码的电路，只有 ``full_return==False`` 或者 ``return_state==True`` 的时候会返回。
         - origin_feature: 经过类别过滤之后的所有特征，并未编码为量子态，只有 ``return_state==True`` 的时候会返回。
         - feature: ``origin_feature`` 经过了补零之后的特征， ``quantum_states`` 就是将 ``feature`` 编码之后的结果。 只有 ``return_state==True`` 的时候会返回。
      :rtype: Tuple[np.ndarray, list, np.ndarray, np.ndarray]

.. py:class:: Iris(encoding: str, num_qubits: int, classes: list, test_rate: float=0.2, need_relabel=True, return_state=True, seed=0)
      
   基类: :py:class:`paddle_quantum.dataset.SimpleDataset`
 
   Iris 数据集。

   :param encoding: ``angle_encoding`` 表示角度编码，一个量子比特编码一个旋转门； ``amplitude_encoding`` 表示振幅编码； ``pauli_rotation_encoding`` 表示SU(3)的角度编码; 还有 ``linear_entangled_encoding`` , ``real_entangled_encoding`` , ``complex_entangled_encoding`` 和 ``IQP_encoding`` 编码。
   :type encoding: str
   :param num_qubits: 量子比特数目。
   :type num_qubits: int
   :param classes: 用列表给出需要的类别，类别用数字标签表示。
   :type classes: list
   :param test_rate: 测试集的占比, 默认为 ``0.2``。
   :type test_rate: float, optional
   :param need_relabel: 将原有类别按照顺序重标记为 0、1、2 等新的名字，比如传入 [1,2] ，重标记之后变为 [0,1] ，主要用于二分类。默认为 ``True``。
   :type need_relabel: bool, optional
   :param return_state: 是否返回量子态，默认为 ``True``。
   :type return_state: bool, optional
   :param seed: 筛选样本的随机种子，默认为 ``0``。
   :type seed: int, optional

.. py:class:: BreastCancer(encoding, num_qubits, test_rate=0.2, return_state=True, seed=0)

   基类: :py:class:`paddle_quantum.dataset.SimpleDataset`

   BreastCancer 数据集。

   :param encoding: ``angle_encoding`` 表示角度编码，一个量子比特编码一个旋转门； ``amplitude_encoding`` 表示振幅编码； ``pauli_rotation_encoding`` 表示SU(3)的角度编码; 还有 ``linear_entangled_encoding`` , ``real_entangled_encoding`` , ``complex_entangled_encoding`` 和 ``IQP_encoding`` 编码。
   :type encoding: _type_
   :param num_qubits: 量子比特数目。
   :type num_qubits: _type_
   :param test_rate: 测试集的占比, 默认为 ``0.2``。
   :type test_rate: float, optional
   :param return_state: 是否返回量子态，默认为 ``True``。
   :type return_state: bool, optional
   :param seed: 筛选样本的随机种子，默认为 ``0``。
   :type seed: int, optional
