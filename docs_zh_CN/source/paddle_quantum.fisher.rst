paddle\_quantum.fisher
=============================

Fisher 信息的功能实现。

.. py:class:: QuantumFisher(cir)

   量子费舍信息及相关量的计算器。

   :param cir: 需要计算量子费舍信息的参数化量子电路。
   :type cir: Circuit

   .. note:: 

      该类利用了二阶参数平移规则计算量子费舍信息矩阵，因此不能直接适用于单比特旋转门的参数间有依赖关系的情况，例如受控旋转门等。

   .. py:method:: get_qfisher_matrix()

      :return: 量子费舍信息矩阵
      :rtype: np.ndarray

   .. code-block:: python

      import paddle
      from paddle_quantum.ansatz import Circuit
      from paddle_quantum.fisher import QuantumFisher

      cir = Circuit(1)
      zero = paddle.zeros([1], dtype="float64")
      cir.ry(param=zero)
      cir.rz(param=zero)

      qf = QuantumFisher(cir)
      qfim = qf.get_qfisher_matrix()
      print(f'The QFIM at {cir.param.tolist()} is \n {qfim}.')

   ::

      The QFIM at [0.0, 0.0] is
      [[1. 0.]
      [0. 0.]].

   .. py:method:: get_qfisher_norm(direction, step_size = 0.01)

      :param direction: 要计算量子费舍信息投影的方向。
      :type direction: np.ndarray
      :param step_size: 有限差分的步长，默认为 ``0.01``。
      :type step_size: float

      :return: 沿给定方向的量子费舍信息的投影
      :rtype: float

   .. code-block:: python

      import paddle
      from paddle_quantum.ansatz import Circuit
      from paddle_quantum.fisher import QuantumFisher

      cir = Circuit(2)
      zero = paddle.zeros([1], dtype="float64")
      cir.ry(0, param=zero)
      cir.ry(1, param=zero)
      cir.cnot(qubits_idx=[0, 1])
      cir.ry(0, param=zero)
      cir.ry(1, param=zero)

      qf = QuantumFisher(cir)
      v = [1,1,1,1]
      qfi_norm = qf.get_qfisher_norm(direction=v)
      print(f'The QFI norm along {v} at {cir.param.tolist()} is {qfi_norm:.7f}')

   ::

      The QFI norm along [1, 1, 1, 1] at [0.0, 0.0, 0.0, 0.0] is 6.0031546

   .. py:method:: get_eff_qdim(num_param_samples = 4, tol = None)

      :param num_param_samples: 用来估计有效量子维数时所用的参数样本量，默认为 4。
      :type num_param_samples: int
      :param tol: 奇异值的最小容差，低于此容差的奇异值认为是 0, 默认为 None, 其含义同 ``numpy.linalg.matrix_rank()``
      :type tol: float

      :return: 给定量子电路对应的有效量子维数。
      :rtype: int

   .. code-block:: python

      import paddle
      from paddle_quantum.ansatz import Circuit
      from paddle_quantum.fisher import QuantumFisher

      cir = Circuit(1)
      cir.rz(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False), which_qubit=0)
      cir.ry(theta=paddle.to_tensor(0., dtype="float64", stop_gradient=False), which_qubit=0)

      qf = QuantumFisher(cir)
      print(cir)
      print(f'The number of parameters of -Rz-Ry- is {len(cir.get_param().tolist())}')
      print(f'The effective quantum dimension -Rz-Ry- is {qf.get_eff_qdim()}')

   ::

      --Rz(0.000)----Ry(0.000)--

      The number of parameters of -Rz-Ry- is 2
      The effective quantum dimension -Rz-Ry- is 1

   .. py:method:: get_qfisher_rank(tol)
   
      :param tol: 奇异值的最小容差，低于此容差的奇异值认为是 0, 默认为 None, 其含义同 ``numpy.linalg.matrix_rank()``
      :type tol: float

      :return: 量子费舍信息矩阵的秩
      :rtype: int

.. py:class:: ClassicalFisher(model, num_thetas, num_inputs, model_type = 'quantum', **kwargs)

   :param model: 经典或量子神经网络模型的实例
   :type model: paddle.nn.Layer
   :param num_thetas: 参数集合的数量
   :type num_thetas: int
   :param num_inputs: 输入的样本数量
   :type num_inputs: int
   :param model_type: 模型是经典 ``classical`` 的还是量子 ``quantum`` 的，默认是量子的
   :type model_type: str
   :param \*\*kwargs: 神经网络参数, 包含如下选项:

      - size (list): 经典神经网络各层神经元的数量 \\
      - num_qubits (int): 量子神经网络量子比特的数量 \\
      - depth (int): 量子神经网络的深度 \\
      - encoding (str): 量子神经网络中经典数据的编码方式，目前支持 ``IQP`` 和 ``re-uploading``
      
   :type \*\*kwargs: Union[List[int], int, str]
 
   .. py:method:: get_gradient(x)

      计算输出层关于变分参数的梯度

      :param x: 输入样本
      :type x: Union[np.ndarray, paddle.Tensor]

      :return: 输出层关于变分参数的梯度，数组形状为（输入样本数量, 输出层维数, 变分参数数量）
      :rtype: np.ndarray

   .. py:method:: get_cfisher(gradients)

      利用雅可比矩阵计算经典费舍信息矩阵

      :param gradients: 输出层关于变分参数的梯度, 数组形状为（输入样本数量, 输出层维数, 变分参数数量）
      :type gradients: np.ndarray

      :return: 经典费舍信息矩阵，数组形状为（输入样本数量, 变分参数数量, 变分参数数量）
      :rtype: np.ndarray

   .. py:method:: get_normalized_cfisher()

      计算归一化的经典费舍信息矩阵

      :return:
         包含如下元素：

         - 归一化的经典费舍信息矩阵，数组形状为（输入样本数量, 变分参数数量, 变分参数数量）
         - 其迹

      :rtype: Tuple[np.ndarray, float]

   .. py:method:: get_eff_dim(normalized_cfisher, list_num_samples, gamma = 1)

      计算经典的有效维数

      :param normalized_cfisher: 归一化的经典费舍信息矩阵
      :type normalized_cfisher: np.ndarray
      :param list_num_samples: 不同样本量构成的列表
      :type list_num_samples: List[int]
      :param gamma: 有效维数定义中包含的一个人为可调参数，默认为 ``1``.
      :type gamma: int
      
      :return: 对于不同样本量的有效维数构成的列表
      :rtype: List[int]
      