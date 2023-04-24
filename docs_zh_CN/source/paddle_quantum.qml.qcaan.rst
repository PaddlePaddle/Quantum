paddle\_quantum.qml.qcaan
==============================================

量子电路关联对抗网络（Quantum-Circuit Associative Adversarial Network, QCAAN）模型

.. py:function:: Data_Load()
   
   加载 MNIST 数据集

   :return: MNIST 数据集
   :rtype: paddle.vision.datasets.MNIST


.. py:class:: ConvBlock(shape, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation=None, normalize=True)
    
   基类：:py:class:`paddle.nn.Layer`
   
   建立神经网络的卷积块。
   每一个卷积块包含若干卷积层、Silu 激活层、层归一化层；并且卷积块保持空间维度不变，即图片的宽和高不变，只改变通道数。
   
   :param shape: 层归一化的形状
   :type shape: List[int]
   :param in_channels: 输入通道数
   :type in_channels: int
   :param out_channels: 输出通道数
   :type out_channels: int
   :param kernel_size: 卷积核的大小
   :type kernel_size: int
   :param stride: stride 的规模
   :type stride: int
   :param padding: padding 的规模
   :type padding: int 
   :param activation: 激活函数
   :type activation: nn.Layer
   :param normalize: 指示是否使用层归一化的标记
   :type normalize: bool

   .. py:method:: forward(x)
   
      前向执行函数

      :param x: 输入张量
      :type x: paddle.Tensor
      
      :return: 输出张量
      :rtype: paddle.Tensor

.. py:class:: Generator(latent_dim=16)
   
   基类：:py:class:`paddle.nn.Layer`
   
   生成器网络
   
   :param latent_dim: 隐藏特征维度
   :type latent_dim: int
   
   .. py:method:: forward(x)
   
      前向执行函数

      :param x: 输入张量
      :type x: paddle.Tensor
      
      :return: 输出张量
      :rtype: paddle.Tensor

.. py:class:: Discriminator(latent_dim=16)
   
   基类：:py:class:`paddle.nn.Layer`
   
   判别器网络
   
   :param latent_dim: 隐藏特征维度
   :type latent_dim: int
   
   .. py:method:: forward(x)
   
      前向执行函数

      :param x: 输入张量
      :type x: paddle.Tensor
      
      :return: 输出张量
      :rtype: paddle.Tensor


.. py:function:: generate_pauli_string_list(num_qubits, num_terms)

   生成测量量子态所需要的可观测量。

   :param num_qubits: 量子比特的数量。
   :type num_qubits: int
   :param num_terms: 生成的可观测量的项数。
   :type num_terms: int

   :return: 返回生成的可观测量。
   :rtype: List[list]
   
   
.. py:class:: QCBM(num_qubits, num_depths, latent_dim=16)
   
   基类：:py:class:`paddle.nn.Layer`
   
   量子玻尔兹曼机，这里即等价于量子神经网络。
   
   :param num_qubits: 量子比特的数量。
   :type num_qubits: int
   :param num_depths: complex entangled layers 的层数
   :type num_depths: int
   :param latent_dim: 隐藏特征维度
   :type latent_dim: int
   
   .. py:method:: forward()
   
      前向执行函数

      :return: 输出在一系列哈密顿量上的观测值，即在 Z0, Z1, ..., X0, X1, ..., Y0, Y1, ...
      :rtype: paddle.Tensor
   
   
.. py:function:: prior_sampling(expec_obs, batch_size)

   模拟从 QCBM 中采样先验概率的过程。

   :param expec_obs: 一系列观测值组成的向量，长度等于隐藏特征维度
   :type expec_obs: paddle.Tensor
   :param batch_size: 一个 batch 中的样本数
   :type batch_size: int

   :return: 采样结果，值在 {-1, 1}中。
   :rtype: paddle.Tensor
   
   
.. py:function:: train(num_qubits = 8, num_depths = 4, lr_qnn = 0.005, batch_size = 128, latent_dim = 16, epochs= 21, lr_g = 0.0002, lr_d = 0.0002, beta1 = 0.5, beta2 = 0.9, manual_seed = 888)
   :noindex:
   
   训练 QCAAN 模型的函数。

   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param num_depths: complex entangled layers 的层数。
   :type num_depths: int
   :param lr_qnn: 更新QNN中参数的学习率。
   :type lr_qnn: paddle.float32
   :param batch_size: 每一步迭代中的批大小。
   :type batch_size: int
   :param latent_dim: 隐藏特征维度。
   :type latent_dim: int
   :param epochs: 训练模型需要的 epoch 数目。
   :type epochs: int
   :param lr_g: 更新生成器的学习率。
   :type lr_g: paddle.float32
   :param lr_d: 更新判别器的学习率。
   :type lr_d: paddle.float32
   :param beta1: 用于生成器和判别器的Adam 中的 beta1 参数。
   :type beta1: paddle.float32
   :param beta2: 用于生成器和判别器的Adam 中的 beta2 参数。
   :type beta2: paddle.float32
   :param manual_seed: 用于可复现的人工种子。
   :type manual_seed: int

   
.. py:function:: model_test(latent_dim = 16, params_path = "params", num_qubits = 8, num_depths = 4, manual_seed = 20230313,)

   加载训练好的 QCAAN 模型参数，生成一些新图。最后存储这些图片。

   :param latent_dim: 隐藏特征维度
   :type latent_dim: int
   :param params_path: 加载模型参数的路径。
   :type params_path: str
   :param num_qubits: 量子电路所包含的量子比特的数量。
   :type num_qubits: int
   :param num_depths: complex entangled layers 的层数。
   :type num_depths: int
   :param manual_seed: 用于可复现的人工种子。
   :type manual_seed: int 
   
   
   
