.. _header-n0:

Paddle Quantum （量桨）
=======================

`Paddle Quantum（量桨） <https://github.com/PaddlePaddle/Quantum>`__\ 是基于百度飞桨开发的量子机器学习工具集，支持量子神经网络的搭建与训练，提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，使得百度飞桨也因此成为国内首个目前也是唯一一个支持量子机器学习的深度学习框架。

.. figure:: https://release-data.cdn.bcebos.com/Paddle%20Quantum.png
   :target: https://github.com/PaddlePaddle/Quantum

量桨建立起了人工智能与量子计算的桥梁，不但可以快速实现量子神经网络的搭建与训练，还提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，并提供多项自研量子机器学习应用。通过百度飞桨深度学习平台赋能量子计算，量桨为领域内的科研人员以及开发者便捷地开发量子人工智能的应用提供了强有力的支撑，同时也为广大量子计算爱好者提供了一条可行的学习途径。

    关于量桨的更多内容可以查看 GitHub 页面：https://github.com/PaddlePaddle/Quantum

.. _header-n6:

特色
----

- 易用性

  - 高效搭建量子神经网络
  - 多种量子神经网络模板
  - 丰富的量子算法教程（10+用例）

- 可拓展性

  - 支持通用量子电路模型
  - 高性能模拟器支持20多个量子比特的模拟运算
  - 提供多种优化工具和 GPU 加速

- 特色工具集

  - 提供组合优化和量子化学等前沿领域的计算工具箱
  - 自研多种量子机器学习算法

.. _header-n15:

安装步骤
--------

.. _header-n16:

安装 PaddlePaddle
~~~~~~~~~~~~~~~~~

请参考
`PaddlePaddle <https://www.paddlepaddle.org.cn/install/quick>`__
安装配置页面。此项目需求 PaddlePaddle 1.8.3 或更高版本。

.. _header-n19:

下载 Paddle Quantum 并安装
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   git clone http://github.com/PaddlePaddle/quantum

.. code:: shell

   cd quantum
   pip install -e .

.. _header-n23:

或使用 requirements.txt 安装依赖包
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

   python -m pip install --upgrade -r requirements.txt

.. _header-n25:

使用 openfermion 读取 xyz 描述文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: 仅在 macOS 和 linux 下可以使用 openfermion 读取 xyz 描述文件。

VQE中调用 openfermion 读取分子 xyz 文件并计算，因此需要安装 openfermion 和
openfermionpyscf。

.. code:: shell

   pip install openfermion
   pip install openfermionpyscf

.. _header-n29:

运行
~~~~

现在，可以试着运行一段程序来验证量桨是否已安装成功。这里我们运行量桨提供的量子近似优化算法
(QAOA) 的例子。

.. code:: shell

   cd paddle_quantum/QAOA/example
   python main.py

..

.. note:: 关于 QAOA 的介绍可以参考我们的 `QAOA 教程 <https://github.com/PaddlePaddle/Quantum/blob/master/tutorial/QAOA>`__。

.. _header-n51:

交流与反馈
----------

- 我们非常欢迎您通过 `Github
  Issues <https://github.com/PaddlePaddle/Quantum/issues>`__
  来提交问题、报告与建议。
- 技术交流QQ群：1076223166

.. _header-n118:

Copyright and License
---------------------

Paddle Quantum 使用 `Apache-2.0 license <https://github.com/PaddlePaddle/Quantum/blob/master/LICENSE>`__ 许可证。
