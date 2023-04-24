.. _header-n0:

Paddle Quantum（量桨）
=======================

`Paddle Quantum（量桨） <https://github.com/PaddlePaddle/Quantum>`__\ 是基于百度飞桨研发的全球首个云量一体的量子机器学习平台。量桨支持量子神经网络的搭建与训练等功能，使得百度飞桨也因此成为国内首个支持量子机器学习的深度学习框架。量桨具备轻松上手、功能丰富等特点，提供了完善的API文档和用例教程，使用户可以快速入门和上手。

.. figure:: https://release-data.cdn.bcebos.com/Paddle%20Quantum.png
   :target: https://github.com/PaddlePaddle/Quantum

量桨建立起了人工智能与量子计算的桥梁，通过百度飞桨深度学习平台赋能量子计算，为领域内的科研人员以及开发者便捷地开发量子人工智能的应用提供了强有力的支撑，同时也为广大量子计算爱好者提供了一条可行的学习途径。

    关于量桨的更多内容可以查看 GitHub 页面：https://github.com/PaddlePaddle/Quantum

.. _header-n6:

特色
----

- 轻松上手

  - 丰富的在线学习资源（近 50 篇教程案例）
  - 通过模板高效搭建量子神经网络
  - 自动微分框架

- 功能丰富

  - 提供多种优化工具和 GPU 模式
  - 高性能模拟器支持25+量子比特的模拟运算
  - 支持多种噪声模型的模拟

- 特色工具集

  - 提供组合优化和量子化学等前沿领域的计算工具箱
  - 分布式量子信息处理模组 LOCCNet
  - 自研多种量子机器学习算法

.. _header-n15:

安装步骤
--------

.. _header-n16:

安装 PaddlePaddle
~~~~~~~~~~~~~~~~~

当用户安装 Paddle Quantum 时会自动下载安装这个关键依赖包。关于 PaddlePaddle 更全面的安装信息请参考
`PaddlePaddle <https://www.paddlepaddle.org.cn/install/quick>`__
安装配置页面。此项目需求 PaddlePaddle 2.2.0 到 2.3.0。

.. _header-n19:

安装 Paddle Quantum
~~~~~~~~~~~~~~~~~~~~~~~~~~

我们推荐通过 ``pip`` 完成安装,

.. code-block:: shell

   pip install paddle-quantum

用户也可以选择下载全部文件后进行本地安装，

.. code-block:: shell

   git clone https://github.com/PaddlePaddle/quantum

.. code-block:: shell

   cd quantum
   pip install -e .

.. _header-n25:

量子化学模块的环境设置
~~~~~~~~~~~~~~~~~~~~~~

我们的量子化学模块是基于 ``Psi4``
进行开发的，所以在运行量子化学模块之前需要先行安装该 Python 包。

.. note::  

   推荐在 Python3.8 环境中安装。

在安装 ``psi4`` 时，我们建议您使用 conda。对于 **MacOS/Linux**
的用户，可以使用如下指令。

.. code-block:: shell

   conda install psi4 -c psi4

对于 **Windows** 用户，请使用

.. code-block:: shell

   conda install psi4 -c psi4 -c conda-forge

**注意：** 更多的下载方法请参考
`Psi4 <https://psicode.org/installs/v14/>`__\ 。

.. _header-n29:

运行
~~~~

现在，可以试着运行一段程序来验证量桨是否已安装成功。这里我们运行量桨提供的量子近似优化算法（QAOA）的例子。

.. code-block:: shell

   cd paddle_quantum/QAOA/example
   python main.py

..

.. note:: 关于 QAOA 的介绍可以参考我们的 `QAOA 教程 </tutorials/combinatorial-optimization/quantum-approximate-optimization-algorithm.html>`__。

.. _header-n51:

交流与反馈
----------

- 我们非常欢迎您通过 `GitHub
  Issues <https://github.com/PaddlePaddle/Quantum/issues>`__
  来提交问题、报告与建议。
- 技术交流QQ群：1076223166

.. _header-n118:

Copyright and License
---------------------

Paddle Quantum 使用 `Apache-2.0 license <https://github.com/PaddlePaddle/Quantum/blob/master/LICENSE>`__ 许可证。
