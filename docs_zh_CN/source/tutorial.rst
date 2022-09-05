入门与教程
=======================

我们准备了入门手册和案例教程，来帮助用户快速学会如何使用量桨（Paddle Quantum）。

.. _header-n33:

入门手册
--------

我们提供了一份 `Paddle Quantum 入门手册 </quick-start/overview.html>`__\ 来方便用户快速上手
Paddle Quantum。目前支持网页阅览和\ `下载运行 Jupyter Notebook <https://github.com/PaddlePaddle/Quantum/tree/master/introduction>`__ 
两种方式。内容上，该手册包括以下几个方面：

- 量子计算和量子神经网络的基础知识介绍
- 变分量子算法的基本思想与算法框架
- 量桨（Paddle Quantum）的使用介绍
- 飞桨（PaddlePaddle）优化器的使用教程
- 量桨中量子化学模块的使用介绍
- 如何基于 GPU 训练量子神经网络

案例教程
--------

我们提供了涵盖量子模拟、机器学习、组合优化、本地操作与经典通讯（local operations and classical communication, LOCC）、量子神经网络等多个领域的案例供大家学习。与\ `入门手册 </quick-start/overview.html>`__\ 类似，每个教程目前支持
\ `网页阅览 </tutorials/overview.html>`__\ 和\ `下载运行 Jupyter Notebook <https://github.com/PaddlePaddle/Quantum/tree/master/tutorials>`__\ 两种方式。我们推荐用户下载 Notebook
后，本地运行进行实践。

- `量子模拟 <https://github.com/PaddlePaddle/Quantum/blob/master/tutorials/quantum_simulation>`__
- `机器学习 <https://github.com/PaddlePaddle/Quantum/blob/master/tutorials/machine_learning>`__
- `组合优化 <https://github.com/PaddlePaddle/Quantum/blob/master/tutorials/combinatorial_optimization>`__
- `LOCCNet <https://github.com/PaddlePaddle/Quantum/blob/master/tutorials/locc>`__
- `量子神经网络研究 <https://github.com/PaddlePaddle/Quantum/blob/master/tutorials/qnn_research>`__

随着 LOCCNet 模组的推出，量桨现已支持分布式量子信息处理任务的高效模拟和开发。感兴趣的读者请参见 `教程 </tutorials/loccnet/loccnet-framework.html>`__。
Paddle Quantum 也支持在 GPU
上进行量子机器学习的训练，具体的方法请参考案例：`在 GPU 上使用 Paddle
Quantum </quick-start/use-paddle-quantum-on-gpu.html>`__。
此外，量桨可以基于噪声模块进行含噪算法的开发以及研究，详情请见 `噪声模块教程 </tutorials/qnn-research/simulating-noisy-quantum-circuits-with-paddle-quantum.html>`__。

在最近的更新中，量桨还加入了基于测量的量子计算（measurement-based
quantum computation, MBQC）模块。与传统的量子电路模型不同，MBQC
具有其独特的运行方式，感兴趣的读者请参见我们提供的\ `多篇教程 </tutorials/measurement-based-quantum-computation/measurement-based-quantum-computation-module.html>`__\ 以了解量桨
MBQC 模块的使用方法和应用案例。
