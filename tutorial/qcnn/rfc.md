81 时间演化电路的性能优化 #17

【任务说明】

任务标题：时间演化电路的性能优化
技术标签：量子计算、哈密顿量
任务难度：中等
详细描述：
哈密顿量模拟，指的是模拟一个量子系统随时间演化的过程。根据量子力学的基本公理，对于不含时的哈密顿量而言，系统的时间演化过程可以由算符 exp(-iHt) 进行描述。目前，量桨中实现了基于 product formula 的数字化哈密顿量模拟，可以根据泡利哈密顿量来创建相应的模拟时间演化电路。在这个任务中，你需要实现对时间演化电路的性能优化。目前，该模块的实现方法是对于泡利哈密顿量中的每一项分别搭建一个旋转电路，其具体方法可以参考 1 中的 4.7.3 节。实际上，对于一些特殊的两量子比特项而言，文献 2 提出了更加高效的电路。因此，你可以将该文献中描述的 special case 进行单独的实现，并将相关代码合入 paddle_quantum.trotter.construct_trotter_circuit() 函数中。

注：对于哈密顿量模拟更加详细的介绍，可以参考量桨官网上的教程：利用 Product Formula 模拟时间演化、模拟一维海森堡链的自旋动力学。

任务要求：

编写单独的函数，使其可以实现文献 2 中提到的量子电路
在搭建时间演化电路时，检测出哈密顿量中可以高效模拟的项并进行单独处理
利用实际系统的哈密顿量对该功能进行验证，确保结果正确
参考资料：

Nielsen, Michael A., and Isaac L. Chuang. "Quantum computing and quantum information." (2000).
Vatan, Farrokh, and Colin Williams. "Optimal quantum circuits for general two-qubit gates." Physical Review A 69.3 (2004): 032315.

82 基于量子卷积神经网络的图片分类 #18

【任务说明】

任务标题：基于量子卷积神经网络 (QCNN) 的图片分类
技术标签：量子卷积神经网络
任务难度：困难
详细描述：
众所周知，卷积神经网络 (CNN) 在图像识别等问题上表现十分出色，受到 CNN 的启发 QCNN 被提出（参考 1）。CNN 核心的操作是卷积和池化，对于 QCNN 可以考虑利用参数化量子电路或者随机电路代替卷积和池化操作。关于 QCNN 的形式有很多（参考 2），目前还处于探索阶段。在这个任务中，你需要尝试实现基于量子卷积神经网络的图片分类。

任务要求：根据参考文献 3 和其它参考文献，形成一篇 QCNN 的教程，教程包括背景知识、方法介绍和代码实验等，具体形式可参考量桨已有的教程（参考资料 4）。

参考资料：

Cong, I., Choi, S. & Lukin, M.D. Quantum convolutional neural networks. Nat. Phys. 15, 1273–1278. (2019) https://doi.org/10.1038/s41567-019-0648-8
Maxwell Henderson, Samriddhi Shakya, Shashindra Pradhan, Tristan Cook. “Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits.”(2019) arXiv:1904.04767
Yanxuan Lü, Qing Gao, Jinhu Lü, Maciej Ogorza?ek, Jin Zheng.A Quantum Convolutional Neural Network for Image Classification.(2021) arxiv:2107.03630
https://qml.baidu.com/tutorials/overview.html
【提交内容】

项目PR到 Quantum
相关技术文档
项目单测文件