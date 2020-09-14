# Paddle Quantum （量桨）

- [特色](#特色)
- [安装步骤](#安装步骤)
   - [安装 PaddlePaddle](#安装-paddlepaddle)
   - [下载 Paddle Quantum 并安装](#下载-paddle-quantum-并安装)
   - [或使用 requirements.txt 安装依赖包](#或使用-requirementstxt-安装依赖包)
   - [使用 openfermion 读取 xyz 描述文件](#使用-openfermion-读取-xyz-描述文件)
   - [运行](#运行)
- [入门与开发](#入门与开发)
   - [教程入门](#教程入门)
   - [案例入门](#案例入门)
   - [API 文档](#api-文档)
   - [开发](#开发)
- [交流与反馈](#交流与反馈)
- [使用 Paddle Quantum 的工作](#使用-paddle-quantum-的工作)
- [FAQ](#faq)
- [Copyright and License](#copyright-and-license)
- [References](#references)

Paddle Quantum（量桨）是基于百度飞桨开发的量子机器学习工具集，支持量子神经网络的搭建与训练，提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，使得百度飞桨也因此成为国内首个目前也是唯一一个支持量子机器学习的深度学习框架。

![](https://release-data.cdn.bcebos.com/Paddle%20Quantum.png)

量桨建立起了人工智能与量子计算的桥梁，不但可以快速实现量子神经网络的搭建与训练，还提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，并提供多项自研量子机器学习应用。通过百度飞桨深度学习平台赋能量子计算，量桨为领域内的科研人员以及开发者便捷地开发量子人工智能的应用提供了强有力的支撑，同时也为广大量子计算爱好者提供了一条可行的学习途径。

## 特色

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

## 安装步骤

### 安装 PaddlePaddle

请参考 [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick) 安装配置页面。此项目需求 PaddlePaddle 1.8.3 或更高版本。

### 下载 Paddle Quantum 并安装

```bash
git clone http://github.com/PaddlePaddle/quantum
```

```bash
cd quantum
pip install -e .
```

### 或使用 requirements.txt 安装依赖包

```bash
python -m pip install --upgrade -r requirements.txt
```

### 使用 openfermion 读取 xyz 描述文件

> 仅在 macOS 和 linux 下可以使用 openfermion 读取 xyz 描述文件。

VQE中调用 openfermion 读取分子 xyz 文件并计算，因此需要安装 openfermion 和 openfermionpyscf。

```bash
pip install openfermion
pip install openfermionpyscf
```

### 运行

现在，可以试着运行一段程序来验证量桨是否已安装成功。这里我们运行量桨提供的量子近似优化算法 (QAOA) 的例子。

```bash
cd paddle_quantum/QAOA/example
python main.py
```

> 关于 QAOA 的介绍可以参考我们的 [QAOA 教程](./tutorial/QAOA)。

## 入门与开发

### 教程入门

量子计算是由量子力学与计算理论交叉而成的全新计算模型，具有强大的信息处理优势和广阔的应用前景，被视作未来计算技术的心脏。量子计算的相关介绍与入门知识可以参考 [1-3]。

量子机器学习是一门结合量子计算与机器学习的交叉学科，一方面利用量子计算的信息处理优势促进人工智能的发展，另一方面也利用现有的人工智能的技术突破量子计算的研发瓶颈。关于量子机器学习的入门资料可以参考 [4-6]。

这里，我们提供了一份[**入门手册**](./introduction)方便用户快速上手 Paddle Quantum。目前支持 PDF 阅读和运行 Jupyter Notebook 两种方式。内容上，该手册包括以下几个方面：

- Paddle Quantum 的详细安装教程
- 量子计算的基础知识介绍
- Paddle Quantum 的使用介绍
- PaddlePaddle 飞桨优化器使用教程
- 具体的量子机器学习案例—VQE

### 案例入门

Paddle Quantum（量桨）建立起了人工智能与量子计算的桥梁，为量子机器学习领域的研发提供强有力的支撑，也提供了丰富的案例供开发者学习。

在这里，我们提供了涵盖量子优化、量子化学、量子机器学习等多个领域的案例供大家学习。与[入门手册](./introduction)类似，每个教程目前支持 PDF 阅读和运行 Jupyter Notebook 两种方式。我们推荐用户下载 Notebook 后，本地运行进行实践。

- [量子近似优化算法 (QAOA)](./tutorial/QAOA)
- [变分量子特征求解器 (VQE)](./tutorial/VQE)
- [量子神经网络的贫瘠高原效应 (Barren Plateaus)](./tutorial/Barren)
- [量子分类器 (Quantum Classifier)](./tutorial/Q-Classifier)
- [量子变分自编码器 (Quantum Autoencoder)](./tutorial/Q-Autoencoder)
- [量子生成对抗网络 (Quantum GAN)](./tutorial/Q-GAN)
- [子空间搜索 - 量子变分特征求解器 (SSVQE)](./tutorial/SSVQE)
- [变分量子态对角化算法 (VQSD)](./tutorial/VQSD)
- [吉布斯态的制备 (Gibbs State Preparation)](./tutorial/Gibbs)
- [变分量子奇异值分解 (VQSVD)](./tutorial/VQSVD)

此外，Paddle Quantum 也支持在 GPU 上进行量子机器学习的训练，具体的方法请参考案例：[在 GPU 上使用 Paddle Quantum](./tutorial/GPU)。

### API 文档

我们为 Paddle Quantum 提供了独立的 [API 文档页面](https://paddle-quantum.readthedocs.io/zh_CN/latest/)，包含了供用户使用的所有函数和类的详细说明与用法。

### 开发

Paddle Quantum 使用 setuptools 的 develop 模式进行安装，相关代码修改可以直接进入`paddle_quantum` 文件夹进行修改。python 文件携带了自说明注释。

## 交流与反馈

- 我们非常欢迎您通过 [Github Issues](https://github.com/PaddlePaddle/Quantum/issues) 来提交问题、报告与建议。

- 技术交流QQ群：1076223166

## 使用 Paddle Quantum 的工作

我们非常欢迎开发者使用 Paddle Quantum 进行量子机器学习的研发，如果您的工作有使用 Paddle Quantum，也非常欢迎联系我们。目前使用 Paddle Quantum 的代表性工作包括了吉布斯态的制备和变分量子奇异值分解：

[1] Wang, Y., Li, G. & Wang, X. Variational quantum Gibbs state preparation with a truncated Taylor series. arXiv:2005.08797 (2020). [[pdf](https://arxiv.org/pdf/2005.08797.pdf)]

[2] Wang, X., Song, Z. & Wang, Y. Variational Quantum Singular Value Decomposition. arXiv:2006.02336 (2020). [[pdf](https://arxiv.org/pdf/2006.02336.pdf)]

## FAQ

1. 问：**研究量子机器学习有什么意义？它有哪些应用场景？**

    答：量子机器学习是将量子计算与机器学习相结合的一门学科，它一方面可以利用现有人工智能技术突破量子计算的研发瓶颈，另一方面也能利用量子计算的信息处理优势促进传统人工智能的发展。量子机器学习不仅适用于量子化学模拟（如[变分量子特征求解器 (VQE)](./tutorial/VQE)）等量子问题，也可以用来解决一些经典问题（如[量子近似优化算法 (QAOA)](./tutorial/QAOA)）。

2. 问：**想做量子机器学习，但对量子计算不是很了解，该如何入门？**

    答：Nielsen 和 Chuang 所著的《量子计算与量子信息》是量子计算领域公认的经典入门教材。建议读者首先学习这本书的第一、二、四章，介绍了量子计算中的基本概念、数学和物理基础、以及量子电路模型。读者也可以阅读量桨的[入门手册](./introduction)，其中包含了对量子计算的简单介绍，并有互动性的例子供读者尝试。对量子计算有了大致了解后，读者可以尝试学习量桨提供的一些前沿[量子机器学习案例](./tutorial)。

3. 问：**现阶段没有规模化的量子硬件，怎么开发量子应用？**

    答：使用量桨，用户可以方便地在经典计算机上模拟量子算法，进行量子应用的开发与验证，为未来使用规模化的量子硬件做技术积累。

4. 问：**量桨有哪些优势？**

    答：量桨是基于百度飞桨开发的量子机器学习工具集。飞桨作为国内首个开源开放的产业级深度学习平台，技术领先且功能完备。拥有飞桨的技术支持，特别是其强大的动态图机制，量桨可以方便地进行机器学习的优化以及 GPU 的加速。同时，基于百度量子计算研究所研发的高性能量子模拟器，量桨在个人笔记本电脑上也能支持20多个量子比特的运算。另外，量桨还有丰富的[量子机器学习案例](./tutorial)供大家参考和学习。

5. 问：**非常想试用量桨，该怎么入门呢？**

    答：建议新用户首先阅读量桨的[入门手册](./introduction)，它包含量桨详细的安装步骤以及入门教程。另外，量桨提供了丰富的[量子机器学习案例](./tutorial)，以 Jupyter Notebook 和 PDF 的方式呈现，方便用户学习和实践。如在学习和使用过程中遇到任何问题，欢迎用户通过 [Github Issues](https://github.com/PaddlePaddle/Quantum/issues) 以及技术交流QQ群（1076223166）与我们交流。

## Copyright and License

Paddle Quantum 使用 [Apache-2.0 license](LICENSE) 许可证。

## References

[1] [量子计算 - 百度百科](https://baike.baidu.com/item/%E9%87%8F%E5%AD%90%E8%AE%A1%E7%AE%97/11035661)

[2] Nielsen, M. A. & Chuang, I. L. Quantum computation and quantum information. (Cambridge university press, 2010).

[3] Phillip Kaye, Laflamme, R. & Mosca, M. An Introduction to Quantum Computing. (2007).

[4] [Biamonte, J. et al. Quantum machine learning. Nature 549, 195–202 (2017).](https://www.nature.com/articles/nature23474)

[5] [Schuld, M., Sinayskiy, I. & Petruccione, F. An introduction to quantum machine learning. Contemp. Phys. 56, 172–185 (2015).](https://www.tandfonline.com/doi/abs/10.1080/00107514.2014.964942)

[6] [Benedetti, M., Lloyd, E., Sack, S. & Fiorentini, M. Parameterized quantum circuits as machine learning models. Quantum Sci. Technol. 4, 043001 (2019).](https://iopscience.iop.org/article/10.1088/2058-9565/ab4eb5)