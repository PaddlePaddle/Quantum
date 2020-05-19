# Paddle Quantum （量桨）

Paddle Quantum（量桨）是基于百度飞桨开发的量子机器学习工具集，支持量子神经网络的搭建与训练，提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，使得百度飞桨也因此成为国内首个目前也是唯一一个支持量子机器学习的深度学习框架。

![](https://release-data.cdn.bcebos.com/Paddle%20Quantum.png)

量桨建立起了人工智能与量子计算的桥梁，不但可以快速实现量子神经网络的搭建与训练，还提供易用的量子机器学习开发套件与量子优化、量子化学等前沿量子应用工具集，并提供多项自研量子机器学习应用。通过百度飞桨深度学习平台赋能量子计算，量桨为领域内的科研人员以及开发者便捷地开发量子人工智能的应用提供了强有力的支撑，同时也为广大量子计算爱好者提供了一条可行的学习途径。



## 特色

- 易用性：提供简洁的神经网络搭建与丰富的量子机器学习案例。
- 通用性与拓展性：支持常用量子电路模型，提供多项优化工具。
- 特色工具集：提供量子优化、量子化学等前沿量子应用工具集，自研多项量子机器学习应用。



## 安装步骤

### Install PaddlePaddle
请参考 [PaddlePaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/index_cn.html) 安装配置页面。此项目需求 PaddlePaddle 1.8.0 或更高版本。



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

### 使用 openfermion 读取xyz 描述文件 （仅可在linux下安装使用）
VQE中调用 openfermion 读取分子xyz文件并计算，因此需要安装 openfermion 和 openfermionpyscf。
```bash
pip install openfermion
pip install openfermionpyscf
```


### 运行

```bash
cd paddle_quantum/QAOA/example
python main.py
```



## 入门与开发

### 教程入门

量子计算是由量子力学与计算理论交叉而成的全新计算模型，具有强大的信息处理优势和广阔的应用前景，被视作未来计算技术的心脏。量子计算的相关介绍与入门知识可以参考 [1-3]。

量子机器学习是一门结合量子计算与机器学习的交叉学科，一方面利用量子计算的信息处理优势促进人工智能的发展，另一方面也利用现有的人工智能的技术突破量子计算的研发瓶颈。关于量子机器学习的入门资料可以参考 [4-6]。Paddle Quantum（量桨）建立起了人工智能与量子计算的桥梁，为量子机器学习领域的研发提供强有力的支撑，也提供了丰富的案例供开发者学习。



### 案例入门

特别的，我们提供了涵盖量子优化、量子化学、量子机器学习等多个领域的案例供大家学习。比如：

- 量子组合优化（QAOA），完成安装步骤后打开 tutorial\QAOA.ipynb 即可进行研究学习。

```bash
cd tutorial
jupyter notebook  QAOA.ipynb
```

- 量子特征求解器（VQE），完成安装步骤后打开 tutorial\VQE.ipynb 即可进行研究学习。

```
cd tutorial
jupyter notebook  VQE.ipynb
```



### 开发

Paddle Quantum 使用 setuptools的develop 模式进行安装，相关代码修改可以直接进入`paddle_quantum` 文件夹进行修改。python 文件携带了自说明注释。



## 交流与反馈

- 我们非常欢迎您欢迎您通过[Github Issues](https://github.com/PaddlePaddle/Quantum/issues)来提交问题、报告与建议。

- QQ技术交流群: 1076223166

## 使用Paddle Quantum的工作

我们非常欢迎开发者使用Paddle Quantum进行量子机器学习的研发，如果您的工作有使用Paddle Quantum，也非常欢迎联系我们。目前使用 Paddle Quantum 的代表性工作关于 Gibbs 态制备如下：

[1] Y. Wang, G. Li, and X. Wang, “Variational quantum Gibbs state preparation with a truncated Taylor series,” arXiv:2005.08797, May 2020. [[pdf](https://arxiv.org/pdf/2005.08797.pdf)]


## Copyright and License

Paddle Quantum 使用 [Apache-2.0 license](LICENSE)许可证。



## References

[1] [量子计算 - 百度百科](https://baike.baidu.com/item/量子计算/11035661?fr=aladdin)

[2] M. A. Nielsen and I. L. Chuang, Quantum computation and quantum information. Cambridge university press, 2010.

[3] Phillip Kaye, R. Laflamme, and M. Mosca, An Introduction to Quantum Computing. 2007.

[4] J. Biamonte, P. Wittek, N. Pancotti, P. Rebentrost, N. Wiebe, and S. Lloyd, “Quantum machine learning,” Nature, vol. 549, no. 7671, pp. 195–202, Sep. 2017. [[pdf](https://arxiv.org/pdf/1611.09347)]

[5] M. Schuld, I. Sinayskiy, and F. Petruccione, “An introduction to quantum machine learning,” Contemp. Phys., vol. 56, no. 2, pp. 172–185, 2015. [[pdf](https://arxiv.org/pdf/1409.3097)]

[6] M. Benedetti, E. Lloyd, S. Sack, and M. Fiorentini, “Parameterized quantum circuits as machine learning models,” Quantum Sci. Technol., vol. 4, no. 4, p. 043001, Nov. 2019. [[pdf](https://arxiv.org/pdf/1906.07682)]
