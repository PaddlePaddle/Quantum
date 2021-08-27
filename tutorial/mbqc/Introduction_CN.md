# 基于测量的量子计算模块

## 概述

基于测量的量子计算（Measurement-Based Quantum Computation, MBQC）模块是基于量桨开发的通用量子计算模拟工具。与传统的量子电路模型不同，MBQC 具有其独特的运行方式 [1-8]，因而常见的电路模拟工具无法直接用于该模型的模拟。为了促进对 MBQC 的进一步研究和探索，我们设计并提供了模拟 MBQC 所需的多个核心模块。

一方面，与其他常见的量子电路模拟器一样，我们可以利用 MBQC 模块快速实现量子神经网络的搭建与训练，在诸如量子化学，量子自然语言处理，量子金融等人工智能领域具有广阔的应用前景。另一方面，MBQC 
模块在某些特殊情形下可以展现出其运算效率上的独特优势，特别是在多比特浅层量子电路模拟时可以大幅度降低计算资源的消耗，从而有潜力实现更大规模的量子算法模拟。

我们的模块是业界首个、目前也是唯一一个支持模拟 MBQC 量子计算模型的系统性工具。我们由衷地期待广大的量子计算爱好者和开发者进行使用和学习。欢迎加入我们，共同探索 MBQC 中更多的奥秘！

## 特色

- 丰富的在线教学示例，理论结合实践，轻松上手
- 完备的核心模块组件，通用扩展性强，调用方便
- 高效的翻译模拟算法，加速执行效率，性能强大

## 零基础入门 MBQC

### 什么是 MBQC 量子计算模型

基于测量的量子计算是平行于量子电路模型的一种通用量子计算技术路线。其核心思想在于对一个量子纠缠态的部分比特进行测量，未被测量的量子系统将会实现相应的演化，并且通过对测量方式的控制，我们可以实现任意需要的演化 [1-8]。

MBQC 的标准计算过程主要分为三个步骤：量子图态准备，单比特测量，副产品纠正。与此过程等价的描述方式还有 MBQC 的测量模式 （或称为 "EMC" 语言）[9]。 详细内容介绍请参见 [MBQC 入门介绍](MBQC_CN.ipynb)。

### 子模块

|子模块|描述|
|:---|:---|
|`simulator`|包含构造 MBQC 模型的常用类和配套的运算模拟工具。|
|`mcalculus`|包含处理 MBQC 测量模式的相关操作。|
|`transpiler`|包含电路模型和 MBQC 测量模式的转义工具。|
|`qobject`|包含量子信息处理的常用对象，如量子态、量子电路、测量模式等。|
|`utils`|包含计算所需的各种常用类和函数。|

### 模块的基本使用方法

下面我们简单介绍一下模块的基本使用方法。

#### 基础调用方法

在 `simulator` 中，我们定义了 `MBQC` 类，用户可以实例化该类来搭建属于自己的 MBQC 模型。大多数 MBQC 模型下的算法都是基于此类的方法来实现的。用户输入图的信息，调用 `measure` 类方法对图中节点进行测量，最后进行相应副产品纠正，即可完成 MBQC 模型的计算过程。这里我们以实现单比特 `Hadamard` 门为例，简要展示其基础用法。

```python
# 引入模块
from paddle_quantum.mbqc.simulator import MBQC
from paddle_quantum.mbqc.utils import basis
# 构造计算相关的图
vertices = ["1", "2"]
edges = [("1", "2")]
graph = [vertices, edges]
# 搭建 MBQC 模型
mbqc = MBQC()
# 输入图信息
mbqc.set_graph(graph)
# 对节点 “1” 进行 X 测量
mbqc.measure("1", basis("X"))
# 对节点 “2” 进行副产品处理
mbqc.correct_byproduct('X', "2", mbqc.sum_outcomes(["1"]))
# 获得量子输出态
state_out = mbqc.get_quantum_output()
```

#### 测量模式调用方法

MBQC 和电路模型一样都可以实现通用量子计算，并且两者之间存在一一对应的关系。然而将电路模型转化为等价的 MBQC 测量模式过程复杂且计算量大 [9]。为此，我们在模块中提供了将量子电路模型自动翻译为测量模式的转义模块 
`transpiler` 。我们可以直接使用类似于 [UAnstaz](https://qml.baidu.com/api/paddle_quantum.circuit.uansatz.html) 的调用格式来构建量子电路，将其自动翻译成测量模式，再接入模拟模块中运行。同样地，这里我们以实现单比特 `Hadamard` 门为例，简要展示其用法。

```python
# 引入模块
from paddle_quantum.mbqc.qobject import Circuit
from paddle_quantum.mbqc.transpiler import transpile
from paddle_quantum.mbqc.simulator import MBQC
# 构造量子电路
width = 1
cir = Circuit(width)
cir.h(0)
# 将电路翻译成测量模式
pat = transpile(cir)
# 调用模拟模块运行
mbqc = MBQC()
mbqc.set_pattern(pat)
mbqc.run_pattern()
# 获得量子输出态
state_out = mbqc.get_quantum_output()
```

以上即为 MBQC 模块当前的主要内容。在实际案例及应用中，我们可以根据需要调用相关模块。另外，模块内部更多丰富的功能请参见示例教程及 API 文档。

### 示例教程

在这里，我们提供了三个示例教程，其中每个教程都包含了关于 MBQC 的基础理论讲解和详细的代码演示。用户可以通过教程学习和代码练习，熟练掌握 MBQC 的运算逻辑和模块的调用方式，为后续进一步探索 MBQC 打下基础。

- [MBQC 入门介绍](MBQC_CN.ipynb)
- [基于测量的量子近似优化算法](QAOA_CN.ipynb)
- [MBQC 模型下求解多项式组合优化问题](PUBO_CN.ipynb)

## 常见问题

- 问：为什么要研究 MBQC ？它有哪些应用场景？

    答：MBQC
    是平行于量子电路模型的一种通用量子计算技术路线。物理实现上，与电路模型相比，单比特测量操作在实验上更容易实现，保真度更高，无适应性的测量部分则可同时进行，从而大幅度减小算法深度，降低相干时间对保真度的影响。经典模拟上，由于不同比特间的测量步骤可以交换次序且不影响测量结果，在模拟时可以通过交换测量顺序来优化计算的执行路线，降低计算资源消耗，提高运算效率。此外，MBQC 中的资源态可以与具体计算任务无关，因此可以应用在量子互联网中用于安全代理计算，保护用户的计算和数据隐私 [10,11]。

- 问：现阶段 MBQC 是通过什么技术手段进行物理实现的？

    答：MBQC 在物理实现上的难点主要是资源态的制备。与量子电路模型中使用的超导技术不同，资源态的制备大多采用线性光学技术或冷原子技术，目前现有的资源态的制备技术请参见 [2,12,13]。


## 参考文献

[1] Gottesman, Daniel, and Isaac L. Chuang. "Demonstrating the viability of universal quantum computation using teleportation and single-qubit operations." [Nature 402.6760 (1999): 390-393.](https://www.nature.com/articles/46503?__hstc=13887208.d9c6f9c40e1956d463f0af8da73a29a7.1475020800048.1475020800050.1475020800051.2&__hssc=13887208.1.1475020800051&__hsfp=1773666937)

[2] Robert Raussendorf, et al. "A one-way quantum computer." [Physical Review Letters 86.22 (2001): 5188.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.86.5188)

[3] Raussendorf, Robert, and Hans J. Briegel. "Computational model underlying the one-way quantum computer." [Quantum Information & Computation 2.6 (2002): 443-486.](https://dl.acm.org/doi/abs/10.5555/2011492.2011495)

[4] Robert Raussendorf, et al. "Measurement-based quantum computation on cluster states." [Physical Review A 68.2 (2003): 022312.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.68.022312)

[5] Nielsen, Michael A. "Quantum computation by measurement and quantum memory." [Physics Letters A 308.2-3 (2003): 96-100.](https://www.sciencedirect.com/science/article/abs/pii/S0375960102018030)

[6] Leung, Debbie W. "Quantum computation by measurements." [International Journal of Quantum Information 2.01 (2004): 33-43.](https://www.worldscientific.com/doi/abs/10.1142/S0219749904000055)

[7] Briegel, Hans J., et al. "Measurement-based quantum computation." [Nature Physics 5.1 (2009): 19-26.](https://www.nature.com/articles/nphys1157)

[8] Aliferis, Panos, and Debbie W. Leung. "Computation by measurements: a unifying picture." [Physical Review A 70.6 (2004): 062314.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.70.062314)

[9] Danos, Vincent, et al. "The measurement calculus." [Journal of the ACM (JACM) 54.2 (2007): 8-es.](https://dl.acm.org/doi/abs/10.1145/1219092.1219096)

[10] Broadbent, Anne, et al. "Universal blind quantum computation." [2009 50th Annual IEEE Symposium on Foundations of Computer Science. IEEE, 2009.](https://arxiv.org/abs/0807.4154)

[11] Morimae, Tomoyuki. "Verification for measurement-only blind quantum computing." [Physical Review A 89.6 (2014): 060302.](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.89.060302)

[12] Larsen, Mikkel V., et al. "Deterministic generation of a two-dimensional cluster state." [Science 366.6463 (2019): 369-372.](https://science.sciencemag.org/content/366/6463/369)

[13] Asavanant, Warit, et al. "Generation of time-domain-multiplexed two-dimensional cluster state." [Science 366.6463 (2019): 373-376.](https://science.sciencemag.org/content/366/6463/373)

