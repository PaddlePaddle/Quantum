# 量子应用模型库

- [特色](#特色)
- [安装](#安装)
- [如何使用](#如何使用)
- [应用列表](#应用列表)

量子应用模型库（**Q**uantum **A**pplication **M**odel **L**ibrary, QAML）是一个开箱即用的实用量子应用模型集合，它由[百度量子计算研究所](https://quantum.baidu.com/)研发，旨在成为企业用户的量子解决方案“超市”。目前，QAML 中的模型已经覆盖了以下领域：

- 人工智能
- 医学制药
- 材料模拟
- 金融科技
- 汽车制造
- 数据分析

QAML 基于量桨这一量子机器学习平台实现，关于量桨的内容可以参考 https://qml.baidu.com 和 https://github.com/PaddlePaddle/Quantum 。

## 特色

- 产业化：10 大应用模型紧贴 6 大产业方向，涵盖人工智能、化工材料、汽车制造、金融套利等热点话题。
- 端到端：打通应用场景到量子算法的全流程，解决量子应用的最后一公里问题。
- 开箱即用：无需特殊配置，通过量桨直接完成模型调用，省去繁琐安装环节。

## 安装

QAML 依赖于量桨（ `paddle-quantum` ）软件包。用户可以通过 pip 来安装：

```shell
pip install paddle-quantum
```

对于那些使用旧版量桨的用户，只需运行 `pip install --upgrade paddle-quantum` 即可安装最新版量桨。

QAML 的内容在 Paddle Quantum 的 GitHub 仓库中，用户可以通过点击[此链接](https://github.com/PaddlePaddle/Quantum/archive/refs/heads/master.zip)下载包含 QAML 源代码的压缩包。QAML 的所有模型都在解压后的文件夹中的 `applications` 文件夹里。

用户也可以使用 git 来获取 QAML 的源码文件。

```shell
git clone https://github.com/PaddlePaddle/Quantum.git
cd Quantum/applications
```

用户可以进入到 `applications` 下的 `handwritten_digits_classification` 文件夹中，然后运行以下代码来检查安装是否成功。

```shell
python vsql_classification.py --example.toml
```

如果上面的程序没有报错、成功运行的话，则说明安装成功了。

## 如何使用

在每个应用模型中，我们都提供了可以直接运行的Python脚本和相应的配置文件。用户可以修改配置文件来实现对应的要求。

以手写数字识别为例，用户可以通过执行 `handwritten_digits_classification` 中的 `python vsql_classification.py --example.toml` 命令来快速使用。我们为每个应用模型提供了教程，方便用户快速理解和上手使用。

## 应用列表

*持续更新中*

我们列出了目前 QAML 的所有应用案例的教程，新开发的应用案例也会持续添加进来。

1. [手写数字识别](./handwritten_digits_classification/introduction_cn.ipynb)
2. [分子基态能量 & 偶极矩计算](./lithium_ion_battery/introduction_cn.ipynb)
3. [中文文本分类](./text_classification/introduction_cn.ipynb)
4. [蛋白质折叠](./protein_folding/introduction_cn.ipynb)
5. [医学影像判别](./medical_image_classification/introduction_cn.ipynb)
6. [材料表面质量检测](./quality_detection/introduction_cn.ipynb)
7. [量子期权定价](./option_pricing/introduction_cn.ipynb)
8. [投资组合优化](./portfolio_optimization/introduction_cn.ipynb)
9. [回归分析](./regression/introduction_cn.ipynb)
10. [线性方程组求解](./linear_solver/introduction_cn.ipynb)
