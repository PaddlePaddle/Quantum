English | [简体中文](README_CN.md)

# Paddle Quantum (量桨)

- [Features](#features)
- [Install](#install)
   - [Install PaddlePaddle](#install-paddlepaddle)
   - [Install Paddle Quantum](#install-paddle-quantum)
   - [Use OpenFermion to read .xyz molecule configuration file](#use-openfermion-to-read-xyz-molecule-configuration-file)
   - [Run programs](#run-programs)
- [Introduction and developments](#introduction-and-developments)
   - [Quick start](#quick-start)
   - [Tutorials](#tutorials)
   - [API documentation](#api-documentation)
- [Feedbacks](#feedbacks)
- [Research with Paddle Quantum](#research-based-on-paddle-quantum)
- [Frequently Asked Questions](#frequently-asked-questions)
- [Copyright and License](#copyright-and-license)
- [References](#references)

[Paddle Quantum (量桨)](https://qml.baidu.com/) is a quantum machine learning (QML) toolkit developed based on Baidu PaddlePaddle. It provides a platform to construct and train quantum neural networks (QNNs) with easy-to-use QML development kits supporting combinatorial optimization, quantum chemistry and other cutting-edge quantum applications, making PaddlePaddle the first and only deep learning framework in China that supports quantum machine learning.

<p align="center">
  <a href="https://qml.baidu.com/">
    <img width=80% src="https://release-data.cdn.bcebos.com/Paddle%20Quantum.png">
  </a>
</p>

<p align="center">
  <!-- docs -->
  <a href="https://qml.baidu.com/api/paddle_quantum.circuit.html">
    <img src="https://img.shields.io/badge/docs-link-green.svg?style=flat-square&logo=read-the-docs"/>
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/paddle-quantum/">
    <img src="https://img.shields.io/badge/pypi-v2.1.0-orange.svg?style=flat-square&logo=pypi"/>
  </a>
  <!-- Python -->
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.6+-blue.svg?style=flat-square&logo=python"/>
  </a>
  <!-- License -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square&logo=apache"/>
  </a>
  <!-- Platform -->
  <a href="https://github.com/PaddlePaddle/Quantum">
    <img src="https://img.shields.io/badge/OS-MacOS%20|%20Windows%20|%20Linux-lightgrey.svg?style=flat-square"/>
  </a>
</p>


Paddle Quantum aims at establishing a bridge between artificial intelligence (AI) and quantum computing (QC). It has been utilized for developing several quantum machine learning applications. With the PaddlePaddle deep learning platform empowering QC, Paddle Quantum provides strong support for scientific research community and developers in the field to easily develop QML applications. Moreover, it provides a learning platform for quantum computing enthusiasts.

## Features

- Easy-to-use
  - Many online learning resources (23+ tutorials)
  - High efficiency in building QNN with various QNN templates
  - Automatic differentiation
- Versatile
  - Multiple optimization tools and GPU mode
  - Simulation with 25+ qubits
  - Flexible noise models
- Featured Toolkits
  - Toolboxes for Chemistry & Optimization
  - LOCCNet for distributed quantum information processing
  - Self-developed QML algorithms

## Install

### Install PaddlePaddle

This dependency will be automatically satisfied when users install Paddle Quantum. Please refer to [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick)'s official installation and configuration page. This project requires PaddlePaddle 2.0.1+.

### Install Paddle Quantum

We recommend the following way of installing Paddle Quantum with `pip`,

```bash
pip install paddle-quantum
```
or download all the files and finish the installation locally,

```bash
git clone http://github.com/PaddlePaddle/quantum
cd quantum
pip install -e .
```


### Use OpenFermion to read .xyz molecule configuration file

> Only macOS and linux users can use OpenFermion to read .xyz description files.

Once the user confirms the above OS constraint, OpenFermion can be installed with the following command. These packages are used for quantum chemistry calculations and could be potentially used in the VQE tutorial.

```bash
pip install openfermion
pip install openfermionpyscf
```

### Run example

Now, you can try to run a program to verify whether Paddle Quantum has been installed successfully. Here we take quantum approximate optimization algorithm (QAOA) as an example.

```bash
cd paddle_quantum/QAOA/example
python main.py
```

> For the introduction of QAOA, please refer to our [QAOA tutorial](https://github.com/PaddlePaddle/Quantum/tree/master/tutorial/combinatorial_optimization/QAOA_EN.ipynb).

## Introduction and developments

### Quick start

[Paddle Quantum Quick Start Manual](https://github.com/PaddlePaddle/Quantum/tree/master/introduction) is probably the best place to get started with Paddle Quantum. Currently, we support online reading and running the Jupyter Notebook locally. The manual includes the following contents:

- Detailed installation tutorials for Paddle Quantum
- Introduction to the basics of quantum computing and QNN
- Introduction on the operation modes of Paddle Quantum
- A quick tutorial on PaddlePaddle's dynamic computational graph and optimizers
- A case study on Quantum Machine Learning -- Variational Quantum Eigensolver (VQE)

### Tutorials

We provide tutorials covering quantum simulation, machine learning, combinatorial optimization, local operations and classical communication (LOCC), and other popular QML research topics. Each tutorial currently supports reading on our website and running Jupyter Notebooks locally. For interested developers, we recommend them to download Jupyter Notebooks and play around with it. Here is the tutorial list,

- [Quantum Simulation](./tutorial/quantum_simulation)
    1. [Variational Quantum Eigensolver (VQE)](./tutorial/quantum_simulation/VQE_EN.ipynb)
    2. [Subspace Search-Quantum Variational Quantum Eigensolver (SSVQE)](./tutorial/quantum_simulation/SSVQE_EN.ipynb)
    3. [Variational Quantum State Diagonalization (VQSD)](./tutorial/quantum_simulation/VQSD_EN.ipynb)
    4. [Gibbs State Preparation](./tutorial/quantum_simulation/GibbsState_EN.ipynb)

- [Machine Learning](./tutorial/machine_learning)
    1. [Encoding Classical Data into Quantum States](./tutorial/machine_learning/DataEncoding_EN.ipynb)
    2. [Quantum Classifier](./tutorial/machine_learning/QClassifier_EN.ipynb)
    3. [Variational Shadow Quantum Learning (VSQL)](./tutorial/machine_learning/VSQL_EN.ipynb)
    4. [Quantum Kernel Methods](./tutorial/machine_learning/QKernel_EN.ipynb)
    5. [Quantum Autoencoder](./tutorial/machine_learning/QAutoencoder_EN.ipynb)
    6. [Quantum GAN](./tutorial/machine_learning/QGAN_EN.ipynb)
    7. [Variational Quantum Singular Value Decomposition (VQSVD)](./tutorial/machine_learning/VQSVD_EN.ipynb)

- [Combinatorial Optimization](./tutorial/combinatorial_optimization)
    1. [Quantum Approximation Optimization Algorithm (QAOA)](./tutorial/combinatorial_optimization/QAOA_EN.ipynb)
    2. [Solving Max-Cut Problem with QAOA](./tutorial/combinatorial_optimization/MAXCUT_EN.ipynb)
    3. [Large-scale QAOA via Divide-and-Conquer](./tutorial/combinatorial_optimization/DC-QAOA_EN.ipynb)
    4. [Travelling Salesman Problem](./tutorial/combinatorial_optimization/TSP_EN.ipynb)

- [LOCC with QNN (LOCCNet)](./tutorial/locc)
    1. [Local Operations and Classical Communication in QNN (LOCCNet)](./tutorial/locc/LOCCNET_Tutorial_EN.ipynb)
    2. [Entanglement Distillation -- the BBPSSW protocol](./tutorial/locc/EntanglementDistillation_BBPSSW_EN.ipynb)
    3. [Entanglement Distillation -- the DEJMPS protocol](./tutorial/locc/EntanglementDistillation_DEJMPS_EN.ipynb)
    4. [Entanglement Distillation -- Protocol design with LOCCNet](./tutorial/locc/EntanglementDistillation_LOCCNET_EN.ipynb)
    5. [Quantum Teleportation](./tutorial/locc/QuantumTeleportation_EN.ipynb)
    6. [Quantum State Discrimination](./tutorial/locc/StateDiscrimination_EN.ipynb)

- [QNN Research](./tutorial/qnn_research)
    1. [The Barren Plateaus Phenomenon on Quantum Neural Networks (Barren Plateaus)](./tutorial/qnn_research/BarrenPlateaus_EN.ipynb)
    2. [Noise Model and Quantum Channel](./tutorial/qnn_research/Noise_EN.ipynb)

With the latest LOCCNet module, Paddle Quantum can efficiently simulate distributed quantum information processing tasks. Interested readers can start with this [tutorial on LOCCNet](./tutorial/locc/LOCCNET_Tutorial_EN.ipynb). In addition, Paddle Quantum supports QNN training on GPU. For users who want to get into more details, please check out the tutorial [Use Paddle Quantum on GPU](./introduction/PaddleQuantum_GPU_EN.ipynb). Moreover, Paddle Quantum could design robust quantum algorithms under noise. For more information, please see [Noise tutorial](./tutorial/qnn_research/Noise_EN.ipynb).

### API documentation

For those who are looking for explanation on the python class and functions provided in Paddle Quantum, we refer to our [API documentation page](https://qml.baidu.com/api/introduction.html).

> We, in particular, denote that the current docstring specified in source code **is written in simplified Chinese**, this will be updated in later versions.

## Feedbacks

Users are encouraged to contact us through [Github Issues](https://github.com/PaddlePaddle/Quantum/issues) or email quantum@baidu.com with general questions, unfixed bugs, and potential improvements. We hope to make Paddle Quantum better together with the community!

## Research based on Paddle Quantum

We also highly encourage developers to use Paddle Quantum as a research tool to develop novel QML algorithms. If your work uses Paddle Quantum, feel free to send us a notice via quantum@baidu.com. We are always excited to hear that! Cite us with the following BibTeX:

> @misc{Paddlequantum,
> title = {{Paddle Quantum}},
> year = {2020},
> url = {https://github.com/PaddlePaddle/Quantum}, }

So far, we have done several projects with the help of Paddle Quantum as a powerful QML development platform.

[1] Wang, Youle, Guangxi Li, and Xin Wang. "Variational quantum gibbs state preparation with a truncated taylor series." arXiv preprint arXiv:2005.08797 (2020). [[pdf](https://arxiv.org/pdf/2005.08797.pdf)]

[2] Wang, Xin, Zhixin Song, and Youle Wang. "Variational Quantum Singular Value Decomposition." arXiv preprint arXiv:2006.02336 (2020). [[pdf](https://arxiv.org/pdf/2006.02336.pdf)]

[3] Li, Guangxi, Zhixin Song, and Xin Wang. "VSQL: Variational Shadow Quantum Learning for Classification." arXiv preprint arXiv:2012.08288 (2020). [[pdf]](https://arxiv.org/pdf/2012.08288.pdf), to appear at **AAAI 2021** conference.

[4] Chen, Ranyiliu, et al. "Variational Quantum Algorithms for Trace Distance and Fidelity Estimation." arXiv preprint arXiv:2012.05768 (2020). [[pdf]](https://arxiv.org/pdf/2012.05768.pdf)

[5] Wang, Kun, et al. "Detecting and quantifying entanglement on near-term quantum devices." arXiv preprint arXiv:2012.14311 (2020). [[pdf]](https://arxiv.org/pdf/2012.14311.pdf)

[6] Zhao, Xuanqiang, et al. "LOCCNet: a machine learning framework for distributed quantum information processing." arXiv preprint arXiv:2101.12190 (2021). [[pdf]](https://arxiv.org/pdf/2101.12190.pdf)

[7] Cao, Chenfeng, and Xin Wang. "Noise-Assisted Quantum Autoencoder." Physical Review Applied 15.5 (2021): 054012. [[pdf]](https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.15.054012)

## Frequently Asked Questions

1. **Question:** What is quantum machine learning? What are the applications?

    **Answer:** Quantum machine learning (QML) is an interdisciplinary subject that combines quantum computing (QC) and machine learning (ML). On the one hand, QML utilizes existing artificial intelligence technology to break through the bottleneck of quantum computing research. On the other hand, QML uses the information processing advantages of quantum computing to promote the development of traditional artificial intelligence. QML is not only suitable for quantum chemical simulations (with Variational Quantum Eigensolver) and other quantum problems. It also help researchers to solve classical optimization problems including knapsack problem, traveling salesman problem, and Max-Cut problem through the Quantum Approximate Optimization Algorithm.

2. **Question:** I want to study QML, but I don't know much about quantum computing. Where should I start?

    **Answer:** *Quantum Computation and Quantum Information* by Nielsen & Chuang is the classic introductory textbook to QC. We recommend readers to study Chapter 1, 2, and 4 of this book first. These chapters introduce the basic concepts, provide solid mathematical and physical foundations, and discuss the quantum circuit model widely used in QC. Readers can also go through Paddle Quantum's quick start guide, which contains a brief introduction to QC and interactive examples. After building a general understanding of QC, readers can try some cutting-edge QML applications provided as tutorials in Paddle Quantum.

3. **Question:** Currently, there is no fault-tolerant large-scale quantum hardware. How can we develop quantum applications? 

    **Answer:** The development of useful algorithms does not necessarily require a perfect hardware. The latter is more of an engineering problem. With Paddle Quantum, one can develop, simulate, and verify the validity of self-innovated quantum algorithms. Then, researchers can choose to implement these tested quantum algorithms in a small scale hardware and see the actual performance of it. Following this line of reasoning, we can prepare ourselves with many candidates of useful quantum algorithms before the age of matured quantum hardware.
    
4. **Question:** What are the advantages of Paddle Quantum?

    **Answer:** Paddle Quantum is an open-source QML toolkit based on Baidu PaddlePaddle. As the first open-source and industrial level deep learning platform in China, PaddlePaddle has the leading ML technology and rich functionality. With the support of PaddlePaddle, especially its dynamic computational graph mechanism, Paddle Quantum could easily train a QNN and with GPU acceleration. In addition, based on the high-performance quantum simulator developed by Institute for Quantum Computing (IQC) at Baidu, Paddle Quantum can simulate more than 20 qubits on personal laptops. Finally, Paddle Quantum provides many open-source QML tutorials for readers from different backgrounds. 

## Copyright and License

Paddle Quantum uses [Apache-2.0 license](LICENSE).

## References

[1] [Quantum Computing - Wikipedia](https://en.wikipedia.org/wiki/Quantum_computing)

[2] Nielsen, M. A. & Chuang, I. L. Quantum computation and quantum information. (2010).

[3] Phillip Kaye, Laflamme, R. & Mosca, M. An Introduction to Quantum Computing. (2007).

[4] [Biamonte, J. et al. Quantum machine learning. Nature 549, 195–202 (2017).](https://www.nature.com/articles/nature23474)

[5] [Schuld, M., Sinayskiy, I. & Petruccione, F. An introduction to quantum machine learning. Contemp. Phys. 56, 172–185 (2015).](https://www.tandfonline.com/doi/abs/10.1080/00107514.2014.964942)

[6] [Benedetti, M., Lloyd, E., Sack, S. & Fiorentini, M. Parameterized quantum circuits as machine learning models. Quantum Sci. Technol. 4, 043001 (2019).](https://iopscience.iop.org/article/10.1088/2058-9565/ab4eb5)