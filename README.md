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

[Paddle Quantum (量桨)](https://qml.baidu.com/) is a quantum machine learning (QML) toolkit developed based on Baidu PaddlePaddle. It provides a platform to construct and train quantum neural networks (QNNs) with easy-to-use QML development kits suporting combinatorial optimization, quantum chemistry and other cutting-edge quantum applications, making PaddlePaddle the first and only deep learning framework in China that supports quantum machine learning.

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
    <img src="https://img.shields.io/badge/pypi-v1.2.0-orange.svg?style=flat-square&logo=pypi"/>
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
  - Many online learning resources (16+ tutorials)
  - High efficiency in building QNN with various QNN templates
  - Automatic differentiation
- Versatile
  - Multiple optimization tools and GPU mode
  - Simulation with 25+ qubits
- Featured Toolkits
  - Toolboxes for Chemistry & Optimization
  - LOCCNet for distributed quantum information processing
  - Self-developed QML algorithms
  

## Install

### Install PaddlePaddle

This dependency will be automatically satisfied when users install Paddle Quantum. Please refer to [PaddlePaddle](https://www.paddlepaddle.org.cn/install/quick)'s official installation and configuration page. This project requires PaddlePaddle 1.8.5.

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
pip install openfermion==0.11.0
pip install openfermionpyscf==0.4
```

### Run example

Now, you can try to run a program to verify whether Paddle Quantum has been installed successfully. Here we take quantum approximate optimization algorithm (QAOA) as an example.

```bash
cd paddle_quantum/QAOA/example
python main.py
```

> For the introduction of QAOA, please refer to our [QAOA tutorial](https://github.com/PaddlePaddle/Quantum/tree/master/tutorial/QAOA).

## Introduction and developments

### Quick start

[Paddle Quantum Quick Start Manual](https://github.com/PaddlePaddle/Quantum/tree/master/introduction) is probably the best place to get started with Paddle Quantum. Currently, we support online reading and running the Jupyter Notebook locally. The manual includes the following contents:

- Detailed installation tutorials for Paddle Quantum
- Introduction to the basics of quantum computing and QNN
- Introduction on the operation modes of Paddle Quantum
- A quick tutorial on PaddlePaddle's dynamic computational graph and optimizers
- A case study on Quantum Machine Learning -- Variational Quantum Eigensolver (VQE)

### Tutorials

We provide tutorials covering combinatorial optimization, quantum chemistry, quantum classification, quantum entanglement manipulation and other popular QML research topics. Each tutorial currently supports reading on our website and running Jupyter Notebooks locally. For interested developers, we recommend them to download Jupyter Notebooks and play around with it. Here is the tutorial list,

1. [Quantum Approximation Optimization Algorithm (QAOA)](./tutorial/QAOA)
2. [Variational Quantum Eigensolver (VQE)](./tutorial/VQE)
3. [Quantum Classifier](./tutorial/Q-Classifier)
4. [The Barren Plateaus Phenomenon on Quantum Neural Networks (Barren Plateaus)](./tutorial/Barren)
5. [Quantum Autoencoder](./tutorial/Q-Autoencoder)
6. [Quantum GAN](./tutorial/Q-GAN)
7. [Subspace Search-Quantum Variational Quantum Eigensolver (SSVQE)](./tutorial/SSVQE)
8. [Variational Quantum State Diagonalization (VQSD)](./tutorial/VQSD)
9. [Gibbs State Preparation](./tutorial/Gibbs)
10. [Variational Quantum Singular Value Decomposition (VQSVD)](./tutorial/VQSVD)
11. [Local Operations and Classical Communication in QNN (LOCCNet)](./tutorial/LOCC/LOCCNET_Tutorial_EN.ipynb)
12. [Entanglement Distillation -- the BBPSSW protocol](./tutorial/LOCC)
13. [Entanglement Distillation -- the DEJMPS protocol](./tutorial/LOCC)
14. [Entanglement Distillation -- Protocol design with LOCCNet](./tutorial/LOCC)
15. [Quantum Teleportation](./tutorial/LOCC)
16. [Quantum State Discrimination](./tutorial/LOCC)

With the latest LOCCNet module, Paddle Quantum can efficiently simulate distributed quantum information processing tasks. Interested readers can start with this [tutorial on LOCCNet](./tutorial/LOCC/LOCCNET_Tutorial_EN.ipynb). In addition, Paddle Quantum supports QNN training on GPU. For users who want to get into more details, please check out the tutorial [Use Paddle Quantum on GPU](./introduction/PaddleQuantum_GPU_EN.ipynb).

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

[1] Wang, Y., Li, G. & Wang, X. Variational quantum Gibbs state preparation with a truncated Taylor series. arXiv:2005.08797 (2020). [[pdf](https://arxiv.org/pdf/2005.08797.pdf)]

[2] Wang, X., Song, Z. & Wang, Y. Variational Quantum Singular Value Decomposition. arXiv:2006.02336 (2020). [[pdf](https://arxiv.org/pdf/2006.02336.pdf)]

[3] Li, G., Song, Z. & Wang, X. VSQL: Variational Shadow Quantum Learning for Classification. arXiv:2012.08288 (2020). [[pdf]](https://arxiv.org/pdf/2012.08288.pdf), to appear at **AAAI 2021** conference.

[4] Chen, R., Song, Z., Zhao, X. & Wang, X. Variational Quantum Algorithms for Trace Distance and Fidelity Estimation. arXiv:2012.05768 (2020). [[pdf]](https://arxiv.org/pdf/2012.05768.pdf)

[5] Wang, K., Song, Z., Zhao, X., Wang Z. & Wang, X. Detecting and quantifying entanglement on near-term quantum devices. arXiv:2012.14311 (2020). [[pdf]](https://arxiv.org/pdf/2012.14311.pdf)

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