# Quantum Application Model Library

- [Features](#features)
- [Installation](#installation)
- [How to Use](#how-to-use)
- [Application List](#application-list)

**Q**uantum **A**pplication **M**odel Library (QAML) is a collection of out-of-box practical quantum algorithms, it is developed by [Institute for Quantum Computing at Baidu](https://quantum.baidu.com/), and aims to be a "supermarket" of quantum solutions for industry users. Currently, models in QAML have covered popular areas listed below:

- Artificial Intelligence
- Medicine and Pharmaceuticals
- Material Simulation
- Financial Technology
- Manufacturing
- Data Analysis

QAML is implemented on Paddle Quantum, a quantum machine learning platform, which can be found at https://qml.baidu.com and https://github.com/PaddlePaddle/Quantum.

## Features

- Industrialization: 10 models closely follow the 6 major industrial directions, covering hot topics such as artificial intelligence, chemical materials, manufacturing, finance, etc.
- End-to-end: Linking the whole process from application scenarios to quantum computing and solving the last mile of quantum applications.
- Out-of-box: No special configuration is required, the model is called directly by the Paddle Quantum, eliminating the tedious installation process.

## Installation

QAML depends on the `paddle-quantum` package. Users can install it by pip.

```shell
pip install paddle-quantum
```

For those who are using old versions of Paddle Quantum, simply run `pip install --upgrade paddle-quantum` to install the latest package.

QAML locates in Paddle Quantum's GitHub repository, you can download the zip file contains QAML source code by clicking [this link](https://github.com/PaddlePaddle/Quantum/archive/refs/heads/master.zip). After unzipping the package, you will find all the models in the `applications` folder in the extracted folder.

You can also use git to get the QAML source code.

```shell
git clone https://github.com/PaddlePaddle/Quantum.git
cd Quantum/applications
```

You can check your installation by going to the `handwritten_digits_classification` folder under `applications` and running

```shell
python vsql_classification.py --example.toml
```

The installation is successful once the program terminates without errors.

## How to Use

In each application model, we provide Python scripts that can be run directly and the corresponding configuration files. The user can modify the configuration file to implement the corresponding requirements.

Take handwritten digit classification as an example, it can be used by executing `python vsql_classification.py --example.toml` in the `handwritten_digits_classification` folder. We provide tutorials for each application model, which allows users to quickly understand and use it.

## Application List

*Continue update*

Below we list instructions for all applications available in QAML, newly developed applications will be continuously integrated into QAML.

1. [Handwritten digits classification](./handwritten_digits_classification/introduction_en.ipynb)
2. [Molecular ground state energy & dipole moment calculation](./lithium_ion_battery/introduction_en.ipynb)
3. [Text classification](./text_classification/introduction_en.ipynb)
4. [Protein folding](./protein_folding/introduction_en.ipynb)
5. [Medical image classification](./medical_image_classification/introduction_en.ipynb)
6. [Quality detection](./quality_detection/introduction_en.ipynb)
7. [Option pricing](./option_pricing/introduction_en.ipynb)
8. [Quantum portfolio optimization](./portfolio_optimization/introduction_en.ipynb)
9. [Regression](./regression/introduction_en.ipynb)
10. [Quantum linear equation solver](./linear_solver/introduction_en.ipynb)
11. [Credit Risk Analysis](./credit_risk_analysis/introduction_en.ipynb)
12. [Deuteron Binding Energy](./deuteron_binding_energy/introduction_en.ipynb)
13. [Handwritten Digits Generation](./handwritten_digits_generation/introduction_en.ipynb)
14. [Intent Classification](./intent_classification/introduction_en.ipynb)
15. [Power Flow Study](./power_flow/introduction_en.ipynb)
16. [Random Number Generation](./random_number/introduction_en.ipynb)
