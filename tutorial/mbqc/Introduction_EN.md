# Measurement-Based Quantum Computation Module

## Introduction

Measurement-Based Quantum Computation (MBQC) module is a simulation tool for universal quantum computing, developed based on Baidu PaddlePaddle. Unlike the conventional quantum circuit model, MBQC has its unique way of computing [1-8] and thus common circuit simulation tools cannot be directly used for the simulation of this model. To facilitate further research and exploration of MBQC, we have designed and provided several core modules in the module that are required for simulation.

On the one hand, as with common quantum circuit simulators, we can use the module to quickly build and train quantum neural networks, which have powerful application prospects in the field of artificial intelligence such as quantum chemistry, quantum natural language processing, and quantum finance. On the other hand, the MBQC module demonstrates its unique advantages in computational efficiency in some special cases, especially in quantum shallow circuit simulations, where the required computational resource can be significantly reduced, leading to a potential opportunity for larger-scale quantum algorithm simulations.

Our module is currently the first and the only systematic tool in the industry that supports the simulation of MBQC. We sincerely look forward to its learning and usage by quantum computing enthusiasts and researchers. Welcome to join us and discover the infinite possibilities of MBQC together!

## Features

- Plenty of online tutorials with rich theory illustration and code demonstrations, making it easy to get started
- Complete core modules for universal simulation with strong extendability and friendly user experience
- Efficient transpiling and simulation algorithms to accelerate the code execution with high performance

## Getting Started with MBQC from Scratch

### What is MBQC

Measurement-based quantum computing is a technique route for universal quantum computing, different from the quantum circuit model. As its name suggests, MBQC controls the computation by measuring part of the qubits of an entangled state, with those remaining unmeasured undergoing the evolution correspondingly. By controlling measurements, we can complete any desired evolution [1-8].

The standard computational process of MBQC is divided into three main steps: graph state preparation, single-qubit measurement, and byproduct correction. An equivalent way to describe this process is the measurement pattern (or "EMC" language) of MBQC [9]. Please refer to [MBQC Quick Start Guide](MBQC_CN.ipynb) for more details.

### Submodules

|Submodule|Description|
|:---|:---|
|`simulator`|contains a frequently used class and accompanying simulation tools for constructing MBQC models.|
|`mcalculus`|contains operations for the manipulation of measurement patterns.|
|`transpiler`|contains tools for transpiling circuits to measurement patterns.|
|`qobject`|contains common objects for quantum information processing, such as quantum states, quantum circuits, measurement patterns, etc.|
|`utils`|contains various common classes and functions used for computation.|

### Basic usage of the module

Here we briefly introduce the basic usage of the module.

#### Running with standard processes

In the `simulator`, we define a class `MBQC` to be instantiated to build your own MBQC model. Most of the MBQC algorithms are implemented by this class. We can first set the underlying graph of their MBQC model, then call the class method `measure` to measure each vertex in the graph, and finally correct the corresponding byproducts to finish the computation. Here we take the implementation of a single-qubit `Hadamard` gate as an example to briefly demonstrate its basic usage.

```python
# Import modules
from paddle_quantum.mbqc.simulator import MBQC
from paddle_quantum.mbqc.utils import basis
# Construct the underlying graph
vertices = ["1", "2"]
edges = [("1", "2")]
graph = [vertices, edges]
# Construct an MBQC model
mbqc = MBQC()
# Set the underlying graph
mbqc.set_graph(graph)
# Measure vertex '1' in Pauli X basis
mbqc.measure("1", basis("X"))
# Correct byproduct on vertex '2'
mbqc.correct_byproduct('X', "2", mbqc.sum_outcomes(["1"]))
# Obtain the quantum output
state_out = mbqc.get_quantum_output()
```

#### Running with measurement patterns

Both MBQC and circuit models can realize universal quantum computation and there is a one-to-one correspondence between them. However, the process of converting a quantum circuit to its MBQC equivalent is complicated and requires numerous calculations [9]. For this, we provide a module `transpiler` in our module for automatic translation from quantum circuits to measurement patterns. We can first call the ``Circuit`` in `qobject` to build a quantum circuit, just like calling [UAnstaz](https://qml.baidu.com/api/paddle_quantum.circuit.uansatz.html), then use the `transpiler` module to get an equivalent measurement pattern of the circuit, and finally send this pattern into the ``simulator`` module to run it. Again, here we briefly demonstrate the usage by the implementation of a single-qubit `Hadamard` gate as an example.

```python
# Import modules
from paddle_quantum.mbqc.qobject import Circuit
from paddle_quantum.mbqc.transpiler import transpile
from paddle_quantum.mbqc.simulator import MBQC
# Construct a circuit
width = 1
cir = Circuit(width)
cir.h(0)
# Transpile it to a pattern
pat = transpile(cir)
# Run the pattern in MBQC
mbqc = MBQC()
mbqc.set_pattern(pat)
mbqc.run_pattern()
# Obtain the quantum output
state_out = mbqc.get_quantum_output()
```

The above are the main contents of the current MBQC module. We can call relevant modules as needed in practice. In terms
of more functionalities of each module, please refer to the tutorials and the API documentation.

### Tutorials

Here, we provide three tutorials, each of which contains basic theory explanations and detailed code demonstrations. By following the tutorials and code practice, you will get familiar with the MBQC model as well as our module
shortly.

- [MBQC Quick Start Guide](MBQC_EN.ipynb)
- [Measurement-based Quantum Approximate Optimization Algorithm](QAOA_EN.ipynb)
- [Polynomial Unconstrained Boolean Optimization Problem in MBQC](PUBO_EN.ipynb)

## Frequently Asked Questions

- Q: Why should I study MBQC? What application scenarios does it have?

    A: MBQC is a universal quantum computing model parallel to the quantum circuit model. Compared to quantum gates, single-qubit measurements are easier to implement in practice with higher fidelity, and the part of non-adaptive measurements can even be realized simultaneously, drastically reducing the algorithm depth and thus the effect of decoherence on fidelity. In terms of classical simulation, since measurements commute between different qubits, the simulation process can be optimized by measurements reordering, leading to less consumption of the computational resource and thus improving the efficiency. In addition, the resource states in MBQC can be independent of specific computational tasks, which can be applied in quantum Internet for secure delegated quantum computing to protect users' privacy [10,11].

- Q: How is the MBQC physically implemented?

    A: The difficulty in the physical implementation of MBQC is mainly the preparation of resource states. Unlike the superconducting techniques used in the quantum circuit model, the resource states are mostly prepared using linear optics or cold atoms, see e.g. [2,12,13] for the currently available techniques for the resource states preparation.

## References

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
