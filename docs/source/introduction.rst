.. _header-n0:

Paddle Quantum
=======================

`Paddle Quantum <https://github.com/PaddlePaddle/Quantum>`__\  is the world's first cloud-integrated quantum machine learning platform based on Baidu PaddlePaddle. It supports the building and training of quantum neural networks, making PaddlePaddle the first deep learning framework in China. Paddle Quantum is feature-rich and easy to use. It provides comprehensive API documentation and tutorials help users get started right away.

.. figure:: https://release-data.cdn.bcebos.com/Paddle%20Quantum.png
   :target: https://github.com/PaddlePaddle/Quantum

Paddle Quantum has established a bridge between artificial intelligence and quantum computing. Through the Baidu PaddlePaddle deep learning platform to empower quantum computing, Paddle Quantum provides a powerful tool for people in the quantum AI industry and a feasible learning path for quantum computing enthusiasts.

    For more information about Paddle Quantum, please check the GitHub page: https://github.com/PaddlePaddle/Quantum

.. _header-n6:

Features
--------

- Easy-to-use

  - Many online learning resources (Nearly 50 tutorials)
  - High efficiency in building QNN with various QNN templates
  - Automatic differentiation

- Versatile

  - Multiple optimization tools and GPU mode
  - Simulation with 25+ qubits
  - Flexible noise models

- Featured Toolkits

  - Toolboxes for Chemistry & Optimization
  - LOCCNet for distributed quantum information processing
  - Self-developed quantum machine learning algorithms

.. _header-n15:

Install
--------

.. _header-n16:

Install PaddlePaddle
~~~~~~~~~~~~~~~~~~~~

This dependency will be automatically satisfied when users install Paddle Quantum. Please refer to `PaddlePaddle <https://www.paddlepaddle.org.cn/install/quick>`__ Install and configuration page. This project requires PaddlePaddle 2.2.0 to 2.3.0.

.. _header-n19:

Install Paddle Quantum
~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend installing Paddle Quantum with ``pip`` ,

.. code:: shell

   pip install paddle-quantum

or download all the files and finish the installation locally,

.. code:: shell

   git clone http://github.com/PaddlePaddle/quantum

.. code:: shell

   cd quantum
   pip install -e .

.. _header-n25:

Environment setup for Quantum Chemistry module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our `qchem` module is based on `Psi4`, so before executing quantum chemistry, we have to install this Python package.

.. note::

   It is recommended that `Psi4` is installed in a Python 3.8 environment.

We highly recommend you to install ``Psi4`` via conda. MacOS/Linux user can use the command

.. code:: shell

   conda install psi4 -c psi4

For **Windows** user, the command is

.. code:: shell

   conda install psi4 -c psi4 -c conda-forge

**Note:** Please refer to `Psi4 <https://psicode.org/installs/v14/>`__\  for more download options.

.. _header-n29:

Run programs
~~~~~~~~~~~~

Now, you can try to run a program to verify whether the Paddle Quantum has been installed successfully. Here we take quantum approximate optimization algorithm (QAOA) as an example.

.. code:: shell

   cd paddle_quantum/QAOA/example
   python main.py

..

.. note:: For the introduction of QAOA, please refer to our `QAOA tutorial </tutorials/combinatorial-optimization/quantum-approximate-optimization-algorithm.html>`__.

.. _header-n51:

Feedbacks
----------

- Users are encouraged to report issues and submit suggestions on `GitHub Issues <https://github.com/PaddlePaddle/Quantum/issues>`__.
- QQ group: 1076223166

.. _header-n118:

Copyright and License
---------------------

Paddle Quantum uses the `Apache-2.0 license <https://github.com/PaddlePaddle/Quantum/blob/master/LICENSE>`__ License.
