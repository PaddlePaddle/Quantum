#!/usr/bin/env python
# Copyright (c) 2020 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Install library to site-packages
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='paddle-quantum',
    version='2.2.1',
    author='Institute for Quantum Computing, Baidu INC.',
    author_email='qml@baidu.com',
    description='Paddle Quantum is a quantum machine learning (QML) toolkit developed based on Baidu PaddlePaddle.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://qml.baidu.com',
    packages=[
        'paddle_quantum',
        'paddle_quantum.ansatz',
        'paddle_quantum.backend',
        'paddle_quantum.channel',
        'paddle_quantum.channel.functional',
        'paddle_quantum.gate',
        'paddle_quantum.gate.functional',
        'paddle_quantum.locc',
        'paddle_quantum.loss',
        'paddle_quantum.operator',
        'paddle_quantum.state',
        'paddle_quantum.qchem',
        'paddle_quantum.qsvt',
        'paddle_quantum.mbqc',
        'paddle_quantum.GIBBS',
        'paddle_quantum.GIBBS.example',
        'paddle_quantum.QAOA',
        'paddle_quantum.QAOA.example',
        'paddle_quantum.SSVQE',
        'paddle_quantum.SSVQE.example',
        'paddle_quantum.VQE',
        'paddle_quantum.VQE.example',
        'paddle_quantum.VQSD',
        'paddle_quantum.VQSD.example',
        'paddle_quantum.mbqc.gates',
        'paddle_quantum.mbqc.gates.mcalculus_tests',
        'paddle_quantum.mbqc.gates.simulator_tests',
        'paddle_quantum.mbqc.GRCS',
        'paddle_quantum.mbqc.GRCS.example',
        'paddle_quantum.mbqc.GRCS.example.rectangular',
        'paddle_quantum.mbqc.GRCS.example.rectangular.depth10',
        'paddle_quantum.mbqc.QAOA',
        'paddle_quantum.mbqc.QAOA.example',
        'paddle_quantum.mbqc.QKernel',
        'paddle_quantum.mbqc.QKernel.example',
        'paddle_quantum.mbqc.VQSVD',
        'paddle_quantum.mbqc.VQSVD.example',
    ],
    package_data={
        'paddle_quantum.VQE': ['*.xyz'],
        'paddle_quantum.VQE.example': ['*.xyz'],
        'paddle_quantum.mbqc.GRCS.example': ['*.txt'],
        'paddle_quantum.mbqc.GRCS.example.rectangular.depth10': ['*.txt'],
        'paddle_quantum.mbqc.QKernel.example': ['*.txt'],
        'paddle_quantum.mbqc.VQSVD.example': ['*.txt'],
    },
    install_requires=[
        'paddlepaddle<=2.3.0',
        'scipy>=1.8.1',
        'protobuf<=3.20.1',
        'networkx>=2.5',
        'qcompute',
        'matplotlib>=3.3.0',
        'tqdm',
        'openfermion',
        'pyscf; platform_system == "Linux" or platform_system == "Darwin"',
        'opencv-python',
        'scikit-learn',
        'fastdtw',
    ],
    python_requires='>=3.6, <4',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    project_urls={
        'Documentation': 'https://qml.baidu.com/api/introduction.html',
        'Source': 'https://github.com/PaddlePaddle/Quantum/',
        'Tracker': 'https://github.com/PaddlePaddle/Quantum/issues',
    },
)
