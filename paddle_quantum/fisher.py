# !/usr/bin/env python3
# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
The source file of the class for the fisher information.
"""

from typing import Optional, Tuple, Union, List
import numpy as np
import paddle
from tqdm import tqdm
from scipy.special import logsumexp
from .ansatz import Circuit
import paddle_quantum


class QuantumFisher:
    r"""Quantum fisher information (QFI) & related calculators.

    Args:
        cir: Parameterized quantum circuits requiring calculation of quantum Fisher information.
        
    Note:
        This class does not fit the situation when parameters among gates are correlated, such as control-gates.
    
    """
    def __init__(self, cir: Circuit):
        self.cir = cir
        paddle_quantum.set_backend('state_vector')

    def get_qfisher_matrix(self) -> np.ndarray:
        r"""Use parameter shift rule of order 2 to calculate the matrix of QFI.

        Returns:
            Matrix of QFI.

        .. code-block:: python

            import paddle
            from paddle_quantum.ansatz import Circuit
            from paddle_quantum.fisher import QuantumFisher

            cir = Circuit(1)
            zero = paddle.zeros([1], dtype="float64")
            cir.ry(0, param=zero)
            cir.rz(0, param=zero)

            qf = QuantumFisher(cir)
            qfim = qf.get_qfisher_matrix()
            print(f'The QFIM at {cir.param.tolist()} is \n {qfim}.')

        ::
        
            The QFIM at [0.0, 0.0] is
            [[1. 0.]
            [0. 0.]].

        """
        # Get the real-time parameters from the Circuit class
        list_param = self.cir.param
        num_param = len(list_param)
        # Initialize a numpy array to record the QFIM
        qfim = np.zeros((num_param, num_param))
        # Assign the signs corresponding to the four terms in a QFIM element
        list_sign = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        # Run the circuit and record the current state vector
        psi = self.cir().numpy()
        # For each QFIM element
        for i in range(0, num_param):
            for j in range(i, num_param):
                # For each term in each element
                for sign_i, sign_j in list_sign:
                    # Shift the parameters by pi/2 * sign
                    list_param[i] += np.pi / 2 * sign_i
                    list_param[j] += np.pi / 2 * sign_j
                    # Update the parameters in the circuit
                    self.cir.update_param(list_param)
                    # Run the shifted circuit and record the shifted state vector
                    psi_shift = self.cir().numpy()
                    # Calculate each term as the fidelity with a sign factor
                    qfim[i][j] += abs(np.vdot(
                        psi_shift, psi))**2 * sign_i * sign_j * (-0.5)
                    # De-shift the parameters
                    list_param[i] -= np.pi / 2 * sign_i
                    list_param[j] -= np.pi / 2 * sign_j
                    self.cir.update_param(list_param)
                if i != j:
                    # The QFIM is symmetric
                    qfim[j][i] = qfim[i][j]

        return qfim

    def get_qfisher_norm(self, direction: np.ndarray, step_size: Optional[float] = 0.01) -> float:
        r"""Use finite difference rule to calculate the projection norm of QFI along particular direction.

        Args:
            direction: A direction represented by a vector.
            step_size: Step size of the finite difference rule. Defaults to ``0.01``。

        Returns:
            Projection norm.

        .. code-block:: python

            import paddle
            from paddle_quantum.ansatz import Circuit
            from paddle_quantum.fisher import QuantumFisher

            cir = Circuit(2)
            zero = paddle.zeros([1], dtype="float64")
            cir.ry(0, param=zero)
            cir.ry(1, param=zero)
            cir.cnot(qubits_idx=[0, 1])
            cir.ry(0, param=zero)
            cir.ry(1, param=zero)

            qf = QuantumFisher(cir)
            v = [1,1,1,1]
            qfi_norm = qf.get_qfisher_norm(direction=v)
            print(f'The QFI norm along {v} at {cir.param.tolist()} is {qfi_norm:.7f}')

        ::

            The QFI norm along [1, 1, 1, 1] at [0.0, 0.0, 0.0, 0.0] is 6.0031546

        """
        # Get the real-time parameters
        list_param = self.cir.param
        # Run the circuit and record the current state vector
        psi = self.cir().numpy()
        # Check whether the length of the input direction vector is equal to the number of the variational parameters
        assert len(list_param) == len(
            direction
        ), "the length of direction vector should be equal to the number of the parameters"
        # Shift the parameters by step_size * direction
        array_params_shift = np.array(
            list_param) + np.array(direction) * step_size
        # Update the parameters in the circuit
        self.cir.update_param(array_params_shift)
        # Run the shifted circuit and record the shifted state vector
        psi_shift = self.cir().numpy()
        # Calculate quantum Fisher-Rao norm along the given direction
        qfisher_norm = (1 - abs(np.vdot(psi_shift, psi))**2) * 4 / step_size**2
        # De-shift the parameters and update
        self.cir.update_param(list_param)

        return qfisher_norm

    def get_eff_qdim(self, num_param_samples: Optional[int] = 4, tol: Optional[float] = None) -> int:
        r"""Calculate the effective quantum dimension, i.e. the maximum rank of QFI matrix in the whole parameter space.

        Args:
            num_param_samples: Number of samples to estimate the dimension. Defaults to ``4``.
            tol: Minimum tolerance of the singular values to be 0. Defaults to ``None``, with the same meaning
                as in ``numpy.linalg.matrix_rank()``.

        Returns:
            Effective quantum dimension of the quantum circuit.

        .. code-block:: python

            import paddle
            from paddle_quantum.ansatz import Circuit
            from paddle_quantum.fisher import QuantumFisher

            cir = Circuit(1)
            zero = paddle.zeros([1], dtype="float64")
            cir.rz(0, param=zero)
            cir.ry(0, param=zero)

            qf = QuantumFisher(cir)
            print(cir)
            print(f'The number of parameters of -Rz-Ry- is {len(cir.param.tolist())}')
            print(f'The effective quantum dimension of -Rz-Ry- is {qf.get_eff_qdim()}')

        ::

            --Rz(0.000)----Ry(0.000)--

            The number of parameters of -Rz-Ry- is 2
            The effective quantum dimension of -Rz-Ry- is 1
        """
        # Get the real-time parameters
        list_param = self.cir.param.tolist()
        num_param = len(list_param)
        # Generate random parameters
        param_samples = 2 * np.pi * np.random.random(
            (num_param_samples, num_param))
        # Record the ranks
        list_ranks = []
        # Here it has been assumed that the set of points that do not maximize the rank of QFIMs, as singularities, form a null set.
        # Thus one can find the maximal rank using a few samples.
        for param in param_samples:
            # Set the random parameters
            self.cir.update_param(param)
            # Calculate the ranks
            list_ranks.append(self.get_qfisher_rank(tol))
        # Recover the original parameters
        self.cir.update_param(list_param)

        return max(list_ranks)

    def get_qfisher_rank(self, tol: Optional[float] = None) -> int:
        r"""Calculate the rank of the QFI matrix.

        Args:
            tol: Minimum tolerance of the singular values to be 0. Defaults to ``None``, with the same meaning
                as in ``numpy.linalg.matrix_rank()``.

        Returns:
            Rank of the QFI matrix.
        """
        qfisher_rank = np.linalg.matrix_rank(self.get_qfisher_matrix().astype('float64'),
                                             1e-6,
                                             hermitian=True)
        return qfisher_rank


class ClassicalFisher:
    r"""Classical fisher information (CFI) & related calculators.

    Args:
        model: Instance of the classical or quantum neural network model.
        num_thetas: Number of the parameter sets.
        num_inputs: Number of the input samples.
        model_type: Model type is ``'classical'`` or ``'quantum'``. Defaults to ``'quantum'``.
        **kwargs: including
        
            - size: list of sizes of classical NN units
            - num_qubits: number of qubits of quantum NN
            - depth: depth of quantum NN
            - encoding: ``IQP`` or ``re-uploading`` encoding of quantum NN
            
    Raises:
        ValueError: Unsupported encoding.
        ValueError: Unsupported model type.
    """
    def __init__(self,
                 model: paddle.nn.Layer,
                 num_thetas: int,
                 num_inputs: int,
                 model_type: str = 'quantum',
                 **kwargs: Union[List[int], int, str]):
        self.model = model
        self.num_thetas = num_thetas
        self.num_inputs = num_inputs
        self._model_type = model_type
        if self._model_type == 'classical':
            layer_dims = kwargs['size']
            self.model_args = [layer_dims]
            self.input_size = layer_dims[0]
            self.output_size = layer_dims[-1]
            self.num_params = sum(layer_dims[i] * layer_dims[i + 1]
                                  for i in range(len(layer_dims) - 1))
        elif self._model_type == 'quantum':
            num_qubits = kwargs['num_qubits']
            depth = kwargs['depth']
            # Supported QNN encoding: ‘IQP' and 're-uploading'
            encoding = kwargs['encoding']
            self.model_args = [num_qubits, depth, encoding]
            self.input_size = num_qubits
            # Default dimension of output layer = 1
            self.output_size = 1
            # Determine the number of model parameters for different encoding types
            if encoding == 'IQP':
                self.num_params = 3 * depth * num_qubits
            elif encoding == 're-uploading':
                self.num_params = 3 * (depth + 1) * num_qubits
            else:
                raise ValueError('Non-existent encoding method')
        else:
            raise ValueError(
                'The model type should be equal to either classical or quantum'
            )

        # Generate random data
        np.random.seed(0)
        x = np.random.normal(0, 1, size=(num_inputs, self.input_size))
        # Use the same input data for each theta set
        self.x = np.tile(x, (num_thetas, 1))

    def get_gradient(self, x: Union[np.ndarray, paddle.Tensor]) -> np.ndarray:
        r"""Calculate the gradients with respect to the variational parameters of the output layer.

        Args:
            x: Input samples.

        Returns:
            Gradient with respect to the variational parameters of the output layer with
            shape [num_inputs, dimension of the output layer, num_thetas].
        """
        if not paddle.is_tensor(x):
            x = paddle.to_tensor(x, stop_gradient=True)
        gradvectors = []
        seed = 0

        pbar = tqdm(desc="running in get_gradient: ",
                    total=len(x),
                    ncols=100,
                    ascii=True)

        for m in range(len(x)):
            pbar.update(1)
            if m % self.num_inputs == 0:
                seed += 1
            paddle.seed(seed)
            net = self.model(*self.model_args)
            output = net(x[m])
            logoutput = paddle.log(output)
            grad = []
            for i in range(self.output_size):
                net.clear_gradients()
                logoutput[i].backward(retain_graph=True)
                grads = []
                for param in net.parameters():
                    grads.append(param.grad.reshape((-1, )))
                gr = paddle.concat(grads)
                grad.append(gr * paddle.sqrt(output[i]))
            jacobian = paddle.concat(grad)
            # Jacobian matrix corresponding to each data point
            jacobian = paddle.reshape(jacobian,
                                      (self.output_size, self.num_params))
            gradvectors.append(jacobian.detach().numpy())

        pbar.close()

        return gradvectors

    def get_cfisher(self, gradients: np.ndarray) -> np.ndarray:
        r"""Use the Jacobian matrix to calculate the CFI matrix.

        Args:
            gradients: Gradients with respect to the variational parameter of the output layer.

        Returns:
            CFI matrix with shape [num_inputs, dimension of the output layer, num_theta].
        """
        fishers = np.zeros((len(gradients), self.num_params, self.num_params))
        for i in range(len(gradients)):
            grads = gradients[i]
            temp_sum = np.zeros(
                (self.output_size, self.num_params, self.num_params))
            for j in range(self.output_size):
                temp_sum[j] += np.array(
                    np.outer(grads[j], np.transpose(grads[j])))
            fishers[i] += np.sum(temp_sum, axis=0)

        return fishers

    def get_normalized_cfisher(self) -> Tuple[np.ndarray, float]:
        r"""Calculate the normalized CFI matrix.

        Returns:
            contains elements

            - CFI matrix with shape [num_inputs, num_theta, num_theta]
            - its trace

        """
        grads = self.get_gradient(self.x)
        fishers = self.get_cfisher(grads)
        fisher_trace = np.trace(np.average(fishers, axis=0))
        # Average over input data
        fisher = np.average(np.reshape(fishers,
                                       (self.num_thetas, self.num_inputs,
                                        self.num_params, self.num_params)),
                            axis=1)
        normalized_cfisher = self.num_params * fisher / fisher_trace

        return normalized_cfisher, fisher_trace

    def get_eff_dim(self, normalized_cfisher: np.ndarray, list_num_samples: List[int], gamma: Optional[int] = 1) -> List[int]:
        r"""Calculate the classical effective dimension.

        Args:
            normalized_cfisher: Normalized CFI matrix.
            list_num_samples: List of different numbers of samples.
            gamma: A parameter in the effective dimension. Defaults to ``1``.

        Returns:
            Classical effective dimensions for different numbers of samples.
        """
        eff_dims = []
        for n in list_num_samples:
            one_plus_F = np.eye(
                self.num_params) + normalized_cfisher * gamma * n / (
                    2 * np.pi * np.log(n))
            det = np.linalg.slogdet(one_plus_F)[1]
            r = det / 2
            eff_dims.append(2 * (logsumexp(r) - np.log(self.num_thetas)) /
                            np.log(gamma * n / (2 * np.pi * np.log(n))))

        return eff_dims
