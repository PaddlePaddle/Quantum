# !/usr/bin/env python3
# Copyright (c) 2023 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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
Quantum-circuit associative adversarial network (QCAAN) model
"""

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.io as data
from paddle.vision.transforms import Compose, Normalize, ToTensor
from paddle.vision.datasets import MNIST
from tqdm import tqdm
from paddle_quantum.ansatz import Circuit
import paddle_quantum
from paddle_quantum.loss import ExpecVal
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import logging

warnings.filterwarnings("ignore", category=Warning)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


def Data_Load() -> paddle.vision.datasets.MNIST:
    r"""
    Load the MNIST dataset

    Returns:
        MNIST dataset

    """
    transform = Compose([ToTensor(), Normalize(mean=[127.5], std=[127.5])])
    dataset = MNIST(mode="train", transform=transform, backend="cv2")
    return dataset


class ConvBlock(nn.Layer):
    r"""
    Convolution block for building neural networks.

    Each ConvBlock consists of several convolution, Silu and layer normalization layers.
    And ConvBlock keeps the spatial dimensions unchanged, i.e., height and weight.

    Args:
        shape: The required shape for layer normalization.
        in_channels: The number of input channels in the input.
        out_channels: The number of output channels produced by the convolution.
        kernel_size: The size of the convolution kernel.
        stride: The stride size.
        padding: The padding size.
        activation: Activation function.
        normalize: A symbol indicating whether layer normalization is needed.
    """

    def __init__(self, shape: List[int], in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 activation: nn.Layer = None, normalize: bool = True) -> None:
        super(ConvBlock, self).__init__()
        self.layer_norm = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2D(in_channels, out_channels,
                               kernel_size, stride, padding)
        self.conv2 = nn.Conv2D(out_channels, out_channels,
                               kernel_size, stride, padding)
        self.conv3 = nn.Conv2D(out_channels, out_channels,
                               kernel_size, stride, padding)
        self.conv4 = nn.Conv2D(out_channels, out_channels,
                               kernel_size, stride, padding)
        self.activation = nn.Silu() if activation is None else activation
        self.normalize = normalize

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        r"""
        The ConvBlock's forward function

        Args:
            x: The input tensor

        Returns:
            The output tensor

        """
        x = self.layer_norm(x) if self.normalize else x
        x = self.conv1(x)
        x = self.activation(x)
        # x = self.conv2(x)
        # x = self.activation(x)
        # x = self.conv3(x)
        # x = self.activation(x)
        # x = self.conv4(x)
        # x = self.activation(x)
        return x


class Generator(nn.Layer):
    r"""
    The Generator network

    Args:
        latent_dim: Latent feature numbers which represents the input dimension of the Generator.
    """

    def __init__(self, latent_dim: int = 16):
        super(Generator, self).__init__()

        self.channels = 128
        self.d2 = self.channels // 2
        self.d4 = self.channels // 4
        self.d8 = self.channels // 8

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.channels * 3 * 3),  # [-1, 128 * 3 * 3]
            nn.Silu(),
            nn.Linear(self.channels * 3 * 3, self.channels * 3 * 3)  # [-1, 128 * 3 * 3]
        )

        self.conv = nn.Sequential(
            ConvBlock([self.channels, 3, 3], self.channels, self.d2),  # [-1, 64, 3, 3]
            nn.Conv2DTranspose(self.d2, self.d2, 3, 2, 0),  # [-1, 64, 7, 7]

            ConvBlock([self.d2, 7, 7], self.d2, self.d4),  # [-1, 32, 7, 7]
            nn.Conv2DTranspose(self.d4, self.d4, 4, 2, 1),  # [-1, 32, 14, 14]

            ConvBlock([self.d4, 14, 14], self.d4, self.d8),  # [-1, 16, 14, 14]
            nn.Conv2DTranspose(self.d8, self.d8, 4, 2, 1),  # [-1, 16, 28, 28]

            ConvBlock([self.d8, 28, 28], self.d8, 1,
                      activation=nn.Tanh(), normalize=False),  # [-1, 1, 28, 28]
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        r"""
        The Generator's forward function

        Args:
            x: The input tensor

        Returns:
            The output tensor

        """
        x = self.fc(x)
        x = self.conv(paddle.reshape(x, [-1, self.channels, 3, 3]))
        return x


class Discriminator(nn.Layer):
    r"""
    The Discriminator network

    Args:
        latent_dim: Latent feature numbers which represents the input dimension of the Generator.
    """

    def __init__(self, latent_dim: int = 16):
        super(Discriminator, self).__init__()

        self.channels = 128
        self.d2 = self.channels // 2
        self.d4 = self.channels // 4
        self.d8 = self.channels // 8

        self.conv = nn.Sequential(
            ConvBlock([1, 28, 28], 1, self.d8, normalize=False),  # [-1, 16, 28, 28]

            nn.Conv2D(self.d8, self.d8, 4, 2, 1),  # [-1, 16, 14, 14]
            ConvBlock([self.d8, 14, 14], self.d8, self.d4),  # [-1, 32, 14, 14]

            nn.Conv2D(self.d4, self.d4, 4, 2, 1),  # [-1, 32, 7, 7]
            ConvBlock([self.d4, 7, 7], self.d4, self.d2),  # [-1, 64, 7, 7]

            nn.Conv2D(self.d2, self.d2, 3, 2, 0),  # [-1, 64, 3, 3]
            ConvBlock([self.d2, 3, 3], self.d2, self.channels),  # [-1, 128, 3, 3]
        )

        self.fc = nn.Sequential(
            nn.Linear(self.channels * 3 * 3, self.channels * 3 * 3),  # [-1, 128 * 3]
            nn.Silu(),
            nn.Linear(self.channels * 3 * 3, latent_dim),  # [-1, 16]
            nn.Tanh(),
        )

        self.output = nn.Sequential(
            nn.Linear(latent_dim, self.channels),  # [-1, 128]
            nn.Silu(),
            nn.Linear(self.channels, 1),  # [-1, 1]
            nn.Sigmoid()
        )

    def forward(self, x: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
        r"""
        The Discriminator's forward function

        Args:
            x: The input tensor

        Returns:
            The output tensor

        """
        x = self.conv(x)
        x_tanh = self.fc(paddle.reshape(x, [-1, self.channels * 3 * 3]))
        x = self.output(x_tanh)
        return x, x_tanh


def generate_pauli_string_list(num_qubits: int, num_terms: int) -> List[list]:
    r"""
    Generate the Pauli string list.
    Args:
        num_qubits: The number of the qubits.
        num_terms: The number of the generated observables.
    Returns:
        Return the generated Pauli string list.
    """
    ob = [[[1.0, f'z{idx:d}']] for idx in range(num_qubits)]
    ob.extend([[1.0, f'x{idx:d}']] for idx in range(num_qubits))
    ob.extend([[1.0, f'y{idx:d}']] for idx in range(num_qubits))
    if len(ob) >= num_terms:
        ob = ob[:num_terms]
    else:
        ob.extend(ob * (num_terms // len(ob) - 1))
        ob.extend(ob[:num_terms % len(ob)])
    return ob


class QCBM(nn.Layer):
    r"""
    Quantum Circuit Boltzmann Machine, which is exactly a quantum neural network here.

    Args:
        num_qubits: The number of qubits which the quantum circuit contains.
        num_depths: The number of depths of complex entangled layers which the quantum circuit contains.
        latent_dim: Latent feature numbers which represents the input dimension of the generator.
    """

    def __init__(self, num_qubits: int, num_depths: int, latent_dim: int = 16):
        super(QCBM, self).__init__()

        self.num_qubits = num_qubits
        self.latent_dim = latent_dim

        # 定义量子电路
        self.cir = Circuit(num_qubits=num_qubits)
        self.cir.complex_entangled_layer(depth=num_depths)
        self.cir.rz(qubits_idx="full")
        self.cir.ry(qubits_idx="full")
        self.cir.rz(qubits_idx="full")

    def forward(self) -> paddle.Tensor:
        r"""
        The forward function of QCBM

        Returns:
            A series of expectation values on Z0, Z1, ..., X0, X1, ..., Y0, Y1, ...

        """
        final_state = self.cir()

        # 定义一系列观测量
        pauli_list = generate_pauli_string_list(num_qubits=self.num_qubits, num_terms=self.latent_dim)
        obs = [paddle_quantum.Hamiltonian(pauli) for pauli in pauli_list]

        # 计算一系列观测量的平均值
        expec_obs = [ExpecVal(ob)(final_state) for ob in obs]
        expec_obs = paddle.concat(expec_obs)
        return expec_obs


def prior_sampling(expec_obs: paddle.Tensor, batch_size: int) -> paddle.Tensor:
    r"""
    Simulate the sampling process of the prior distribution, from QCBM.

    Args:
        expec_obs: The vector of expectation values whose length is the same as 'latent_dim'.
        batch_size: The number of samples in a batch.

    Returns:
        The sampling results, values in {-1, 1}.

    """
    prior_samples = paddle.rand([batch_size, len(expec_obs)])
    expec_obs_probs = (expec_obs + 1) / 2.
    prior_samples = paddle.cast(expec_obs_probs - prior_samples >= 0, dtype="float32")
    prior_samples = prior_samples * 2 - 1
    return prior_samples


def train(
        # Define the hyperparameters
        num_qubits: int = 8,
        num_depths: int = 4,
        lr_qnn: paddle.float32 = 0.005,
        batch_size: int = 128,
        latent_dim: int = 16,
        epochs: int = 21,
        lr_g: paddle.float32 = 0.0002,
        lr_d: paddle.float32 = 0.0002,
        beta1: paddle.float32 = 0.5,
        beta2: paddle.float32 = 0.9,
        manual_seed: int = 888,
) -> None:
    r"""
    The training function of QCAAN model

    Args:
        num_qubits: The number of qubits which the quantum circuit contains.
        num_depths: The number of depths of complex entangled layers which the quantum circuit contains.
        lr_qnn: The learning rate used to update the QNN parameters, default to 0.005.
        batch_size: The batch size in each iteration.
        latent_dim: Latent feature numbers which represents the input dimension of the generator.
        epochs: The number of epochs to train the model.
        lr_g: The learning rate used to update the generator parameters, default to 0.0002.
        lr_d: The learning rate used to update the discriminator parameters, default to 0.0002.
        beta1: The beta1 used in Adam optimizer of generator and discriminator, default to 0.5.
        beta2: The beta2 used in Adam optimizer of generator and discriminator, default to 0.9.
        manual_seed: The manual seed for reproducibility.

    Returns:
        The parameters of QNN, Generator and Discriminator are saved.

    """
    # Set random seed for reproducibility
    paddle.seed(manual_seed)
    np.random.seed(manual_seed)

    paddle.set_device(paddle.device.get_device())  # Run on GPU if available
    print(f"\nThis program is running on your {paddle.device.get_device()}!\n")

    # Define the data loader
    dataset = Data_Load()
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the G and D networks and quantum circuit for sampling
    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator(latent_dim=latent_dim)
    cir = QCBM(num_qubits=num_qubits, num_depths=num_depths, latent_dim=latent_dim)

    # Define the loss functions and optimizers
    bce_loss = nn.BCELoss()
    mse_loss = nn.loss.MSELoss()

    opt_g = optim.Adam(parameters=generator.parameters(),
                       learning_rate=lr_g, beta1=beta1, beta2=beta2)
    opt_d = optim.Adam(parameters=discriminator.parameters(),
                       learning_rate=lr_d, beta1=beta1, beta2=beta2)
    opt_qnn = optim.Adam(parameters=cir.parameters(),
                         learning_rate=lr_qnn)

    # Train the GAN
    for epoch in tqdm(range(epochs), ncols=80):
        qnn_update_interval = paddle.randint(50, 100).numpy()[0]
        print("QNN update interval in this epoch:", qnn_update_interval)
        for i, (real_data, _) in enumerate(tqdm(loader, ncols=80)):
            # Define the labels for real and fake data
            real_labels = paddle.ones((len(real_data), 1))
            fake_labels = paddle.zeros((len(real_data), 1))

            # Train the qnn
            if i % qnn_update_interval == 0:
                opt_qnn.clear_grad()
                expec_obs = cir()
                _, real_tanh = discriminator(real_data)
                qnn_loss = mse_loss(expec_obs, paddle.mean(real_tanh.detach(), axis=0))
                qnn_loss.backward()
                opt_qnn.step()

            # Train the discriminator
            opt_d.clear_grad()
            # Generate fake data and compute the discriminator
            prior_samples = prior_sampling(expec_obs.detach(), batch_size=len(real_data))
            fake_data = generator(prior_samples)
            fake_scores, _ = discriminator(fake_data.detach())
            real_scores, _ = discriminator(real_data)
            d_loss = bce_loss(fake_scores, fake_labels) + bce_loss(real_scores, real_labels)
            d_loss.backward()
            opt_d.step()

            # Train the generator
            opt_g.clear_grad()
            fake_scores, _ = discriminator(fake_data)
            g_loss = bce_loss(fake_scores, real_labels)
            g_loss.backward()
            opt_g.step()

        # # Print the losses every 100 steps
        # if i % 100 == 0:
        print("\nEpoch [%d/%d]," % (epoch, epochs),
              "Discriminator Loss: :%.4f, Generator Loss: %.4f, " % (d_loss.numpy(), g_loss.numpy()),
              "QCBM Loss: %.4f" % qnn_loss.numpy())
        if epoch == epochs - 1:
            if not os.path.exists("params"):
                os.makedirs("params")
            paddle.save(cir.state_dict(), "params/params_qnn_epoch%d.pdparams" % epoch)
            paddle.save(generator.state_dict(), "params/params_G_epoch%d.pdparams" % epoch)
            # paddle.save(discriminator.state_dict(), "params/params_D_epoch%d.pdparams" % epoch)


# 加载保存的模型参数，并生成一些图片，最后将这些图片保存。
def model_test(
        model_name: str = 'qcaan-model',
        latent_dim: int = 16,
        params_path: str = 'params',
        num_qubits: int = 8,
        num_depths: int = 4,
        manual_seed: int = 20230313,
) -> None:
    r"""
    Load the saved model parameters, and generate some pictures. Finally save these pictures.

    Args:
        model_name: The name of the model, which is used to save the model.
        latent_dim: Latent feature numbers which represents the input dimension of the generator.
        params_path: The path to load the parameters of the trained model. Both relative and absolute paths are allowed.
        num_qubits: The number of qubits which the quantum circuit contains.
        num_depths: The number of depths of complex entangled layers which the quantum circuit contains.
        manual_seed: The manual seed for reproducibility.

    Returns:
        The generated pictures are saved to a .png file.

    """

    logging.basicConfig(
        filename='./qcaan_inference.log',
        filemode='w',
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO
    )
    logging.info(
        f"\nThe full config for inferring the QC-AAN model.\n"
        f"The mode of this config. Available values: 'train' | 'inference'.\n"
        f"  mode = 'inference'\n"
        f"The name of the model, which is used to save the model.\n"
        f"  model_name = '{model_name}'\n"
        f"The path to load the parameters of the trained model. Both relative and absolute paths are allowed.\n"
        f"  params_path = '{params_path}'\n"
        f"The number of qubits which the quantum circuit contains.\n"
        f"  num_qubits = {num_depths}\n"
        f"The number of depths of complex entangled layers which the quantum circuit contains.\n"
        f"  num_depths = {num_qubits}\n"
        f"Latent feature numbers which represents the input dimension of the generator.\n"
        f"  latent_dim = {latent_dim}\n"
        f"The manual seed for reproducibility.\n"
        f"  manual_seed = {manual_seed}\n"
    )

    # Set random seed for reproducibility
    paddle.seed(manual_seed)
    np.random.seed(manual_seed)

    paddle.set_device(paddle.device.get_device())  # Run on GPU if available
    print(f"\nThis program is running on your {paddle.device.get_device()}!\n")

    # load models
    gen = Generator(latent_dim=latent_dim)
    cir = QCBM(num_qubits=num_qubits, num_depths=num_depths, latent_dim=latent_dim)
    # load weights
    load_epoch = 20
    gen.set_state_dict(paddle.load(f"{params_path}/params_G_epoch{load_epoch}.pdparams"))
    gen.eval()
    cir.set_state_dict(paddle.load(f"{params_path}/params_qnn_epoch{load_epoch}.pdparams"))
    cir.eval()
    print("Model loaded, generating new images...")

    # 定义总生成图片数量
    num_shows_rows, num_shows_cols = 6, 10
    # 总共展示 11x10 张图片。上半部分是生成的，下面 40 张是原图。
    fig, axs = plt.subplots(num_shows_rows + 5, num_shows_cols)
    # plt.rcParams["figure.figsize"] = [28, 30]
    plt.gray()

    # generating pics using trained Generator
    expec_obs = cir()
    prior_samples = prior_sampling(expec_obs.detach(), batch_size=num_shows_rows * num_shows_cols)
    print("The first 10 QNN samples are listed below:\n", prior_samples.numpy()[0:10])

    with paddle.no_grad():
        gen_pics = gen(prior_samples).numpy()

    for r in range(num_shows_rows):
        for c in range(num_shows_cols):
            axs[r, c].imshow(gen_pics[r * num_shows_cols + c].reshape(28, 28))
            axs[r, c].axis("off")

    for c in range(num_shows_cols):
        axs[num_shows_rows, c].axis("off")

    # 获取一些初始图片并展示，作为对比
    # Define the data loader
    dataset = Data_Load()
    loader = data.DataLoader(dataset, batch_size=40, shuffle=True)
    rand_i = np.random.randint(low=1, high=10, size=1)
    for i, (real_data, _) in enumerate(loader):
        if i == rand_i:
            original_pics = real_data.numpy()
            break

    for r in range(4):
        for c in range(num_shows_cols):
            axs[r + num_shows_rows + 1, c].imshow(original_pics[r * num_shows_cols + c].reshape(28, 28))
            axs[r + num_shows_rows + 1, c].axis("off")

    axs[0, 4].set_title("Generated pictures by QCAAN")
    axs[7, 4].set_title("Original pictures from MNIST")

    if not os.path.exists("gen_pics"):
        os.makedirs("gen_pics")
    plt.savefig(f"gen_pics/qcaan_generated_vs_original_{manual_seed}.png")
    print(f"\nThe generated pictures are saved to the file "
          f"named 'gen_pics/qcaan_generated_vs_original_{manual_seed}.png'.\n")
    plt.show()


if __name__ == "__main__":
    exit(0)
