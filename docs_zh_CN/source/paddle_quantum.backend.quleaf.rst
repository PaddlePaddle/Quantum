paddle\_quantum.backend.quleaf
=====================================

量易伏后端的功能实现。

.. py:function:: set_quleaf_backend(backend)

   设置量易伏的后端实现。

   :param backend: 你想要设置的后端名称。
   :type backend: str

.. py:function:: get_quleaf_backend()

   得到量易伏的当前后端。

   :return: 量易伏当前的后端名称。
   :rtype: QCompute.BackendName

.. py:function:: set_quleaf_token(token)

   设置量易伏的 token。

   当使用云端服务器的时候，需要输入 token 才能使用。

   :param token: 你的 token。
   :type token: str

.. py:function:: get_quleaf_token()

   得到量易伏的当前 token。

   :return: 你所设置的 token。
   :rtype: str

.. py:class:: ExpecValOp(paddle.autograd.PyLayer)

   基类：:py:class:`paddle.autograd.PyLayer`

   .. py:staticmethod:: forward(ctx, param, state, hamiltonian, shots)

      前向函数，用于在量易伏后端中实现可观测量对于量子态的期望值的算子。

      :param ctx: 用于保持在反向传播过程中可能用到的变量。
      :type ctx: paddle.autograd.PyLayerContext
      :param param: 在先前的量子门中所包含的参数。
      :type param: paddle.Tensor
      :param state: 要被测量的量子态。
      :type state: paddle_quantum.State
      :param hamiltonian: 可观测量。
      :type hamiltonian: paddle_quantum.Hamiltonian
      :param shots: 测量次数。
      :type shots: int
      :return: 可观测量对量子态的期望值。
      :rtype: paddle.Tensor

   .. py:staticmethod:: backward(ctx, expec_val_grad)

      反向传播函数，用于计算输入参数的梯度。

      :param ctx: 得到前向函数中存储的变量。
      :type ctx: paddle.autograd.PyLayerContext
      :param expec_val_grad: 期望值本身已有的梯度。
      :type expec_val_grad: paddle.Tensor
      :return: 量子门的参数的梯度值。
      :rtype: paddle.Tensor
