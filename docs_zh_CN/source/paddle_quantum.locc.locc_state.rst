paddle\_quantum.locc.locc\_state
=======================================

LOCC中量子态的功能实现。

.. py:class:: LoccState(data=None, prob=None, measured_result=None, num_qubits=None, backend=None, dtype=None)

   基类：:py:class:`paddle_quantum.state.state.State`

   LOCCNet 中的一个 LOCC 态。

   由于我们在 LOCC 中不仅关心量子态的解析形式，同时还关心得到它的概率，以及是经过怎样的测量而得到的。
   因此该类包含三个成员变量：量子态 ``data``、得到这个态的概率 ``prob``，和得到这个态的测量的测量结果是什么，
   即 ``measured_result``。

   :param data: 量子态的矩阵形式，默认为 ``None``。
   :type data: paddle.Tensor, optional
   :param prob: 得到该量子态的概率，默认为 ``None``。
   :type prob: paddle.Tensor, optional
   :param measured_result: 得到该量子态的测量的测量结果，默认为 ``None``。
   :type measured_result: str, optional
   :param num_qubits: 总的量子比特个数，默认为 ``None``。
   :type num_qubits: int, optional
   :param backend: 量桨的后端，默认为 ``None``。
   :type backend: paddle_quantum.Backend, optional
   :param dtype: 数据的类型，默认为 ``None``。
   :type dtype: str, optional

   .. py:method:: clone()

      创建一个当前对象的副本。

      :return: 当前对象的副本。
      :rtype: LoccState
