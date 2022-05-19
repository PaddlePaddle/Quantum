paddle\_quantum.locc.locc\_net
=====================================

LOCCNet的功能实现。

.. py:class:: LoccNet(state, qubits_idx)

   基类：:py:class:`paddle.fluid.dygraph.layers.Layer`

   用于设计 LOCC 协议，并进行训练或验证。

   .. py:method:: set_init_state(state, qubits_idx)

      初始化 LoccNet 的 LoccState。

      :param state: 输入的量子态的矩阵形式。
      :type state: paddle_quantum.State
      :param qubits_idx: 输入的量子态对应的量子比特的编号，其形式为 ``(party_id, qubit_id)`` 的 ``tuple`` ，或者由其组成的 ``list``。
      :type qubits_idx: Iterable
      :raises ValueError: 参与方的 ID 应为 ``str`` 或 ``int``。

   .. py:method:: partial_state(state, qubits_idx, is_desired=True)

      得到指定的量子比特的量子态。

      :param state: 输入的 LOCC 态。
      :type state: Union[List[LoccState], LoccState]
      :param qubits_idx: 指定的量子比特的编号，其形式为 ``(party_id, qubit_id)`` 的 ``tuple`` ，或者由其组成的 ``list``。
      :type qubits_idx: Iterable
      :param is_desired: 若为 ``True``，返回指定的量子比特的量子态；若为 ``False``，返回剩下的量子比特的量子态。 默认为 ``True``。
      :type is_desired: bool, optional
      :raises ValueError: 参与方的 ID 应为 ``str`` 或 ``int``。
      :raises ValueError: ``state`` 应为 ``LoccState`` 或者由其组成的 ``list``。
      :return: 得到部分量子态后的 LOCC 态。
      :rtype: Union[List[LoccState], LoccState]

   .. py:method:: reset_state(status, state, which_qubits)

      将指定的量子比特重置为输入的量子态。

      :param status: 重置前的 LOCC 态。
      :type status: Union[List[LoccState], LoccState]
      :param state: 输入的量子态。
      :type state: paddle_quantum.State
      :param which_qubits: 指定的量子比特的编号，其形式为 ``(party_id, qubit_id)`` 的 ``tuple`` ，或者由其组成的 ``list``。
      :type which_qubits: Iterable
      :raises ValueError: 参与方的 ID 应为 ``str`` 或 ``int``。
      :raises ValueError:  ``state`` 应为 ``LoccState`` 或者由其组成的 ``list``。
      :return: 重置部分量子比特后的 LOCC 态。
      :rtype: Union[List[LoccState], LoccState]

   .. py:method:: add_new_party(qubits_number, party_name=None)

      添加一个新的 LOCC 的参与方。

      :param qubits_number: 该参与方的量子比特个数。
      :type qubits_number: int
      :param party_name: 该参与方的名字，默认为 ``None``。
      :type party_name: str, optional

      .. note::
         可以使用字符串或者数字对参与方进行索引。如果想使用字符串索引，需要每次指定 ``party_name``；
         如果想使用数字索引，则不需要指定 ``party_name``，其索引数字会从 0 开始依次增长。

      :raises ValueError: ``party_name`` 应为 ``str``。
      :return: 该参与方的 ID。
      :rtype: Union[int, str]

   .. py:method:: create_ansatz(party_id)

      创建一个新的本地电路模板。

      :param party_id: 参与方的 ID。
      :type party_id: Union[int, str]
      :raises ValueError: 参与方的 ID 应为 ``str`` 或 ``int``。
      :return: 创建的本地电路模板。
      :rtype: LoccAnsatz

   .. py:method:: measure(status, which_qubits, results_desired, theta=None)

      对 LOCC 态进行 0-1 测量或含参测量。

      :param status: 待测量的 LOCC 态。
      :type status: Union[List[LoccState], LoccState]
      :param which_qubits: 测量的量子比特的编号。
      :type which_qubits: Iterable
      :param results_desired: 期望得到的测量结果。
      :type results_desired: Union[List[str], str]
      :param theta: 测量运算的参数，默认为 ``None``，表示 0-1 测量。
      :type theta: paddle.Tensor, optional
      :raises ValueError: ``results_desired`` 应为 ``str`` 或者由其组成的 ``list``。
      :raises ValueError: 参与方的 ID 应为 ``str`` 或 ``int``。
      :raises ValueError: ``status`` 应为 ``LoccState`` 或者由其组成的 ``list``。
      :return: 测量后的 LOCC 态。
      :rtype: Union[List[LoccState], LoccState]

   .. py:method:: get_num_qubits()

      得到该 LoccNet 的量子比特个数。

      :return: 该 LoccNet 的量子比特个数。
      :rtype: int
