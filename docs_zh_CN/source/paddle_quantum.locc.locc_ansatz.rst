paddle\_quantum.locc.locc\_ansatz
========================================

LOCC的电路的功能实现。

.. py:class:: LoccAnsatz(party)

   基类：:py:class:`paddle_quantum.ansatz.circuit.Circuit`

   继承 ``Circuit`` 类，目的是建立在 LOCC 任务上的电路模板。

   :param party: 参与方。
   :type party: LoccParty

   .. py:method:: append(operator)

      增加一个 Operator 类。

      :param operator: 一个（附带名字的）Operator 类。
      :type operator: Union[Iterable, paddle_quantum.Operator]

   .. py:method:: extend(operators)

      增加一组 Operator 类。

      :param operators: 一组 Operator 类。
      :type operators: List[Operator]
   
   .. py:method:: insert(index, operator)

      在指定位置插入一个 Operator 类。

      :param index: 插入的位置。
      :type index: int
      :param operator: 一个 Operator。
      :type operator: Operator
   
   .. py:method:: pop(operator)

      在指定 Operator 下删除一个 Operator 类。

      :param operator: 指定删除的 Operator。
      :type operator: Operator
   
   .. py:method:: forward(state)

      前向传播输入数据。

      :param state: 输入数据。
      :type state: LoccState
      
      :return: 输出数据。
      :rtype: LoccState

   .. py:method:: h(qubits_idx='full', num_qubits=None, depth=1)

      添加单量子比特 Hadamard 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: s(qubits_idx='full', num_qubits=None, depth=1)

      添加单量子比特 S 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: t(qubits_idx='full', num_qubits=None, depth=1)

      添加单量子比特 T 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: x(qubits_idx='full', num_qubits=None, depth=1)

      添加单量子比特 X 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: y(qubits_idx='full', num_qubits=None, depth=1)

      添加单量子比特 Y 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: z(qubits_idx='full', num_qubits=None, depth=1)

      添加单量子比特 Z 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: p(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加单量子比特 P 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: rx(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加关于 x 轴的单量子比特旋转门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: ry(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加关于 y 轴的单量子比特旋转门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: rz(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加关于 z 轴的单量子比特旋转门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: u3(qubits_idx='full', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加单量子比特旋转门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'full'``。
      :type qubits_idx: Union[Iterable, int, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: cnot(qubits_idx='cycle', num_qubits=None, depth=1)

      添加 CNOT 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: cx(qubits_idx='cycle', num_qubits=None, depth=1)

      同 cnot。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: cy(qubits_idx='cycle', num_qubits=None, depth=1)

      添加受控 Y 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: cz(qubits_idx='cycle', num_qubits=None, depth=1)

      添加受控 Z 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: swap(qubits_idx='cycle', num_qubits=None, depth=1)

      添加 SWAP 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: cp(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加受控 P 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: crx(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加关于 x 轴的受控单量子比特旋转门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: cry(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加关于 y 轴的受控单量子比特旋转门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: crz(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加关于 z 轴的受控单量子比特旋转门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: cu(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加受控单量子比特旋转门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: rxx(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加 RXX 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: ryy(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加 RYY 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: rzz(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加 RZZ 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: ms(qubits_idx='cycle', num_qubits=None, depth=1)

      添加 Mølmer-Sørensen (MS) 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: cswap(qubits_idx='cycle', num_qubits=None, depth=1)

      添加 CSWAP (Fredkin) 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: ccx(qubits_idx='cycle', num_qubits=None, depth=1)

      添加 CCX 门。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: universal_two_qubits(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加两量子比特通用门，该通用门需要 15 个参数。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: universal_three_qubits(qubits_idx='cycle', num_qubits=None, depth=1, param=None, param_sharing=False)

      添加三量子比特通用门，该通用门需要 81 个参数。

      :param qubits_idx: 作用在的量子比特的编号，默认为 ``'cycle'``。
      :type qubits_idx: Union[Iterable, str], optional
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
      :param param: 量子门参数，默认为 ``None``。
      :type param: Union[paddle.Tensor, float], optional
      :param param_sharing: 同一层中的量子门是否共享参数，默认为 ``False``。
      :type param_sharing: bool, optional

   .. py:method:: oracle(oracle, qubits_idx, num_qubits=None, depth=1)

      添加一个 oracle 门。

      :param oracle: 要实现的 oracle。
      :type oracle: paddle.tensor
      :param qubits_idx: 作用在的量子比特的编号。
      :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional

   .. py:method:: control_oracle(oracle, qubits_idx, num_qubits=None, depth=1)

      添加一个受控 oracle 门。

      :param oracle: 要实现的 oracle。
      :type oracle: paddle.tensor
      :param qubits_idx: 作用在的量子比特的编号。
      :type qubits_idx: Union[Iterable[Iterable[int]], Iterable[int], int]
      :param num_qubits: 总的量子比特个数，默认为 ``None``。
      :type num_qubits: int, optional
      :param depth: 层数，默认为 ``1``。
      :type depth: int, optional
