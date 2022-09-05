paddle\_quantum.hamiltonian
==================================

哈密顿量的功能实现。

.. py:class:: Hamiltonian(pauli_str, compress=True)

   基类: :py:class:`object`

   Paddle Quantum 中的 Hamiltonian ``class``。

   用户可以通过一个 Pauli string 来实例化该 ``class``。

   :param pauli_str: 用列表定义的 Hamiltonian，如 ``[(1, 'Z0, Z1'), (2, 'I')]``。
   :type pauli_str: list
   :param compress: 是否对输入的 list 进行自动合并（例如 ``(1, 'Z0, Z1')`` 和 ``(2, 'Z1, Z0')`` 这两项将被自动合并），默认为 ``True``。
   :type compress: bool, optional

   .. note::
      
      若 ``compress`` 为 ``False``, 则不会检查输入的合法性。

   .. py:property:: n_terms()

      该哈密顿量的项数。

   .. py:property:: pauli_str()

      该哈密顿量对应的 Pauli string。

   .. py:property:: terms()

      该哈密顿量中的每一项，i.e. ``[['Z0, Z1'], ['I']]``。
   
   .. py:property:: coefficients()

      该哈密顿量中每一项的系数，i.e. ``[1.0, 2.0]``。

   .. py:property:: pauli_words()

      该哈密顿量中每一项对应的 Pauli word 构成的列表，i.e. ``['ZIZ', 'IIX']``。
   
   .. py:property:: pauli_words_r()

      该哈密顿量中每一项对应的简化（不包含 I） Pauli word 组成的列表，i.e. ``['ZXZZ', 'Z', 'X']``。

   .. py:property:: pauli_words_matrix()

      该哈密顿量中每一项对应的简化（不包含 I） Pauli word 对应的 matrix 组成的列表。

   .. py:property:: sites()

      该哈密顿量中的每一项对应的量子比特编号组成的列表。
   
   .. py:property:: n_qubits()

      该哈密顿量对应的量子比特数。
   
   .. py:method:: decompose_with_sites()

      将 pauli_str 分解为系数、泡利字符串的简化形式以及它们分别作用的量子比特下标。

      :return: 包含如下元素的 tuple:
                  - coefficients: 元素为每一项的系数。
                  - pauli_words_r: 元素为每一项的泡利字符串的简化形式，例如 'Z0, Z1, X3' 这一项的泡利字符串为 'ZZX'。
                  - sites: 元素为每一项作用的量子比特下标，例如 'Z0, Z1, X3' 这一项的 site 为 [0, 1, 3]。
      :rtype: Tuple[list]

   .. py:method:: decompose_pauli_words()

      将 pauli_str 分解为系数和泡利字符串。

      :return: 包含如下元素的 tuple:
                  - coefficients: 元素为每一项的系数。
                  - pauli_words: 元素为每一项的泡利字符串，例如 'Z0, Z1, X3' 这一项的泡利字符串为 'ZZIX'。
      :rtype: Tuple[list]
   
   .. py:method:: construct_h_matrix(qubit_num=None)
        
      构建 Hamiltonian 在 Z 基底下的矩阵。

      :param qubit_num: 量子比特数量，默认值为 ``1``。
      :type qubit_num: int, optional

      :return: Z 基底下的哈密顿量矩阵形式。
      :rtype: np.ndarray


.. py:class:: SpinOps(size, use_sparse=False)
   
   基类: :py:class:`object`

   矩阵表示下的自旋算符，可以用来构建哈密顿量矩阵或者自旋可观测量。
   
   :param size: 系统的大小
   :type size: int
   :param use_sparse: 是否使用 sparse matrix 计算，默认为 ``False``。
   :type use_sparse: bool, optional

   .. py:property:: sigz_p()

      :math:`S^z_i` 算符组成的列表，其中每一项对应不同的 :math:`i`。
   
   .. py:property:: sigy_p()

      :math:`S^y_i` 算符组成的列表，其中每一项对应不同的 :math:`i`。

   .. py:property:: sigx_p()

      :math:`S^x_i` 算符组成的列表，其中每一项对应不同的 :math:`i`。
