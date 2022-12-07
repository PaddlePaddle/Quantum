paddle\_quantum.qpp.laurent
===============================

劳伦类的定义和它的函数

.. py:class:: Laurent

   基类: :py:class:`object`

   为劳伦多项式定义的类，定义为 :math:`P:\mathbb{C}[X, X^{-1}] \to \mathbb{C} :x \mapsto \sum_{j = -L}^{L} p_j X^j`

   :param coef: 劳伦多项式系数的列表，排列为 :math:`\{p_{-L}, ..., p_{-1}, p_0, p_1, ..., p_L\}`
   :type coef: np.ndarray

   .. py:method:: __call__(X)

      计算P(X)的值
      :param X: 输入X
      :type X: Union[int, float, complex]

      :return: P(X)
      :rtype: complex

   .. py:property:: coef()

      以上升顺序给出多项式的系数序列
      
   .. py:property:: conj()

      给出多项式的共轭

   .. py:property:: roots()

      给出多项式根的列表
      
   .. py:property:: norm()

      给出多项式系数的绝对值平方之和

   .. py:property:: max_norm()

      给出多项式的系数的绝对值的最大值

   .. py:property:: parity()

      给出多项式的宇称
      
   .. py:method:: __copy__()

      复制劳伦多项式

      :return: 复制后的多项式
      :rtype: Laurent

   .. py:method:: __add__(other)

      劳伦多项式的相加

      :param other: 一个标量或一个劳伦多项式 :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`
      :type other: Any

      :raises TypeError: 不支持劳伦多项式和others的相加

   .. py:method:: __mul__(other)

      劳伦多项式的相乘

      :param other: 一个标量或一个劳伦多项式 :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`
      :type other: Any

      :raises TypeError: 不支持劳伦多项式和others的相乘

   .. py:method:: __sub__(other)

      劳伦多项式的相减

      :param other: 一个标量或一个劳伦多项式 :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`
      :type other: Any

   .. py:method:: __eq__(other)

      劳伦多项式的相等

      :param other: 一个标量或一个劳伦多项式 :math:`Q(x) = \sum_{j = -L}^{L} q_{j} X^j`
      :type other: Any

      :raises TypeError: 不支持劳伦多项式和 ``others`` 的相等

   .. py:method:: __str__()

      打印劳伦多项式

   .. py:method:: is_parity(p)

      检验劳伦多项式是否有确定宇称

      :param p: 宇称
      :type p: int

      :return:
        包含以下元素：

        -宇称是否是 ``p mod 2``
        -如果不是，返回破坏该宇称的绝对值最大的系数
        -如果不是，返回破坏该宇称的绝对值最小的系数
      :rtype: Tuple[bool, complex]

.. py:function:: revise_tol(t)

   回顾 ``TOL`` 的值

   :param t: TOL的值
   :type t: float

.. py:function:: ascending_coef(coef)

   通过 ``coef`` 从 ``P`` 和 ``Q`` 中计算角度

   :param coef: 排列成 :math:`\{ p_0, ..., p_L, p_{-L}, ..., p_{-1} \}` 的系数列表
   :type coef: np.ndarray

   :return: 排列成 :math:`\{ p_{-L}, ..., p_{-1}, p_0, p_1, ..., p_L \}` 的系数列表
   :rtype: np.ndarray

.. py:function:: remove_abs_error(data, tol)

   移除数据中的错误

   :param data: 数据数组
   :type data: np.ndarray
   :param tol: 容错率
   :type tol: Optional[float] = None

   :return: 除错后的数据
   :rtype: np.ndarray

.. py:function:: random_laurent_poly(deg, parity, is_real)

   随机生成一个劳伦多项式

   :param deg: 该多项式的度数
   :type deg: int
   :param parity: 该多项式的宇称，默认为 ``none``
   :type parity: Optional[int] = None
   :param is_real: 该多项式系数是否是实数，默认为 ``false``
   :type is_real: Optional[bool] = False

   :return: 一个模小于等于1的劳伦多项式
   :rtype: Laurent

.. py:function:: sqrt_generation(A)

   生成劳伦多项式 :math:`A` 的平方根

   :param A: 一个劳伦多项式
   :type A: Laurent

   :return: 一个模小于等于1的劳伦多项式
   :rtype: Laurent

.. py:function:: Q_generation(P)

   生成劳伦多项式 :math:`P` 的互补多项式

   :param P: 一个宇称为 :math:`deg` ，度数为 :math:`L` 的劳伦多项式
   :type P: Laurent

   :return: 一个宇称为 :math:`deg` ，度数为 :math:`L` 的劳伦多项式  :math:`Q` ，使得 :math:`PP^* + QQ^* = 1`
   :rtype: Laurent

.. py:function:: pair_generation(f)

   生成劳伦多项式 :math:`f` 的劳伦对

   :param f: 一个实的，偶次的，max_norm小于1的劳伦多项式
   :type f: Laurent

   :return: 劳伦多项式 :math:`P, Q` 使得  :math:`P = \sqrt{1 + f / 2}, Q = \sqrt{1 - f / 2}`
   :rtype: Laurent

.. py:function:: laurent_generator(fn, dx, deg, L)

   生成劳伦多项式 :math:`f` 的劳伦对

   :param fn: 要近似的函数
   :type fn: Callable[[np.ndarray], np.ndarray]
   :param dx: 数据点的采样频率
   :type dx: float
   :param deg: 劳伦多项式的度数
   :type deg: int
   :param L: 近似宽度的一半
   :type L: float

   :return: 一个度数为 ``deg`` 的，在区间 :math:`[-L, L]` 内近似`fn` 的劳伦多项式
   :rtype: Laurent

.. py:function:: deg_finder(fn, delta, l)

   找到一个度数，使得由 ``laurent_generator`` 生成的劳伦多项式具有小于1的max_norm

   :param fn: 要近似的函数
   :type fn: Callable[[np.ndarray], np.ndarray]
   :param delta: 数据点的采样频率，默认值为 :math:`0.00001 \pi`
   :type delta: Optional[float] = 0.00001 * np.pi
   :param l: 近似宽度的一半，默认值为 :math:`\pi`
   :type l: Optional[float] = np.pi

   :return: 该近似的度数
   :rtype: int

.. py:function:: step_laurent(deg)

   生成一个近似阶梯函数的劳伦多项式

   :param deg: 输出劳伦多项式的度数（为偶数）
   :type deg: int

   :return: 一个估计 :math:`f(x) = 0.5` if :math:`x <= 0` else :math:`0` 的劳伦多项式
   :rtype: Laurent

   .. note::
       在哈密顿量能量计算器中使用

.. py:function:: hamiltonian_laurent(t, deg)

   生成一个近似哈密顿量演化函数的劳伦多项式

   :param t: 演化常数（时间）
   :type t: float
   :param deg: 输出劳伦多项式的度数（为偶数）
   :type deg: int

   :return: 一个估计 :math:`e^{it \cos(x)}` 的劳伦多项式
   :rtype: Laurent

   .. note::
       -起源于Jacobi-Anger展开： :math:`y(x) = \sum_n i^n Bessel(n, x) e^{inx}`
       -在哈密顿量模拟中使用

.. py:function:: ln_laurent(deg, t)

   生成一个近似ln函数的劳伦多项式

   :param deg: 劳伦多项式的度数（是4的因子）
   :type deg: int
   :param t: 归一化常数
   :type t: float
   

   :return: 一个估计 :math:`ln(cos(x)^2) / t` 的劳伦多项式
   :rtype: Laurent

   .. note::
       在冯诺依曼熵的估计中使用。

.. py:function:: comb(n, k)

   计算nCr(n, k)

   :param n: 输入参数
   :type n: float
   :param k: 输入参数
   :type k: int
   

   :return: nCr(n, k)
   :rtype: float

.. py:function:: power_laurent(deg, alpha, t)

   生成近似幂函数的劳伦多项式

   :param deg: 劳伦多项式的度数（是4的因子）
   :type deg: int
   :param alpha: 幂函数的幂次
   :type alpha: int
   :param t: 归一化常数
   :type t: float
   

   :return: 一个估计 :math:`(cos(x)^2)^{\alpha / 2} / t` 的劳伦多项式
   :rtype: Laurent
