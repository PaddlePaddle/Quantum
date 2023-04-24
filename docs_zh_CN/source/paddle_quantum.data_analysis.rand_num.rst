paddle\_quantum.data_analysis.rand_num
==============================================

量子随机数生成器，

.. py:function:: random_number_generation(bit_len, backend, token, extract, security, min_entr_1, min_entr_2, log_path)

   封装的随机数生成函数，隐私增强部分可以参考 https://arxiv.org/abs/1311.5322。

   :param bit_len: 所需比特串长度
   :type bit_len: int
   :param backend: 物理后端，包括
               | `local_baidu_sim2`
               | `cloud_baidu_sim2_water`
               | `cloud_baidu_sim2_earth`
               | `cloud_baidu_sim2_thunder`
               | `cloud_baidu_sim2_heaven`
               | `cloud_baidu_sim2_wind`
               | `cloud_baidu_sim2_lake`
               | `cloud_aer_at_bd`
               | `cloud_baidu_qpu_qian`
               | `cloud_iopcas`
               | `cloud_ionapm`
               | `service_ubqc`
   :type backend: str
   :param token: 云服务所需的用户 token 
   :type token: str
   :param extract: 是否执行隐私增强后处理
   :type extract: bool
   :param security: 隐私指数
   :type security: float
   :param min_entr_1: 第一个物理后端的最小熵
   :type min_entr_1: float
   :param min_entr_2: 第二个物理后端的最小熵
   :type min_entr_2: float
   :param log_path: 日志文件存储路径
   :type log_path: str

   :return: 返回给定长度的 0，1 比特串
   :rtype: List[int]


