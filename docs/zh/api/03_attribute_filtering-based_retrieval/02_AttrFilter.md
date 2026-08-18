# AttrFilter<a id="ZH-CN_TOPIC_0000001458687398"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001507967265"></a>

特征属性过滤器，该结构体需要结合AscendIndexTS实例来使用，在特征检索时作为输入参数。

调用检索接口的所有query向量共享同一个过滤器，该过滤器会和底库中的每一个底库特征对应的属性进行匹配，可以比较的信息例如：时间、token ID。

匹配成功的底库特征会参与接下来的检索流程，即向量距离比对与TopK排序等。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。当前的特征检索内部会使用OMP做性能加速，OMP不支持与其他多线程机制混用。反复创建新线程使用OMP会导致内存持续累加，因此建议使用固定的线程来运行检索任务。

## timesEnd接口<a name="ZH-CN_TOPIC_0000001458367566"></a>

int32_t：过滤时间段的结束时间。

## timesStart接口<a name="ZH-CN_TOPIC_0000001507647493"></a>

int32_t：过滤时间段的开始时间。

## tokenBitSet接口<a name="ZH-CN_TOPIC_0000001507887177"></a>

uint8\_t\*：特征token ID的列表，每个uint8\_t成员从低位到高位，按位记录token信息，1代表选中，0代表token未选中。

例如：一个过滤器的token列表包含两个非零的uint8_t成员：\[7, 15, 0, 0, ……, 0\]，非零成员的二进制表示为00000111、00001111，则它们表达的有效token ID为：0，1，2，8，9，10，11。

> [!NOTE]
>“tokenBitSet”长度应为“tokenBitSetLen”，否则可能出现越界读写错误并引起程序崩溃。

## tokenBitSetLen接口<a name="ZH-CN_TOPIC_0000001458687402"></a>

uint32_t：指定过滤器AttrFilter中tokenBitSet字段的长度。
