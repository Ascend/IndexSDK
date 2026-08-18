# FeatureAttr<a id="ZH-CN_TOPIC_0000001507967381"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001458367674"></a>

特征属性信息，入库时和特征向量一起添加。该结构体需要结合AscendIndexTS实例来使用。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

## time接口<a name="ZH-CN_TOPIC_0000001507647601"></a>

int32_t：记录当前特征的时间信息，以时间戳（秒级）形式表示。

> [!NOTE]
>由于昇腾硬件限制，只能处理int32类型数据，因此用户需要保证当前时间戳不会超过int32的最大值，建议在实际操作时，将当前实际时间戳减去固定的一个历史时间戳，然后再存入。

## tokenId接口<a name="ZH-CN_TOPIC_0000001507887269"></a>

uint32_t：特征token ID，一个token ID对应多个特征，一个特征对应一个token ID，需要小于用户初始化AscendIndexTS时传入的tokenNum的值。
