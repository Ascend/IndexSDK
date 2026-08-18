# ExtraValFilter<a id="ZH-CN_TOPIC_0000002013200765"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001976640904"></a>

附加属性过滤器，该结构体需要结合AscendIndexTS实例来使用，在特征检索时作为输入参数。

不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则检索接口可能导致异常。并且不支持不同线程间共享一个Device。

## filterVal接口<a name="ZH-CN_TOPIC_0000001976481180"></a>

int16_t：待查询的附加属性，用二进制表示，“1”表示保留附加属性，“0”表示过滤附加属性。

## matchVal接口<a name="ZH-CN_TOPIC_0000002013041289"></a>

int16_t：附加属性查询模式，分为模式0和模式1。

- 对于模式0，匹配条件为：**`(ExtraValAttr::val & ExtraValFilter::filterVal) == ExtraValFilter::filterVal`**
- 对于模式1，匹配条件为：**`(ExtraValAttr::val & ExtraValFilter::filterVal) > 0`**
