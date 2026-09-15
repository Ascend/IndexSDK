# `FeatureAttr`<a id="en-us_TOPIC_0000001507967381"></a>

## Function Description<a name="en-us_TOPIC_0000001458367674"></a>

Feature attribute information. It is added together with the feature vector during insertion. This structure must be used together with an `AscendIndexTS` instance.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use; otherwise, retrieval APIs may fail. Different threads cannot share one device.

## `time`<a name="en-us_TOPIC_0000001507647601"></a>

`int32_t`: Records the time information of the current feature as a timestamp in seconds.

> [!NOTE]
> Due to Ascend hardware limitations, only `int32` type data can be processed. Therefore, you need to ensure that the current timestamp does not exceed the maximum value of `int32`. In actual operations, subtract a fixed historical timestamp from the current timestamp before storing it.

## `tokenId`<a name="en-us_TOPIC_0000001507887269"></a>

`uint32_t`: Feature token ID. One token ID corresponds to multiple features, and one feature corresponds to one token ID. The value must be less than `tokenNum` passed during `AscendIndexTS` initialization.
