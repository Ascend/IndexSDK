# `ExtraValAttr`<a id="en-us_TOPIC_0000002013198657"></a>

## Function Description<a name="en-us_TOPIC_0000002013039153"></a>

Additional attribute information. It is added together with the feature vector during insertion. This structure must be used together with an `AscendIndexTS` instance.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use; otherwise, retrieval APIs may fail. Different threads cannot share one device.

## `val`<a name="en-us_TOPIC_0000001976479160"></a>

`int16_t`: Records the additional attribute information of the current feature. In binary representation, `1` indicates yes and `0` indicates no.
