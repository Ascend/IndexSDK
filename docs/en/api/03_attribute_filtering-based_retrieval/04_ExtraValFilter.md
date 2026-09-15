# `ExtraValFilter`<a id="en-us_TOPIC_0000002013200765"></a>

## Function Description<a name="en-us_TOPIC_0000001976640904"></a>

Additional attribute filter. This structure must be used together with an `AscendIndexTS` instance and acts as an input parameter during feature retrieval.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use; otherwise, retrieval APIs may fail. Different threads cannot share one device.

## `filterVal`<a name="en-us_TOPIC_0000001976481180"></a>

`int16_t`: Additional attributes to query. In binary representation, `1` indicates that the additional attribute is retained and `0` indicates that it is filtered out.

## `matchVal`<a name="en-us_TOPIC_0000002013041289"></a>

`int16_t`: Additional attribute query mode. Two modes are supported: mode 0 and mode 1.

- For mode 0, the matching condition is **`(ExtraValAttr::val & ExtraValFilter::filterVal) == ExtraValFilter::filterVal`**.
- For mode 1, the matching condition is **`(ExtraValAttr::val & ExtraValFilter::filterVal) > 0`**.
