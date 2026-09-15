# `AttrFilter`<a id="en-us_TOPIC_0000001458687398"></a>

## Function Description<a name="en-us_TOPIC_0000001507967265"></a>

Feature attribute filter. This structure must be used together with an `AscendIndexTS` instance and acts as an input parameter during feature retrieval.

All query vectors in a retrieval call share the same filter. The filter matches the attributes of each feature in the base library. The information that can be matched includes time and token ID.

Matched base library features participate in the subsequent retrieval process, including vector distance comparison and TopK sorting.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must acquire a lock before use; otherwise, retrieval APIs may cause exceptions. Different threads cannot share one device. The retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks using fixed threads.

## `timesEnd`<a name="en-us_TOPIC_0000001458367566"></a>

`int32_t`: End time of the filter range.

## `timesStart`<a name="en-us_TOPIC_0000001507647493"></a>

`int32_t`: Start time of the filter range.

## `tokenBitSet`<a name="en-us_TOPIC_0000001507887177"></a>

`uint8_t*`: List of feature token IDs. Each `uint8_t` member records token information bitwise, from the least significant bit (LSB) to the most significant bit (MSB). 1 indicates that the token is selected, and 0 indicates that the token is not selected.

For example, if a filter token list contains two non-zero `uint8_t` members, `[7, 15, 0, 0, ..., 0]`, and the binary representations of the non-zero members are `00000111` and `00001111`, the valid token IDs they represent are 0, 1, 2, 8, 9, 10, and 11.

> [!NOTE]
> The length of `tokenBitSet` should be `tokenBitSetLen`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.

## `tokenBitSetLen`<a name="en-us_TOPIC_0000001458687402"></a>

`uint32_t`: Length of the `tokenBitSet` field in `AttrFilter`.
