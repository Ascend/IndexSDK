# Attribute Filtering-based Retrieval<a name="ZH-CN_TOPIC_0000001482844454"></a>

## AscendIndexTS<a name="ZH-CN_TOPIC_0000001507640105"></a>

### Overview<a name="ZH-CN_TOPIC_0000001507879785"></a>

Spatiotemporal index API class. When you add base library features, you can configure a `FeatureAttr` for each feature. When you run retrieval, you can configure an `AttrFilter` for each batch of query vectors. The filter first screens the entire base library and then compares the vectors that meet the conditions.

The following algorithms are supported:

- Binary feature retrieval (Hamming distance). Before use, manually generate the `BinaryFlat` and `Mask` operators and move them to the corresponding `modelpath` directory.
- `Int8Flat` (cosine distance), `FP16Flat` (IP distance), and `Int8Flat` (L2 distance). Before use, manually generate the `Mask` operator and move it to the corresponding `modelpath` directory.
- Multithreaded concurrent calls are supported. Set the `MX_INDEX_MULTITHREAD` environment variable to 1, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `AddFeature`<a name="ZH-CN_TOPIC_0000001458360182"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddFeature(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const uint8_t *customAttr = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds features.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to add.<br><code>const void *features</code>: Features to add. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.<br><code>const FeatureAttr *attributes</code>: Feature attributes to add. For details, see <code>FeatureAttr</code>.<br><code>const int64_t *labels</code>: Feature labels to add. Ensure that each label is unique within the <code>Index</code> instance.<br><code>const uint8_t *customAttr</code>: User-defined feature attributes to add.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The base library capacity is <code>1e9</code>. The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>attributes</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and not already exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>customAttr</code> must be a null pointer or have a length of <code>count * customAttrLen</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>customAttrLen</code> is set in <code>Init</code> or <code>InitWithExtraVal</code>.</td></tr>
</tbody></table>

> [!NOTE]
> `AddFeature` cannot be used together with `AddWithExtraVal`.

### `AddFeatureByIndice`<a name="ZH-CN_TOPIC_0000002411433020"></a>

> [!NOTE]
>
> - `AddFeatureByIndice` cannot be used together with `AddFeature` or `AddWithExtraVal`.
> - After you use `AddFeatureByIndice` to add base library features by position, you cannot use APIs such as `GetExtraValAttrByLabel` that depend on labels. `AddFeatureByIndice` and `GetFeatureByIndice` must be used as a pair.

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddFeatureByIndice(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *indices, const ExtraValAttr *extraVal = nullptr, const uint8_t *customAttr = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds base library features by position. This API currently supports only <code>FlatIP</code> and <code>Int8Flat</code> (cosine distance).</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to add.<br><code>const void *features</code>: Features to add. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.<br><code>const FeatureAttr *attributes</code>: Feature attributes to add.<br><code>const int64_t *indices</code>: Positions of the features in the base library.<br><code>const ExtraValAttr *extraVal</code>: Additional feature attributes to add.<br><code>const uint8_t *customAttr</code>: User-defined feature attributes to add.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The base library capacity is <code>1e9</code>. The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>attributes</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>indices</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The values must be strictly increasing and non-negative. If a value is smaller than the number of features in the base library, it indicates replacement. If a value is greater than or equal to the number of features in the base library, it indicates addition, and the values must be consecutive. <code>extraVal</code> must be a null pointer or have a length of <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. A null pointer means that no additional attributes need to be added. <code>customAttr</code> must be a null pointer or have a length of <code>count * customAttrLen</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. A null pointer means that no custom attributes need to be added.</td></tr>
</tbody></table>

### `AddWithExtraVal`<a name="ZH-CN_TOPIC_0000001976650872"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR AddWithExtraVal(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const ExtraValAttr *extraVal, const uint8_t *customAttr = nullptr);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Adds features with additional attributes.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to add.<br><code>const void *features</code>: Features to add. The Hamming distance uses <code>uint8_t</code> data, and <code>Int8Flat</code> uses <code>int8_t</code>.<br><code>const FeatureAttr *attributes</code>: Feature attributes to add. For details, see <code>FeatureAttr</code>.<br><code>const int64_t *labels</code>: Feature labels to add. Ensure that each label is unique within the <code>Index</code> instance.<br><code>const ExtraValAttr *extraVal</code>: Additional feature attributes to add. For details, see <code>ExtraValAttr</code>.<br><code>const uint8_t *customAttr</code>: User-defined feature attributes to add.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The base library capacity is <code>1e9</code>. The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>attributes</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and not already exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>extraVal</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>customAttr</code> must be a null pointer or have a length of <code>count * customAttrLen</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>customAttrLen</code> is set in <code>Init</code> or <code>InitWithExtraVal</code>.</td></tr>
</tbody></table>

### `AscendIndexTS`<a name="ZH-CN_TOPIC_0000001458200394"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexTS() = default;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Constructor of <code>AscendIndexTS</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexTS(const AscendIndexTS &amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Copy constructor of <code>AscendIndexTS</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>const AscendIndexTS &amp;</code>: <code>AscendIndexTS</code> object.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `~AscendIndexTS`<a name="ZH-CN_TOPIC_0000001507760865"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>virtual ~AscendIndexTS() = default;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Destructor of <code>AscendIndexTS</code>. It destroys the feature management object.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `DeleteFeatureByLabel`<a name="ZH-CN_TOPIC_0000001458200398"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR DeleteFeatureByLabel(int64_t count, const int64_t *labels);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Deletes features by label.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to delete.<br><code>const int64_t *labels</code>: Feature labels.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `DeleteFeatureByToken`<a id="ZH-CN_TOPIC_0000001458680018"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR DeleteFeatureByToken(int64_t count, const uint32_t *tokens);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Deletes features by token.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to delete.<br><code>const uint32_t *tokens</code>: Feature tokens.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The length of <code>tokens</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `FastDeleteFeatureByIndice`<a name="ZH-CN_TOPIC_0000002445152089"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR FastDeleteFeatureByIndice(int64_t count, const int64_t *indices);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Quickly deletes features by position.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to delete.<br><code>const int64_t *indices</code>: Positions of the features in the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The length of <code>indices</code> must be <code>count</code>, and all values must be unique, non-negative, and smaller than the number of features in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `FastDeleteFeatureByRange`<a name="ZH-CN_TOPIC_0000002445960745"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR FastDeleteFeatureByRange(int64_t start, int64_t count);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Quickly deletes <code>count</code> base library features starting from <code>start</code>. This API supports only the additional similarity scenarios of <code>TSFlatIP</code> and <code>TSInt8FlatCos</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t start</code>: Start position of the features to delete.<br><code>int64_t count</code>: Number of features to delete.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>start</code> must be greater than or equal to 0 and smaller than the number of features in the base library. <code>count</code> must be greater than 0 and less than or equal to the number of features in the base library. The sum of <code>start</code> and <code>count</code> must be less than or equal to the number of features in the base library.</td></tr>
</tbody></table>

### `GetBaseByRange`<a name="ZH-CN_TOPIC_0000001818301380"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the base library by range.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t offset</code>: Initial offset for retrieving base library features.<br><code>uint32_t num</code>: Number of features.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int64_t *labels</code>: Feature labels.<br><code>void *features</code>: Features. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.<br><code>FeatureAttr *attributes</code>: Feature attributes.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 &lt; offset &lt;= 8.0e8</code>. <code>0 &lt; num &lt;= 8.0e8</code>. <code>offset + num &lt;= ntotal</code>. The length of <code>labels</code> must be <code>num</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>features</code> must be <code>num * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>attributes</code> must be <code>num</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetBaseByRangeWithExtraVal`<a name="ZH-CN_TOPIC_0000001976495686"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetBaseByRangeWithExtraVal(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes, ExtraValAttr *extraVal) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Queries the base library with additional attributes by range.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t offset</code>: Initial offset for retrieving base library features.<br><code>uint32_t num</code>: Number of features.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int64_t *labels</code>: Feature labels.<br><code>void *features</code>: Features. The Hamming distance uses <code>uint8_t</code> data, and <code>Int8Flat</code> uses <code>int8_t</code>.<br><code>FeatureAttr *attributes</code>: Feature attributes.<br><code>ExtraValAttr *extraVal</code>: Additional attributes.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>0 &lt;= offset &lt; 8.0e8</code>. <code>0 &lt; num &lt;= 8.0e8</code>. <code>offset + num &lt;= ntotal</code>. The length of <code>labels</code> must be <code>num</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>features</code> must be <code>num * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>attributes</code> must be <code>num</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>extraVal</code> must be <code>num</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetBaseMask`<a name="ZH-CN_TOPIC_0000002445112157"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetBaseMask(int64_t count, uint8_t *mask);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Obtains the flag that indicates whether the base library has been quickly deleted. If a bit is 0, the base library entry at that position has been deleted and is invalid.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Valid length of the <code>mask</code> array.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>uint8_t *mask</code>: Array that marks whether the base library entry has been deleted.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, ceil(<code>ntotal</code>/8)]. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. Here, <code>ntotal</code> is the number of features in the base library. The length of <code>mask</code> must be greater than or equal to <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetCustomAttrByBlockId`<a name="ZH-CN_TOPIC_0000001736682593"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetCustomAttrByBlockId(uint32_t blockId, uint8_t *&amp;customAttr) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Obtains the custom attributes of the specified <code>blockId</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t blockId</code>: <code>blockId</code> to retrieve.<br><code>uint8_t *&amp;customAttr</code>: User-defined feature attributes on the device side.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">The length of <code>customAttr</code> must be <code>customAttrBlockSize * customAttrLen</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>customAttrBlockSize</code> and <code>customAttrLen</code> are set in <code>Init</code> or <code>InitWithExtraVal</code>.</td></tr>
</tbody></table>

### `GetExtraValAttrByLabel`<a name="ZH-CN_TOPIC_0000001976655414"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetExtraValAttrByLabel(int64_t count, const int64_t *labels, ExtraValAttr *extraVal) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Obtains the additional attributes of the features with the specified labels.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to retrieve.<br><code>const int64_t *labels</code>: Feature labels.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>ExtraValAttr *extraVal</code>: Additional attributes.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. If the input <code>labels</code> do not exist in the base library, the <code>val</code> field in the returned additional attributes is <code>INT16_MIN</code>. The length of <code>extraVal</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetFeatureAttrByLabel`<a name="ZH-CN_TOPIC_0000001594544301"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatureAttrByLabel(int64_t count, const int64_t *labels, FeatureAttr *attributes) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Obtains the attributes of the features with the specified labels.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to retrieve.<br><code>const int64_t *labels</code>: Feature labels.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>FeatureAttr *attributes</code>: Feature attributes.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. If the input <code>labels</code> do not exist in the base library, the returned feature attributes contain <code>time = INT32_MIN</code> and <code>tokenId = UINT32_MAX</code>. The length of <code>attributes</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetFeatureByIndice`<a name="ZH-CN_TOPIC_0000002411592888"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels = nullptr, void *features = nullptr, FeatureAttr *attributes = nullptr, ExtraValAttr *extraVal = nullptr) const;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Obtains base library features by position.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to retrieve.<br><code>const int64_t *indices</code>: Positions of the features in the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int64_t *labels</code>: Labels of the features to retrieve.<br><code>void *features</code>: Feature vectors to retrieve.<br><code>FeatureAttr *attributes</code>: Spatiotemporal attributes of the features to retrieve.<br><code>ExtraValAttr *extraVal</code>: Additional attributes of the features to retrieve.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The length of <code>indices</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The values must be greater than or equal to 0 and less than the number of features in the base library. When <code>labels</code> is <code>nullptr</code>, no labels are retrieved. Otherwise, the length must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>features</code> is <code>nullptr</code>, no features are retrieved. Otherwise, the length must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>attributes</code> is <code>nullptr</code>, no attributes are retrieved. Otherwise, the length must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>extraVal</code> is <code>nullptr</code>, no additional attributes are retrieved. Otherwise, the length must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetFeatureByLabel`<a name="ZH-CN_TOPIC_0000001507879789"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatureByLabel(int64_t count, const int64_t *labels, void *features);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Retrieves the features with the specified labels.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>int64_t count</code>: Number of features to retrieve.<br><code>const int64_t *labels</code>: Feature labels.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>void *features</code>: Features retrieved by the specified labels. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 1e6]. The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `GetFeatureNum`<a name="ZH-CN_TOPIC_0000001544946953"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR GetFeatureNum(int64_t *totalNum);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Obtains the number of features in this <code>Index</code> instance.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int64_t *totalNum</code>: Number of features in the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `Init`<a id="ZH-CN_TOPIC_0000001458680014"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Init(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, AlgorithmType algType = AlgorithmType::FLAT_COS_INT8, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits&lt;uint64_t&gt;::max());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initializes the instance.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t deviceId</code>: Device ID used by the <code>Index</code>.<br><code>uint32_t dim</code>: Dimension of the base library vectors.<br><code>uint32_t tokenNum</code>: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated <code>Mask</code> operator.<br><code>AlgorithmType algType</code>: Distance comparison algorithm used by the backend. The default value is <code>AlgorithmType::FLAT_COS_INT8</code>. For supported algorithms, see the following list.<br> <code>AlgorithmType::FLAT_HAMMING</code>: Binary feature retrieval (Hamming distance). <code>AlgorithmType::FLAT_COS_INT8</code>: <code>Int8Flat</code> (cosine distance). <code>AlgorithmType::FLAT_L2_INT8</code>: <code>Int8Flat</code> (L2 distance). <code>AlgorithmType::FLAT_IP_FP16</code>: <code>FP16Flat</code> (IP distance). <code>AlgorithmType::FLAT_HPP_COS_INT8</code>: <code>Int8Flat</code> (cosine distance).<br><code>MemoryStrategy memoryStrategy</code>: Memory strategy used by the backend. The default value is <code>MemoryStrategy::PURE_DEVICE_MEMORY</code>. Supported strategies are listed below. <code>MemoryStrategy::PURE_DEVICE_MEMORY</code>: Pure device memory strategy. <code>MemoryStrategy::HETERO_MEMORY</code>: Heterogeneous memory strategy. <code>MemoryStrategy::HPP</code>: HPP heterogeneous memory strategy.<br><code>customAttrLen</code>: Length of custom attributes.<br><code>customAttrBlockSize</code>: Block size of custom attributes.<br><code>maxFeatureRowCount</code>: Maximum number of vectors in the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">Call this API immediately after the constructor. <code>deviceId</code> must be a valid device ID in the range [0, 1024]. <code>tokenNum</code> must be in the range (0, 3e5]. For binary feature retrieval (Hamming distance), <code>dim</code> must be one of {256, 512, 1024}. For the <code>Int8Flat</code> algorithm (cosine distance or L2 distance), <code>dim</code> must be one of {64, 128, 256, 384, 512, 768, 1024}. For the <code>FP16Flat</code> algorithm (IP distance), <code>dim</code> must be one of {64, 128, 256, 384, 512, 768, 1024}. <code>memoryStrategy::HETERO_MEMORY</code> currently supports only <code>AlgorithmType::FLAT_COS_INT8</code>. <code>customAttrLen</code> must be in the range [0, 32]. The default value is 0. A value of 0 means that no custom attributes exist. <code>customAttrBlockSize</code> must be in the range [0, 262144*64] and must be an integer multiple of 1024*256. The default value is 0. A value of 0 means that no custom attributes exist. <code>maxFeatureRowCount</code> must be in the range [262144 \* 64, 262144 \* 550 \* 3] and must be an integer multiple of 256. The default value is the maximum value of <code>uint64</code>. This parameter is valid only when <code>memoryStrategy</code> is set to <code>MemoryStrategy::HPP</code>. When <code>memoryStrategy</code> is set to <code>MemoryStrategy::HPP</code>, the available Host memory must be at least 250 GB, the number of free physical CPU cores must be at least 15, and only 256-dimensional vector retrieval is supported.</td></tr>
</tbody></table>

### `InitWithExtraVal`<a id="ZH-CN_TOPIC_0000002013206217"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR InitWithExtraVal(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, AlgorithmType algType = AlgorithmType::FLAT_HAMMING, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits&lt;uint64_t&gt;::max());</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initializes an instance with additional attributes.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t deviceId</code>: Device ID used by the <code>Index</code>.<br><code>uint32_t dim</code>: Dimension of the base library vectors.<br><code>uint32_t tokenNum</code>: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated <code>Mask</code> operator.<br><code>uint64_t resources</code>: Shared memory size.<br><code>AlgorithmType algType</code>: Distance comparison algorithm used by the backend. The default value is <code>AlgorithmType::FLAT_HAMMING</code>. Supported algorithms are listed below. <code>AlgorithmType::FLAT_HAMMING</code>: Binary feature retrieval (Hamming distance). <code>AlgorithmType::FLAT_COS_INT8</code>: <code>Int8Flat</code> (cosine distance).<br><code>MemoryStrategy memoryStrategy</code>: Memory strategy used by the backend. The default value is <code>MemoryStrategy::PURE_DEVICE_MEMORY</code>. Supported strategies are listed below. <code>MemoryStrategy::PURE_DEVICE_MEMORY</code>: Pure device memory strategy. <code>MemoryStrategy::HETERO_MEMORY</code>: Heterogeneous memory strategy.<br><code>customAttrLen</code>: Length of custom attributes.<br><code>customAttrBlockSize</code>: Block size of custom attributes.<br><code>maxFeatureRowCount</code>: Maximum number of vectors in the base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">Call this API immediately after the constructor. <code>deviceId</code> must be a valid device ID in the range [0, 1024]. <code>tokenNum</code> must be in the range (0, 3e5]. <code>resources</code> must be in the range [1 \* 1024 \* 1024 \* 1024, 32 \* 1024 \* 1024 \* 1024]. When you use additional attributes, 4 GB is recommended. For binary feature retrieval (Hamming distance), <code>dim</code> must be one of {256, 512, 1024}. For the <code>Int8Flat</code> algorithm (cosine distance), <code>dim</code> must be one of {64, 128, 256, 384, 512, 768, 1024}. <code>customAttrLen</code> must be in the range [0, 32]. The default value is 0. A value of 0 means that no custom attributes exist. <code>customAttrBlockSize</code> must be in the range [0, 262144 \* 64] and must be an integer multiple of 1024*256. The default value is 0. A value of 0 means that no custom attributes exist. <code>maxFeatureRowCount</code> does not support HPP when additional attributes are used, and the default value is the maximum value of <code>uint64</code>.</td></tr>
</tbody></table>

### `InitWithQuantify`<a name="ZH-CN_TOPIC_0000002458673509"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR InitWithQuantify(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, const float *scale, AlgorithmType algType = AlgorithmType::FLAT_IP_FP16, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Initializes the vectorized base library.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t deviceId</code>: Device ID used by the <code>Index</code>.<br><code>uint32_t dim</code>: Dimension of the base library vectors.<br><code>uint32_t tokenNum</code>: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated <code>Mask</code> operator.<br><code>uint64_t resources</code>: Shared memory size.<br><code>const float *scale</code>: Scaling factor for base library vectorization. After the scaling factor is multiplied by the base library, the result is converted to the <code>int8_t</code> type.<br><code>AlgorithmType algType</code>: Distance comparison algorithm used by the backend. The default value is <code>AlgorithmType::FLAT_IP_FP16</code>, which means <code>FP16Flat</code> (IP distance). Currently, only <code>AlgorithmType::FLAT_IP_FP16</code> is supported.<br><code>uint32_t customAttrLen</code>: Length of custom attributes.<br><code>uint32_t customAttrBlockSize</code>: Block size of custom attributes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">Call this API immediately after the constructor. <code>deviceId</code> must be a valid device ID in the range [0, 1024]. <code>tokenNum</code> must be in the range (0, 3e5]. <code>resources</code> must be greater than 0 and less than or equal to 4 \* 1024 \* 1024 \* 1024. The <code>scale</code> array is used for division during dequantization and must not be close to 0. The absolute value of each factor in <code>scale</code> must be greater than or equal to 1e-6f. For the <code>FP16Flat</code> algorithm (IP distance), <code>dim</code> must be one of {64, 128, 256, 384, 512, 768, 1024}. Only the non-shared mode of the <code>FP16Flat</code> algorithm (IP distance) is supported. This API must be used together with <code>AddFeatureByIndice</code>. <code>customAttrLen</code> must be in the range [0, 32]. The default value is 0. A value of 0 means that no custom attributes exist. <code>customAttrBlockSize</code> must be in the range [0, 262144 * 64] and must be an integer multiple of 1024 \* 256. The default value is 0. A value of 0 means that no custom attributes exist.</td></tr>
</tbody></table>

### `operator =`<a name="ZH-CN_TOPIC_0000001507959881"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>AscendIndexTS &amp;operator=(const AscendIndexTS &amp;) = delete;</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Declares the assignment operator for this <code>Index</code> as deleted, which means that the type is non-copyable.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>const AscendIndexTS &amp;</code>: Constant <code>AscendIndexTS</code>.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">None</td></tr>
</tbody></table>

### `Search`<a name="ZH-CN_TOPIC_0000001507640109"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR Search(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Calculates the distance between the input features and the base library vectors filtered by <code>AttrFilter</code>, sorts the distances by TopK, and returns the corresponding distances and indices.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t count</code>: Number of features to compare.<br><code>const void *features</code>: Features to compare. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.<br><code>const AttrFilter *attrFilter</code>: Attribute filter information. For details, see <code>AttrFilter</code>.<br><code>bool shareAttrFilter</code>: Whether different queries share the same mask.<br><code>uint32_t topk</code>: TopK size to keep after cosine distance calculation.<br><code>bool enableTimeFilter</code>: Time-stamp attribute filter switch. The default value is <code>true</code>. When <code>enableTimeFilter = false</code>, time-stamp attribute filtering is disabled.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int64_t *labels</code>: Labels of the TopK features.<br><code>float *distances</code>: Distances of the TopK features.<br><code>uint32_t *validNums</code>: Number of valid results obtained after each query vector is compared.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 10240]. The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>attrFilter</code> must be 1. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>topk</code> must be in the range [1, 100000]. The length of <code>labels</code> must be <code>count * topk</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>distances</code> must be <code>count * topk</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>validNums</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `SearchWithExtraMask`<a name="ZH-CN_TOPIC_0000001494506850"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Calculates the distance between the input features and the base library vectors filtered by <code>AttrFilter</code> and an external <code>Mask</code>, sorts the distances by TopK, and returns the corresponding distances and indices.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t count</code>: Number of features to compare.<br><code>const void *features</code>: Features to compare. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.<br><code>const AttrFilter *attrFilter</code>: Attribute filter information. For details, see <code>AttrFilter</code>.<br><code>bool shareAttrFilter</code>: Whether the same query shares one <code>Mask</code>.<br><code>uint32_t topk</code>: TopK size to keep after cosine distance calculation.<br><code>const uint8_t *extraMask</code>: Additional filter <code>Mask</code> provided from outside. The value is expressed in bits, where 0 and 1 indicate filtering or selecting the feature respectively.<br><code>uint64_t extraMaskLenEachQuery</code>: Length of the external <code>Mask</code>, in bytes.<br><code>bool extraMaskIsAtDevice</code>: Whether the external <code>Mask</code> already exists on the device side.<br><code>bool enableTimeFilter</code>: Time-stamp attribute filter switch. The default value is <code>true</code>. When <code>enableTimeFilter = false</code>, time-stamp attribute filtering is disabled.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int64_t *labels</code>: Labels of the TopK features.<br><code>float *distances</code>: Distances of the TopK features.<br><code>uint32_t *validNums</code>: Number of valid results obtained after each query vector is compared.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 10240]. <code>topk</code> must be in the range [1, 100000]. The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>attrFilter</code> must be 1. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>distances</code> must be <code>count * topk</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>validNums</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>labels</code> must be <code>count * topk</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>extraMask</code> must be <code>extraMaskLenEachQuery</code>. When <code>shareAttrFilter</code> is <code>false</code>, the length of <code>extraMask</code> must be <code>count * extraMaskLenEachQuery</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

### `SearchWithExtraMask` with Extra Similarity<a name="ZH-CN_TOPIC_0000002373091106"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, const uint16_t *extraScore, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Calculates the distance between the input features and the base library vectors filtered by <code>AttrFilter</code> and an external <code>Mask</code>, sorts the distances by TopK, and returns the corresponding distances and indices.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t count</code>: Number of features to compare.<br><code>const void *features</code>: Features to compare. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.<br><code>const AttrFilter *attrFilter</code>: Attribute filter information. For details, see <code>AttrFilter</code>.<br><code>bool shareAttrFilter</code>: Whether the same query shares one <code>Mask</code>.<br><code>uint32_t topk</code>: TopK size to keep after cosine distance calculation.<br><code>const uint8_t *extraMask</code>: Additional filter <code>Mask</code> provided from outside. The value is expressed in bits, where 0 and 1 indicate filtering or selecting the feature respectively.<br><code>uint64_t extraMaskLenEachQuery</code>: Length of the external <code>Mask</code>, in bytes.<br><code>bool extraMaskIsAtDevice</code>: Whether the external <code>Mask</code> already exists on the device side.<br><code>const uint16_t *extraScore</code>: Additional similarity provided by the user. The length is <code>count * totalPad</code>, where <code>totalPad</code> is the base library length aligned to 16 bytes.<br><code>bool enableTimeFilter</code>: Time-stamp attribute filter switch. The default value is <code>true</code>. When <code>enableTimeFilter = false</code>, time-stamp attribute filtering is disabled.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>int64_t *labels</code>: Labels of the TopK features. If the base library is added by using <code>AddFeatureByIndice</code>, the output here is the base library position (<code>indices</code>).<br><code>float *distances</code>: Distances of the TopK features.<br><code>uint32_t *validNums</code>: Number of valid results obtained after each query vector is compared.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 10240]. <code>topk</code> must be in the range [1, 100000]. The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>attrFilter</code> must be 1. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>distances</code> must be <code>count * topk</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>validNums</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>labels</code> must be <code>count * topk</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>extraMask</code> must be <code>extraMaskLenEachQuery</code>. When <code>shareAttrFilter</code> is <code>false</code>, the length of <code>extraMask</code> must be <code>count * extraMaskLenEachQuery</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>extraScore</code> must be <code>count * totalPad</code>, where <code>totalPad</code> is the base library length aligned to 16 bytes. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. It actually corresponds to the <code>float16_t</code> type, and its values range from <code>-1.0</code> to <code>1.0</code>. It is currently effective only for non-shared masks in <code>Int8FlatCos</code> and <code>FlatIP</code>. Otherwise, <code>extraScore</code> does not take part in the calculation.</td></tr>
</tbody></table>

### `SearchWithExtraVal`<a name="ZH-CN_TOPIC_0000002013215285"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SearchWithExtraVal(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, const ExtraValFilter *extraValFilter, bool enableTimeFilter = true);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Calculates the distance between the input features and the base library vectors filtered by <code>AttrFilter</code> and <code>ExtraValFilter</code>, sorts the distances by TopK, and returns the corresponding distances and indices.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>uint32_t count</code>: Number of features to compare.<br><code>const void *features</code>: Features to compare. The Hamming distance uses <code>uint8_t</code> data, and <code>Int8cos</code> uses <code>int8_t</code>. Currently, only <code>int8cos</code> is supported, including heterogeneous memory scenarios, together with the Hamming distance.<br><code>const AttrFilter *attrFilter</code>: Attribute filter information. For details, see <code>AttrFilter</code>.<br><code>bool shareAttrFilter</code>: Additional attributes currently support only <code>false</code>. Different queries do not share the same mask.<br><code>uint32_t topk</code>: TopK size to keep after cosine distance calculation.<br><code>const ExtraValFilter *extraValFilter</code>: Additional attribute filter information. For details, see <code>ExtraValFilter</code>.<br><code>bool enableTimeFilter</code>: Time-stamp attribute filter switch. The default value is <code>true</code>. When <code>enableTimeFilter = false</code>, time-stamp attribute filtering is disabled.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle"><code>uint32_t *validNums</code>: Number of valid results obtained after each query vector is compared.<br><code>int64_t *labels</code>: Labels of the TopK features.<br><code>float *distances</code>: Distances of the TopK features.</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle"><code>count</code> must be in the range [1, 10240]. The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>attrFilter</code> must be 1. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>topk</code> must be in the range [1, 100000]. The length of <code>labels</code> must be <code>count * topk</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>distances</code> must be <code>count * topk</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. The length of <code>validNums</code> must be <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash. <code>extraValFilter</code> must be a null pointer or have a length of <code>count</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</td></tr>
</tbody></table>

> [!NOTE]
>
> `SearchWithExtraVal` cannot be used together with `Search`.

### `SetHeteroParam`<a name="ZH-CN_TOPIC_0000001630850578"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetHeteroParam(size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity);</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Sets the parameters of the heterogeneous storage strategy.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle"><code>size_t deviceCapacity</code>: Base library capacity stored on the device side when the heterogeneous memory strategy is used, in bytes.<br><code>size_t deviceBuffer</code>: Cache capacity on the device side when the heterogeneous memory strategy is used, in bytes.<br><code>size_t hostCapacity</code>: Base library capacity stored on the Host side when the heterogeneous memory strategy is used, in bytes.</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">Use this API after you set the memory strategy to <code>MemoryStrategy::HETERO_MEMORY</code> in the <code>Init</code> API. The minimum value of <code>deviceCapacity</code> is <code>1G</code>, and the maximum value is the actual remaining device memory. The minimum value of <code>deviceBuffer</code> is <code>2 * 262144 * dim</code>, and the maximum value is <code>8G</code>. Set it according to the actual remaining device memory. <code>deviceCapacity + deviceBuffer</code> must be smaller than the actual remaining device memory on the device. The value range of <code>hostCapacity</code> is <code>[1G, 512G]</code>. Configure it according to the amount of actual memory that can be allocated on the Host side.</td></tr>
</tbody></table>

### `SetSaveHostMemory`<a name="ZH-CN_TOPIC_0000002106649489"></a>

<table><tbody>
<tr><td width="150" align="center" valign="middle">API Definition</td><td valign="middle"><strong><code>APP_ERROR SetSaveHostMemory();</code></strong></td></tr>
<tr><td width="150" align="center" valign="middle">Description</td><td valign="middle">Sets the host memory saving mode. This mode is disabled by default.</td></tr>
<tr><td width="150" align="center" valign="middle">Parameters</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Output</td><td valign="middle">None</td></tr>
<tr><td width="150" align="center" valign="middle">Returns</td><td valign="middle"><code>APP_ERROR</code>: Operation status. For details, see the API return value reference.</td></tr>
<tr><td width="150" align="center" valign="middle">Constraints</td><td valign="middle">Use this API after the <code>Init</code> API when the base library size is 0. This API can save host memory, but it reduces the performance of delete-type and retrieve-type APIs. When you use this mode, you cannot use the <code>DeleteFeatureByToken</code> API. This API supports only the Hamming distance.</td></tr>
</tbody></table>

## `AttrFilter`<a id="ZH-CN_TOPIC_0000001458687398"></a>

### Overview<a name="ZH-CN_TOPIC_0000001507967265"></a>

Feature attribute filter. This structure must be used together with an `AscendIndexTS` instance and acts as an input parameter during feature retrieval.

All query vectors in a retrieval call share the same filter. The filter matches the attributes of each base library feature. The comparable information includes time and token ID.

Matched base library features participate in the retrieval process that follows, including vector distance comparison and TopK sorting.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use, otherwise retrieval APIs may fail. Different threads cannot share one device. The retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Therefore, repeatedly creating new threads with OMP causes memory usage to keep increasing. You are advised to run retrieval tasks with fixed threads.

### `timesEnd`<a name="ZH-CN_TOPIC_0000001458367566"></a>

`int32_t`: End time of the filter interval.

### `timesStart`<a name="ZH-CN_TOPIC_0000001507647493"></a>

`int32_t`: Start time of the filter interval.

### `tokenBitSet`<a name="ZH-CN_TOPIC_0000001507887177"></a>

`uint8_t*`: List of feature token IDs. Each `uint8_t` member records token information bit by bit from low-order bits to high-order bits. 1 indicates selected, and 0 indicates that the token is not selected.

For example, if a filter token list contains two non-zero `uint8_t` members, `[7, 15, 0, 0, ..., 0]`, and the binary representations of the non-zero members are 00000111 and 00001111, the valid token IDs they represent are 0, 1, 2, 8, 9, 10, and 11.

> [!NOTE]
> The length of `tokenBitSet` should be `tokenBitSetLen`. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.

### `tokenBitSetLen`<a name="ZH-CN_TOPIC_0000001458687402"></a>

`uint32_t`: Length of the `tokenBitSet` field in `AttrFilter`.

## `ExtraValAttr`<a id="ZH-CN_TOPIC_0000002013198657"></a>

### Overview<a name="ZH-CN_TOPIC_0000002013039153"></a>

Additional attribute information. It is added together with the feature vector when the feature is stored. This structure must be used together with an `AscendIndexTS` instance.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use, otherwise retrieval APIs may fail. Different threads cannot share one device.

### `val`<a name="ZH-CN_TOPIC_0000001976479160"></a>

`int16_t`: Records the additional attribute information of the current feature. The binary representation uses 1 to indicate `yes` and 0 to indicate `no`.

## `ExtraValFilter`<a id="ZH-CN_TOPIC_0000002013200765"></a>

### Overview<a name="ZH-CN_TOPIC_0000001976640904"></a>

Additional attribute filter. This structure must be used together with an `AscendIndexTS` instance and acts as an input parameter during feature retrieval.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use, otherwise retrieval APIs may fail. Different threads cannot share one device.

### `filterVal`<a name="ZH-CN_TOPIC_0000001976481180"></a>

`int16_t`: Additional attributes to query. The binary representation uses 1 to indicate that the additional attribute is retained and 0 to indicate that it is filtered out.

### `matchVal`<a name="ZH-CN_TOPIC_0000002013041289"></a>

`int16_t`: Additional attribute query mode. Two modes are supported, mode 0 and mode 1.

- For mode 0, the matching condition is `ExtraValAttr::val & ExtraValFilter::filterVal == ExtraValFilter::filterVal`.
- For mode 1, the matching condition is `ExtraValAttr::val & ExtraValFilter::filterVal > 0`.

## `FeatureAttr`<a id="ZH-CN_TOPIC_0000001507967381"></a>

### Overview<a name="ZH-CN_TOPIC_0000001458367674"></a>

Feature attribute information. It is added together with the feature vector when the feature is stored. This structure must be used together with an `AscendIndexTS` instance.

Multithreaded concurrent calls are not supported. Therefore, in multithreaded scenarios, you must lock before use, otherwise retrieval APIs may fail. Different threads cannot share one device.

### `time`<a name="ZH-CN_TOPIC_0000001507647601"></a>

`int32_t`: Records the time information of the current feature as a time stamp in seconds.

> [!NOTE]
> Due to Ascend hardware limitations, only `int32` type data can be processed. Therefore, you need to ensure that the current time stamp does not exceed the maximum value of `int32`. In actual operations, subtract a fixed historical time stamp from the current time stamp before you store it.

### `tokenId`<a name="ZH-CN_TOPIC_0000001507887269"></a>

`uint32_t`: Feature token ID. One token ID corresponds to multiple features, and one feature corresponds to one token ID. The value must be smaller than `tokenNum` passed when the user initializes `AscendIndexTS`.
