# `AscendIndexTS`<a name="en-us_TOPIC_0000001507640105"></a>

## Function Description<a name="en-us_TOPIC_0000001507879785"></a>

Spatiotemporal index API class. When you add base library features, you can configure a `FeatureAttr` for each feature. When you run retrieval, you can configure an `AttrFilter` for each batch of query vectors. The filter first screens the entire base library and then compares the vectors that meet the conditions.

The following algorithms are supported:

- Binary feature retrieval (Hamming distance): Before use, manually generate the <a href="../../05_user_guide.md#binaryflat"><code>BinaryFlat</code></a> and <a href="../../05_user_guide.md#mask"><code>Mask</code></a> operators and move them to the corresponding <code>modelpath</code> directory.
- Int8Flat (cosine distance), FP16Flat (IP distance), Int8Flat (L2 distance): Before use, manually generate the <a href="../../05_user_guide.md#mask"><code>Mask</code></a> operator and move it to the corresponding <code>modelpath</code> directory.
- Multithreaded concurrent calls are supported. To enable this feature, set the `MX_INDEX_MULTITHREAD` environment variable to `1`, that is, run `export MX_INDEX_MULTITHREAD=1`. If you set it to any other value or do not set it, multithreading remains disabled. The current feature retrieval implementation uses OMP for performance acceleration. OMP does not support being mixed with other multithreading mechanisms. Repeatedly creating new threads with OMP causes memory usage to keep increasing. Therefore, you are advised to use fixed threads to run retrieval tasks.

## `AddFeature`<a name="en-us_TOPIC_0000001458360182"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p645895412284"><a name="p645895412284"></a><a name="p645895412284"></a><code>APP_ERROR AddFeature(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const uint8_t *customAttr = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Adds features.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p1994144463213"><a name="p1994144463213"></a><a name="p1994144463213"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a><code>int64_t count</code></strong>: Number of features to add.</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a><code>const void *features</code></strong>: Features to add. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1733711363162"><a name="b1733711363162"></a><a name="b1733711363162"></a><code>const FeatureAttr *attributes</code></strong>: Feature attributes to add. For details, see <a href="./05_FeatureAttr.md#en-us_TOPIC_0000001507967381"><code>FeatureAttr</code></a>.</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b11413152220159"><a name="b11413152220159"></a><a name="b11413152220159"></a><code>const int64_t *labels</code></strong>: Feature labels to add. Ensure that each label is unique within the <code>Index</code> instance.</p>
<p id="p1087211592617"><a name="p1087211592617"></a><a name="p1087211592617"></a><strong id="b1387213364459"><a name="b1387213364459"></a><a name="b1387213364459"></a><code>const uint8_t *customAttr</code></strong>: User-defined feature attributes to add.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><code>count</code> must be in the range <code>[1, 1e6]</code>. The base library capacity is <code>1e9</code>.</li><li>The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>attributes</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and not already exist in the base library. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>customAttr</code> must be a null pointer or have a length of <code>count * customAttrLen</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>customAttrLen</code> is set in <a href="#en-us_TOPIC_0000001458680014"><code>Init</code></a> or <a href="#en-us_TOPIC_0000002013206217"><code>InitWithExtraVal</code></a>.</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
> `AddFeature` cannot be used together with `AddWithExtraVal`.

## `AddFeatureByIndice`<a name="en-us_TOPIC_0000002411433020"></a>

> [!NOTE]
>
> - `AddFeatureByIndice` cannot be used together with `AddFeature` or `AddWithExtraVal`.
> - After you use `AddFeatureByIndice` to add base library features by position, you cannot use APIs such as `GetExtraValAttrByLabel` that depend on labels. `AddFeatureByIndice` and `GetFeatureByIndice` must be used as a pair.

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p4977325172214"><a name="p4977325172214"></a><a name="p4977325172214"></a><code>APP_ERROR AddFeatureByIndice(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *indices, const ExtraValAttr *extraVal = nullptr, const uint8_t *customAttr = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p84931034102319"><a name="p84931034102319"></a><a name="p84931034102319"></a>Adds base library features by position. This API currently supports only <code>FlatIP</code> and <code>Int8Flat</code> (cosine distance).</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p16400852182313"><a name="p16400852182313"></a><a name="p16400852182313"></a><strong id="b12681434245"><a name="b12681434245"></a><a name="b12681434245"></a><code>int64_t count</code></strong>: Number of features to add.</p>
<p id="p1640015524237"><a name="p1640015524237"></a><a name="p1640015524237"></a><strong id="b3961719172416"><a name="b3961719172416"></a><a name="b3961719172416"></a><code>const void *features</code></strong>: Features to add. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.</p>
<p id="p9400155214239"><a name="p9400155214239"></a><a name="p9400155214239"></a><strong id="b117145534246"><a name="b117145534246"></a><a name="b117145534246"></a><code>const FeatureAttr *attributes</code></strong>: Feature attributes to add.</p>
<p id="p940013526232"><a name="p940013526232"></a><a name="p940013526232"></a><strong id="b192668514250"><a name="b192668514250"></a><a name="b192668514250"></a><code>const int64_t *indices</code></strong>: Positions of the features in the base library.</p>
<p id="p740065211237"><a name="p740065211237"></a><a name="p740065211237"></a><strong id="b11521184710255"><a name="b11521184710255"></a><a name="b11521184710255"></a><code>const ExtraValAttr *extraVal</code></strong>: Additional feature attributes to add.</p>
<p id="p2400165210237"><a name="p2400165210237"></a><a name="p2400165210237"></a><strong id="b2498951122512"><a name="b2498951122512"></a><a name="b2498951122512"></a><code>const uint8_t *customAttr</code></strong>: User-defined feature attributes to add.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.6.1 "><a name="ul101361314193612"></a><a name="ul101361314193612"></a><ul id="ul101361314193612"><li><code>count</code> must be in the range <code>[1, 1e6]</code>. The base library capacity is <code>1e9</code>.</li><li>The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>attributes</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>indices</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. The values must be strictly increasing and non-negative. If a value is smaller than the number of features in the base library, it indicates replacement. If a value is greater than or equal to the number of features in the base library, it indicates addition, and the values must be consecutive.</li><li><code>extraVal</code> must be a null pointer or have a length of <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. A null pointer means that no additional attributes need to be added.</li><li><code>customAttr</code> must be a null pointer or have a length of <code>count * customAttrLen</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. A null pointer means that no custom attributes need to be added.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AddWithExtraVal`<a name="en-us_TOPIC_0000001976650872"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p737617493548"><a name="p737617493548"></a><a name="p737617493548"></a><code>APP_ERROR AddWithExtraVal(int64_t count, const void *features, const FeatureAttr *attributes, const int64_t *labels, const ExtraValAttr *extraVal, const uint8_t *customAttr = nullptr);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Adds features with additional attributes.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1994144463213"><a name="p1994144463213"></a><a name="p1994144463213"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a><code>int64_t count</code></strong>: Number of features to add.</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a><code>const void *features</code></strong>: Features to add. The Hamming distance uses <code>uint8_t</code> data, and <code>Int8Flat</code> uses <code>int8_t</code>.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b1733711363162"><a name="b1733711363162"></a><a name="b1733711363162"></a><code>const FeatureAttr *attributes</code></strong>: Feature attributes to add. For details, see <a href="./05_FeatureAttr.md#en-us_TOPIC_0000001507967381"><code>FeatureAttr</code></a>.</p>
<p id="p32462050775"><a name="p32462050775"></a><a name="p32462050775"></a><strong id="b11413152220159"><a name="b11413152220159"></a><a name="b11413152220159"></a><code>const int64_t *labels</code></strong>: Feature labels to add. Ensure that each label is unique within the <code>Index</code> instance.</p>
<p id="p14651655103714"><a name="p14651655103714"></a><a name="p14651655103714"></a><strong id="b48516115382"><a name="b48516115382"></a><a name="b48516115382"></a><code>const ExtraValAttr *extraVal</code></strong>: Additional feature attributes to add. For details, see <a href="./03_ExtraValAttr.md#en-us_TOPIC_0000002013198657"><code>ExtraValAttr</code></a>.</p>
<p id="p1087211592617"><a name="p1087211592617"></a><a name="p1087211592617"></a><strong id="b9522152852710"><a name="b9522152852710"></a><a name="b9522152852710"></a><code>const uint8_t *customAttr</code></strong>: User-defined feature attributes to add.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><code>count</code> must be in the range <code>[1, 1e6]</code>. The base library capacity is <code>1e9</code>.</li><li>The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>attributes</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and not already exist in the base library. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>extraVal</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>customAttr</code> must be a null pointer or have a length of <code>count * customAttrLen</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>customAttrLen</code> is set in <a href="#en-us_TOPIC_0000001458680014"><code>Init</code></a> or <a href="#en-us_TOPIC_0000002013206217"><code>InitWithExtraVal</code></a>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `AscendIndexTS`<a name="en-us_TOPIC_0000001458200394"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1960115394717"><a name="p1960115394717"></a><a name="p1960115394717"></a><code>AscendIndexTS() = default;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p196741716104810"><a name="p196741716104810"></a><a name="p196741716104810"></a>Constructor of <code>AscendIndexTS</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

<a name="table91172211633"></a>
<table><tbody><tr id="row21174211319"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p2117202114312"><a name="p2117202114312"></a><a name="p2117202114312"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1642685832412"><a name="p1642685832412"></a><a name="p1642685832412"></a><code>AscendIndexTS(const AscendIndexTS &amp;) = delete;</code></p>
</td>
</tr>
<tr id="row81173211232"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p611714212318"><a name="p611714212318"></a><a name="p611714212318"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the copy constructor of this <code>Index</code> as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row61178219313"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p131171921232"><a name="p131171921232"></a><a name="p131171921232"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b20557153917269"><a name="b20557153917269"></a><a name="b20557153917269"></a><code>const AscendIndexTS &amp;</code></strong>: <code>AscendIndexTS</code> object.</p>
</td>
</tr>
<tr id="row14117162110310"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p91177215313"><a name="p91177215313"></a><a name="p91177215313"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1011742114316"><a name="p1011742114316"></a><a name="p1011742114316"></a>None</p>
</td>
</tr>
<tr id="row1911772112317"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1511742113312"><a name="p1511742113312"></a><a name="p1511742113312"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p8117221239"><a name="p8117221239"></a><a name="p8117221239"></a>None</p>
</td>
</tr>
<tr id="row141174211735"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1411719210317"><a name="p1411719210317"></a><a name="p1411719210317"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~AscendIndexTS`<a name="en-us_TOPIC_0000001507760865"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p2734420479"><a name="p2734420479"></a><a name="p2734420479"></a><code>virtual ~AscendIndexTS() = default;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Destructor of <code>AscendIndexTS</code>. It destroys the feature management object.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `DeleteFeatureByLabel`<a name="en-us_TOPIC_0000001458200398"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p71517584492"><a name="p71517584492"></a><a name="p71517584492"></a><code>APP_ERROR DeleteFeatureByLabel(int64_t count, const int64_t *labels);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Deletes features by label.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p1018221113350"><a name="p1018221113350"></a><a name="p1018221113350"></a><strong id="b518231114356"><a name="b518231114356"></a><a name="b518231114356"></a><code>int64_t count</code></strong>: Number of features to delete.</p>
<p id="p0703157114817"><a name="p0703157114817"></a><a name="p0703157114817"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a><code>const int64_t *labels</code></strong>: Feature labels.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><code>count</code> must be in the range <code>[1, 1e6]</code>.</li><li>The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `DeleteFeatureByToken`<a id="en-us_TOPIC_0000001458680018"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p610285755019"><a name="p610285755019"></a><a name="p610285755019"></a><code>APP_ERROR DeleteFeatureByToken(int64_t count, const uint32_t *tokens);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p121015103512"><a name="p121015103512"></a><a name="p121015103512"></a>Deletes features by token.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p960115103339"><a name="p960115103339"></a><a name="p960115103339"></a><strong id="b29989101715"><a name="b29989101715"></a><a name="b29989101715"></a><code>int64_t count</code></strong>: Number of features to delete.</p>
<p id="p871394423613"><a name="p871394423613"></a><a name="p871394423613"></a><strong id="b2814139201"><a name="b2814139201"></a><a name="b2814139201"></a><code>const uint32_t *tokens</code></strong>: Feature tokens.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul981181435015"></a><a name="ul981181435015"></a><ul id="ul981181435015"><li><code>count</code> must be in the range <code>[1, 1e6]</code>.</li><li>The length of <code>tokens</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `FastDeleteFeatureByIndice`<a name="en-us_TOPIC_0000002445152089"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p6761137134011"><a name="p6761137134011"></a><a name="p6761137134011"></a><code>APP_ERROR FastDeleteFeatureByIndice(int64_t count, const int64_t *indices);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p188718287407"><a name="p188718287407"></a><a name="p188718287407"></a>Quickly deletes features by position. This API supports only the additional similarity scenarios of <code>TSFlatIP</code> and <code>TSInt8FlatCos</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p830817388409"><a name="p830817388409"></a><a name="p830817388409"></a><strong id="b7314134311402"><a name="b7314134311402"></a><a name="b7314134311402"></a><code>int64_t count</code></strong>: Number of features to delete.</p>
<p id="p030813385400"><a name="p030813385400"></a><a name="p030813385400"></a><strong id="b280944614401"><a name="b280944614401"></a><a name="b280944614401"></a><code>const int64_t *indices</code></strong>: Positions of the features in the base library.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1991445165319"></a><a name="ul1991445165319"></a><ul id="ul1991445165319"><li><code>count</code> must be greater than <code>0</code> and less than or equal to the number of features in the base library.</li><li>The length of <code>indices</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. The values must be greater than or equal to <code>0</code> and less than the number of features in the base library.</li></ul>
</td>
</tr>
</tbody>
</table>

## `FastDeleteFeatureByRange`<a name="en-us_TOPIC_0000002445960745"></a>

<a name="table18950829154115"></a>
<table><tbody><tr id="row12950172911415"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p1950162910415"><a name="p1950162910415"></a><a name="p1950162910415"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1124533454213"><a name="p1124533454213"></a><a name="p1124533454213"></a><code>APP_ERROR FastDeleteFeatureByRange(int64_t start, int64_t count);</code></p>
</td>
</tr>
<tr id="row295015292419"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p179503291419"><a name="p179503291419"></a><a name="p179503291419"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p9695104615427"><a name="p9695104615427"></a><a name="p9695104615427"></a>Quickly deletes <code>count</code> base library features starting from <code>start</code>. This API supports only the additional similarity scenarios of <code>TSFlatIP</code> and <code>TSInt8FlatCos</code>.</p>
</td>
</tr>
<tr id="row119504292414"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p09501729114112"><a name="p09501729114112"></a><a name="p09501729114112"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p114134413433"><a name="p114134413433"></a><a name="p114134413433"></a><strong id="b124422894315"><a name="b124422894315"></a><a name="b124422894315"></a><code>int64_t start</code></strong>: Start position of the features to delete.</p>
<p id="p1741320420433"><a name="p1741320420433"></a><a name="p1741320420433"></a><strong id="b1149672012430"><a name="b1149672012430"></a><a name="b1149672012430"></a><code>int64_t count</code></strong>: Number of features to delete.</p>
</td>
</tr>
<tr id="row1795015292412"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p4950102984115"><a name="p4950102984115"></a><a name="p4950102984115"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p4950102910412"><a name="p4950102910412"></a><a name="p4950102910412"></a>None</p>
</td>
</tr>
<tr id="row1295062964115"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p12950102912412"><a name="p12950102912412"></a><a name="p12950102912412"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p995022912414"><a name="p995022912414"></a><a name="p995022912414"></a><strong id="b19950029124112"><a name="b19950029124112"></a><a name="b19950029124112"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row8950132974112"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p2095102954118"><a name="p2095102954118"></a><a name="p2095102954118"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul121991916566"></a><a name="ul121991916566"></a><ul id="ul121991916566"><li><code>start</code> must be greater than or equal to <code>0</code> and less than the number of features in the base library.</li><li><code>count</code> must be greater than <code>0</code> and less than or equal to the number of features in the base library.</li><li>The sum of <code>start</code> and <code>count</code> must be less than or equal to the number of features in the base library.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetBaseByRange`<a name="en-us_TOPIC_0000001818301380"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.1.1 "><p id="p312319560281"><a name="p312319560281"></a><a name="p312319560281"></a><code>APP_ERROR GetBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Queries the base library by range.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.3.1 "><p id="p133574814429"><a name="p133574814429"></a><a name="p133574814429"></a><strong id="b168047345217"><a name="b168047345217"></a><a name="b168047345217"></a><code>uint32_t offset</code></strong>: Initial offset for retrieving base library features.</p>
<p id="p580191612428"><a name="p580191612428"></a><a name="p580191612428"></a><strong id="b19447237624"><a name="b19447237624"></a><a name="b19447237624"></a><code>uint32_t num</code></strong>: Number of features.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.4.1 "><p id="p1916155584617"><a name="p1916155584617"></a><a name="p1916155584617"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a><code>int64_t *labels</code></strong>: Feature labels.</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a><code>void *features</code></strong>: Features. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.</p>
<p id="p12993104793912"><a name="p12993104793912"></a><a name="p12993104793912"></a><strong id="b2042017420403"><a name="b2042017420403"></a><a name="b2042017420403"></a><code>FeatureAttr *attributes</code></strong>: Feature attributes.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.6.1 "><a name="ul1652965944519"></a><a name="ul1652965944519"></a><ul id="ul1652965944519"><li><code>0 &lt;= offset &lt; 8.0e8</code>.</li><li><code>0 &lt; num &lt;= 8.0e8</code>.</li><li><code>offset + num &lt;= ntotal</code>.</li><li>The length of <code>labels</code> must be <code>num</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>features</code> must be <code>num * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>attributes</code> must be <code>num</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetBaseByRangeWithExtraVal`<a name="en-us_TOPIC_0000001976495686"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.1.1 "><p id="p1695861915578"><a name="p1695861915578"></a><a name="p1695861915578"></a><code>APP_ERROR GetBaseByRangeWithExtraVal(uint32_t offset, uint32_t num, int64_t *labels, void *features, FeatureAttr *attributes, ExtraValAttr *extraVal) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Queries the base library with additional attributes by range.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.3.1 "><p id="p133574814429"><a name="p133574814429"></a><a name="p133574814429"></a><strong id="b138871842534"><a name="b138871842534"></a><a name="b138871842534"></a><code>uint32_t offset</code></strong>: Initial offset for retrieving base library features.</p>
<p id="p580191612428"><a name="p580191612428"></a><a name="p580191612428"></a><strong id="b13710463319"><a name="b13710463319"></a><a name="b13710463319"></a><code>uint32_t num</code></strong>: Number of features.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.4.1 "><p id="p1916155584617"><a name="p1916155584617"></a><a name="p1916155584617"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a><code>int64_t *labels</code></strong>: Feature labels.</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a><code>void *features</code></strong>: Features. The Hamming distance uses <code>uint8_t</code> data, and <code>Int8Flat</code> uses <code>int8_t</code>.</p>
<p id="p12993104793912"><a name="p12993104793912"></a><a name="p12993104793912"></a><strong id="b2042017420403"><a name="b2042017420403"></a><a name="b2042017420403"></a><code>FeatureAttr *attributes</code></strong>: Feature attributes.</p>
<p id="p584735635516"><a name="p584735635516"></a><a name="p584735635516"></a><strong id="b024346145617"><a name="b024346145617"></a><a name="b024346145617"></a><code>ExtraValAttr *extraVal</code></strong>: Additional attributes.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.919999999999998%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.08%" headers="mcps1.1.3.6.1 "><a name="ul1652965944519"></a><a name="ul1652965944519"></a><ul id="ul1652965944519"><li><code>0 &lt;= offset &lt; 8.0e8</code>.</li><li><code>0 &lt; num &lt;= 8.0e8</code>.</li><li><code>offset + num &lt;= ntotal</code>.</li><li>The length of <code>labels</code> must be <code>num</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>features</code> must be <code>num * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>attributes</code> must be <code>num</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>extraVal</code> must be <code>num</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetBaseMask`<a name="en-us_TOPIC_0000002445112157"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1257715237445"><a name="p1257715237445"></a><a name="p1257715237445"></a><code>APP_ERROR GetBaseMask(int64_t count, uint8_t *mask);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p1228610371441"><a name="p1228610371441"></a><a name="p1228610371441"></a>Obtains the flag that indicates whether the base library has been quickly deleted. If a bit is <code>0</code>, the base library entry at that position has been deleted and is invalid.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p15465289457"><a name="p15465289457"></a><a name="p15465289457"></a><strong id="b166204535358"><a name="b166204535358"></a><a name="b166204535358"></a><code>int64_t count</code></strong>: Valid length of the <code>mask</code> array.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p2022018482356"><a name="p2022018482356"></a><a name="p2022018482356"></a><strong id="b99311008366"><a name="b99311008366"></a><a name="b99311008366"></a><code>uint8_t *mask</code></strong>: Array that marks whether the base library entry has been deleted.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul11392164455912"></a><a name="ul11392164455912"></a><ul id="ul11392164455912"><li><code>count</code> must be in the range <code>[1, ceil(ntotal/8)]</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. Here, <code>ntotal</code> is the number of features in the base library.</li><li>The length of <code>mask</code> must be greater than or equal to <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetCustomAttrByBlockId`<a name="en-us_TOPIC_0000001736682593"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2398651133314"><a name="p2398651133314"></a><a name="p2398651133314"></a><code>APP_ERROR GetCustomAttrByBlockId(uint32_t blockId, uint8_t *&amp;customAttr) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p121015103512"><a name="p121015103512"></a><a name="p121015103512"></a>Obtains the custom attributes of the specified <code>blockId</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p960115103339"><a name="p960115103339"></a><a name="p960115103339"></a><strong id="b64714843417"><a name="b64714843417"></a><a name="b64714843417"></a><code>uint32_t blockId</code></strong>: <code>blockId</code> to retrieve.</p>
<p id="p871394423613"><a name="p871394423613"></a><a name="p871394423613"></a><strong id="b1231449164011"><a name="b1231449164011"></a><a name="b1231449164011"></a><code>uint8_t *&amp;customAttr</code></strong>: User-defined feature attributes on the Device side.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p2142161113218"><a name="p2142161113218"></a><a name="p2142161113218"></a>The length of <code>customAttr</code> must be <code>customAttrBlockSize * customAttrLen</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. <code>customAttrBlockSize</code> and <code>customAttrLen</code> are set in <a href="#en-us_TOPIC_0000001458680014"><code>Init</code></a> or <a href="#en-us_TOPIC_0000002013206217"><code>InitWithExtraVal</code></a>.</p>
</td>
</tr>
</tbody>
</table>

## `GetExtraValAttrByLabel`<a name="en-us_TOPIC_0000001976655414"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1723162213589"><a name="p1723162213589"></a><a name="p1723162213589"></a><code>APP_ERROR GetExtraValAttrByLabel(int64_t count, const int64_t *labels, ExtraValAttr *extraVal) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Obtains the additional attributes of the features with the specified labels.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p92901840173714"><a name="p92901840173714"></a><a name="p92901840173714"></a><strong id="b274355824717"><a name="b274355824717"></a><a name="b274355824717"></a><code>int64_t count</code></strong>: Number of features to retrieve.</p>
<p id="p4672543173915"><a name="p4672543173915"></a><a name="p4672543173915"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a><code>const int64_t *labels</code></strong>: Feature labels.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p12993104793912"><a name="p12993104793912"></a><a name="p12993104793912"></a><strong id="b12012305910"><a name="b12012305910"></a><a name="b12012305910"></a><code>ExtraValAttr *extraVal</code></strong>: Additional attributes.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><code>count</code> must be in the range <code>[1, 1e6]</code>.</li><li>The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. If the input <code>labels</code> do not exist in the base library, the <code>val</code> field in the returned additional attributes is <code>INT16_MIN</code>.</li><li>The length of <code>extraVal</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetFeatureAttrByLabel`<a name="en-us_TOPIC_0000001594544301"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p129110403377"><a name="p129110403377"></a><a name="p129110403377"></a><code>APP_ERROR GetFeatureAttrByLabel(int64_t count, const int64_t *labels, FeatureAttr *attributes) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Obtains the attributes of the features with the specified labels.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p92901840173714"><a name="p92901840173714"></a><a name="p92901840173714"></a><strong id="b274355824717"><a name="b274355824717"></a><a name="b274355824717"></a><code>int64_t count</code></strong>: Number of features to retrieve.</p>
<p id="p4672543173915"><a name="p4672543173915"></a><a name="p4672543173915"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a><code>const int64_t *labels</code></strong>: Feature labels.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p12993104793912"><a name="p12993104793912"></a><a name="p12993104793912"></a><strong id="b2042017420403"><a name="b2042017420403"></a><a name="b2042017420403"></a><code>FeatureAttr *attributes</code></strong>: Feature attributes.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><code>count</code> must be in the range <code>[1, 1e6]</code>.</li><li>The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. If the input <code>labels</code> do not exist in the base library, the returned feature attributes contain <code>time = INT32_MIN</code> and <code>tokenId = UINT32_MAX</code>.</li><li>The length of <code>attributes</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetFeatureByIndice`<a name="en-us_TOPIC_0000002411592888"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.1.1 "><p id="p987171553412"><a name="p987171553412"></a><a name="p987171553412"></a><code>APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels = nullptr, void *features = nullptr, FeatureAttr *attributes = nullptr, ExtraValAttr *extraVal = nullptr) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.2.1 "><p id="p14469247153411"><a name="p14469247153411"></a><a name="p14469247153411"></a>Obtains base library features by position.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.3.1 "><p id="p204841192352"><a name="p204841192352"></a><a name="p204841192352"></a><strong id="b217916181358"><a name="b217916181358"></a><a name="b217916181358"></a><code>int64_t count</code></strong>: Number of features to retrieve.</p>
<p id="p648439143515"><a name="p648439143515"></a><a name="p648439143515"></a><strong id="b1925652253519"><a name="b1925652253519"></a><a name="b1925652253519"></a><code>const int64_t *indices</code></strong>: Positions of the features in the base library.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.4.1 "><p id="p184841691354"><a name="p184841691354"></a><a name="p184841691354"></a><strong id="b6544142883511"><a name="b6544142883511"></a><a name="b6544142883511"></a><code>int64_t *labels</code></strong>: Labels of the features to retrieve.</p>
<p id="p16484892356"><a name="p16484892356"></a><a name="p16484892356"></a><strong id="b1177389353"><a name="b1177389353"></a><a name="b1177389353"></a><code>void *features</code></strong>: Feature vectors to retrieve.</p>
<p id="p1448489143512"><a name="p1448489143512"></a><a name="p1448489143512"></a><strong id="b157791442193515"><a name="b157791442193515"></a><a name="b157791442193515"></a><code>FeatureAttr *attributes</code></strong>: Spatiotemporal attributes of the features to retrieve.</p>
<p id="p124841495355"><a name="p124841495355"></a><a name="p124841495355"></a><strong id="b1896914513357"><a name="b1896914513357"></a><a name="b1896914513357"></a><code>ExtraValAttr *extraVal</code></strong>: Additional attributes of the features to retrieve.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.01%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.99000000000001%" headers="mcps1.1.3.6.1 "><a name="ul1365255411394"></a><a name="ul1365255411394"></a><ul id="ul1365255411394"><li><code>count</code> must be in the range <code>[1, 1e6]</code>.</li><li>The length of <code>indices</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. The values must be greater than or equal to <code>0</code> and less than the number of features in the base library.</li><li>When <code>labels</code> is <code>nullptr</code>, no labels are retrieved. Otherwise, the length must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>features</code> is <code>nullptr</code>, no features are retrieved. Otherwise, the length must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>attributes</code> is <code>nullptr</code>, no attributes are retrieved. Otherwise, the length must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>extraVal</code> is <code>nullptr</code>, no additional attributes are retrieved. Otherwise, the length must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetFeatureByLabel`<a name="en-us_TOPIC_0000001507879789"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p410994614718"><a name="p410994614718"></a><a name="p410994614718"></a><code>APP_ERROR GetFeatureByLabel(int64_t count, const int64_t *labels, void *features);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Retrieves the features with the specified labels.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p6644175415341"><a name="p6644175415341"></a><a name="p6644175415341"></a><strong id="b274355824717"><a name="b274355824717"></a><a name="b274355824717"></a><code>int64_t count</code></strong>: Number of features to retrieve.</p>
<p id="p0703157114817"><a name="p0703157114817"></a><a name="p0703157114817"></a><strong id="b649315397142"><a name="b649315397142"></a><a name="b649315397142"></a><code>const int64_t *labels</code></strong>: Feature labels.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p136503012544"><a name="p136503012544"></a><a name="p136503012544"></a><strong id="b818316576311"><a name="b818316576311"></a><a name="b818316576311"></a><code>void *features</code></strong>: Features retrieved by the specified labels. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><code>count</code> must be in the range <code>[1, 1e6]</code>.</li><li>The length of <code>labels</code> must be <code>count</code>, and all elements must be unique and exist in the base library. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `GetFeatureNum`<a name="en-us_TOPIC_0000001544946953"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p312319560281"><a name="p312319560281"></a><a name="p312319560281"></a><code>APP_ERROR GetFeatureNum(int64_t *totalNum);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Obtains the number of features in this <code>Index</code> instance.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p0703157114817"><a name="p0703157114817"></a><a name="p0703157114817"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p1658819198309"><a name="p1658819198309"></a><a name="p1658819198309"></a><strong id="b10588219113017"><a name="b10588219113017"></a><a name="b10588219113017"></a><code>int64_t *totalNum</code></strong>: Number of features in the base library.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><p id="p20431123883017"><a name="p20431123883017"></a><a name="p20431123883017"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Init`<a id="en-us_TOPIC_0000001458680014"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1177454219213"><a name="p1177454219213"></a><a name="p1177454219213"></a><code>APP_ERROR Init(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, AlgorithmType algType = AlgorithmType::FLAT_COS_INT8, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits&lt;uint64_t&gt;::max());</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Initializes the instance.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p133574814429"><a name="p133574814429"></a><a name="p133574814429"></a><strong id="b57154112218"><a name="b57154112218"></a><a name="b57154112218"></a><code>uint32_t deviceId</code></strong>: Device ID used by the <code>Index</code>.</p>
<p id="p580191612428"><a name="p580191612428"></a><a name="p580191612428"></a><strong id="b1137774510218"><a name="b1137774510218"></a><a name="b1137774510218"></a><code>uint32_t dim</code></strong>: Dimension of the base library vectors.</p>
<p id="p1916155584617"><a name="p1916155584617"></a><a name="p1916155584617"></a><strong id="b1192712462213"><a name="b1192712462213"></a><a name="b1192712462213"></a><code>uint32_t tokenNum</code></strong>: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated <code>Mask</code> operator.</p>
<p id="p13858143610478"><a name="p13858143610478"></a><a name="p13858143610478"></a><strong id="b12145453824"><a name="b12145453824"></a><a name="b12145453824"></a><code>AlgorithmType algType</code></strong>: Distance comparison algorithm used by the backend. The default value is <code>AlgorithmType::FLAT_COS_INT8</code>. Supported algorithms are listed below.</p>
<a name="ul7984114704715"></a><a name="ul7984114704715"></a><ul id="ul7984114704715"><li><code>AlgorithmType::FLAT_HAMMING</code>: Binary feature retrieval (Hamming distance).</li><li><code>AlgorithmType::FLAT_COS_INT8</code>: <code>Int8Flat</code> (cosine distance).</li><li><code>AlgorithmType::FLAT_L2_INT8</code>: <code>Int8Flat</code> (L2 distance).</li><li><code>AlgorithmType::FLAT_IP_FP16</code>: <code>FP16Flat</code> (IP distance).</li><li><code>AlgorithmType::FLAT_HPP_COS_INT8</code>: <code>Int8Flat</code> (cosine distance).</li></ul>
<div class="p" id="p511515103547"><a name="p511515103547"></a><a name="p511515103547"></a><strong id="b18177046143817"><a name="b18177046143817"></a><a name="b18177046143817"></a><code>MemoryStrategy memoryStrategy</code></strong>: Memory strategy used by the backend. The default value is <code>MemoryStrategy::PURE_DEVICE_MEMORY</code>. Supported strategies are listed below.<a name="ul6481121833912"></a><a name="ul6481121833912"></a><ul id="ul6481121833912"><li><code>MemoryStrategy::PURE_DEVICE_MEMORY</code>: Pure Device memory strategy.</li><li><code>MemoryStrategy::HETERO_MEMORY</code>: Heterogeneous memory strategy.</li><li><code>MemoryStrategy::HPP</code>: HPP heterogeneous memory strategy.</li></ul>
</div>
<p id="p14841631173010"><a name="p14841631173010"></a><a name="p14841631173010"></a><strong id="b1383382923017"><a name="b1383382923017"></a><a name="b1383382923017"></a><code>customAttrLen</code></strong>: Length of custom attributes.</p>
<p id="p133252511308"><a name="p133252511308"></a><a name="p133252511308"></a><strong id="b92382483110"><a name="b92382483110"></a><a name="b92382483110"></a><code>customAttrBlockSize</code></strong>: Block size of custom attributes.</p>
<p id="p195521819124"><a name="p195521819124"></a><a name="p195521819124"></a><strong id="b19493104084719"><a name="b19493104084719"></a><a name="b19493104084719"></a><code>maxFeatureRowCount</code></strong>: Maximum number of vectors in the base library.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p157591724172420"><a name="p157591724172420"></a><a name="p157591724172420"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1233201818419"></a><a name="ul1233201818419"></a><ul id="ul1233201818419"><li>Call this API immediately after the constructor.</li><li><code>deviceId</code> must be a valid device ID in the range <code>[0, 1024]</code>.</li><li><code>tokenNum</code> must be in the range <code>(0, 3e5]</code>.</li><li>For binary feature retrieval (Hamming distance), <code>dim</code> must be in <code>{256, 512, 1024}</code>.</li><li>For the <code>Int8Flat</code> algorithm (cosine distance or L2 distance), <code>dim</code> must be in <code>{64, 128, 256, 384, 512, 768, 1024}</code>. For the <code>FP16Flat</code> algorithm (IP distance), <code>dim</code> must be in <code>{64, 128, 256, 384, 512, 768, 1024}</code>.</li><li><code>memoryStrategy::HETERO_MEMORY</code> currently supports only <code>AlgorithmType::FLAT_COS_INT8</code>.</li><li><code>customAttrLen</code> must be in the range <code>[0, 32]</code>. The default value is <code>0</code>. A value of <code>0</code> means that no custom attributes exist.</li><li><code>customAttrBlockSize</code> must be in the range <code>[0, 262144*64]</code> and must be an integer multiple of <code>1024*256</code>. The default value is <code>0</code>. A value of <code>0</code> means that no custom attributes exist.</li><li><code>maxFeatureRowCount</code> must be in the range <code>[262144 * 64, 262144 * 550 * 3]</code> and must be an integer multiple of <code>256</code>. The default value is the maximum value of <code>uint64</code>. This parameter is valid only when <code>MemoryStrategy memoryStrategy</code> is set to <code>MemoryStrategy::HPP</code>.</li><li>When <strong id="b44445251175"><a name="b44445251175"></a><a name="b44445251175"></a><code>MemoryStrategy memoryStrategy</code></strong> is set to <code>MemoryStrategy::HPP</code>, the available Host memory must be at least <code>250 GB</code>, the number of free physical CPU cores must be at least <code>15</code>, and only 256-dimensional vector retrieval is supported.</li></ul>
</td>
</tr>
</tbody>
</table>

## `InitWithExtraVal`<a id="en-us_TOPIC_0000002013206217"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p201951631145314"><a name="p201951631145314"></a><a name="p201951631145314"></a><code>APP_ERROR InitWithExtraVal(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, AlgorithmType algType = AlgorithmType::FLAT_HAMMING, MemoryStrategy memoryStrategy = MemoryStrategy::PURE_DEVICE_MEMORY, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0, uint64_t maxFeatureRowCount = std::numeric_limits&lt;uint64_t&gt;::max());</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Initializes an instance with additional attributes.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p133574814429"><a name="p133574814429"></a><a name="p133574814429"></a><strong id="b57154112218"><a name="b57154112218"></a><a name="b57154112218"></a><code>uint32_t deviceId</code></strong>: Device ID used by the <code>Index</code>.</p>
<p id="p580191612428"><a name="p580191612428"></a><a name="p580191612428"></a><strong id="b1137774510218"><a name="b1137774510218"></a><a name="b1137774510218"></a><code>uint32_t dim</code></strong>: Dimension of the base library vectors.</p>
<p id="p1916155584617"><a name="p1916155584617"></a><a name="p1916155584617"></a><strong id="b1192712462213"><a name="b1192712462213"></a><a name="b1192712462213"></a><code>uint32_t tokenNum</code></strong>: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated <code>Mask</code> operator.</p>
<p id="p19371341202917"><a name="p19371341202917"></a><a name="p19371341202917"></a><strong id="b88105533316"><a name="b88105533316"></a><a name="b88105533316"></a><code>uint64_t resources</code></strong>: Shared memory size.</p>
<p id="p13858143610478"><a name="p13858143610478"></a><a name="p13858143610478"></a><strong id="b12145453824"><a name="b12145453824"></a><a name="b12145453824"></a><code>AlgorithmType algType</code></strong>: Distance comparison algorithm used by the backend. The default value is <code>AlgorithmType::FLAT_HAMMING</code>. Supported algorithms are listed below.</p>
<a name="ul7984114704715"></a><a name="ul7984114704715"></a><ul id="ul7984114704715"><li><code>AlgorithmType::FLAT_HAMMING</code>: Binary feature retrieval (Hamming distance).</li><li><code>AlgorithmType::FLAT_COS_INT8</code>: <code>Int8Flat</code> (cosine distance).</li></ul>
<div class="p" id="p511515103547"><a name="p511515103547"></a><a name="p511515103547"></a><strong id="b18177046143817"><a name="b18177046143817"></a><a name="b18177046143817"></a><code>MemoryStrategy memoryStrategy</code></strong>: Memory strategy used by the backend. The default value is <code>MemoryStrategy::PURE_DEVICE_MEMORY</code>. Supported strategies are listed below.<a name="ul6481121833912"></a><a name="ul6481121833912"></a><ul id="ul6481121833912"><li><code>MemoryStrategy::PURE_DEVICE_MEMORY</code>: Pure Device memory strategy.</li><li><code>MemoryStrategy::HETERO_MEMORY</code>: Heterogeneous memory strategy.</li></ul>
</div>
<p id="p14841631173010"><a name="p14841631173010"></a><a name="p14841631173010"></a><strong id="b1383382923017"><a name="b1383382923017"></a><a name="b1383382923017"></a><code>customAttrLen</code></strong>: Length of custom attributes.</p>
<p id="p133252511308"><a name="p133252511308"></a><a name="p133252511308"></a><strong id="b92382483110"><a name="b92382483110"></a><a name="b92382483110"></a><code>customAttrBlockSize</code></strong>: Block size of custom attributes.</p>
<p id="p195521819124"><a name="p195521819124"></a><a name="p195521819124"></a><strong id="b19493104084719"><a name="b19493104084719"></a><a name="b19493104084719"></a><code>maxFeatureRowCount</code></strong>: Maximum number of vectors in the base library.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1233201818419"></a><a name="ul1233201818419"></a><ul id="ul1233201818419"><li>Call this API immediately after the constructor.</li><li><code>deviceId</code> must be a valid device ID in the range <code>[0, 1024]</code>.</li><li><code>tokenNum</code> must be in the range <code>(0, 3e5]</code>.</li><li><code>resources</code> must be in the range <code>[1 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024]</code>. When you use additional attributes, <code>4 GB</code> is recommended.</li><li>For binary feature retrieval (Hamming distance), <code>dim</code> must be in <code>{256, 512, 1024}</code>.</li><li>For the <code>Int8Flat</code> algorithm (cosine distance), <code>dim</code> must be in <code>{64, 128, 256, 384, 512, 768, 1024}</code>.</li><li><code>customAttrLen</code> must be in the range <code>[0, 32]</code>. The default value is <code>0</code>. A value of <code>0</code> means that no custom attributes exist.</li><li><code>customAttrBlockSize</code> must be in the range <code>[0, 262144 * 64]</code> and must be an integer multiple of <code>1024 * 256</code>. The default value is <code>0</code>. A value of <code>0</code> means that no custom attributes exist.</li><li><code>maxFeatureRowCount</code> does not support HPP when additional attributes are used, and the default value is the maximum value of <code>uint64</code>.</li></ul>
</td>
</tr>
</tbody>
</table>

## `InitWithQuantify`<a name="en-us_TOPIC_0000002458673509"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p2056224963619"><a name="p2056224963619"></a><a name="p2056224963619"></a><code>APP_ERROR InitWithQuantify(uint32_t deviceId, uint32_t dim, uint32_t tokenNum, uint64_t resources, const float *scale, AlgorithmType algType = AlgorithmType::FLAT_IP_FP16, uint32_t customAttrLen = 0, uint32_t customAttrBlockSize = 0);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Initializes the vectorized base library.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p11571112514013"><a name="p11571112514013"></a><a name="p11571112514013"></a><strong id="b3472938164012"><a name="b3472938164012"></a><a name="b3472938164012"></a><code>uint32_t deviceId</code></strong>: Device ID used by the <code>Index</code>.</p>
<p id="p957192544013"><a name="p957192544013"></a><a name="p957192544013"></a><strong id="b118141148114017"><a name="b118141148114017"></a><a name="b118141148114017"></a><code>uint32_t dim</code></strong>: Dimension of the base library vectors.</p>
<p id="p1857110253409"><a name="p1857110253409"></a><a name="p1857110253409"></a><strong id="b1497619074111"><a name="b1497619074111"></a><a name="b1497619074111"></a><code>uint32_t tokenNum</code></strong>: Maximum number of tokens in the current spatiotemporal library. It must match the token count of the generated <code>Mask</code> operator.</p>
<p id="p457112584010"><a name="p457112584010"></a><a name="p457112584010"></a><strong id="b153364211415"><a name="b153364211415"></a><a name="b153364211415"></a><code>uint64_t resources</code></strong>: Shared memory size.</p>
<p id="p1257132594013"><a name="p1257132594013"></a><a name="p1257132594013"></a><strong id="b567513917419"><a name="b567513917419"></a><a name="b567513917419"></a><code>const float *scale</code></strong>: Scaling factor for base library vectorization. After the scaling factor is multiplied by the base library, the result is converted to the <code>int8_t</code> type.</p>
<p id="p1357118253407"><a name="p1357118253407"></a><a name="p1357118253407"></a><strong id="b116721651194112"><a name="b116721651194112"></a><a name="b116721651194112"></a><code>AlgorithmType algType</code></strong>: Distance comparison algorithm used by the backend. The default value is <code>AlgorithmType::FLAT_IP_FP16</code>, which means <code>FP16Flat</code> (IP distance). Currently, only <code>AlgorithmType::FLAT_IP_FP16</code> is supported.</p>
<p id="p668831612439"><a name="p668831612439"></a><a name="p668831612439"></a><strong id="b8861135274318"><a name="b8861135274318"></a><a name="b8861135274318"></a><code>uint32_t customAttrLen</code></strong>: Length of custom attributes.</p>
<p id="p1649111854311"><a name="p1649111854311"></a><a name="p1649111854311"></a><strong id="b146321459440"><a name="b146321459440"></a><a name="b146321459440"></a><code>uint32_t customAttrBlockSize</code></strong>: Block size of custom attributes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1233201818419"></a><a name="ul1233201818419"></a><ul id="ul1233201818419"><li>Call this API immediately after the constructor.</li><li><code>deviceId</code> must be a valid device ID in the range <code>[0, 1024]</code>.</li><li><code>tokenNum</code> must be in the range <code>(0, 3e5]</code>.</li><li><code>resources</code> must be greater than <code>0</code> and less than or equal to <code>4 * 1024 * 1024 * 1024</code>.</li><li>The <code>scale</code> array is used for division during dequantization and must not be close to <code>0</code>. The absolute value of each factor in <code>scale</code> must be greater than or equal to <code>1e-6f</code>.</li><li>For the <code>FP16Flat</code> algorithm (IP distance), <code>dim</code> must be in <code>{64, 128, 256, 384, 512, 768, 1024}</code>.</li><li>Only the non-shared mode of the <code>FP16Flat</code> algorithm (IP distance) is supported.</li><li>This API must be used together with <code>AddFeatureByIndice</code>.</li><li><code>customAttrLen</code> must be in the range <code>[0, 32]</code>. The default value is <code>0</code>. A value of <code>0</code> means that no custom attributes exist.</li><li><code>customAttrBlockSize</code> must be in the range <code>[0, 262144 * 64]</code> and must be an integer multiple of <code>1024 * 256</code>. The default value is <code>0</code>. A value of <code>0</code> means that no custom attributes exist.</li></ul>
</td>
</tr>
</tbody>
</table>

## `operator=`<a name="en-us_TOPIC_0000001507959881"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p26981713282"><a name="p26981713282"></a><a name="p26981713282"></a><code>AscendIndexTS &amp;operator=(const AscendIndexTS &amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the assignment operator of this <code>Index</code> as deleted, which means that the type is non-copyable.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b6551145181714"><a name="b6551145181714"></a><a name="b6551145181714"></a><code>const AscendIndexTS &amp;</code></strong>: Constant <code>AscendIndexTS</code>.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `Search`<a name="en-us_TOPIC_0000001507640109"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1722511394427"><a name="p1722511394427"></a><a name="p1722511394427"></a><code>APP_ERROR Search(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Calculates the distance between the input features and the base library vectors filtered by <code>AttrFilter</code>, sorts the distances by TopK, and returns the corresponding distances and indices.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p2360715171619"><a name="p2360715171619"></a><a name="p2360715171619"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a><code>uint32_t count</code></strong>: Number of features to compare.</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a><code>const void *features</code></strong>: Features to compare. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b161793465431"><a name="b161793465431"></a><a name="b161793465431"></a><code>const AttrFilter *attrFilter</code></strong>: Attribute filter information. For details, see <a href="./02_AttrFilter.md#en-us_TOPIC_0000001458687398"><code>AttrFilter</code></a>.</p>
<p id="p638962561711"><a name="p638962561711"></a><a name="p638962561711"></a><strong id="b72082417127"><a name="b72082417127"></a><a name="b72082417127"></a><code>bool shareAttrFilter</code></strong>: Whether different queries share the same mask.</p>
<p id="p263216178448"><a name="p263216178448"></a><a name="p263216178448"></a><strong id="b9237122811443"><a name="b9237122811443"></a><a name="b9237122811443"></a><code>uint32_t topk</code></strong>: TopK size to keep after cosine distance calculation.</p>
<p id="p198449441221"><a name="p198449441221"></a><a name="p198449441221"></a><strong id="b194349562317"><a name="b194349562317"></a><a name="b194349562317"></a><code>bool enableTimeFilter</code></strong>: Time-stamp attribute filter switch. The default value is <code>true</code>. When <strong id="b12428111462413"><a name="b12428111462413"></a><a name="b12428111462413"></a><code>enableTimeFilter = false</code></strong>, time-stamp attribute filtering is disabled.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p33413632111"><a name="p33413632111"></a><a name="p33413632111"></a><strong id="b170516359452"><a name="b170516359452"></a><a name="b170516359452"></a><code>int64_t *labels</code></strong>: Labels of the TopK features.</p>
<p id="p1861116507441"><a name="p1861116507441"></a><a name="p1861116507441"></a><strong id="b1356612271453"><a name="b1356612271453"></a><a name="b1356612271453"></a><code>float *distances</code></strong>: Distances of the TopK features.</p>
<p id="p1767014017194"><a name="p1767014017194"></a><a name="p1767014017194"></a><strong id="b2308133192313"><a name="b2308133192313"></a><a name="b2308133192313"></a><code>uint32_t *validNums</code></strong>: Number of valid results obtained after each query vector is compared.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><code>count</code> must be in the range <code>[1, 10240]</code>.</li><li>The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>attrFilter</code> must be <code>1</code>. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>topk</code> must be in the range <code>[1, 100000]</code>.</li><li>The length of <code>labels</code> must be <code>count * topk</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>distances</code> must be <code>count * topk</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>validNums</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchWithExtraMask`<a name="en-us_TOPIC_0000001494506850"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p3463193419203"><a name="p3463193419203"></a><a name="p3463193419203"></a><code>APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Calculates the distance between the input features and the base library vectors filtered by <code>AttrFilter</code> and an external <code>Mask</code>, sorts the distances by TopK, and returns the corresponding distances and indices.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p2360715171619"><a name="p2360715171619"></a><a name="p2360715171619"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a><code>uint32_t count</code></strong>: Number of features to compare.</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a><code>const void *features</code></strong>: Features to compare. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b161793465431"><a name="b161793465431"></a><a name="b161793465431"></a><code>const AttrFilter *attrFilter</code></strong>: Attribute filter information. For details, see <a href="./02_AttrFilter.md#en-us_TOPIC_0000001458687398"><code>AttrFilter</code></a>.</p>
<p id="p638962561711"><a name="p638962561711"></a><a name="p638962561711"></a><strong id="b72082417127"><a name="b72082417127"></a><a name="b72082417127"></a><code>bool shareAttrFilter</code></strong>: Whether the same query shares one <code>Mask</code>.</p>
<p id="p263216178448"><a name="p263216178448"></a><a name="p263216178448"></a><strong id="b9237122811443"><a name="b9237122811443"></a><a name="b9237122811443"></a><code>uint32_t topk</code></strong>: TopK size to keep after cosine distance calculation.</p>
<p id="p1792132311"><a name="p1792132311"></a><a name="p1792132311"></a><strong id="b10193171182320"><a name="b10193171182320"></a><a name="b10193171182320"></a><code>const uint8_t *extraMask</code></strong>: Additional filter <code>Mask</code> provided from outside. The value is expressed in bits, where <code>0</code> and <code>1</code> indicate filtering or selecting the feature respectively.</p>
<p id="p1426510564258"><a name="p1426510564258"></a><a name="p1426510564258"></a><strong id="b11418152210274"><a name="b11418152210274"></a><a name="b11418152210274"></a><code>uint64_t extraMaskLenEachQuery</code></strong>: Length of the external <code>Mask</code>, in bytes.</p>
<p id="p1165364810261"><a name="p1165364810261"></a><a name="p1165364810261"></a><strong id="b171528264278"><a name="b171528264278"></a><a name="b171528264278"></a><code>bool extraMaskIsAtDevice</code></strong>: Whether the external <code>Mask</code> already exists on the Device side.</p>
<p id="p198449441221"><a name="p198449441221"></a><a name="p198449441221"></a><strong id="b194349562317"><a name="b194349562317"></a><a name="b194349562317"></a><code>bool enableTimeFilter</code></strong>: Time-stamp attribute filter switch. The default value is <code>true</code>. When <strong id="b12428111462413"><a name="b12428111462413"></a><a name="b12428111462413"></a><code>enableTimeFilter = false</code></strong>, time-stamp attribute filtering is disabled.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p695214202452"><a name="p695214202452"></a><a name="p695214202452"></a><strong id="b170516359452"><a name="b170516359452"></a><a name="b170516359452"></a><code>int64_t *labels</code></strong>: Labels of the TopK features.</p>
<p id="p1861116507441"><a name="p1861116507441"></a><a name="p1861116507441"></a><strong id="b1356612271453"><a name="b1356612271453"></a><a name="b1356612271453"></a><code>float *distances</code></strong>: Distances of the TopK features.</p>
<p id="p33413632111"><a name="p33413632111"></a><a name="p33413632111"></a><strong id="b2308133192313"><a name="b2308133192313"></a><a name="b2308133192313"></a><code>uint32_t *validNums</code></strong>: Number of valid results obtained after each query vector is compared.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul19870412737"></a><a name="ul19870412737"></a><ul id="ul19870412737"><li><code>count</code> must be in the range <code>[1, 10240]</code>.</li><li><code>topk</code> must be in the range <code>[1, 100000]</code>.</li><li>The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>attrFilter</code> must be <code>1</code>. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>distances</code> must be <code>count * topk</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>validNums</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>labels</code> must be <code>count * topk</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>extraMask</code> must be <code>extraMaskLenEachQuery</code>. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count * extraMaskLenEachQuery</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchWithExtraMask` with Extra Similarity<a name="en-us_TOPIC_0000002373091106"></a>

<a name="table197013362381"></a>
<table><tbody><tr id="row597023693810"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p199703365385"><a name="p199703365385"></a><a name="p199703365385"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p99701836173815"><a name="p99701836173815"></a><a name="p99701836173815"></a><code>APP_ERROR SearchWithExtraMask(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk,const uint8_t *extraMask, uint64_t extraMaskLenEachQuery, bool extraMaskIsAtDevice, const uint16_t *extraScore, int64_t *labels,float *distances, uint32_t *validNums, bool enableTimeFilter = true);</code></p>
</td>
</tr>
<tr id="row109701336173810"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p17970153683816"><a name="p17970153683816"></a><a name="p17970153683816"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p169706365385"><a name="p169706365385"></a><a name="p169706365385"></a>Calculates the distance between the input features and the base library vectors filtered by <code>AttrFilter</code> and an external <code>Mask</code>, sorts the distances by TopK, and returns the corresponding distances and indices.</p>
</td>
</tr>
<tr id="row19702366386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p18970113673814"><a name="p18970113673814"></a><a name="p18970113673814"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p119701036143819"><a name="p119701036143819"></a><a name="p119701036143819"></a><strong id="b1897053618384"><a name="b1897053618384"></a><a name="b1897053618384"></a><code>uint32_t count</code></strong>: Number of features to compare.</p>
<p id="p1697015367382"><a name="p1697015367382"></a><a name="p1697015367382"></a><strong id="b49704363384"><a name="b49704363384"></a><a name="b49704363384"></a><code>const void *features</code></strong>: Features to compare. The Hamming distance uses <code>uint8_t</code> data, <code>Int8Flat</code> uses <code>int8_t</code>, and <code>FP16Flat</code> uses <code>float</code>.</p>
<p id="p12970173633811"><a name="p12970173633811"></a><a name="p12970173633811"></a><strong id="b2097015361386"><a name="b2097015361386"></a><a name="b2097015361386"></a><code>const AttrFilter *attrFilter</code></strong>: Attribute filter information. For details, see <a href="./02_AttrFilter.md#en-us_TOPIC_0000001458687398"><code>AttrFilter</code></a>.</p>
<p id="p18970193617382"><a name="p18970193617382"></a><a name="p18970193617382"></a><strong id="b16970133693819"><a name="b16970133693819"></a><a name="b16970133693819"></a><code>bool shareAttrFilter</code></strong>: Whether the same query shares one <code>Mask</code>.</p>
<p id="p497013613389"><a name="p497013613389"></a><a name="p497013613389"></a><strong id="b3970123613810"><a name="b3970123613810"></a><a name="b3970123613810"></a><code>uint32_t topk</code></strong>: TopK size to keep after cosine distance calculation.</p>
<p id="p119709368387"><a name="p119709368387"></a><a name="p119709368387"></a><strong id="b109708366381"><a name="b109708366381"></a><a name="b109708366381"></a><code>const uint8_t *extraMask</code></strong>: Additional filter <code>Mask</code> provided from outside. The value is expressed in bits, where <code>0</code> and <code>1</code> indicate filtering or selecting the feature respectively.</p>
<p id="p5970123623814"><a name="p5970123623814"></a><a name="p5970123623814"></a><strong id="b2970103610387"><a name="b2970103610387"></a><a name="b2970103610387"></a><code>uint64_t extraMaskLenEachQuery</code></strong>: Length of the external <code>Mask</code>, in bytes.</p>
<p id="p1970103663815"><a name="p1970103663815"></a><a name="p1970103663815"></a><strong id="b19701936173815"><a name="b19701936173815"></a><a name="b19701936173815"></a><code>bool extraMaskIsAtDevice</code></strong>: Whether the external <code>Mask</code> already exists on the Device side.</p>
<p id="p472715355451"><a name="p472715355451"></a><a name="p472715355451"></a><strong id="b367692814812"><a name="b367692814812"></a><a name="b367692814812"></a><code>const uint16_t *extraScore</code></strong>: Additional similarity provided by the user. The length is <code>count * totalPad</code>, where <code>totalPad</code> is the base library length aligned to 16.</p>
<p id="p1397083623812"><a name="p1397083623812"></a><a name="p1397083623812"></a><strong id="b69700361388"><a name="b69700361388"></a><a name="b69700361388"></a><code>bool enableTimeFilter</code></strong>: Time-stamp attribute filter switch. The default value is <code>true</code>. When <strong id="b11970153611385"><a name="b11970153611385"></a><a name="b11970153611385"></a><code>enableTimeFilter = false</code></strong>, time-stamp attribute filtering is disabled.</p>
</td>
</tr>
<tr id="row199701366383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p09701636163810"><a name="p09701636163810"></a><a name="p09701636163810"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p897011361383"><a name="p897011361383"></a><a name="p897011361383"></a><strong id="b15970336173812"><a name="b15970336173812"></a><a name="b15970336173812"></a><code>int64_t *labels</code></strong>: Labels of the TopK features. If the base library is added by using <code>AddFeatureByIndice</code>, the output here is the base library position (<code>indices</code>).</p>
<p id="p1497083653817"><a name="p1497083653817"></a><a name="p1497083653817"></a><strong id="b6970436153820"><a name="b6970436153820"></a><a name="b6970436153820"></a><code>float *distances</code></strong>: Distances of the TopK features.</p>
<p id="p19701536113820"><a name="p19701536113820"></a><a name="p19701536113820"></a><strong id="b159701536183810"><a name="b159701536183810"></a><a name="b159701536183810"></a><code>uint32_t *validNums</code></strong>: Number of valid results obtained after each query vector is compared.</p>
</td>
</tr>
<tr id="row3970143643812"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p0970936143817"><a name="p0970936143817"></a><a name="p0970936143817"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p13970536113820"><a name="p13970536113820"></a><a name="p13970536113820"></a><strong id="b15970133653813"><a name="b15970133653813"></a><a name="b15970133653813"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row097033617385"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p20970153617385"><a name="p20970153617385"></a><a name="p20970153617385"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul16970136203817"></a><a name="ul16970136203817"></a><ul id="ul16970136203817"><li><code>count</code> must be in the range <code>[1, 10240]</code>.</li><li><code>topk</code> must be in the range <code>[1, 100000]</code>.</li><li>The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>attrFilter</code> must be <code>1</code>. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>distances</code> must be <code>count * topk</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>validNums</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>labels</code> must be <code>count * topk</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>extraMask</code> must be <code>extraMaskLenEachQuery</code>. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count * extraMaskLenEachQuery</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>extraScore</code> must be <code>count * totalPad</code>, where <code>totalPad</code> is the base library length aligned to <code>16</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash. It actually corresponds to the <code>float16_t</code> type, and its values range from <code>-1.0</code> to <code>1.0</code>. It is currently effective only for non-shared masks in <code>Int8FlatCos</code> and <code>FlatIP</code>. Otherwise, <code>extraScore</code> does not take part in the calculation.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SearchWithExtraVal`<a name="en-us_TOPIC_0000002013215285"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p1394085711557"><a name="p1394085711557"></a><a name="p1394085711557"></a><code>APP_ERROR SearchWithExtraVal(uint32_t count, const void *features, const AttrFilter *attrFilter, bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums, const ExtraValFilter *extraValFilter, bool enableTimeFilter = true);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p0795142313481"><a name="p0795142313481"></a><a name="p0795142313481"></a>Calculates the distance between the input features and the base library vectors filtered by <code>AttrFilter</code> and <code>ExtraValFilter</code>, sorts the distances by TopK, and returns the corresponding distances and indices.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p2360715171619"><a name="p2360715171619"></a><a name="p2360715171619"></a><strong id="b125611612173311"><a name="b125611612173311"></a><a name="b125611612173311"></a><code>uint32_t count</code></strong>: Number of features to compare.</p>
<p id="p874812810555"><a name="p874812810555"></a><a name="p874812810555"></a><strong id="b17401300315"><a name="b17401300315"></a><a name="b17401300315"></a><code>const void *features</code></strong>: Features to compare. The Hamming distance uses <code>uint8_t</code> data, and <code>Int8Flat</code> uses <code>int8_t</code>. Currently, only <code>Int8Flat</code> (including heterogeneous memory scenarios) and Hamming distance are supported.</p>
<p id="p661314244382"><a name="p661314244382"></a><a name="p661314244382"></a><strong id="b161793465431"><a name="b161793465431"></a><a name="b161793465431"></a><code>const AttrFilter *attrFilter</code></strong>: Attribute filter information. For details, see <a href="./02_AttrFilter.md#en-us_TOPIC_0000001458687398"><code>AttrFilter</code></a>.</p>
<p id="p638962561711"><a name="p638962561711"></a><a name="p638962561711"></a><strong id="b72082417127"><a name="b72082417127"></a><a name="b72082417127"></a><code>bool shareAttrFilter</code></strong>: Additional attributes currently support only <code>false</code>. Different queries do not share the same mask.</p>
<p id="p263216178448"><a name="p263216178448"></a><a name="p263216178448"></a><strong id="b9237122811443"><a name="b9237122811443"></a><a name="b9237122811443"></a><code>uint32_t topk</code></strong>: TopK size to keep after cosine distance calculation.</p>
<p id="p576645315463"><a name="p576645315463"></a><a name="p576645315463"></a><strong id="b1599945784610"><a name="b1599945784610"></a><a name="b1599945784610"></a><code>const ExtraValFilter *extraValFilter</code></strong>: Additional attribute filter information. For details, see <a href="./04_ExtraValFilter.md#en-us_TOPIC_0000002013200765"><code>ExtraValFilter</code></a>.</p>
<p id="p198449441221"><a name="p198449441221"></a><a name="p198449441221"></a><strong id="b194349562317"><a name="b194349562317"></a><a name="b194349562317"></a><code>bool enableTimeFilter</code></strong>: Time-stamp attribute filter switch. The default value is <code>true</code>. When <strong id="b12428111462413"><a name="b12428111462413"></a><a name="b12428111462413"></a><code>enableTimeFilter = false</code></strong>, time-stamp attribute filtering is disabled.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p33413632111"><a name="p33413632111"></a><a name="p33413632111"></a><strong id="b2308133192313"><a name="b2308133192313"></a><a name="b2308133192313"></a><code>uint32_t *validNums</code></strong>: Number of valid results obtained after each query vector is compared.</p>
<p id="p695214202452"><a name="p695214202452"></a><a name="p695214202452"></a><strong id="b170516359452"><a name="b170516359452"></a><a name="b170516359452"></a><code>int64_t *labels</code></strong>: Labels of the TopK features.</p>
<p id="p1861116507441"><a name="p1861116507441"></a><a name="p1861116507441"></a><strong id="b1356612271453"><a name="b1356612271453"></a><a name="b1356612271453"></a><code>float *distances</code></strong>: Distances of the TopK features.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul13705385331"></a><a name="ul13705385331"></a><ul id="ul13705385331"><li><code>count</code> must be in the range <code>[1, 10240]</code>.</li><li>The length of <code>features</code> must be <code>count * dim</code>, where <code>dim</code> is the vector dimension. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>When <code>shareAttrFilter</code> is <code>true</code>, the length of <code>attrFilter</code> must be <code>1</code>. When <code>shareAttrFilter</code> is <code>false</code>, the length must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>topk</code> must be in the range <code>[1, 100000]</code>.</li><li>The length of <code>labels</code> must be <code>count * topk</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>distances</code> must be <code>count * topk</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li>The length of <code>validNums</code> must be <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li><li><code>extraValFilter</code> must be a null pointer or have a length of <code>count</code>. Otherwise, out-of-bounds read and write errors may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

> [!NOTE]
>
> `SearchWithExtraVal` cannot be used together with `Search`.

## `SetHeteroParam`<a name="en-us_TOPIC_0000001630850578"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.1.1 "><p id="p312319560281"><a name="p312319560281"></a><a name="p312319560281"></a><code>APP_ERROR SetHeteroParam(size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Sets the parameters of the heterogeneous storage strategy.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.3.1 "><p id="p2908312183018"><a name="p2908312183018"></a><a name="p2908312183018"></a><strong id="b156074619418"><a name="b156074619418"></a><a name="b156074619418"></a><code>size_t deviceCapacity</code></strong>: Base library capacity stored on the Device side when the heterogeneous memory strategy is used, in bytes.</p>
<p id="p1534173611301"><a name="p1534173611301"></a><a name="p1534173611301"></a><strong id="b81425114419"><a name="b81425114419"></a><a name="b81425114419"></a><code>size_t deviceBuffer</code></strong>: Cache capacity on the Device side when the heterogeneous memory strategy is used, in bytes.</p>
<p id="p194803440119"><a name="p194803440119"></a><a name="p194803440119"></a><strong id="b208671753121117"><a name="b208671753121117"></a><a name="b208671753121117"></a><code>size_t hostCapacity</code></strong>: Base library capacity stored on the Host side when the heterogeneous memory strategy is used, in bytes.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.4.1 "><p id="p1658819198309"><a name="p1658819198309"></a><a name="p1658819198309"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: The operation status. For details, see the interface return value reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.02%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.97999999999999%" headers="mcps1.1.3.6.1 "><a name="ul19200250124118"></a><a name="ul19200250124118"></a><ul id="ul19200250124118"><li>Use this API after you set the memory strategy to <code>MemoryStrategy::HETERO_MEMORY</code> in the <a href="#en-us_TOPIC_0000001458680014"><code>Init</code></a> API.</li><li>The minimum value of <code>deviceCapacity</code> is <code>1GB</code>, and the maximum value is the actual remaining Device memory.</li><li>The minimum value of <code>deviceBuffer</code> is <code>2 * 262144 * dim</code>, and the maximum value is <code>8GB</code>. Set it according to the actual remaining Device memory.</li><li><strong id="b16232174415416"><a name="b16232174415416"></a><a name="b16232174415416"></a><code>deviceCapacity + deviceBuffer</code></strong> must be smaller than the actual remaining Device memory.</li><li>The value range of <code>hostCapacity</code> is <code>[1GB, 512GB]</code>. Configure it according to the amount of memory that can be allocated on the Host side.</li></ul>
</td>
</tr>
</tbody>
</table>

## `SetSaveHostMemory`<a name="en-us_TOPIC_0000002106649489"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.1.1 "><p id="p45681112145513"><a name="p45681112145513"></a><a name="p45681112145513"></a><code>APP_ERROR SetSaveHostMemory();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.2.1 "><p id="p1984752010553"><a name="p1984752010553"></a><a name="p1984752010553"></a>Sets the host memory saving mode. This mode is disabled by default.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.3.1 "><p id="p2908312183018"><a name="p2908312183018"></a><a name="p2908312183018"></a>None</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.4.1 "><p id="p1658819198309"><a name="p1658819198309"></a><a name="p1658819198309"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b51473590249"><a name="b51473590249"></a><a name="b51473590249"></a><code>APP_ERROR</code></strong>: Operation status. For details, see the API Return Value Reference.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.1%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.9%" headers="mcps1.1.3.6.1 "><a name="ul13741324723"></a><a name="ul13741324723"></a><ul id="ul13741324723"><li>Use this API after the <a href="#en-us_TOPIC_0000001458680014">Init</a> API when the base library size is 0.</li><li>This API saves host memory but reduces the performance of delete-type and retrieve-type APIs.</li><li>When this mode is enabled, the <a href="#en-us_TOPIC_0000001458680018">DeleteFeatureByToken</a> API cannot be used.</li><li>This API supports only the Hamming distance.</li></ul>
</td>
</tr>
</tbody>
</table>
