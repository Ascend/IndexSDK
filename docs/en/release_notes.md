# Version Mapping<a name="ZH-CN_TOPIC_0000002524441743"></a>

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-30T02:42:53.095Z -->

## Product Version<a name="ZH-CN_TOPIC_0000002492442016"></a>

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a>Product</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>Index SDK</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a>26.0.0</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>Release Version</p>
</td>
</tr>
</tbody>
</table>

## Related Product Versions<a name="ZH-CN_TOPIC_0000002524561713"></a>

|Product|Version|
|--|--|
|Ascend HDK|26.0.RC1|
|CANN|9.0.0|

## Virus Scan Results<a name="ZH-CN_TOPIC_0000002492442006"></a>

Virus scan passed.

# Version Compatibility<a name="ZH-CN_TOPIC_0000002492442012"></a>

- Index SDK: After upgrading to this version, applications developed based on Index SDK need to be recompiled and relevant operators need to be regenerated.

**Table 1**  Software Version Compatibility Description

|MindSDK Software Version|MindSDK Version to Upgrade|CANN Version Compatibility|Ascend HDK Version Compatibility|
|--|--|--|--|
|Index SDK 26.0.0 |<li>MindSDK 6.0.RC3 and patch versions</li><li>MindSDK 6.0.0 and patch versions</li><li>MindSDK 7.0.RC1 and patch versions</li><li>MindSDK 7.1.RC1 and patch versions</li><li>MindSDK 7.2.RC1 and patch versions</li><li>MindSDK 7.3.0 and patch versions</li>|<li>CANN 8.1.RC1 and patch versions</li><li>CANN 8.2.RC1 and patch versions</li><li>CANN 8.3.RC1 and patch versions</li><li>CANN 8.5.0 and patch versions</li><li>CANN 9.0.0 and patch versions</li>|<li>Ascend HDK 25.0.RC1 and patch versions</li><li>Ascend HDK 25.2.0 and patch versions</li><li>Ascend HDK 25.3.RC1 and patch versions</li><li>Ascend HDK 25.5.0 and patch versions</li><li>Ascend HDK 26.0.RC1 and patch versions</li>|

> [!NOTE]
> Software version compatibility means that when the product software version is upgraded, other related software does not need to be upgraded or patched at the same time, and existing functions remain supported.

# Important Notes<a name="ZH-CN_TOPIC_0000002492282032"></a>

None

# Update Notes<a name="ZH-CN_TOPIC_0000002524441747"></a>

## New Features<a name="ZH-CN_TOPIC_0000002492282034"></a>

|Feature|Description|Supported Product Model|
|---------|--------------------------------------------|----------|
|Index SDK|<li>ILFlat standard-state performance optimization: For a base library with 5 million entries and 256 dimensions, the time required to randomly retrieve 40,000 entries from the base library with GetFeature is reduced to within 25 ms.</li><li>Heterogeneous memory support for additional attributes in the spatiotemporal library: The feature supports TSInt8FlatCos in the spatiotemporal library. At 1024 dimensions, you can add additional attributes in heterogeneous memory scenarios, and retrieval can correctly filter by additional attributes.</li>|Atlas 300I Pro Inference Card<br>Atlas 300V Video Analysis Card<br>Atlas 300V Pro Video Analysis Card<br>Atlas 300I Duo Inference Card<br>Atlas 200I SoC A1 Core Board<br>Atlas 300I Inference Card (Model 3000)<br>Atlas 300I Inference Card (Model 3010)<br>Atlas 800I A2 Inference Server|

## Service Interface Changes<a name="ZH-CN_TOPIC_0000002492442008"></a>

**Index SDK<a name="zh-cn_topic_0000001963197973_section3125124045019"></a>**

- No interface changes are involved.

## Key Feature Changes<a name="ZH-CN_TOPIC_0000002524441749"></a>

None

## Resolved Issues<a name="ZH-CN_TOPIC_0000002492442002"></a>

None

## Known Issues<a name="ZH-CN_TOPIC_0000002524561719"></a>

- No known issues.

# Upgrade Impact<a name="ZH-CN_TOPIC_0000002524561715"></a>

## Impact on the System During the Upgrade<a name="ZH-CN_TOPIC_0000002492282026"></a>

None

## Impact on the System After the Upgrade<a name="ZH-CN_TOPIC_0000002524441745"></a>

None

# 26.0.0 Documentation<a name="ZH-CN_TOPIC_0000002524561717"></a>

|Document|Description|Release Notes|
|--|--|--|
|*Index SDK 26.0.0 User Guide*|Mainly includes the usage process of Index SDK, algorithm introduction, operator generation instructions, API interface descriptions, and other common operations.|For changes, see *[Index SDK 26.0.0 User Guide](introduction.md#software-architecture)*.|

# Fixed Vulnerabilities<a name="ZH-CN_TOPIC_0000002492282030"></a>

None
