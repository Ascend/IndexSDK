# Version Mapping<a name="ZH-CN_TOPIC_0000002524441743"></a>

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-05-30T02:42:53.095Z -->

## Product Version<a name="ZH-CN_TOPIC_0000002492442016"></a>

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody>
<tr><td width="150" align="center" valign="middle">Product</td><td valign="middle">Index SDK</td></tr>
<tr><td width="150" align="center" valign="middle">Product Version</td><td valign="middle">26.0.0</td></tr>
<tr><td width="150" align="center" valign="middle">Version Type</td><td valign="middle">Release Version</td></tr>
</tbody></table>

## Related Product Versions<a name="ZH-CN_TOPIC_0000002524561713"></a>

<table><tbody>
<tr><td width="140" align="center" valign="middle">Product</td><td valign="middle">Version</td></tr>
<tr><td width="140" align="center" valign="middle">Ascend HDK</td><td valign="middle">26.0.RC1</td></tr>
<tr><td width="140" align="center" valign="middle">CANN</td><td valign="middle">9.0.0</td></tr>
</tbody></table>

## Virus Scan Results<a name="ZH-CN_TOPIC_0000002492442006"></a>

Virus scan passed.

# Version Compatibility<a name="ZH-CN_TOPIC_0000002492442012"></a>

- Index SDK: After upgrading to this version, applications developed based on Index SDK need to be recompiled and relevant operators need to be regenerated.

**Table 1**  Software Version Compatibility Description

<table><tbody>
<tr><th align="center" valign="middle" width="240">MindSDK Software Version</th><th align="center" valign="middle">MindSDK Version to Upgrade</th><th align="center" valign="middle">CANN Version Compatibility</th><th align="center" valign="middle">Ascend HDK Version Compatibility</th></tr>
<tr><td align="center" valign="middle" width="240">Index SDK 26.0.0</td><td valign="middle">● MindSDK 6.0.RC3 and patch versions<br>● MindSDK 6.0.0 and patch versions<br>● MindSDK 7.0.RC1 and patch versions<br>● MindSDK 7.1.RC1 and patch versions<br>● MindSDK 7.2.RC1 and patch versions<br>● MindSDK 7.3.0 and patch versions</td><td valign="middle">● CANN 8.1.RC1 and patch versions<br>● CANN 8.2.RC1 and patch versions<br>● CANN 8.3.RC1 and patch versions<br>● CANN 8.5.0 and patch versions<br>● CANN 9.0.0 and patch versions</td><td valign="middle">● Ascend HDK 25.0.RC1 and patch versions<br>● Ascend HDK 25.2.0 and patch versions<br>● Ascend HDK 25.3.RC1 and patch versions<br>● Ascend HDK 25.5.0 and patch versions<br>● Ascend HDK 26.0.RC1 and patch versions</td></tr>
</tbody></table>

> [!NOTE]
> Software version compatibility means that when the product software version is upgraded, other related software does not need to be upgraded or patched at the same time, and existing functions remain supported.

# Important Notes<a name="ZH-CN_TOPIC_0000002492282032"></a>

None

# Update Notes<a name="ZH-CN_TOPIC_0000002524441747"></a>

## New Features<a name="ZH-CN_TOPIC_0000002492282034"></a>

<table><tbody>
<tr><td align="center" valign="middle" width="140"><strong>Feature</strong></td><td align="center" valign="middle"><strong>Description</strong></td><td align="center" valign="middle" width="360"><strong>Supported Product Model</strong></td></tr>
<tr><td valign="middle" width="140">Index SDK</td><td valign="middle">● ILFlat standard-state performance optimization: For a base library with 5 million entries and 256 dimensions, the time required to randomly retrieve 40,000 entries from the base library with GetFeature is reduced to within 25 ms.<br>● Heterogeneous memory support for additional attributes in the spatiotemporal library: The feature supports TSInt8FlatCos in the spatiotemporal library. At 1024 dimensions, you can add additional attributes in heterogeneous memory scenarios, and retrieval can correctly filter by additional attributes.</td><td valign="middle" width="360">Atlas 300I Pro Inference Card<br>Atlas 300V Video Analysis Card<br>Atlas 300V Pro Video Analysis Card<br>Atlas 300I Duo Inference Card<br>Atlas 200I SoC A1 Core Board<br>Atlas 300I Inference Card (Model 3000)<br>Atlas 300I Inference Card (Model 3010)<br>Atlas 800I A2 Inference Server</td></tr>
</tbody></table>

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

<table><tbody>
<tr><td align="center" valign="middle" width="280"><strong>Document</strong></td><td align="center" valign="middle"><strong>Description</strong></td><td align="center" valign="middle"><strong>Release Notes</strong></td></tr>
<tr><td valign="middle" width="280">*Index SDK 26.0.0 User Guide*</td><td valign="middle">Mainly includes the usage process of Index SDK, algorithm introduction, operator generation instructions, API interface descriptions, and other common operations.</td><td valign="middle">For changes, see *<a href="introduction.md#software-architecture">Index SDK 26.0.0 User Guide</a>*.</td></tr>
</tbody></table>

# Fixed Vulnerabilities<a name="ZH-CN_TOPIC_0000002492282030"></a>

None
