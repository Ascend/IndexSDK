# 版本配套说明<a name="ZH-CN_TOPIC_0000002524441743"></a>

## 产品版本信息<a name="ZH-CN_TOPIC_0000002492442016"></a>

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>Index SDK</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a>26.0.0</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>Release版本</p>
</td>
</tr>
</tbody>
</table>

## 相关产品版本配套说明<a name="ZH-CN_TOPIC_0000002524561713"></a>

|产品名称|版本|
|--|--|
|Ascend HDK|26.0.RC1|
|CANN|9.0.0|

## 病毒扫描结果<a name="ZH-CN_TOPIC_0000002492442006"></a>

病毒扫描通过。

# 版本兼容性说明<a name="ZH-CN_TOPIC_0000002492442012"></a>

- Index SDK：在升级本版本后，基于Index SDK开发的应用需要重新编译，并重新生成相关算子。

**表 1**  软件版本兼容性说明

|MindSDK软件版本|MindSDK待升级版本|CANN版本兼容性|Ascend HDK版本兼容性|
|--|--|--|--|
|Index SDK 26.0.0 |<li>MindSDK 6.0.RC3及补丁版本</li><li>MindSDK 6.0.0及补丁版本</li><li>MindSDK 7.0.RC1及补丁版本</li><li>MindSDK 7.1.RC1及补丁版本</li><li>MindSDK 7.2.RC1及补丁版本</li><li>MindSDK 7.3.0及补丁版本</li>|<li>CANN 8.1.RC1及补丁版本</li><li>CANN 8.2.RC1及补丁版本</li><li>CANN 8.3.RC1及补丁版本</li><li>CANN 8.5.0及补丁版本</li><li>CANN 9.0.0及补丁版本</li>|<li>Ascend HDK 25.0.RC1及补丁版本</li><li>Ascend HDK 25.2.0及补丁版本</li><li>Ascend HDK 25.3.RC1及补丁版本</li><li>Ascend HDK 25.5.0及补丁版本</li><li>Ascend HDK 26.0.RC1及补丁版本</li>|

> [!NOTE]
>软件版本兼容性是指产品软件版本升级时，其他关联软件不需要联动升级或打补丁，仍然可以支持已有功能。

# 版本使用注意事项<a name="ZH-CN_TOPIC_0000002492282032"></a>

无

# 更新说明<a name="ZH-CN_TOPIC_0000002524441747"></a>

## 新增特性<a name="ZH-CN_TOPIC_0000002492282034"></a>

|特性名称|特性描述|配套产品型号|
|---------|--------------------------------------------|----------|
|Index SDK|<li>ILFlat标准态性能优化：底库500w，256维度，通过GetFeature随机获取4w底库，耗时减少到25ms内。</li><li>异构内存支持时空库附加属性：功能支持时空库TSInt8FlatCos，1024维度下，异构内存场景能添加附加属性，检索能正常过滤附加属性。</li>|Atlas 300I Pro 推理卡<br>Atlas 300V 视频解析卡<br>Atlas 300V Pro 视频解析卡<br>Atlas 300I Duo 推理卡<br>Atlas 200I SoC A1 核心板<br>Atlas 300I 推理卡（型号 3000）<br>Atlas 300I 推理卡（型号 3010）<br>Atlas 800I A2 推理服务器|

## 业务接口变更<a name="ZH-CN_TOPIC_0000002492442008"></a>

**Index SDK<a name="zh-cn_topic_0000001963197973_section3125124045019"></a>**

- 不涉及接口变更。

## 关键特性变更<a name="ZH-CN_TOPIC_0000002524441749"></a>

无

## 已解决的问题<a name="ZH-CN_TOPIC_0000002492442002"></a>

无

## 遗留问题<a name="ZH-CN_TOPIC_0000002524561719"></a>

- 无遗留问题。

# 升级影响<a name="ZH-CN_TOPIC_0000002524561715"></a>

## 升级过程对现行系统的影响<a name="ZH-CN_TOPIC_0000002492282026"></a>

无

## 升级后对现行系统的影响<a name="ZH-CN_TOPIC_0000002524441745"></a>

无

# 26.0.0版本配套文档<a name="ZH-CN_TOPIC_0000002524561717"></a>

|文档名称|内容简介|更新说明|
|--|--|--|
|《Index SDK 26.0.0 用户指南》|主要包括Index SDK的使用流程、算法介绍、算子生成说明、API接口说明以及其他常用的操作。|变更详见《[Index SDK 26.0.0 用户指南](introduction.md#软件架构)》。|

# 漏洞修补列表<a name="ZH-CN_TOPIC_0000002492282030"></a>

无
