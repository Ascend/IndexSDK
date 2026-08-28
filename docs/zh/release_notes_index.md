# 版本说明

## 版本配套说明

### 产品版本信息

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody>
<tr><td width="140" align="center" valign="middle">产品名称</td><td valign="middle">Index SDK</td></tr>
<tr><td width="140" align="center" valign="middle">产品版本</td><td valign="middle">26.1.0</td></tr>
<tr><td width="140" align="center" valign="middle">版本类型</td><td valign="middle">Release版本</td></tr>
</tbody></table>

### 相关产品版本配套说明

**表 1** Index SDK 软件版本配套表

<table><tbody>
<tr><th width="110" align="center" valign="middle">Index SDK</th><th width="110" align="center" valign="middle">CANN 版本</th><th width="160" align="center" valign="middle">Ascend HDK 版本</th></tr>
<tr><td width="110" align="center" valign="middle">26.1.0</td><td width="110" align="center" valign="middle">9.1.0</td><td width="160" align="center" valign="middle">26.1.0</td></tr>
</tbody></table>

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 2** Index SDK 与 CANN 版本兼容

<table>
<thead>
  <tr>
    <th rowspan="2" valign="middle" align="center">Index SDK</th>
    <th colspan="3" valign="middle" align="center">CANN 版本</th>
  </tr>
  <tr>
    <th valign="middle" align="center">8.5.0</th>
    <th valign="middle" align="center">9.0.0</th>
    <th valign="middle" align="center">9.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td valign="middle" align="center">7.3.0</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">/</td>
    <td valign="middle" align="center">/</td>
  </tr>
  <tr>
    <td valign="middle" align="center">26.0.0</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">/</td>
  </tr>
  <tr>
    <td valign="middle" align="center">26.1.0</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">Y</td>
  </tr>
</tbody>
</table>

**表 3** Index SDK 与 Ascend HDK 版本兼容

<table>
<thead>
  <tr>
    <th rowspan="2" valign="middle" align="center">Index SDK</th>
    <th colspan="3" valign="middle" align="center">Ascend HDK 版本</th>
  </tr>
  <tr>
    <th valign="middle" align="center">25.5.0</th>
    <th valign="middle" align="center">26.0.RC1</th>
    <th valign="middle" align="center">26.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td valign="middle" align="center">7.3.0</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">/</td>
    <td valign="middle" align="center">/</td>
  </tr>
  <tr>
    <td valign="middle" align="center">26.0.0</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">/</td>
  </tr>
  <tr>
    <td valign="middle" align="center">26.1.0</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">Y</td>
    <td valign="middle" align="center">Y</td>
  </tr>
</tbody>
</table>

## 版本使用注意事项

无

## 更新说明

### 新增特性

|特性名称|特性描述|配套产品型号|
|---------|--------------------------------------------|----------|
|TS FlatIP 与 Int8Cos 支持 A2/A3 平台|TS FlatIP, Int8Cos 支持 A2 A3：A2 底库 6 千万，A3 底库 1.25 亿，256 维度，batch1-256，topk200。|Atlas 800I A3 超节点服务器<br>Atlas 800I A2 推理服务器|
|IVF-RabitQ 索引支持 A2/A3 平台|A2, A3 支持 IVF-RabitQ：底库 1000 万，128 维度，topk300，数据精度 FP32。|Atlas 800I A3 超节点服务器<br>Atlas 800I A2 推理服务器|

### 业务接口变更

**Index SDK**

- IVF-RabitQ新增train，remove_ids，copyFrom，copyTo，update。

### 关键特性变更

**Index SDK**

- 不涉及关键特性变更。

### 已解决的问题

无

### 遗留问题

无

## 升级影响

### 升级过程对现行系统的影响

无

### 升级后对现行系统的影响

无

## 26.1.0版本配套文档

|文档名称|内容简介|更新说明|
|--|--|--|
|《Index SDK 26.1.0 用户指南》|主要包括Index SDK的使用流程、算法介绍、算子生成说明、API接口说明以及其他常用的操作。|详见《[Index SDK 26.1.0 用户指南](01_introduction.md#软件架构)》。|

## 病毒扫描结果

病毒扫描通过。

## 漏洞修补列表

无
