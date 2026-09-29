# 版本说明

## 关键特性

新增了 IVFPQ 索引在 Atlas A3推理系列产品和 Ascend 950PR系列产品上的支持。

## 版本配套说明

### 产品版本信息

<table><tbody><tr><th class="firstcol" valign="top" width="25%"><p>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="75%"><p><span>Index SDK</span></p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="25%"><p>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="75%"><p>26.2.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="25%"><p>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="75%"><p>Release版本</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="25%"><p>维护周期</p>
</th>
<td class="cellrowborder" valign="top" width="75%"><p>参考<a href="https://gitcode.com/Ascend/IndexSDK/blob/master/README.md#%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5">维护策略</a></p>
</td>
</tr>
</tbody>
</table>

### 相关产品版本配套说明

**表 1** Index SDK 软件版本配套表

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="25%"><p>产品名称</p>
</th>
<th class="cellrowborder" valign="top" width="75%"><p>版本</p>
</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="25%"><p>Index SDK</p>
</td>
<td class="cellrowborder" valign="top" width="75%"><p>26.2.0</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="25%"><p>Ascend HDK</p>
</td>
<td class="cellrowborder" valign="top" width="75%"><p>26.2.0</p>
</td>
</tr>
<tr><td class="cellrowborder" valign="top" width="25%"><p>CANN</p>
</td>
<td class="cellrowborder" valign="top" width="75%"><p>9.2.0</p>
</td>
</tr>
</tbody>
</table>

## 版本兼容性说明

> [!NOTE]
>
> 本节表格中“/”表示不可配套，“Y”表示可配套。

**表 2** Index SDK 与 CANN 版本兼容

<table style="table-layout: fixed; width: 900px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">Index SDK</th>
      <th colspan="5">CANN 版本</th>
    </tr>
    <tr>
      <th>8.3.RC1</th>
      <th>8.5.0</th>
      <th>9.0.0</th>
      <th>9.1.0</th>
      <th>9.2.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7.3.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.0.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <td>26.2.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

**表 3** Index SDK 与 Ascend HDK 版本兼容

<table style="table-layout: fixed; width: 900px; text-align:center">
  <colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 150px">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2">Index SDK</th>
      <th colspan="5">Ascend HDK 版本</th>
    </tr>
    <tr>
      <th>25.3.0</th>
      <th>25.5.0</th>
      <th>26.0.0</th>
      <th>26.1.0</th>
      <th>26.2.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7.3.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.0.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>/</td>
      <td>/</td>
    </tr>
    <tr>
      <td>26.1.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
    <tr>
      <td>26.2.0</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
      <td>Y</td>
    </tr>
  </tbody>
</table>

## 版本使用注意事项

新开发的 IVFPQ 索引算法兼容 CANN 8.5.0 及以上版本，低于 CANN 8.5.0 的版本不在适配范围内。

## 更新说明

### 新增特性说明

**Index SDK**

- IVFPQ 索引支持 Atlas A3推理系列产品和 Ascend 950PR系列产品

### 关键特性变更

**Index SDK**

- 不涉及关键特性变更。

### 业务接口变更

**Index SDK**

- 不涉及业务接口变更。

### 已解决的问题

无。

### 遗留问题

无。

## 升级影响

### 升级过程对现行系统的影响

无。

### 升级后对现行系统的影响

无。

## 版本配套文档

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="25%"><p>文档名称</p>
</th>
<th class="cellrowborder" valign="top" width="50%"><p>内容简介</p>
</th>
<th class="cellrowborder" valign="top" width="25%"><p>更新说明</p>
</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="25%"><p>《Index SDK 26.2.0 用户指南》</p>
</td>
<td class="cellrowborder" valign="top" width="50%"><p>主要包括 Index SDK 的使用流程、算法介绍、算子生成说明、API 接口说明以及其他常用的操作。</p>
</td>
<td class="cellrowborder" valign="top" width="25%"><p>详见《<a href="01_introduction.md#软件架构">Index SDK 26.2.0 用户指南</a>》。</p>
</td>
</tr>
</tbody>
</table>

## 病毒扫描结果

病毒扫描通过。

## 漏洞修补列表

无。

## 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| --- | --- | --- |
| 01 | 2026-09-30 | 第一次正式发布。 |
