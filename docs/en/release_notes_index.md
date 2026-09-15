# Release Notes

## Version Information

### Product Version Information

<a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p246mcpsimp"><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p246mcpsimp"></a><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p246mcpsimp"></a>Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p1684675795511"><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p1684675795511"></a><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>Index SDK</span></p>
</td>
</tr>
<tr id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p257mcpsimp"><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p257mcpsimp"></a><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p257mcpsimp"></a>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p233mcpsimp"><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p233mcpsimp"></a><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p233mcpsimp"></a>26.1.0</p>
</td>
</tr>
<tr id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p7260182135013"><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p7260182135013"></a><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p7260182135013"></a>Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p72606219501"><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p72606219501"></a><a name="en-us_topic_0000001938532254_en-us_topic_0000001935094108_p72606219501"></a>Release Version</p>
</td>
</tr>
</tbody>
</table>

### Related Product Version Compatibility

**Table 1**  Index SDK Software Version Compatibility

| Index SDK | CANN Version | Ascend HDK Version |
| ------------ | ------------- | ------------ |
| 26.1.0 | 9.1.0 | 26.1.0 |

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates that the versions are incompatible, and "Y" indicates that the versions are compatible.

**Table 2**  Index SDK and CANN Version Compatibility

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Index SDK</th>
    <th colspan="3">CANN Version</th>
  </tr>
  <tr>
    <th>8.5.0</th>
    <th>9.0.0</th>
    <th>9.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>7.3.0</td>
    <td>Y</td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

**Table 3**  Index SDK and Ascend HDK Version Compatibility

<table style="table-layout: fixed; width: 433px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Index SDK</th>
    <th colspan="3">Ascend HDK Version</th>
  </tr>
  <tr>
    <th>25.5.0</th>
    <th>26.0.RC1</th>
    <th>26.1.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>7.3.0</td>
    <td>Y</td>
    <td>/</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.0.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>/</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td>Y</td>
    <td>Y</td>
    <td>Y</td>
  </tr>
</tbody>
</table>

## Important Usage Notes

None

## Update Notes

### New Features

| Feature | Description | Supported Product Model |
|---------|--------------------------------------------|----------|
| TS FlatIP and Int8Cos Support on A2/A3 Platforms | TS FlatIP and Int8Cos support A2 and A3 platforms: a base library with 60 million entries on A2 and 125 million entries on A3, 256 dimensions, batch sizes of 1 to 256, and TopK of 200. | Atlas 800I A3 SuperPoD Server<br>Atlas 800I A2 Inference Server |
| IVF-RabitQ Index Support on A2/A3 Platforms | IVF-RabitQ supports A2 and A3 platforms: a base library with 10 million entries, 128 dimensions, TopK of 300, and FP32 data precision. | Atlas 800I A3 SuperPoD Server<br>Atlas 800I A2 Inference Server |

### Service Interface Changes

**Index SDK**

- IVF-RabitQ adds `train`, `remove_ids`, `copyFrom`, `copyTo`, and `update`.

### Key Feature Changes

**Index SDK**

- No key feature changes are involved.

### Resolved Issues

None

### Known Issues

None

## Upgrade Impact

### Impact on the System During the Upgrade

None

### Impact on the System After the Upgrade

None

## 26.1.0 Documentation

| Document | Description | Update Notes |
|--|--|--|
| *Index SDK 26.1.0 User Guide* | Mainly includes the Index SDK usage process, algorithm descriptions, operator generation instructions, API descriptions, and other commonly used operations. | For details, see [Index SDK 26.1.0 User Guide](01_introduction.md#software-architecture). |

## Virus Scan Results

Virus scan passed.

## Vulnerability Fixes

None
