# Release Notes

## Key Features

Adds support for the IVFPQ index on Atlas A3 inference products and Atlas 950PR products.

## Version Information

### Product Version Information

<table><tbody><tr><th class="firstcol" valign="top" width="25%"><p>Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="75%"><p>Index SDK</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="25%"><p>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="75%"><p>26.2.0</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="25%"><p>Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="75%"><p>Release Version</p>
</td>
</tr>
<tr><th class="firstcol" valign="top" width="25%"><p>Maintenance Period</p>
</th>
<td class="cellrowborder" valign="top" width="75%"><p>For details, see the <a href="https://atomgit.com/Ascend/IndexSDK/blob/master/README.md#%E7%89%88%E6%9C%AC%E7%BB%B4%E6%8A%A4%E7%AD%96%E7%95%A5">Maintenance Policy</a>.</p>
</td>
</tr>
</tbody>
</table>

### Related Product Version Compatibility

**Table 1**  Index SDK Software Version Compatibility

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="25%"><p>Product Name</p>
</th>
<th class="cellrowborder" valign="top" width="75%"><p>Version</p>
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

## Version Compatibility

> [!NOTE]
>
> In the tables in this section, "/" indicates that the versions are incompatible, and "Y" indicates that the versions are compatible.

**Table 2**  Index SDK and CANN Version Compatibility

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
      <th colspan="5">CANN Version</th>
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

**Table 3**  Index SDK and Ascend HDK Version Compatibility

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
      <th colspan="5">Ascend HDK Version</th>
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

## Important Usage Notes

The newly developed IVFPQ index algorithm is compatible with CANN 8.5.0 and later versions. Versions earlier than CANN 8.5.0 are not supported.

## Update Notes

### New Features

**Index SDK**

- The IVFPQ index supports Atlas A3 inference products and Atlas 950PR products.

### Key Feature Changes

**Index SDK**

- No key feature changes are involved.

### Service Interface Changes

**Index SDK**

- No service interface changes are involved.

### Resolved Issues

None

### Known Issues

None

## Upgrade Impact

### Impact on the System During the Upgrade

None

### Impact on the System After the Upgrade

None

## Documentation

<table><thead align="left"><tr><th class="cellrowborder" valign="top" width="25%"><p>Document Name</p>
</th>
<th class="cellrowborder" valign="top" width="50%"><p>Description</p>
</th>
<th class="cellrowborder" valign="top" width="25%"><p>Update Notes</p>
</th>
</tr>
</thead>
<tbody><tr><td class="cellrowborder" valign="top" width="25%"><p>Index SDK 26.2.0 User Guide</p>
</td>
<td class="cellrowborder" valign="top" width="50%"><p>Mainly includes the Index SDK usage process, algorithm descriptions, operator generation instructions, API descriptions, and other commonly used operations.</p>
</td>
<td class="cellrowborder" valign="top" width="25%"><p>For details, see the <a href="01_introduction.md#software-architecture">Index SDK 26.2.0 User Guide</a>.</p>
</td>
</tr>
</tbody>
</table>

## Virus Scan Results

Virus scan passed.

## Vulnerability Fixes

None

## Revision History

| Document Version | Release Date | Change Description |
| --- | --- | --- |
| 01 | 2026-09-30 | First official release. |
