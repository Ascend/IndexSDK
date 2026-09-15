# `AscendNNInference`<a name="en-us_TOPIC_0000001456375320"></a>

## Function Description<a name="en-us_TOPIC_0000001456535204"></a>

Performs inference through a neural network.

## `AscendNNInference`<a name="en-us_TOPIC_0000001456854780"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a><code>AscendNNInference(std::vector&lt;int&gt; deviceList, const char* model, uint64_t modelSize);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Constructor of <code>AscendNNInference</code>. It creates the <code>AscendNNInference</code> object and configures the Ascend AI Processor resources on the device side and the model based on the values in <code>deviceList</code>.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><code>std::vector&lt;int&gt; deviceList</code>: Device IDs on the device side.</p>
<p id="p187869217128"><a name="p187869217128"></a><a name="p187869217128"></a><code>const char* model</code>: Deep neural network inference model.</p>
<p id="p11661833191215"><a name="p11661833191215"></a><a name="p11661833191215"></a><code>uint64_t modelSize</code>: Size of the deep neural network inference model.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul246474615523"></a><a name="ul246474615523"></a><ul id="ul246474615523"><li>The valid range of <code>deviceList</code> is (0, 32].</li><li><code>model</code> must be a valid memory pointer to a deep neural network inference model, and its size must be <code>modelSize</code>. The valid range of <code>modelSize</code> is (0, 128 MB]. Parameter mismatches may cause model instantiation or inference to fail. Invalid models may harm the system. Ensure that the model source is valid and effective.<a name="ul29631955112419"></a><a name="ul29631955112419"></a><ul id="ul29631955112419"><li><code>dimIn</code> ∈ {64, 128, 256, 384, 512, 768, 1024}.</li><li><code>dimOut</code> ∈ {32, 64, 96, 128, 256}.</li><li><code>batches</code> ∈ {1, 2, 4, 8, 16, 32, 64, 128}.</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1246213101873"></a>
<table><tbody><tr id="row1462121015717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p84621510171"><a name="p84621510171"></a><a name="p84621510171"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p17980142452716"><a name="p17980142452716"></a><a name="p17980142452716"></a><code>AscendNNInference(const AscendNNInference&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row164624102073"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p12462161014713"><a name="p12462161014713"></a><a name="p12462161014713"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1462161010718"><a name="p1462161010718"></a><a name="p1462161010718"></a>Declares the copy constructor of <code>AscendNNInference</code> as deleted. Therefore, <code>AscendNNInference</code> is a non-copyable type.</p>
</td>
</tr>
<tr id="row5462101013718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p04623106716"><a name="p04623106716"></a><a name="p04623106716"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendNNInference&amp;</code>: Constant <code>AscendNNInference</code>.</p>
</td>
</tr>
<tr id="row12462610671"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p94628106715"><a name="p94628106715"></a><a name="p94628106715"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p946213108719"><a name="p946213108719"></a><a name="p946213108719"></a>None</p>
</td>
</tr>
<tr id="row194623101670"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1546320101573"><a name="p1546320101573"></a><a name="p1546320101573"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1346321016716"><a name="p1346321016716"></a><a name="p1346321016716"></a>None</p>
</td>
</tr>
<tr id="row104631810275"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1046391017711"><a name="p1046391017711"></a><a name="p1046391017711"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `~AscendNNInference`<a name="en-us_TOPIC_0000001506495737"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a><code>~AscendNNInference();</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Destructor of <code>AscendNNInference</code>. It destroys the <code>AscendNNInference</code> object and releases resources.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>None</p>
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

## `getDimBatch`<a name="en-us_TOPIC_0000001506334797"></a>

<a name="en-us_topic_0000001287392566_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001287392566_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001287392566_p12559123810"><a name="en-us_topic_0000001287392566_p12559123810"></a><a name="en-us_topic_0000001287392566_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001287392566_p13466158111313"><a name="en-us_topic_0000001287392566_p13466158111313"></a><a name="en-us_topic_0000001287392566_p13466158111313"></a><code>int getDimBatch() const;</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001287392566_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001287392566_p1212599383"><a name="en-us_topic_0000001287392566_p1212599383"></a><a name="en-us_topic_0000001287392566_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001287392566_p131714208358"><a name="en-us_topic_0000001287392566_p131714208358"></a><a name="en-us_topic_0000001287392566_p131714208358"></a>Gets the number of samples or query vectors in a single inference pass.</p>
</td>
</tr>
<tr id="en-us_topic_0000001287392566_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001287392566_p112195910383"><a name="en-us_topic_0000001287392566_p112195910383"></a><a name="en-us_topic_0000001287392566_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001287392566_p1963814585141"><a name="en-us_topic_0000001287392566_p1963814585141"></a><a name="en-us_topic_0000001287392566_p1963814585141"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001287392566_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001287392566_p17235973820"><a name="en-us_topic_0000001287392566_p17235973820"></a><a name="en-us_topic_0000001287392566_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001287392566_p8451184515218"><a name="en-us_topic_0000001287392566_p8451184515218"></a><a name="en-us_topic_0000001287392566_p8451184515218"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001287392566_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001287392566_p182459113812"><a name="en-us_topic_0000001287392566_p182459113812"></a><a name="en-us_topic_0000001287392566_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001287392566_p132314362521"><a name="en-us_topic_0000001287392566_p132314362521"></a><a name="en-us_topic_0000001287392566_p132314362521"></a>The number of samples or query vectors in a single inference pass.</p>
</td>
</tr>
<tr id="en-us_topic_0000001287392566_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001287392566_p423590386"><a name="en-us_topic_0000001287392566_p423590386"></a><a name="en-us_topic_0000001287392566_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001287392566_p991611401713"><a name="en-us_topic_0000001287392566_p991611401713"></a><a name="en-us_topic_0000001287392566_p991611401713"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getInputType`<a name="en-us_TOPIC_0000001456854776"></a>

<a name="en-us_topic_0000001340072289_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001340072289_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001340072289_p12559123810"><a name="en-us_topic_0000001340072289_p12559123810"></a><a name="en-us_topic_0000001340072289_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001340072289_p13466158111313"><a name="en-us_topic_0000001340072289_p13466158111313"></a><a name="en-us_topic_0000001340072289_p13466158111313"></a><code>int getInputType() const;</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001340072289_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001340072289_p1212599383"><a name="en-us_topic_0000001340072289_p1212599383"></a><a name="en-us_topic_0000001340072289_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001340072289_p131714208358"><a name="en-us_topic_0000001340072289_p131714208358"></a><a name="en-us_topic_0000001340072289_p131714208358"></a>Gets the input data type of the model.</p>
</td>
</tr>
<tr id="en-us_topic_0000001340072289_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001340072289_p112195910383"><a name="en-us_topic_0000001340072289_p112195910383"></a><a name="en-us_topic_0000001340072289_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001340072289_p1963814585141"><a name="en-us_topic_0000001340072289_p1963814585141"></a><a name="en-us_topic_0000001340072289_p1963814585141"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001340072289_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001340072289_p17235973820"><a name="en-us_topic_0000001340072289_p17235973820"></a><a name="en-us_topic_0000001340072289_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001340072289_p8451184515218"><a name="en-us_topic_0000001340072289_p8451184515218"></a><a name="en-us_topic_0000001340072289_p8451184515218"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001340072289_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001340072289_p182459113812"><a name="en-us_topic_0000001340072289_p182459113812"></a><a name="en-us_topic_0000001340072289_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001340072289_p132314362521"><a name="en-us_topic_0000001340072289_p132314362521"></a><a name="en-us_topic_0000001340072289_p132314362521"></a>The input data type of the model.</p>
</td>
</tr>
<tr id="en-us_topic_0000001340072289_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001340072289_p423590386"><a name="en-us_topic_0000001340072289_p423590386"></a><a name="en-us_topic_0000001340072289_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001340072289_p991611401713"><a name="en-us_topic_0000001340072289_p991611401713"></a><a name="en-us_topic_0000001340072289_p991611401713"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getOutputType`<a name="en-us_TOPIC_0000001456854868"></a>

<a name="en-us_topic_0000001340232437_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001340232437_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001340232437_p12559123810"><a name="en-us_topic_0000001340232437_p12559123810"></a><a name="en-us_topic_0000001340232437_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001340232437_p13466158111313"><a name="en-us_topic_0000001340232437_p13466158111313"></a><a name="en-us_topic_0000001340232437_p13466158111313"></a><code>int getOutputType() const;</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001340232437_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001340232437_p1212599383"><a name="en-us_topic_0000001340232437_p1212599383"></a><a name="en-us_topic_0000001340232437_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001340232437_p131714208358"><a name="en-us_topic_0000001340232437_p131714208358"></a><a name="en-us_topic_0000001340232437_p131714208358"></a>Gets the output data type of the model.</p>
</td>
</tr>
<tr id="en-us_topic_0000001340232437_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001340232437_p112195910383"><a name="en-us_topic_0000001340232437_p112195910383"></a><a name="en-us_topic_0000001340232437_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001340232437_p1963814585141"><a name="en-us_topic_0000001340232437_p1963814585141"></a><a name="en-us_topic_0000001340232437_p1963814585141"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001340232437_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001340232437_p17235973820"><a name="en-us_topic_0000001340232437_p17235973820"></a><a name="en-us_topic_0000001340232437_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001340232437_p8451184515218"><a name="en-us_topic_0000001340232437_p8451184515218"></a><a name="en-us_topic_0000001340232437_p8451184515218"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001340232437_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001340232437_p182459113812"><a name="en-us_topic_0000001340232437_p182459113812"></a><a name="en-us_topic_0000001340232437_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001340232437_p132314362521"><a name="en-us_topic_0000001340232437_p132314362521"></a><a name="en-us_topic_0000001340232437_p132314362521"></a>The output data type of the model.</p>
</td>
</tr>
<tr id="en-us_topic_0000001340232437_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001340232437_p423590386"><a name="en-us_topic_0000001340232437_p423590386"></a><a name="en-us_topic_0000001340232437_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001340232437_p991611401713"><a name="en-us_topic_0000001340232437_p991611401713"></a><a name="en-us_topic_0000001340232437_p991611401713"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getDimIn`<a name="en-us_TOPIC_0000001456535128"></a>

<a name="en-us_topic_0000001287712442_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001287712442_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001287712442_p12559123810"><a name="en-us_topic_0000001287712442_p12559123810"></a><a name="en-us_topic_0000001287712442_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001287712442_p13466158111313"><a name="en-us_topic_0000001287712442_p13466158111313"></a><a name="en-us_topic_0000001287712442_p13466158111313"></a><code>int getDimIn() const;</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001287712442_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001287712442_p1212599383"><a name="en-us_topic_0000001287712442_p1212599383"></a><a name="en-us_topic_0000001287712442_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001287712442_p131714208358"><a name="en-us_topic_0000001287712442_p131714208358"></a><a name="en-us_topic_0000001287712442_p131714208358"></a>Gets the input data dimension of the model.</p>
</td>
</tr>
<tr id="en-us_topic_0000001287712442_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001287712442_p112195910383"><a name="en-us_topic_0000001287712442_p112195910383"></a><a name="en-us_topic_0000001287712442_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001287712442_p1963814585141"><a name="en-us_topic_0000001287712442_p1963814585141"></a><a name="en-us_topic_0000001287712442_p1963814585141"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001287712442_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001287712442_p17235973820"><a name="en-us_topic_0000001287712442_p17235973820"></a><a name="en-us_topic_0000001287712442_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001287712442_p8451184515218"><a name="en-us_topic_0000001287712442_p8451184515218"></a><a name="en-us_topic_0000001287712442_p8451184515218"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001287712442_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001287712442_p182459113812"><a name="en-us_topic_0000001287712442_p182459113812"></a><a name="en-us_topic_0000001287712442_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001287712442_p132314362521"><a name="en-us_topic_0000001287712442_p132314362521"></a><a name="en-us_topic_0000001287712442_p132314362521"></a>The input data dimension.</p>
</td>
</tr>
<tr id="en-us_topic_0000001287712442_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001287712442_p423590386"><a name="en-us_topic_0000001287712442_p423590386"></a><a name="en-us_topic_0000001287712442_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001287712442_p991611401713"><a name="en-us_topic_0000001287712442_p991611401713"></a><a name="en-us_topic_0000001287712442_p991611401713"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `getDimOut`<a name="en-us_TOPIC_0000001456695056"></a>

<a name="en-us_topic_0000001287552486_table7235918388"></a>
<table><tbody><tr id="en-us_topic_0000001287552486_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="en-us_topic_0000001287552486_p12559123810"><a name="en-us_topic_0000001287552486_p12559123810"></a><a name="en-us_topic_0000001287552486_p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="en-us_topic_0000001287552486_p13466158111313"><a name="en-us_topic_0000001287552486_p13466158111313"></a><a name="en-us_topic_0000001287552486_p13466158111313"></a><code>int getDimOut() const;</code></p>
</td>
</tr>
<tr id="en-us_topic_0000001287552486_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="en-us_topic_0000001287552486_p1212599383"><a name="en-us_topic_0000001287552486_p1212599383"></a><a name="en-us_topic_0000001287552486_p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="en-us_topic_0000001287552486_p131714208358"><a name="en-us_topic_0000001287552486_p131714208358"></a><a name="en-us_topic_0000001287552486_p131714208358"></a>Gets the output data dimension of the model.</p>
</td>
</tr>
<tr id="en-us_topic_0000001287552486_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="en-us_topic_0000001287552486_p112195910383"><a name="en-us_topic_0000001287552486_p112195910383"></a><a name="en-us_topic_0000001287552486_p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="en-us_topic_0000001287552486_p1963814585141"><a name="en-us_topic_0000001287552486_p1963814585141"></a><a name="en-us_topic_0000001287552486_p1963814585141"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001287552486_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="en-us_topic_0000001287552486_p17235973820"><a name="en-us_topic_0000001287552486_p17235973820"></a><a name="en-us_topic_0000001287552486_p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="en-us_topic_0000001287552486_p8451184515218"><a name="en-us_topic_0000001287552486_p8451184515218"></a><a name="en-us_topic_0000001287552486_p8451184515218"></a>None</p>
</td>
</tr>
<tr id="en-us_topic_0000001287552486_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="en-us_topic_0000001287552486_p182459113812"><a name="en-us_topic_0000001287552486_p182459113812"></a><a name="en-us_topic_0000001287552486_p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="en-us_topic_0000001287552486_p132314362521"><a name="en-us_topic_0000001287552486_p132314362521"></a><a name="en-us_topic_0000001287552486_p132314362521"></a>The output data dimension of the model.</p>
</td>
</tr>
<tr id="en-us_topic_0000001287552486_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="en-us_topic_0000001287552486_p423590386"><a name="en-us_topic_0000001287552486_p423590386"></a><a name="en-us_topic_0000001287552486_p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="en-us_topic_0000001287552486_p991611401713"><a name="en-us_topic_0000001287552486_p991611401713"></a><a name="en-us_topic_0000001287552486_p991611401713"></a>None</p>
</td>
</tr>
</tbody>
</table>

## `infer`<a name="en-us_TOPIC_0000001506495709"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a><code>void infer(size_t n, const char* inputData, char* outputData) const;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Performs inference using the neural network model.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.3.1 "><p id="p1963814585141"><a name="p1963814585141"></a><a name="p1963814585141"></a><code>size_t n</code>: Number of inputs for inference.</p>
<p id="p1633753171511"><a name="p1633753171511"></a><a name="p1633753171511"></a><code>const char* inputData</code>: Feature vectors for inference.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.4.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><code>char* outputData</code>: Feature vector results from inference.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.6.1 "><a name="ul58419974316"></a><a name="ul58419974316"></a><ul id="ul58419974316"><li>The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9.</li><li>Pointer <code>inputData</code> must be non-null, and its length must be <code>dimIn * n</code>. Pointer <code>outputData</code> must be non-null, and its length must be <code>dimOut * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `operator=`<a name="en-us_TOPIC_0000001456535156"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p17980142452716"><a name="p17980142452716"></a><a name="p17980142452716"></a><code>AscendNNInference&amp; operator=(const AscendNNInference&amp;) = delete;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>Declares the copy assignment operator of <code>AscendNNInference</code> as deleted. Therefore, <code>AscendNNInference</code> is a non-copyable type.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><code>const AscendNNInference&amp;</code>: Constant <code>AscendNNInference</code>.</p>
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
