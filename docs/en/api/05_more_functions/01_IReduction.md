# IReduction<a name="en-us_TOPIC_0000001456694992"></a>

## Function Description<a name="en-us_TOPIC_0000001506615161"></a>

`IReduction` is the unified interface for dimensionality reduction methods in the feature retrieval component. It currently supports the `PCAR` and `NN` dimensionality reduction algorithms.

## `CreateReduction`<a name="en-us_TOPIC_0000001456695108"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p034518361750"><a name="p034518361750"></a><a name="p034518361750"></a><code>IReduction *CreateReduction(std::string typeName, const ReductionConfig &amp;config);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p966663212512"><a name="p966663212512"></a><a name="p966663212512"></a>Creates a specific dimensionality reduction algorithm.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><code>std::string typeName</code>: Dimensionality reduction algorithm parameter. Valid values are <code>{"NN", "PCAR"}</code>.</p>
<p id="p1579483519305"><a name="p1579483519305"></a><a name="p1579483519305"></a><code>ReductionConfig &amp;config</code>: Dimensionality reduction configuration.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><code>IReduction *CreateReduction</code>: Created dimensionality reduction instance.</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>Currently, only the <code>NN</code> and <code>PCAR</code> dimensionality reduction parameters are supported. Using any other parameter causes an exception.</p>
<p id="p18899829194617"><a name="p18899829194617"></a><a name="p18899829194617"></a>After you finish using this instance, remember to delete this pointer to release the corresponding memory.</p>
</td>
</tr>
</tbody>
</table>

## `reduce`<a name="en-us_TOPIC_0000001456375280"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p107777311038"><a name="p107777311038"></a><a name="p107777311038"></a><code>virtual void reduce(idx_t n, const float *x, float *res) const = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>Dimensionality reduction interface. This function does not provide a concrete implementation.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1963814585141"><a name="p1963814585141"></a><a name="p1963814585141"></a><code>idx_t n</code>: Number of inputs for inference.</p>
<p id="p1633753171511"><a name="p1633753171511"></a><a name="p1633753171511"></a><code>const float *x</code>: Feature vectors for inference.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><code>float *res</code>: Feature-vector results from inference.</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul58419974316"></a><a name="ul58419974316"></a><ul id="ul58419974316"><li>The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9.</li><li>Pointer <code>x</code> must be non-null, and its length must be <code>dimIn * n</code>. Pointer <code>res</code> must be non-null, and its length must be <code>dimOut * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>

## `ReductionConfig`<a name="en-us_TOPIC_0000001456375264"></a>

| Member | Type | Description |
|--|--|--|
| dimIn | int | Input feature dimension, that is, the dimension before reduction. `PCAR` requires this parameter. |
| dimOut | int | Output feature dimension, that is, the dimension after reduction. `PCAR` requires this parameter. |
| eigenPower | float | Power of the singular values. `PCAR` requires this parameter. |
| randomRotation | bool | Whether to perform random rotation. `PCAR` requires this parameter. |
| deviceList | std::vector\<int> | Device-side resource configuration. `NN` requires this parameter. |
| model | const char * | Neural network dimensionality reduction model. `NN` requires this parameter. |
| modelSize | uint64_t | Model size. `NN` requires this parameter. |

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p580615115812"><a name="p580615115812"></a><a name="p580615115812"></a><code>inline ReductionConfig(int dimIn, int dimOut, float eigenPower, bool randomRotation);</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p196741716104810"><a name="p196741716104810"></a><a name="p196741716104810"></a>Constructor of <code>ReductionConfig</code>. Use this function when you use <code>PCAR</code> dimensionality reduction.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p493524721114"><a name="p493524721114"></a><a name="p493524721114"></a><code>int dimIn</code>: Input feature dimension, that is, the dimension before reduction. <code>PCAR</code> requires this parameter.</p>
<p id="p863214151491"><a name="p863214151491"></a><a name="p863214151491"></a><code>int dimOut</code>: Output feature dimension, that is, the dimension after reduction. <code>PCAR</code> requires this parameter.</p>
<p id="p19166202181110"><a name="p19166202181110"></a><a name="p19166202181110"></a><code>float eigenPower</code>: Power of the singular values. <code>PCAR</code> requires this parameter.</p>
<p id="p1731541105814"><a name="p1731541105814"></a><a name="p1731541105814"></a><code>bool randomRotation</code>: Whether to perform random rotation. <code>PCAR</code> requires this parameter.</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul163002012286"></a><a name="ul163002012286"></a><ul id="ul163002012286"><li>When you use different dimensionality reduction algorithms, configure the corresponding parameters and ensure that the dimension after reduction satisfies the dimension limit of the downstream index that uses the reduced data.</li><li>When you use <code>PCAR</code> dimensionality reduction, ensure that <code>dimOut</code> &gt; 0 and <code>dimIn</code> ≥ <code>dimOut</code>. The range of <code>eigenPower</code> is [-0.5, 0].</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table2034112619"></a>
<table><tbody><tr id="row140641961"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p19018411664"><a name="p19018411664"></a><a name="p19018411664"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p111981596192"><a name="p111981596192"></a><a name="p111981596192"></a><code>inline ReductionConfig(std::vector&lt;int&gt; deviceList, const char *model, uint64_t modelSize);</code></p>
</td>
</tr>
<tr id="row160141769"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p110441762"><a name="p110441762"></a><a name="p110441762"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p90154119617"><a name="p90154119617"></a><a name="p90154119617"></a>Constructor of <code>ReductionConfig</code>. Use this function when you use <code>NN</code> dimensionality reduction.</p>
</td>
</tr>
<tr id="row19015411615"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p17019411869"><a name="p17019411869"></a><a name="p17019411869"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p7166132141113"><a name="p7166132141113"></a><a name="p7166132141113"></a><code>std::vector&lt;int&gt; deviceList</code>: Device-side resource configuration.</p>
<p id="p1316615215113"><a name="p1316615215113"></a><a name="p1316615215113"></a><code>const char *model</code>: Neural network dimensionality reduction model.</p>
<p id="p981527104513"><a name="p981527104513"></a><a name="p981527104513"></a><code>uint64_t modelSize</code>: Model size.</p>
</td>
</tr>
<tr id="row8010412616"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1102411964"><a name="p1102411964"></a><a name="p1102411964"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p170841061"><a name="p170841061"></a><a name="p170841061"></a>None</p>
</td>
</tr>
<tr id="row2005417619"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p7012418619"><a name="p7012418619"></a><a name="p7012418619"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1909416612"><a name="p1909416612"></a><a name="p1909416612"></a>None</p>
</td>
</tr>
<tr id="row9011417617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p8010419616"><a name="p8010419616"></a><a name="p8010419616"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul29631955112419"></a><a name="ul29631955112419"></a><ul id="ul29631955112419"><li>The valid range of <code>deviceList</code> is (0, 32].</li><li>When you use different dimensionality reduction algorithms, configure the corresponding parameters and ensure that the dimension after reduction satisfies the dimension limit of the downstream index that uses the reduced data.</li><li><code>model</code> must be a valid memory pointer to a deep neural network dimensionality reduction model, and its size must be <code>modelSize</code>. The valid range of <code>modelSize</code> is (0, 128 MB]. Parameter mismatches may cause model instantiation or inference to fail. Invalid models may harm the system. Ensure that the model source is valid and effective.<a name="ul78321143192514"></a><a name="ul78321143192514"></a><ul id="ul78321143192514"><li><code>dimIn</code> ∈ {64, 128, 256, 384, 512, 768, 1024}.</li><li><code>dimOut</code> ∈ {32, 64, 96, 128, 256}.</li><li><code>batches</code> ∈ {1, 2, 4, 8, 16, 32, 64, 128}.</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## `~IReduction`<a name="en-us_TOPIC_0000001714244661"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58311930112818"><a name="p58311930112818"></a><a name="p58311930112818"></a><code>virtual ~IReduction() = default;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p118311130182814"><a name="p118311130182814"></a><a name="p118311130182814"></a>Destructor of <code>IReduction</code>. It destroys the <code>IReduction</code> object and releases resources.</p>
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

## `train`<a name="en-us_TOPIC_0000001506495753"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API Definition</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p102917819313"><a name="p102917819313"></a><a name="p102917819313"></a><code>virtual void train(idx_t n, const float *x) const = 0;</code></p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>Description</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>Abstract training interface. This function does not provide a concrete implementation.</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>Input</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><code>idx_t n</code>: Number of feature vectors in the training set.</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><code>const float *x</code>: Feature-vector data.</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>Output</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>None</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>Return Value</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a>None</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>Constraints</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul047418624016"></a><a name="ul047418624016"></a><ul id="ul047418624016"><li>The value of <code>n</code> must be in the range 0 &lt; <code>n</code> &lt; 1e9.</li><li>Pointer <code>x</code> must be non-null, and its length must be <code>dimIn * n</code>. Otherwise, out-of-bounds reads or writes may occur and cause the program to crash.</li></ul>
</td>
</tr>
</tbody>
</table>
