# IReduction<a name="ZH-CN_TOPIC_0000001456694992"></a>

## 功能介绍<a name="ZH-CN_TOPIC_0000001506615161"></a>

IReduction是特征检索组件中降维方法的统一接口，目前支持**PCAR**和**NN**两种降维算法。

## CreateReduction接口<a name="ZH-CN_TOPIC_0000001456695108"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p034518361750"><a name="p034518361750"></a><a name="p034518361750"></a>IReduction *CreateReduction(std::string typeName, const ReductionConfig &amp;config);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p966663212512"><a name="p966663212512"></a><a name="p966663212512"></a>创建具体的降维算法。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1220621175115"><a name="p1220621175115"></a><a name="p1220621175115"></a><strong id="b104261847125210"><a name="b104261847125210"></a><a name="b104261847125210"></a>std::string typeName</strong>：降维算法参数，可选{"NN", "PCAR"}。</p>
<p id="p1579483519305"><a name="p1579483519305"></a><a name="p1579483519305"></a><strong id="b119959353307"><a name="b119959353307"></a><a name="b119959353307"></a>ReductionConfig &amp;config</strong>：降维参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a><strong id="b193311933154019"><a name="b193311933154019"></a><a name="b193311933154019"></a>IReduction *CreateReduction</strong>：创建的具体的降维实例。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p1792162134913"><a name="p1792162134913"></a><a name="p1792162134913"></a>目前仅支持使用NN、PCAR两种降维参数，使用其他参数降维会抛异常。</p>
<p id="p18899829194617"><a name="p18899829194617"></a><a name="p18899829194617"></a>使用完毕该实例后请注意delete此指针，释放对应的空间。</p>
</td>
</tr>
</tbody>
</table>

## reduce接口<a name="ZH-CN_TOPIC_0000001456375280"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p107777311038"><a name="p107777311038"></a><a name="p107777311038"></a>virtual void reduce(idx_t n, const float *x, float *res) const = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1974334115476"><a name="p1974334115476"></a><a name="p1974334115476"></a>降维接口，本函数中不提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1963814585141"><a name="p1963814585141"></a><a name="p1963814585141"></a><strong id="b104261847125210"><a name="b104261847125210"></a><a name="b104261847125210"></a>idx_t n</strong>：待执行推理的输入数量。</p>
<p id="p1633753171511"><a name="p1633753171511"></a><a name="p1633753171511"></a><strong id="b1937815534524"><a name="b1937815534524"></a><a name="b1937815534524"></a>const float *x</strong>：待执行推理的特征向量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b29431145430"><a name="b29431145430"></a><a name="b29431145430"></a>float* res</strong>：执行推理得到的特征向量结果。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul58419974316"></a><a name="ul58419974316"></a><ul id="ul58419974316"><li>此处<span class="parmname" id="parmname1589434893110"><a name="parmname1589434893110"></a><a name="parmname1589434893110"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname1017013169439"><a name="parmname1017013169439"></a><a name="parmname1017013169439"></a>“x”</span>需要为非空指针，且长度应该为dimIn * <strong id="b1658485663114"><a name="b1658485663114"></a><a name="b1658485663114"></a>n</strong>，<span class="parmname" id="parmname10352171914316"><a name="parmname10352171914316"></a><a name="parmname10352171914316"></a>“res”</span>需要为非空指针，且长度应该为dimOut * <strong id="b952335293118"><a name="b952335293118"></a><a name="b952335293118"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>

## ReductionConfig接口<a name="ZH-CN_TOPIC_0000001456375264"></a>

|成员|类型|说明|
|--|--|--|
|dimIn|int|输入特征维度，即降维前的维度。PCAR需要配置此参数。|
|dimOut|int|输出特征维度，即降维后的维度。PCAR需要配置此参数。|
|eigenPower|float|奇异值的power数。PCAR需要配置此参数。|
|randomRotation|bool|是否进行随机旋转。PCAR需要配置此参数。|
|deviceList|std::vector\<int>|Device侧资源配置。NN需要配置此参数。|
|model|const char *|神经网络降维模型。NN需要配置此参数。|
|modelSize|uint64_t|模型的大小。NN需要配置此参数。|

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p580615115812"><a name="p580615115812"></a><a name="p580615115812"></a>inline ReductionConfig(int dimIn, int dimOut, float eigenPower, bool randomRotation);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p196741716104810"><a name="p196741716104810"></a><a name="p196741716104810"></a>ReductionConfig的构造函数，当用户使用“PCAR”降维时，使用该函数。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p493524721114"><a name="p493524721114"></a><a name="p493524721114"></a><strong id="b15987175822420"><a name="b15987175822420"></a><a name="b15987175822420"></a>int dimIn</strong>：输入特征维度，即降维前的维度，PCAR需要配置此参数。</p>
<p id="p863214151491"><a name="p863214151491"></a><a name="p863214151491"></a><strong id="b139707539247"><a name="b139707539247"></a><a name="b139707539247"></a>int dimOut</strong>：输出特征维度，即降维后的维度，PCAR需要配置此参数。</p>
<p id="p19166202181110"><a name="p19166202181110"></a><a name="p19166202181110"></a><strong id="b1328919276247"><a name="b1328919276247"></a><a name="b1328919276247"></a>float eigenPower</strong>：奇异值的power数，PCAR需要配置此参数。</p>
<p id="p1731541105814"><a name="p1731541105814"></a><a name="p1731541105814"></a><strong id="b108481935192313"><a name="b108481935192313"></a><a name="b108481935192313"></a>bool randomRotation</strong>：是否进行随机旋转，PCAR需要配置此参数。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul163002012286"></a><a name="ul163002012286"></a><ul id="ul163002012286"><li>使用不同的降维算法，需要配置对应的参数并且降维后的维度需要满足后续使用降维数据Index的维度限制。</li><li>使用PCAR降维时，需要保证dimOut&gt;0，dimIn ≥ dimOut。<strong id="b1147755119416"><a name="b1147755119416"></a><a name="b1147755119416"></a>eigenPower</strong>的范围为[-0.5, 0]。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table2034112619"></a>
<table><tbody><tr id="row140641961"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p19018411664"><a name="p19018411664"></a><a name="p19018411664"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p111981596192"><a name="p111981596192"></a><a name="p111981596192"></a>inline ReductionConfig(std::vector&lt;int&gt; deviceList, const char *model, uint64_t modelSize);</p>
</td>
</tr>
<tr id="row160141769"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p110441762"><a name="p110441762"></a><a name="p110441762"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p90154119617"><a name="p90154119617"></a><a name="p90154119617"></a>ReductionConfig的构造函数，当用户使用“NN”降维时，使用该函数。</p>
</td>
</tr>
<tr id="row19015411615"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p17019411869"><a name="p17019411869"></a><a name="p17019411869"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p7166132141113"><a name="p7166132141113"></a><a name="p7166132141113"></a><strong id="b153488499243"><a name="b153488499243"></a><a name="b153488499243"></a>std::vector&lt;int&gt; deviceList</strong>：Device侧资源配置。</p>
<p id="p1316615215113"><a name="p1316615215113"></a><a name="p1316615215113"></a><strong id="b18743194419245"><a name="b18743194419245"></a><a name="b18743194419245"></a>const char *model</strong>：神经网络降维模型。</p>
<p id="p981527104513"><a name="p981527104513"></a><a name="p981527104513"></a><strong id="b1475873814244"><a name="b1475873814244"></a><a name="b1475873814244"></a>uint64_t modelSize</strong>：模型的大小。</p>
</td>
</tr>
<tr id="row8010412616"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p1102411964"><a name="p1102411964"></a><a name="p1102411964"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p170841061"><a name="p170841061"></a><a name="p170841061"></a>无</p>
</td>
</tr>
<tr id="row2005417619"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p7012418619"><a name="p7012418619"></a><a name="p7012418619"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1909416612"><a name="p1909416612"></a><a name="p1909416612"></a>无</p>
</td>
</tr>
<tr id="row9011417617"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p8010419616"><a name="p8010419616"></a><a name="p8010419616"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul29631955112419"></a><a name="ul29631955112419"></a><ul id="ul29631955112419"><li>deviceList取值范围(0, 32]。</li><li>使用不同的降维算法，需要配置对应的参数并且降维后的维度需要满足后续使用降维数据Index的维度限制。</li><li><span class="parmname" id="parmname17928145516416"><a name="parmname17928145516416"></a><a name="parmname17928145516416"></a>“model”</span>需要为合法有效的深度神经网络推理模型的内存指针，大小为<span class="parmname" id="parmname13470125964114"><a name="parmname13470125964114"></a><a name="parmname13470125964114"></a>“modelSize”</span>，modelSize取值范围为(0, 128MB]，参数不匹配可能造成模型实例化或推理失败。非法的模型可能会对系统造成危害，请确保模型的来源合法有效。<a name="ul78321143192514"></a><a name="ul78321143192514"></a><ul id="ul78321143192514"><li>dimsIn ∈ {64, 128, 256, 384, 512, 768, 1024}。</li><li>dimsOut ∈ {32, 64, 96, 128, 256}。</li><li>batches ∈ {1, 2, 4, 8, 16, 32, 64, 128}</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

## \~IReduction接口<a name="ZH-CN_TOPIC_0000001714244661"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p58311930112818"><a name="p58311930112818"></a><a name="p58311930112818"></a>virtual ~IReduction() = default;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p118311130182814"><a name="p118311130182814"></a><a name="p118311130182814"></a>IReduction的析构函数，销毁IReduction对象，释放资源。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>

## train接口<a name="ZH-CN_TOPIC_0000001506495753"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p102917819313"><a name="p102917819313"></a><a name="p102917819313"></a>virtual void train(idx_t n, const float *x) const = 0;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p18122312578"><a name="p18122312578"></a><a name="p18122312578"></a>训练的抽象接口，本函数中不提供具体实现。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1464472124510"><a name="p1464472124510"></a><a name="p1464472124510"></a><strong id="b104261847125210"><a name="b104261847125210"></a><a name="b104261847125210"></a>idx_t n</strong>：训练集中特征向量的条数。</p>
<p id="p426592383"><a name="p426592383"></a><a name="p426592383"></a><strong id="b1937815534524"><a name="b1937815534524"></a><a name="b1937815534524"></a>const float *x</strong>：特征向量数据。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p973225082318"><a name="p973225082318"></a><a name="p973225082318"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1127672510562"><a name="p1127672510562"></a><a name="p1127672510562"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul047418624016"></a><a name="ul047418624016"></a><ul id="ul047418624016"><li>此处<span class="parmname" id="parmname255132043110"><a name="parmname255132043110"></a><a name="parmname255132043110"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname179498218312"><a name="parmname179498218312"></a><a name="parmname179498218312"></a>“x”</span>需要为非空指针，且长度应该为dimIn * <strong id="b191965249319"><a name="b191965249319"></a><a name="b191965249319"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>
