## 其他功能<a name="ZH-CN_TOPIC_0000001482684458"></a>

### IReduction<a name="ZH-CN_TOPIC_0000001456694992"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001506615161"></a>

IReduction是特征检索组件中降维方法的统一接口，目前支持**PCAR**和**NN**两种降维算法。


#### CreateReduction接口<a name="ZH-CN_TOPIC_0000001456695108"></a>

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
<p id="p1579483519305"><a name="p1579483519305"></a><a name="p1579483519305"></a><strong id="b119959353307"><a name="b119959353307"></a><a name="b119959353307"></a>Reduction &amp;config</strong>：降维参数。</p>
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


#### reduce接口<a name="ZH-CN_TOPIC_0000001456375280"></a>

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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1963814585141"><a name="p1963814585141"></a><a name="p1963814585141"></a><strong id="b104261847125210"><a name="b104261847125210"></a><a name="b104261847125210"></a>idx_t n</strong>：输入的待执行推理的输入数量。</p>
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


#### ReductionConfig接口<a name="ZH-CN_TOPIC_0000001456375264"></a>

|成员|类型|说明|
|--|--|--|
|dimIn|int|输入特征维度，即降维前的维度。PCAR需要配置此参数。|
|dimOut|int|输出特征维度，即降维后的维度。PCAR需要配置此参数。|
|eigenPower|float|奇异值的power数。PCAR需要配置此参数。|
|randomRotation|bool|是否进行随机旋转。PCAR需要配置此参数。|
|deviceList|std::vector<int>|Device侧资源配置。NN需要配置此参数。|
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p196741716104810"><a name="p196741716104810"></a><a name="p196741716104810"></a>ReductionConfig的默认构造函数，当用户使用“PCAR”降维时，使用该函数。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p90154119617"><a name="p90154119617"></a><a name="p90154119617"></a>ReductionConfig的默认构造函数，当用户使用“NN”降维时，使用该函数。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul29631955112419"></a><a name="ul29631955112419"></a><ul id="ul29631955112419"><li>deviceList取值范围(0, 32]。</li><li>使用不同的降维算法，需要配置对应的参数并且降维后的维度需要满足后续使用降维数据Index的维度限制。</li><li><span class="parmname" id="parmname17928145516416"><a name="parmname17928145516416"></a><a name="parmname17928145516416"></a>“model”</span>需要为合法有效的深度神经网络降维模型的内存指针，大小为<span class="parmname" id="parmname13470125964114"><a name="parmname13470125964114"></a><a name="parmname13470125964114"></a>“modelSize”</span>，modelSize取值范围为(0, 128MB]，参数不匹配可能造成模型实例化或推理失败。非法的模型可能会对系统造成危害，请确保模型的来源合法有效。<a name="ul78321143192514"></a><a name="ul78321143192514"></a><ul id="ul78321143192514"><li>dimsIn ∈ {64, 128, 256, 384, 512, 768, 1024}。</li><li>dimsOut ∈ {32, 64, 96, 128, 256,}。</li><li>batches ∈ {1, 2, 4, 8, 16, 32, 64, 128}</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>


#### \~IReduction接口<a name="ZH-CN_TOPIC_0000001714244661"></a>

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


#### train接口<a name="ZH-CN_TOPIC_0000001506495753"></a>

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



### AscendNNInference<a name="ZH-CN_TOPIC_0000001456375320"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001456535204"></a>

通过神经网络执行推理。


#### AscendNNInference接口<a name="ZH-CN_TOPIC_0000001456854780"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12834184020192"><a name="p12834184020192"></a><a name="p12834184020192"></a>AscendNNInference(std::vector&lt;int&gt; deviceList, const char* model, uint64_t modelSize);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendNNInference的构造函数，生成AscendNNInference，此时根据<span class="parmname" id="parmname8437181062119"><a name="parmname8437181062119"></a><a name="parmname8437181062119"></a>“deviceList”</span>中配置的值设置Device侧<span id="ph2098120577332"><a name="ph2098120577332"></a><a name="ph2098120577332"></a>昇腾AI处理器</span>资源以及模型路径等。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1869835152319"><a name="p1869835152319"></a><a name="p1869835152319"></a><strong id="b18268454414"><a name="b18268454414"></a><a name="b18268454414"></a>std::vector&lt;int&gt; deviceList</strong>：Device侧设备ID。</p>
<p id="p187869217128"><a name="p187869217128"></a><a name="p187869217128"></a><strong id="b9239165094116"><a name="b9239165094116"></a><a name="b9239165094116"></a>const char* model</strong>：深度神经网络降维模型。</p>
<p id="p11661833191215"><a name="p11661833191215"></a><a name="p11661833191215"></a><strong id="b165044526414"><a name="b165044526414"></a><a name="b165044526414"></a>uint64_t modelSize</strong>：深度神经网络降维模型的大小。</p>
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
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul246474615523"></a><a name="ul246474615523"></a><ul id="ul246474615523"><li>deviceList取值范围(0, 32]。</li><li><span class="parmname" id="parmname17928145516416"><a name="parmname17928145516416"></a><a name="parmname17928145516416"></a>“model”</span>需要为合法有效的深度神经网络降维模型的内存指针，大小为<span class="parmname" id="parmname13470125964114"><a name="parmname13470125964114"></a><a name="parmname13470125964114"></a>“modelSize”</span>，modelSize取值范围为(0, 128MB]，参数不匹配可能造成模型实例化或推理失败。非法的模型可能会对系统造成危害，请确保模型的来源合法有效。<a name="ul29631955112419"></a><a name="ul29631955112419"></a><ul id="ul29631955112419"><li>dimsIn ∈ {64, 128, 256, 384, 512, 768, 1024}。</li><li>dimsOut ∈ {32, 64, 96, 128, 256,}。</li><li>batches ∈ {1, 2, 4, 8, 16, 32, 64, 128}</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table1246213101873"></a>
<table><tbody><tr id="row1462121015717"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p84621510171"><a name="p84621510171"></a><a name="p84621510171"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p17980142452716"><a name="p17980142452716"></a><a name="p17980142452716"></a>AscendNNInference(const AscendNNInference&amp;) = delete;</p>
</td>
</tr>
<tr id="row164624102073"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p12462161014713"><a name="p12462161014713"></a><a name="p12462161014713"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p1462161010718"><a name="p1462161010718"></a><a name="p1462161010718"></a>声明此index拷贝构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row5462101013718"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p04623106716"><a name="p04623106716"></a><a name="p04623106716"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b05634141428"><a name="b05634141428"></a><a name="b05634141428"></a>const AscendNNInference&amp;</strong>：常量AscendNNInference。</p>
</td>
</tr>
<tr id="row12462610671"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p94628106715"><a name="p94628106715"></a><a name="p94628106715"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p946213108719"><a name="p946213108719"></a><a name="p946213108719"></a>无</p>
</td>
</tr>
<tr id="row194623101670"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p1546320101573"><a name="p1546320101573"></a><a name="p1546320101573"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p1346321016716"><a name="p1346321016716"></a><a name="p1346321016716"></a>无</p>
</td>
</tr>
<tr id="row104631810275"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p1046391017711"><a name="p1046391017711"></a><a name="p1046391017711"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p182559163813"><a name="p182559163813"></a><a name="p182559163813"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### \~AscendNNInference接口<a name="ZH-CN_TOPIC_0000001506495737"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a>~AscendNNInference();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>AscendNNInference的析构函数，销毁AscendNNInference对象，释放资源。</p>
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


#### getDimBatch接口<a name="ZH-CN_TOPIC_0000001506334797"></a>

<a name="zh-cn_topic_0000001287392566_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001287392566_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001287392566_p12559123810"><a name="zh-cn_topic_0000001287392566_p12559123810"></a><a name="zh-cn_topic_0000001287392566_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001287392566_p13466158111313"><a name="zh-cn_topic_0000001287392566_p13466158111313"></a><a name="zh-cn_topic_0000001287392566_p13466158111313"></a>int getDimBatch() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001287392566_p1212599383"><a name="zh-cn_topic_0000001287392566_p1212599383"></a><a name="zh-cn_topic_0000001287392566_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001287392566_p131714208358"><a name="zh-cn_topic_0000001287392566_p131714208358"></a><a name="zh-cn_topic_0000001287392566_p131714208358"></a>获取模型的单次推理的样本或查询向量的数量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001287392566_p112195910383"><a name="zh-cn_topic_0000001287392566_p112195910383"></a><a name="zh-cn_topic_0000001287392566_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001287392566_p1963814585141"><a name="zh-cn_topic_0000001287392566_p1963814585141"></a><a name="zh-cn_topic_0000001287392566_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001287392566_p17235973820"><a name="zh-cn_topic_0000001287392566_p17235973820"></a><a name="zh-cn_topic_0000001287392566_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001287392566_p8451184515218"><a name="zh-cn_topic_0000001287392566_p8451184515218"></a><a name="zh-cn_topic_0000001287392566_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001287392566_p182459113812"><a name="zh-cn_topic_0000001287392566_p182459113812"></a><a name="zh-cn_topic_0000001287392566_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001287392566_p132314362521"><a name="zh-cn_topic_0000001287392566_p132314362521"></a><a name="zh-cn_topic_0000001287392566_p132314362521"></a>模型单次推理的样本或查询向量的数量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287392566_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001287392566_p423590386"><a name="zh-cn_topic_0000001287392566_p423590386"></a><a name="zh-cn_topic_0000001287392566_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001287392566_p991611401713"><a name="zh-cn_topic_0000001287392566_p991611401713"></a><a name="zh-cn_topic_0000001287392566_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### getInputType接口<a name="ZH-CN_TOPIC_0000001456854776"></a>

<a name="zh-cn_topic_0000001340072289_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001340072289_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001340072289_p12559123810"><a name="zh-cn_topic_0000001340072289_p12559123810"></a><a name="zh-cn_topic_0000001340072289_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001340072289_p13466158111313"><a name="zh-cn_topic_0000001340072289_p13466158111313"></a><a name="zh-cn_topic_0000001340072289_p13466158111313"></a>int getInputType() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001340072289_p1212599383"><a name="zh-cn_topic_0000001340072289_p1212599383"></a><a name="zh-cn_topic_0000001340072289_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001340072289_p131714208358"><a name="zh-cn_topic_0000001340072289_p131714208358"></a><a name="zh-cn_topic_0000001340072289_p131714208358"></a>获取模型的输入数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001340072289_p112195910383"><a name="zh-cn_topic_0000001340072289_p112195910383"></a><a name="zh-cn_topic_0000001340072289_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001340072289_p1963814585141"><a name="zh-cn_topic_0000001340072289_p1963814585141"></a><a name="zh-cn_topic_0000001340072289_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001340072289_p17235973820"><a name="zh-cn_topic_0000001340072289_p17235973820"></a><a name="zh-cn_topic_0000001340072289_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001340072289_p8451184515218"><a name="zh-cn_topic_0000001340072289_p8451184515218"></a><a name="zh-cn_topic_0000001340072289_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001340072289_p182459113812"><a name="zh-cn_topic_0000001340072289_p182459113812"></a><a name="zh-cn_topic_0000001340072289_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001340072289_p132314362521"><a name="zh-cn_topic_0000001340072289_p132314362521"></a><a name="zh-cn_topic_0000001340072289_p132314362521"></a>模型的输入数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340072289_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001340072289_p423590386"><a name="zh-cn_topic_0000001340072289_p423590386"></a><a name="zh-cn_topic_0000001340072289_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001340072289_p991611401713"><a name="zh-cn_topic_0000001340072289_p991611401713"></a><a name="zh-cn_topic_0000001340072289_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### getOutputType接口<a name="ZH-CN_TOPIC_0000001456854868"></a>

<a name="zh-cn_topic_0000001340232437_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001340232437_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001340232437_p12559123810"><a name="zh-cn_topic_0000001340232437_p12559123810"></a><a name="zh-cn_topic_0000001340232437_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001340232437_p13466158111313"><a name="zh-cn_topic_0000001340232437_p13466158111313"></a><a name="zh-cn_topic_0000001340232437_p13466158111313"></a>int getOutputType() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001340232437_p1212599383"><a name="zh-cn_topic_0000001340232437_p1212599383"></a><a name="zh-cn_topic_0000001340232437_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001340232437_p131714208358"><a name="zh-cn_topic_0000001340232437_p131714208358"></a><a name="zh-cn_topic_0000001340232437_p131714208358"></a>获取模型的输出数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001340232437_p112195910383"><a name="zh-cn_topic_0000001340232437_p112195910383"></a><a name="zh-cn_topic_0000001340232437_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001340232437_p1963814585141"><a name="zh-cn_topic_0000001340232437_p1963814585141"></a><a name="zh-cn_topic_0000001340232437_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001340232437_p17235973820"><a name="zh-cn_topic_0000001340232437_p17235973820"></a><a name="zh-cn_topic_0000001340232437_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001340232437_p8451184515218"><a name="zh-cn_topic_0000001340232437_p8451184515218"></a><a name="zh-cn_topic_0000001340232437_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001340232437_p182459113812"><a name="zh-cn_topic_0000001340232437_p182459113812"></a><a name="zh-cn_topic_0000001340232437_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001340232437_p132314362521"><a name="zh-cn_topic_0000001340232437_p132314362521"></a><a name="zh-cn_topic_0000001340232437_p132314362521"></a>模型的输出数据类型。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340232437_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001340232437_p423590386"><a name="zh-cn_topic_0000001340232437_p423590386"></a><a name="zh-cn_topic_0000001340232437_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001340232437_p991611401713"><a name="zh-cn_topic_0000001340232437_p991611401713"></a><a name="zh-cn_topic_0000001340232437_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### getDimIn接口<a name="ZH-CN_TOPIC_0000001456535128"></a>

<a name="zh-cn_topic_0000001287712442_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001287712442_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001287712442_p12559123810"><a name="zh-cn_topic_0000001287712442_p12559123810"></a><a name="zh-cn_topic_0000001287712442_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001287712442_p13466158111313"><a name="zh-cn_topic_0000001287712442_p13466158111313"></a><a name="zh-cn_topic_0000001287712442_p13466158111313"></a>int getDimIn() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001287712442_p1212599383"><a name="zh-cn_topic_0000001287712442_p1212599383"></a><a name="zh-cn_topic_0000001287712442_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001287712442_p131714208358"><a name="zh-cn_topic_0000001287712442_p131714208358"></a><a name="zh-cn_topic_0000001287712442_p131714208358"></a>获取模型的输入数据维度。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001287712442_p112195910383"><a name="zh-cn_topic_0000001287712442_p112195910383"></a><a name="zh-cn_topic_0000001287712442_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001287712442_p1963814585141"><a name="zh-cn_topic_0000001287712442_p1963814585141"></a><a name="zh-cn_topic_0000001287712442_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001287712442_p17235973820"><a name="zh-cn_topic_0000001287712442_p17235973820"></a><a name="zh-cn_topic_0000001287712442_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001287712442_p8451184515218"><a name="zh-cn_topic_0000001287712442_p8451184515218"></a><a name="zh-cn_topic_0000001287712442_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001287712442_p182459113812"><a name="zh-cn_topic_0000001287712442_p182459113812"></a><a name="zh-cn_topic_0000001287712442_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001287712442_p132314362521"><a name="zh-cn_topic_0000001287712442_p132314362521"></a><a name="zh-cn_topic_0000001287712442_p132314362521"></a>输入数据维度。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287712442_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001287712442_p423590386"><a name="zh-cn_topic_0000001287712442_p423590386"></a><a name="zh-cn_topic_0000001287712442_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001287712442_p991611401713"><a name="zh-cn_topic_0000001287712442_p991611401713"></a><a name="zh-cn_topic_0000001287712442_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### getDimOut接口<a name="ZH-CN_TOPIC_0000001456695056"></a>

<a name="zh-cn_topic_0000001287552486_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001287552486_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001287552486_p12559123810"><a name="zh-cn_topic_0000001287552486_p12559123810"></a><a name="zh-cn_topic_0000001287552486_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001287552486_p13466158111313"><a name="zh-cn_topic_0000001287552486_p13466158111313"></a><a name="zh-cn_topic_0000001287552486_p13466158111313"></a>int getDimOut() const;</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001287552486_p1212599383"><a name="zh-cn_topic_0000001287552486_p1212599383"></a><a name="zh-cn_topic_0000001287552486_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001287552486_p131714208358"><a name="zh-cn_topic_0000001287552486_p131714208358"></a><a name="zh-cn_topic_0000001287552486_p131714208358"></a>获取模型的输出数据维度。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001287552486_p112195910383"><a name="zh-cn_topic_0000001287552486_p112195910383"></a><a name="zh-cn_topic_0000001287552486_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001287552486_p1963814585141"><a name="zh-cn_topic_0000001287552486_p1963814585141"></a><a name="zh-cn_topic_0000001287552486_p1963814585141"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001287552486_p17235973820"><a name="zh-cn_topic_0000001287552486_p17235973820"></a><a name="zh-cn_topic_0000001287552486_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001287552486_p8451184515218"><a name="zh-cn_topic_0000001287552486_p8451184515218"></a><a name="zh-cn_topic_0000001287552486_p8451184515218"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001287552486_p182459113812"><a name="zh-cn_topic_0000001287552486_p182459113812"></a><a name="zh-cn_topic_0000001287552486_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001287552486_p132314362521"><a name="zh-cn_topic_0000001287552486_p132314362521"></a><a name="zh-cn_topic_0000001287552486_p132314362521"></a>模型的输出数据维度。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001287552486_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001287552486_p423590386"><a name="zh-cn_topic_0000001287552486_p423590386"></a><a name="zh-cn_topic_0000001287552486_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001287552486_p991611401713"><a name="zh-cn_topic_0000001287552486_p991611401713"></a><a name="zh-cn_topic_0000001287552486_p991611401713"></a>无</p>
</td>
</tr>
</tbody>
</table>


#### infer接口<a name="ZH-CN_TOPIC_0000001506495709"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.1.1 "><p id="p13466158111313"><a name="p13466158111313"></a><a name="p13466158111313"></a>void infer(size_t n, const char* inputData, char* outputData) const;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>根据网络模型执行推理。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.3.1 "><p id="p1963814585141"><a name="p1963814585141"></a><a name="p1963814585141"></a><strong id="b33632034317"><a name="b33632034317"></a><a name="b33632034317"></a>size_t n</strong>：输入的待执行推理的输入数量。</p>
<p id="p1633753171511"><a name="p1633753171511"></a><a name="p1633753171511"></a><strong id="b12723522435"><a name="b12723522435"></a><a name="b12723522435"></a>const char* inputData</strong>：待执行推理的特征向量。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.4.1 "><p id="p8451184515218"><a name="p8451184515218"></a><a name="p8451184515218"></a><strong id="b29431145430"><a name="b29431145430"></a><a name="b29431145430"></a>char* outputData</strong>：执行推理得到的特征向量结果。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.5.1 "><p id="p132314362521"><a name="p132314362521"></a><a name="p132314362521"></a>无</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="19.96%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="80.04%" headers="mcps1.1.3.6.1 "><a name="ul58419974316"></a><a name="ul58419974316"></a><ul id="ul58419974316"><li>此处<span class="parmname" id="parmname1589434893110"><a name="parmname1589434893110"></a><a name="parmname1589434893110"></a>“n”</span>的取值范围：0 &lt; n &lt; 1e9。</li><li>此处指针<span class="parmname" id="parmname1017013169439"><a name="parmname1017013169439"></a><a name="parmname1017013169439"></a>“inputData”</span>需要为非空指针，且长度应该为dimIn * <strong id="b1658485663114"><a name="b1658485663114"></a><a name="b1658485663114"></a>n</strong>，<span class="parmname" id="parmname10352171914316"><a name="parmname10352171914316"></a><a name="parmname10352171914316"></a>“outputData”</span>需要为非空指针，且长度应该为dimOut * <strong id="b952335293118"><a name="b952335293118"></a><a name="b952335293118"></a>n</strong>，否则可能出现越界读写错误并引起程序崩溃。</li></ul>
</td>
</tr>
</tbody>
</table>


#### operator = 接口<a name="ZH-CN_TOPIC_0000001456535156"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p17980142452716"><a name="p17980142452716"></a><a name="p17980142452716"></a>AscendNNInference&amp; operator=(const AscendNNInference&amp;) = delete;</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>声明此Index赋值构造函数为空，即不可拷贝类型。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p867213174418"><a name="p867213174418"></a><a name="p867213174418"></a><strong id="b19703385424"><a name="b19703385424"></a><a name="b19703385424"></a>const AscendNNInference&amp;</strong>：常量AscendNNInference。</p>
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



### AscendClonerOptions<a name="ZH-CN_TOPIC_0000001456854804"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001456535196"></a>

AscendCloner接口的配置参数。

**成员介绍<a name="section1372191465013"></a>**

|成员|类型|说明|
|--|--|--|
|reserveVecs|long|当前无效，预留内存的特征数。|
|verbose|bool|是否打印拷贝日志。|
|resourceSize|int64_t|资源池大小。|
|slim|bool|AscendIndexSQConfig成员变量，是否动态增加内存。默认为false。|
|filterable|bool|AscendIndexSQConfig成员变量，是否按照id进行过滤。默认为false。|
|indexMode|uint32_t|Index int8检索模式，默认值为0 （DEFAULT_MODE）。|
|blockSize|uint32_t|配置Device侧的blockSize，默认值“BLOCK_SIZE”为16384 * 16 = 262144。|



#### AscendClonerOptions接口<a name="ZH-CN_TOPIC_0000001506414885"></a>

<a name="zh-cn_topic_0000001340833369_table7235918388"></a>
<table><tbody><tr id="zh-cn_topic_0000001340833369_row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001340833369_p12559123810"><a name="zh-cn_topic_0000001340833369_p12559123810"></a><a name="zh-cn_topic_0000001340833369_p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001340833369_p1212917467412"><a name="zh-cn_topic_0000001340833369_p1212917467412"></a><a name="zh-cn_topic_0000001340833369_p1212917467412"></a>AscendClonerOptions()</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001340833369_p1212599383"><a name="zh-cn_topic_0000001340833369_p1212599383"></a><a name="zh-cn_topic_0000001340833369_p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001340833369_p131714208358"><a name="zh-cn_topic_0000001340833369_p131714208358"></a><a name="zh-cn_topic_0000001340833369_p131714208358"></a>AscendClonerOptions的构造函数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001340833369_p112195910383"><a name="zh-cn_topic_0000001340833369_p112195910383"></a><a name="zh-cn_topic_0000001340833369_p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001340833369_p1745111429517"><a name="zh-cn_topic_0000001340833369_p1745111429517"></a><a name="zh-cn_topic_0000001340833369_p1745111429517"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="zh-cn_topic_0000001340833369_p17235973820"><a name="zh-cn_topic_0000001340833369_p17235973820"></a><a name="zh-cn_topic_0000001340833369_p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="zh-cn_topic_0000001340833369_p6295973819"><a name="zh-cn_topic_0000001340833369_p6295973819"></a><a name="zh-cn_topic_0000001340833369_p6295973819"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="zh-cn_topic_0000001340833369_p182459113812"><a name="zh-cn_topic_0000001340833369_p182459113812"></a><a name="zh-cn_topic_0000001340833369_p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="zh-cn_topic_0000001340833369_p912917864516"><a name="zh-cn_topic_0000001340833369_p912917864516"></a><a name="zh-cn_topic_0000001340833369_p912917864516"></a>无</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001340833369_row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="zh-cn_topic_0000001340833369_p423590386"><a name="zh-cn_topic_0000001340833369_p423590386"></a><a name="zh-cn_topic_0000001340833369_p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="zh-cn_topic_0000001340833369_p3947640162619"><a name="zh-cn_topic_0000001340833369_p3947640162619"></a><a name="zh-cn_topic_0000001340833369_p3947640162619"></a>无</p>
</td>
</tr>
</tbody>
</table>



### AscendCloner<a name="ZH-CN_TOPIC_0000001506334577"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000001456375412"></a>

Index SDK提供了将NPU上的检索Index资源拷贝到CPU侧Faiss的操作，拷贝过程发生在内存中，原始NPU的Index上加载的数据会被拷贝到CPU侧的内存中，方便用户在CPU上使用相同的底库执行检索。

>![](public_sys-resources/icon-note.gif) **说明：** 
>部分版本的Faiss中提供了将内存中的Index落盘（内存中的数据保存到本地硬盘）的方法，用户在基于Index SDK和Faiss处理某些敏感数据时需要特别注意提供对应的权限控制和加密保护。


#### index\_ascend\_to\_cpu接口<a name="ZH-CN_TOPIC_0000001506334821"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p01708458179"><a name="p01708458179"></a><a name="p01708458179"></a>faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>根据Ascend上的检索index资源，拷贝生成一个CPU上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><strong id="b1522416109533"><a name="b1522416109533"></a><a name="b1522416109533"></a>const faiss::Index *ascend_index</strong>：Ascend上的Index资源。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>生成一个CPU上的检索Index。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p3947640162619"><a name="p3947640162619"></a><a name="p3947640162619"></a>使用完毕该<span class="parmname" id="parmname18261913175312"><a name="parmname18261913175312"></a><a name="parmname18261913175312"></a>“index”</span>后请注意delete掉此指针，释放对应的空间。</p>
</td>
</tr>
</tbody>
</table>


#### index\_cpu\_to\_ascend接口<a name="ZH-CN_TOPIC_0000001456695032"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p1915092217197"><a name="p1915092217197"></a><a name="p1915092217197"></a>faiss::Index *index_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>根据CPU上的检索Index资源，拷贝生成一个Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><strong id="b1464710223531"><a name="b1464710223531"></a><a name="b1464710223531"></a>std::initializer_list&lt;int&gt; devices</strong>：NPU上待配置的设备ID。</p>
<p id="p142883052016"><a name="p142883052016"></a><a name="p142883052016"></a><strong id="b18287724145316"><a name="b18287724145316"></a><a name="b18287724145316"></a>const faiss::Index *index</strong>：CPU上的检索Index资源。</p>
<p id="p15735115410199"><a name="p15735115410199"></a><a name="p15735115410199"></a><strong id="b883722705317"><a name="b883722705317"></a><a name="b883722705317"></a>const AscendClonerOptions *options = nullptr</strong>：待配置的AscendClonerOptions资源。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>生成一个Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul37673419536"></a><a name="ul37673419536"></a><ul id="ul37673419536"><li>使用完毕该<span class="parmname" id="parmname1876546145310"><a name="parmname1876546145310"></a><a name="parmname1876546145310"></a>“index”</span>后请注意delete掉此指针，释放对应的空间。</li><li><span class="parmname" id="parmname11908183575319"><a name="parmname11908183575319"></a><a name="parmname11908183575319"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname1260113705320"><a name="parmname1260113705320"></a><a name="parmname1260113705320"></a>“index”</span>需要为合法有效的CPU Index指针。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table22143401019"></a>
<table><tbody><tr id="row122113471017"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p0263414104"><a name="p0263414104"></a><a name="p0263414104"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p12818145232014"><a name="p12818145232014"></a><a name="p12818145232014"></a>faiss::Index *index_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</p>
</td>
</tr>
<tr id="row1629341102"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212340103"><a name="p1212340103"></a><a name="p1212340103"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p82153419105"><a name="p82153419105"></a><a name="p82153419105"></a>根据CPU上的检索Index资源，拷贝生成一个Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row10211342101"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p182234111012"><a name="p182234111012"></a><a name="p182234111012"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p122133411107"><a name="p122133411107"></a><a name="p122133411107"></a><strong id="b1675612491543"><a name="b1675612491543"></a><a name="b1675612491543"></a>std::vector&lt;int&gt; devices</strong>：NPU上待配置的设备ID。</p>
<p id="p02113412104"><a name="p02113412104"></a><a name="p02113412104"></a><strong id="b1646135211547"><a name="b1646135211547"></a><a name="b1646135211547"></a>const faiss::Index *index</strong>：CPU上的检索Index资源。</p>
<p id="p521734121010"><a name="p521734121010"></a><a name="p521734121010"></a><strong id="b8416205519544"><a name="b8416205519544"></a><a name="b8416205519544"></a>const AscendClonerOptions *options = nullptr</strong>：待配置的AscendClonerOptions资源。</p>
</td>
</tr>
<tr id="row20253411010"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p12734121017"><a name="p12734121017"></a><a name="p12734121017"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p22203431014"><a name="p22203431014"></a><a name="p22203431014"></a>无</p>
</td>
</tr>
<tr id="row15210342102"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p12283431010"><a name="p12283431010"></a><a name="p12283431010"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p132173418105"><a name="p132173418105"></a><a name="p132173418105"></a>生成一个Ascend上的检索Index。</p>
</td>
</tr>
<tr id="row112143411101"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5210340101"><a name="p5210340101"></a><a name="p5210340101"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1030011345514"></a><a name="ul1030011345514"></a><ul id="ul1030011345514"><li>使用完毕该<span class="parmname" id="parmname6242191414559"><a name="parmname6242191414559"></a><a name="parmname6242191414559"></a>“index”</span>后请注意delete掉此指针，释放对应的空间。</li><li><span class="parmname" id="parmname41511577559"><a name="parmname41511577559"></a><a name="parmname41511577559"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname1362238115514"><a name="parmname1362238115514"></a><a name="parmname1362238115514"></a>“index”</span>需要为合法有效的CPU Index指针。</li></ul>
</td>
</tr>
</tbody>
</table>


#### index\_int8\_ascend\_to\_cpu接口<a name="ZH-CN_TOPIC_0000001506414761"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p01708458179"><a name="p01708458179"></a><a name="p01708458179"></a>faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>根据Ascend上的INT8的检索Index资源，拷贝生成一个CPU上的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><strong id="b6155153813552"><a name="b6155153813552"></a><a name="b6155153813552"></a>const AscendIndexInt8 *index</strong>：Ascend上的Index资源。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>生成一个CPU上的检索Index。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20237164255510"></a><a name="ul20237164255510"></a><ul id="ul20237164255510"><li>使用完毕该<span class="parmname" id="parmname1866724645517"><a name="parmname1866724645517"></a><a name="parmname1866724645517"></a>“index”</span>后请注意delete此指针，释放对应的空间。</li><li><span class="parmname" id="parmname1298514443554"><a name="parmname1298514443554"></a><a name="parmname1298514443554"></a>“index”</span>需要为合法有效的AscendIndexInt8指针。</li></ul>
</td>
</tr>
</tbody>
</table>


#### index\_int8\_cpu\_to\_ascend接口<a name="ZH-CN_TOPIC_0000001456375248"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p19640556122110"><a name="p19640556122110"></a><a name="p19640556122110"></a>AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p894441552217"><a name="p894441552217"></a><a name="p894441552217"></a>根据CPU上的检索Index资源，拷贝生成一个Ascend上的INT8的检索Index。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p21724117137"><a name="p21724117137"></a><a name="p21724117137"></a><strong id="b3395198135618"><a name="b3395198135618"></a><a name="b3395198135618"></a>std::initializer_list&lt;int&gt; devices</strong>：NPU上待配置的设备ID。</p>
<p id="p142883052016"><a name="p142883052016"></a><a name="p142883052016"></a><strong id="b183851411105619"><a name="b183851411105619"></a><a name="b183851411105619"></a>const faiss::Index *index</strong>：CPU上的检索Index资源。</p>
<p id="p15735115410199"><a name="p15735115410199"></a><a name="p15735115410199"></a><strong id="b851411319563"><a name="b851411319563"></a><a name="b851411319563"></a>const AscendClonerOptions *options = nullptr</strong>：待配置的AscendClonerOptions资源。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p1246923411510"><a name="p1246923411510"></a><a name="p1246923411510"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>生成一个Ascend上的INT8的检索Index。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul142782285619"></a><a name="ul142782285619"></a><ul id="ul142782285619"><li>使用完毕该<span class="parmname" id="parmname11343264375"><a name="parmname11343264375"></a><a name="parmname11343264375"></a>“index”</span>后请注意delete此指针，释放对应的空间。</li><li><span class="parmname" id="parmname454423155617"><a name="parmname454423155617"></a><a name="parmname454423155617"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname12364173425615"><a name="parmname12364173425615"></a><a name="parmname12364173425615"></a>“index”</span>需要为合法有效的CPU Index指针。</li></ul>
</td>
</tr>
</tbody>
</table>

<a name="table161071151116"></a>
<table><tbody><tr id="row0610181121116"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p166107115116"><a name="p166107115116"></a><a name="p166107115116"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p561010115113"><a name="p561010115113"></a><a name="p561010115113"></a>AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector&lt;int&gt; devices, const faiss::Index *index, const AscendClonerOptions *options = nullptr);</p>
</td>
</tr>
<tr id="row1161011116113"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p461011116112"><a name="p461011116112"></a><a name="p461011116112"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p6610181101117"><a name="p6610181101117"></a><a name="p6610181101117"></a>根据CPU上的检索Index资源，拷贝生成一个Ascend上的INT8的检索Index。</p>
</td>
</tr>
<tr id="row166109141110"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p14610131191111"><a name="p14610131191111"></a><a name="p14610131191111"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p36107118118"><a name="p36107118118"></a><a name="p36107118118"></a><strong id="b411385010568"><a name="b411385010568"></a><a name="b411385010568"></a>std::vector&lt;int&gt; devices</strong>：NPU上待配置的设备ID。</p>
<p id="p1561010110117"><a name="p1561010110117"></a><a name="p1561010110117"></a><strong id="b7139524560"><a name="b7139524560"></a><a name="b7139524560"></a>const faiss::Index *index</strong>：CPU上的检索Index资源。</p>
<p id="p561031161112"><a name="p561031161112"></a><a name="p561031161112"></a><strong id="b0814145395615"><a name="b0814145395615"></a><a name="b0814145395615"></a>const AscendClonerOptions *options = nullptr</strong>：待配置的AscendClonerOptions资源。</p>
</td>
</tr>
<tr id="row1461012111111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p36107113117"><a name="p36107113117"></a><a name="p36107113117"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p161071131117"><a name="p161071131117"></a><a name="p161071131117"></a>无</p>
</td>
</tr>
<tr id="row14611161171112"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p106113111111"><a name="p106113111111"></a><a name="p106113111111"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p16111161110"><a name="p16111161110"></a><a name="p16111161110"></a>生成一个Ascend上的INT8的检索Index。</p>
</td>
</tr>
<tr id="row146118119111"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p5611101151114"><a name="p5611101151114"></a><a name="p5611101151114"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul20972759205619"></a><a name="ul20972759205619"></a><ul id="ul20972759205619"><li>使用完毕该<span class="parmname" id="parmname187433510576"><a name="parmname187433510576"></a><a name="parmname187433510576"></a>“index”</span>后请注意delete此指针，释放对应的空间。</li><li><span class="parmname" id="parmname152210225710"><a name="parmname152210225710"></a><a name="parmname152210225710"></a>“devices”</span>需要为合法有效不重复的设备ID，最大数量为64。</li><li><span class="parmname" id="parmname13280475716"><a name="parmname13280475716"></a><a name="parmname13280475716"></a>“index”</span>需要为合法有效的CPU Index指针。</li></ul>
</td>
</tr>
</tbody>
</table>



### DiskPQ<a name="ZH-CN_TOPIC_0000002382802364"></a>

#### 功能介绍<a name="ZH-CN_TOPIC_0000002382647580"></a>

Index SDK提供PQ（Product Quantization）量化的训练和检索功能。PQ接口不支持多线程并发调用，因此在多线程的场景中需要用户在使用前加锁，否则可能导致功能异常。


#### DiskPQParams接口<a name="ZH-CN_TOPIC_0000002382807444"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p784562919563"><a name="p784562919563"></a><a name="p784562919563"></a>DiskPQParams {</p>
<p id="p584518296562"><a name="p584518296562"></a><a name="p584518296562"></a>int pqChunks = 512;</p>
<p id="p17845192912562"><a name="p17845192912562"></a><a name="p17845192912562"></a>int funcType = 1;</p>
<p id="p1884502910567"><a name="p1884502910567"></a><a name="p1884502910567"></a>int dim = 1;</p>
<p id="p8845122945614"><a name="p8845122945614"></a><a name="p8845122945614"></a>char *pqTable = nullptr;</p>
<p id="p13845202914564"><a name="p13845202914564"></a><a name="p13845202914564"></a>uint32_t *offsets = nullptr;</p>
<p id="p28451029185617"><a name="p28451029185617"></a><a name="p28451029185617"></a>char *tablesTransposed = nullptr;</p>
<p id="p12845102915560"><a name="p12845102915560"></a><a name="p12845102915560"></a>char *centroids = nullptr;</p>
<p id="p584532905613"><a name="p584532905613"></a><a name="p584532905613"></a>}</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>PQ量化结构体。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>参数值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><strong id="b3354181515167"><a name="b3354181515167"></a><a name="b3354181515167"></a>int pqChunks：</strong>表示将原始向量维度dim切分为pqChunks块。</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><strong id="b13953183012163"><a name="b13953183012163"></a><a name="b13953183012163"></a>int funcType：</strong>表示进行PQ查表距离计算时使用的计算标准。</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><strong id="b229253513169"><a name="b229253513169"></a><a name="b229253513169"></a>int dim：</strong>表示原始数据维度。</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><strong id="b1712474051618"><a name="b1712474051618"></a><a name="b1712474051618"></a>char *pqTable：</strong>表示存储码本数据的指针。默认值为nullptr。</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><strong id="b76015463165"><a name="b76015463165"></a><a name="b76015463165"></a>uint32_t *offsets：</strong>表示存储每个chunk在原始维度上起始和截止的维度。默认值为nullptr。</p>
<p id="p16645243598"><a name="p16645243598"></a><a name="p16645243598"></a><strong id="b632142913336"><a name="b632142913336"></a><a name="b632142913336"></a>char *tablesTransposed：</strong>表示存储码本数据的转置形态指针。默认值为nullptr。</p>
<p id="p864518435917"><a name="p864518435917"></a><a name="p864518435917"></a><strong id="b172846021716"><a name="b172846021716"></a><a name="b172846021716"></a>char *centroids：</strong>表示存储每个维度的平均值，用于对数据进行中心化处理。默认值为nullptr。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1368219905119"></a><a name="ul1368219905119"></a><ul id="ul1368219905119"><li>1 &lt;= pqChunks &lt;= dim。使用较小pqChunks将使用更少内存，但会带来相应的精度损失。一般情况下，推荐使用pqChunks为dim / 8或者dim / 16（均向上取整）。默认值为512。</li><li>funcType取值范围为1~3。1表示使用L2距离；2表示使用IP距离；3表示使用cosine距离。默认值为1。</li><li>1 &lt;= dim &lt;= 2000。默认值为1。</li><li>pqTable目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。</li><li>tablesTransposed目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。</li></ul>
</td>
</tr>
</tbody>
</table>


#### VectorArrayData接口<a name="ZH-CN_TOPIC_0000002416326913"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p455275425615"><a name="p455275425615"></a><a name="p455275425615"></a>VectorArrayData {</p>
<p id="p055225485618"><a name="p055225485618"></a><a name="p055225485618"></a>int length;</p>
<p id="p20552454105617"><a name="p20552454105617"></a><a name="p20552454105617"></a>int maxlen;</p>
<p id="p1355211547565"><a name="p1355211547565"></a><a name="p1355211547565"></a>int dim;</p>
<p id="p15552155445611"><a name="p15552155445611"></a><a name="p15552155445611"></a>size_t itemsize;</p>
<p id="p655255413561"><a name="p655255413561"></a><a name="p655255413561"></a>char *items;</p>
<p id="p75521654195610"><a name="p75521654195610"></a><a name="p75521654195610"></a>}</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p139815474417"><a name="p139815474417"></a><a name="p139815474417"></a>数据封装结构体。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1639825419446"><a name="p1639825419446"></a><a name="p1639825419446"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p203971254164419"><a name="p203971254164419"></a><a name="p203971254164419"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>参数值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p11645124317919"><a name="p11645124317919"></a><a name="p11645124317919"></a><strong id="b168113245547"><a name="b168113245547"></a><a name="b168113245547"></a>int length：</strong>表示结构体中存储的向量条数。</p>
<p id="p1264516437919"><a name="p1264516437919"></a><a name="p1264516437919"></a><strong id="b13953183012163"><a name="b13953183012163"></a><a name="b13953183012163"></a>int maxlen：</strong>表示结构体中存储的最大向量条数。</p>
<p id="p564519434918"><a name="p564519434918"></a><a name="p564519434918"></a><strong id="b229253513169"><a name="b229253513169"></a><a name="b229253513169"></a>int dim：</strong>表示结构体中存储的向量维度。</p>
<p id="p164518432910"><a name="p164518432910"></a><a name="p164518432910"></a><strong id="b1712474051618"><a name="b1712474051618"></a><a name="b1712474051618"></a>size_t itemsize：</strong>保留字段，用户可以选择不设置。</p>
<p id="p164510431912"><a name="p164510431912"></a><a name="p164510431912"></a><strong id="b346491725412"><a name="b346491725412"></a><a name="b346491725412"></a>char *items：</strong>表示存储VectorArrayData中数据的指针。默认值为nullptr。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>参数约束</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1368219905119"></a><a name="ul1368219905119"></a><ul id="ul1368219905119"><li>1 &lt;= length &lt;= 100000000。</li><li>maxlen是OpenGauss侧保留字段，非OpenGauss用户设置该值等于length值即可。</li><li>1 &lt;= dim &lt;= 2000。</li><li>对于不同接口，用户需要确保items指向不同大小的数据。</li></ul>
</td>
</tr>
</tbody>
</table>


#### ComputePQTable接口<a name="ZH-CN_TOPIC_0000002416446741"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p122214507454"><a name="p122214507454"></a><a name="p122214507454"></a>int ComputePQTable(VectorArrayData *sample, DiskPQParams *params);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>使用sample中存储的采样底库数据计算PQ码本，并将码本相关的数据存储在参数params中的对应参数里。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b1110246113812"><a name="b1110246113812"></a><a name="b1110246113812"></a>VectorArrayData *sample：</strong>指向填充好采样底库数据的VectorArrayData实例的指针。不能为空指针。</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><strong id="b5117650173817"><a name="b5117650173817"></a><a name="b5117650173817"></a>DiskPQParams *params：</strong>指向仅包含PQ参数，未填充训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>sample数据填充要求如下：<p id="p14941713111016"><a name="p14941713111016"></a><a name="p14941713111016"></a>items指向的数据大小为(8 + dim) * length * sizeof(float)字节，即每条向量前有8字节的metadata。非OpenGauss用户使用时，需在每条向量数据添加8字节的任意数据。</p>
</li><li>params成员变量填充要求如下：<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>dim除满足上述的范围限制要求之外，还需确保与sample中对应的dim字段保持一致。</li><li>pqTable必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * 256 （256为每个chunk内的聚类数）* sizeof(float)字节。</li><li>offsets必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于(pqChunks + 1) * sizeof(uint32_t)字节。</li><li>tablesTransposed必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * 256 * sizeof(float)字节。</li><li>centroids必须为nullptr，在动态库内部将使用new []关键字进行内存申请，需要使用者在外部对申请的内存进行释放（使用delete []）。内部申请的内存大小确保等于dim * sizeof(float)字节。</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>


#### ComputeVectorPQCode接口<a name="ZH-CN_TOPIC_0000002382647584"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a>int ComputeVectorPQCode(VectorArrayData *baseData, const DiskPQParams *params, uint8_t *pqCode);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>使用填充好PQ数据的params，对baseData中的底库数据进行量化，并将量化数据写入pqCode指向的缓存区中。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b89499121237"><a name="b89499121237"></a><a name="b89499121237"></a>VectorArrayData *baseData：</strong>指向填充好底库数据的VectorArrayData实例的指针。不能为空指针。用户可以根据自身内存的限制，在外层决定baseData中底库数据的大小。</p>
<p id="p8607152012439"><a name="p8607152012439"></a><a name="p8607152012439"></a><strong id="b617081772316"><a name="b617081772316"></a><a name="b617081772316"></a>const DiskPQParams *params：</strong>指向填充好PQ参数和训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b293063113232"><a name="b293063113232"></a><a name="b293063113232"></a>uint8_t *pqCode：</strong>接收返回的压缩好的底库向量的指针。不能为空指针。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>baseData数据填充要求如下：<p id="p71308396175"><a name="p71308396175"></a><a name="p71308396175"></a>items指向的数据大小为length * dim * sizeof(float)字节。注意此处与ComputePQTable接口不同， 无需在每条数据前填充代替metadata的数据。</p>
</li><li>params成员变量填充要求如下：<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>dim除满足上述的范围限制要求之外，还需确保与sample中对应的dim字段保持一致。</li><li>pqTable必须指向内存大小为dim * 256 * sizeof(float)字节数的码本数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li><li>offsets必须指向内存大小为(pqChunks + 1) * sizeof(uint32_t)字节数的offsets数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li><li>对tablesTransposed填充值无要求。</li><li>centroids必须指向内存大小为dim * sizeof(float)字节数的centroids数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li></ul>
</li><li>用户需保证pqCode指向的空间大小至少有length * pqChunks字节数。其中，length为VectorArrayData参数；pqChunks为DiskPQParams参数。</li></ul>
</td>
</tr>
</tbody>
</table>


#### GetPQDistanceTable接口<a name="ZH-CN_TOPIC_0000002382807448"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a>int GetPQDistanceTable(char *vec, const DiskPQParams *params, float *pqDistanceTable);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.2.1 "><p id="p2019105094517"><a name="p2019105094517"></a><a name="p2019105094517"></a>使用填充好PQ数据的params，对vec指向的query数据进行ADC PQ距离计算，并将PQ距离表写入pqDistanceTable指向的缓存区中。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.3.1 "><p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b510342493616"><a name="b510342493616"></a><a name="b510342493616"></a>char *vec：</strong>指向待计算的query数据的指针。</p>
<p id="p86231040131614"><a name="p86231040131614"></a><a name="p86231040131614"></a><strong id="b23454284363"><a name="b23454284363"></a><a name="b23454284363"></a>const DiskPQParams *params：</strong>指向填充好PQ参数和训练好的PQ数据的DiskPQParams实例的指针。不能为空指针。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b1548973317364"><a name="b1548973317364"></a><a name="b1548973317364"></a>float *pqDistanceTable：</strong>接收返回的query与每个chunk内每个centroid距离的指针。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.03%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.97%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>用户需保证vec指向的空间大小至少有dim * sizeof(float)字节数。目前仅支持float数据类型，即OpenGauss数据类型中的Vector数据类型。</li><li>params成员变量填充要求如下：<a name="ul156691381133"></a><a name="ul156691381133"></a><ul id="ul156691381133"><li>pqTable指向值无要求。</li><li>offsets必须指向内存大小为(pqChunks + 1) * sizeof(uint32_t)字节数的offsets数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li><li>tablesTransposed必须指向内存大小为dim * 256 * sizeof(float)字节数的码本数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li><li>centroids必须指向内存大小为dim * sizeof(float)字节数的centroids数据。用户需要保证指向的内存大小符合，否则有段错误风险。</li></ul>
</li><li>用户需保证pqDistanceTable指向的空间大小至少有pqChunks * 256 * sizeof(float)字节数。</li></ul>
</td>
</tr>
</tbody>
</table>


#### GetPQDistance接口<a name="ZH-CN_TOPIC_0000002416326917"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p953585410711"><a name="p953585410711"></a><a name="p953585410711"></a>int GetPQDistance(const uint8_t *basecode, const DiskPQParams *params, const float *pqDistanceTable, float &amp;pqDistance);</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p169090129309"><a name="p169090129309"></a><a name="p169090129309"></a>使用basecode指向的底库向量对应的压缩码字数据和GetPQDistanceTable接口中获取的pqDistanceTable，计算query与该底库向量的PQ距离。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p1533165363012"><a name="p1533165363012"></a><a name="p1533165363012"></a><strong id="b10761103265015"><a name="b10761103265015"></a><a name="b10761103265015"></a>const uint8_t *basecode：</strong>指向一个底库向量对应的压缩码字数据的指针。</p>
<p id="p12607112054318"><a name="p12607112054318"></a><a name="p12607112054318"></a><strong id="b1811010371509"><a name="b1811010371509"></a><a name="b1811010371509"></a>const DiskPQParams *params：</strong>指向填充好pqChunks数值的DiskPQParams实例的指针。不能为空指针。</p>
<p id="p18804047171617"><a name="p18804047171617"></a><a name="p18804047171617"></a><strong id="b144854125013"><a name="b144854125013"></a><a name="b144854125013"></a>const float *pqDistanceTable：</strong>指向query对应的ADC PQ距离表的指针。</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p116514331439"><a name="p116514331439"></a><a name="p116514331439"></a><strong id="b19388104715016"><a name="b19388104715016"></a><a name="b19388104715016"></a>float &amp;pqDistance：</strong>接收最终输出的PQ距离的引用值。</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p161015064519"><a name="p161015064519"></a><a name="p161015064519"></a>int：返回值为0时表示流程正常；返回值为-1时表示流程异常，且会将异常日志信息打印到cerr中。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><a name="ul1693011362444"></a><a name="ul1693011362444"></a><ul id="ul1693011362444"><li>用户需保证basecode指向的数据大小至少有pqChunks个字节。</li><li>在params中，仅需填充pqChunks值，且与basecode中提到的pqChunks值对应。</li><li>用户需保证pqDistanceTable指向的数据大小至少有pqChunks * 256 * sizeof(float)字节数。</li><li>接口中不会在使用前对pqDistance置零，pqDistance最终结果为原pqDistance值 + 输出的query与basecode的PQ距离，因此推荐输入值为0。</li></ul>
</td>
</tr>
</tbody>
</table>



### GetVersionInfo<a name="ZH-CN_TOPIC_0000001456535080"></a>

<a name="table7235918388"></a>
<table><tbody><tr id="row1721359113814"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.1.1"><p id="p12559123810"><a name="p12559123810"></a><a name="p12559123810"></a>API定义</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.1.1 "><p id="p293175163620"><a name="p293175163620"></a><a name="p293175163620"></a>std::string GetVersionInfo();</p>
</td>
</tr>
<tr id="row421759103816"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.2.1"><p id="p1212599383"><a name="p1212599383"></a><a name="p1212599383"></a>功能描述</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.2.1 "><p id="p131714208358"><a name="p131714208358"></a><a name="p131714208358"></a>获取版本信息。会根据环境变量MX_INDEX_HOME来获取对应的版本信息，软件包安装时该环境变量会自动设置，无需修改。</p>
</td>
</tr>
<tr id="row122155911383"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.3.1"><p id="p112195910383"><a name="p112195910383"></a><a name="p112195910383"></a>输入</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.3.1 "><p id="p98111229152410"><a name="p98111229152410"></a><a name="p98111229152410"></a>无</p>
</td>
</tr>
<tr id="row5219599386"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.4.1"><p id="p17235973820"><a name="p17235973820"></a><a name="p17235973820"></a>输出</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.4.1 "><p id="p38831623193613"><a name="p38831623193613"></a><a name="p38831623193613"></a>无</p>
</td>
</tr>
<tr id="row102185913388"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.5.1"><p id="p182459113812"><a name="p182459113812"></a><a name="p182459113812"></a>返回值</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.5.1 "><p id="p912917864516"><a name="p912917864516"></a><a name="p912917864516"></a>版本信息。</p>
</td>
</tr>
<tr id="row22159193815"><th class="firstcol" valign="top" width="20.05%" id="mcps1.1.3.6.1"><p id="p423590386"><a name="p423590386"></a><a name="p423590386"></a>约束说明</p>
</th>
<td class="cellrowborder" valign="top" width="79.95%" headers="mcps1.1.3.6.1 "><p id="p3947640162619"><a name="p3947640162619"></a><a name="p3947640162619"></a>无</p>
</td>
</tr>
</tbody>
</table>



