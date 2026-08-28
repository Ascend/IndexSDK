# 接口调用返回值参考<a id="ZH-CN_TOPIC_0000001456375228"></a>

**表 1**  APP\_ERR，Device侧接口调用返回值表

<table><tbody>
<tr><td width="410" align="center" valign="middle"><strong>返回码</strong></td><td align="center" valign="middle"><strong>返回说明</strong></td></tr>
<tr><td width="410" valign="middle">APP_ERR_OK = 0</td><td valign="middle">success</td></tr>
<tr><td width="410" valign="middle">APP_ACL_BASE = 1000</td><td valign="middle">define the error code of ACL, Error codes 1001~1999</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_BAD_ALLOC = 1001</td><td valign="middle">ACL: memory allocation failed</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_LOAD_MODEL_FAILED = 1002</td><td valign="middle">ACL: model load failed</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_UNLOAD_MODEL_FAILED = 1003</td><td valign="middle">ACL: model unload failed</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_GET_DIMS_FAILURE = 1004</td><td valign="middle">ACL: failed to get dimension information</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_CREATE_MODEL_DESC_FAILED = 1005</td><td valign="middle">ACL: failed to create model information</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_DESTROY_MODEL_DESC_FAILED = 1006</td><td valign="middle">ACL: failed to destroy model information</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_GET_MODEL_DESC_FAILED = 1007</td><td valign="middle">ACL: failed to get model information</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_MODEL_EXEC_FAILURE = 1008</td><td valign="middle">ACL: model inference failed</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_CREATE_DATA_SET_FAILED = 1009</td><td valign="middle">ACL: failed to create dataset</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_DESTROY_DATA_SET_FAILED = 1010</td><td valign="middle">ACL: failed to destroy dataset</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_CREATE_DATA_BUF_FAILED = 1011</td><td valign="middle">ACL: failed to create databuffer</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_DESTROY_DATA_BUF_FAILED = 1012</td><td valign="middle">ACL: failed to destroy databuffer</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_ADD_DATA_BUF_FAILED = 1013</td><td valign="middle">ACL: failed to add databuffer to dataset</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_GET_DATA_BUF_ADDR_NULL = 1014</td><td valign="middle">ACL: failed to get the address of databuffer</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_OP_NOT_FOUND = 1015</td><td valign="middle">ACL: op model not found</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_OP_LOAD_MODEL_FAILED = 1016</td><td valign="middle">ACL: op model load failed</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_OP_EXEC_FAILED = 1017</td><td valign="middle">ACL: op model execute failed, not running in aicore</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_SET_DEVICE_FAILED = 1018</td><td valign="middle">ACL: failed to set device</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ACL_END = 1019</td><td valign="middle">ACL: end of ACL ERR</td></tr>
<tr><td width="410" valign="middle">APP_BASE = 2000</td><td valign="middle">define the APP error code, range: 2001~2999</td></tr>
<tr><td width="410" valign="middle">APP_ERR_INVALID_PARAM = 2001</td><td valign="middle">invalid parameter</td></tr>
<tr><td width="410" valign="middle">APP_ERR_INVALID_HDC_DATA = 2002</td><td valign="middle">invalid HDC transport data</td></tr>
<tr><td width="410" valign="middle">APP_ERR_INDEX_NOT_FOUND = 2003</td><td valign="middle">index not found</td></tr>
<tr><td width="410" valign="middle">APP_ERR_TRANSFORMER_NOT_FOUND = 2004</td><td valign="middle">transformer not found</td></tr>
<tr><td width="410" valign="middle">APP_ERR_CLUSTERING_NOT_FOUND = 2005</td><td valign="middle">clustering not found</td></tr>
<tr><td width="410" valign="middle">APP_ERR_INFERENCE_NOT_FOUND = 2006</td><td valign="middle">inference not found</td></tr>
<tr><td width="410" valign="middle">APP_ERR_REQUEST_ERROR = 2007</td><td valign="middle">request parameter error</td></tr>
<tr><td width="410" valign="middle">APP_ERR_NOT_IMPLEMENT = 2008</td><td valign="middle">method not implement</td></tr>
<tr><td width="410" valign="middle">APP_ERR_ILLEGAL_OPERATION = 2009</td><td valign="middle">illegal operation</td></tr>
<tr><td width="410" valign="middle">APP_ERR_INNER_ERROR = 2010</td><td valign="middle">internal error</td></tr>
<tr><td width="410" valign="middle">APP_ERR_TIMEOUT = 2011</td><td valign="middle">timeout</td></tr>
<tr><td width="410" valign="middle">APP_CREATE_INDEX_FAILED = 2012</td><td valign="middle">create index failed</td></tr>
<tr><td width="410" valign="middle">APP_CREATE_TRANSFORM_FAILED = 2013</td><td valign="middle">create transform failed</td></tr>
<tr><td width="410" valign="middle">APP_CREATE_INFERENCE_FAILED = 2014</td><td valign="middle">create inference failed</td></tr>
<tr><td width="410" valign="middle">APP_ERR_INVALID_TABLE_INDEX = 2015</td><td valign="middle">Idx exceeds tableLen, table mapping failed</td></tr>
<tr><td width="410" valign="middle">APP_ERR_INDEX_NOT_INIT = 2016</td><td valign="middle">index not initialize</td></tr>
<tr><td width="410" valign="middle">APP_ERR_END = 2017</td><td valign="middle">end of APP ERR</td></tr>
</tbody></table>
