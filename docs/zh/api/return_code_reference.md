
## 接口调用返回值参考<a id="ZH-CN_TOPIC_0000001456375228"></a>

**表 1**  APP\_ERR，Device侧接口调用返回值表

|返回码|返回说明|
|--|--|
|APP_ERR_OK = 0|success|
|APP_ACL_BASE = 1000|define the error code of ACL, Error codes 1001~1999|
|APP_ERR_ACL_BAD_ALLOC = 1001|ACL: memory allocation failed|
|APP_ERR_ACL_LOAD_MODEL_FAILED = 1002|ACL: model load failed|
|APP_ERR_ACL_UNLOAD_MODEL_FAILED = 1003|ACL: model unload failed|
|APP_ERR_ACL_GET_DIMS_FAILURE = 1004|ACL: failed to get dimension information|
|APP_ERR_ACL_CREATE_MODEL_DESC_FAILED = 1005|ACL: failed to create model information|
|APP_ERR_ACL_DESTORY_MODEL_DESC_FAILED = 1006|ACL: failed to destory model information|
|APP_ERR_ACL_GET_MODEL_DESC_FAILED = 1007|ACL: failed to get model information|
|APP_ERR_ACL_MODEL_EXEC_FAILURE = 1008|ACL: model inference failed|
|APP_ERR_ACL_CREATE_DATA_SET_FAILED = 1009|ACL: failed to create dataset|
|APP_ERR_ACL_DESTORY_DATA_SET_FAILED = 1010|ACL: failed to destory dataset|
|APP_ERR_ACL_CREATE_DATA_BUF_FAILED = 1011|ACL: failed to create databuffer|
|APP_ERR_ACL_DESTORY_DATA_BUF_FAILED = 1012|ACL: failed to destory databuffer|
|APP_ERR_ACL_ADD_DATA_BUF_FAILED = 1013|ACL: failed to add databuffer to dataset|
|APP_ERR_ACL_GET_DATA_BUF_ADDR_NULL = 1014|ACL: failed to get the address of databuffer|
|APP_ERR_ACL_OP_NOT_FOUND = 1015|ACL: op model not found|
|APP_ERR_ACL_OP_LOAD_MODEL_FAILED = 1016|ACL: op model load failed|
|APP_ERR_ACL_OP_EXEC_FAILED = 1017|ACL: op model execute failed, not running in aicore|
|APP_ERR_ACL_SET_DEVICE_FAILED = 1018|ACL: failed to set device|
|APP_ERR_ACL_END = 1019|ACL: end of ACL ERR|
|APP_BASE = 2000|define the APP error code, range: 2001~2999|
|APP_ERR_INVALID_PARAM = 2001|invalid parameter|
|APP_ERR_INVALID_HDC_DATA = 2002|invalid HDC transport data|
|APP_ERR_INDEX_NOT_FOUND = 2003|index not found|
|APP_ERR_TRANSFORMER_NOT_FOUND = 2004|transformer not found|
|APP_ERR_CLUSTERING_NOT_FOUND = 2005|clustering not found|
|APP_ERR_INFERENCE_NOT_FOUND = 2006|inference not found|
|APP_ERR_REQUEST_ERROR = 2007|request parameter error|
|APP_ERR_NOT_IMPLEMENT = 2008|method not implement|
|APP_ERR_ILLEGAL_OPERATION = 2009|illegal operation|
|APP_ERR_INNER_ERROR = 2010|internal error|
|APP_ERR_TIMEOUT = 2011|timeout|
|APP_CREATE_INDEX_FAILED = 2012|create index failed|
|APP_CREATE_TRANSFORM_FAILED = 2013|create transform failed|
|APP_CREATE_INFERENCE_FAILED = 2014|create inference failed|
|APP_ERR_INVALID_TABLE_INDEX = 2015|Idx exceeds tableLen, table mapping failed|
|APP_ERR_INDEX_NOT_INIT = 2016|index not initialize|
|APP_ERR_END = 2017|end of APP ERR|



