# API Call Return Code Reference<a id="ZH-CN_TOPIC_0000001456375228"></a>

**Table 1**  APP_ERR, device-side API call return values

|Return Code|Description|
|--|--|
|APP_ERR_OK = 0|Success.|
|APP_ACL_BASE = 1000|Defines the ACL error code range. Error codes 1001 to 1999.|
|APP_ERR_ACL_BAD_ALLOC = 1001|ACL: failed to allocate memory.|
|APP_ERR_ACL_LOAD_MODEL_FAILED = 1002|ACL: failed to load model.|
|APP_ERR_ACL_UNLOAD_MODEL_FAILED = 1003|ACL: failed to unload model.|
|APP_ERR_ACL_GET_DIMS_FAILURE = 1004|ACL: failed to get dimension information.|
|APP_ERR_ACL_CREATE_MODEL_DESC_FAILED = 1005|ACL: failed to create model information.|
|APP_ERR_ACL_DESTROY_MODEL_DESC_FAILED = 1006|ACL: failed to destroy model information.|
|APP_ERR_ACL_GET_MODEL_DESC_FAILED = 1007|ACL: failed to get model information.|
|APP_ERR_ACL_MODEL_EXEC_FAILURE = 1008|ACL: failed to run model inference.|
|APP_ERR_ACL_CREATE_DATA_SET_FAILED = 1009|ACL: failed to create dataset.|
|APP_ERR_ACL_DESTROY_DATA_SET_FAILED = 1010|ACL: failed to destroy dataset.|
|APP_ERR_ACL_CREATE_DATA_BUF_FAILED = 1011|ACL: failed to create data buffer.|
|APP_ERR_ACL_DESTROY_DATA_BUF_FAILED = 1012|ACL: failed to destroy data buffer.|
|APP_ERR_ACL_ADD_DATA_BUF_FAILED = 1013|ACL: failed to add data buffer to dataset.|
|APP_ERR_ACL_GET_DATA_BUF_ADDR_NULL = 1014|ACL: failed to get the address of the data buffer.|
|APP_ERR_ACL_OP_NOT_FOUND = 1015|ACL: op model not found.|
|APP_ERR_ACL_OP_LOAD_MODEL_FAILED = 1016|ACL: failed to load op model.|
|APP_ERR_ACL_OP_EXEC_FAILED = 1017|ACL: failed to execute the model. Not running on AI Core.|
|APP_ERR_ACL_SET_DEVICE_FAILED = 1018|ACL: failed to set device.|
|APP_ERR_ACL_END = 1019|ACL: end of ACL errors.|
|APP_BASE = 2000|Defines the APP error code range. Error codes 2001 to 2999.|
|APP_ERR_INVALID_PARAM = 2001|Invalid parameter.|
|APP_ERR_INVALID_HDC_DATA = 2002|Invalid HDC transport data.|
|APP_ERR_INDEX_NOT_FOUND = 2003|Index not found.|
|APP_ERR_TRANSFORMER_NOT_FOUND = 2004|Transformer not found.|
|APP_ERR_CLUSTERING_NOT_FOUND = 2005|Clustering not found.|
|APP_ERR_INFERENCE_NOT_FOUND = 2006|Inference not found.|
|APP_ERR_REQUEST_ERROR = 2007|Request parameter error.|
|APP_ERR_NOT_IMPLEMENT = 2008|Method not implement.|
|APP_ERR_ILLEGAL_OPERATION = 2009|Illegal operation.|
|APP_ERR_INNER_ERROR = 2010|Internal error.|
|APP_ERR_TIMEOUT = 2011|Timeout.|
|APP_CREATE_INDEX_FAILED = 2012|Failed to create index.|
|APP_CREATE_TRANSFORM_FAILED = 2013|Failed to create transform.|
|APP_CREATE_INFERENCE_FAILED = 2014|Failed to create inference.|
|APP_ERR_INVALID_TABLE_INDEX = 2015|`Idx` exceeds `tableLen`. Failed to map table.|
|APP_ERR_INDEX_NOT_INIT = 2016|Index not initialized.|
|APP_ERR_END = 2017|End of APP errors.|
