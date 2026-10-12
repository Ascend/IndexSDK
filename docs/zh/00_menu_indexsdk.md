# Index SDK 用户指南

<!-- npu="950,A3,910b,310p" id6 -->
- [简介](./01_introduction.md)
- [软件安装](./04_installation_guide.md)
- [快速入门](./03_quick_start.md)
- [生成算子](./05_user_guide.md)
- [API参考](./api/README.md)
  - [接口变更说明](./api/README.md)
  - [全量检索]()
    - [AscendIndex](./api/01_full_retrieval/01_AscendIndex.md)
    <!-- npu="310p" id2 -->
    - [AscendIndexCluster](./api/01_full_retrieval/02_AscendIndexCluster.md)
    <!-- end id2 -->
    - [AscendIndexConfig](./api/01_full_retrieval/03_AscendIndexConfig.md)
    - [AscendIndexInt8](./api/01_full_retrieval/04_AscendIndexInt8.md)
    - [AscendIndexInt8Config](./api/01_full_retrieval/05_AscendIndexInt8Config.md)
    - [AscendIndexInt8Flat](./api/01_full_retrieval/06_AscendIndexInt8Flat.md)
    - [AscendIndexInt8FlatConfig](./api/01_full_retrieval/07_AscendIndexInt8FlatConfig.md)
    - [AscendIndexFlat](./api/01_full_retrieval/08_AscendIndexFlat.md)
    - [AscendIndexFlatConfig](./api/01_full_retrieval/09_AscendIndexFlatConfig.md)
    - [AscendIndexFlatL2](./api/01_full_retrieval/10_AscendIndexFlatL2.md)
    - [AscendIndexSQ](./api/01_full_retrieval/11_AscendIndexSQ.md)
    - [AscendIndexSQConfig](./api/01_full_retrieval/12_AscendIndexSQConfig.md)
    - [IndexIL](./api/01_full_retrieval/13_IndexIL.md)
    <!-- npu="310p" id3 -->
    - [IndexILFlat](./api/01_full_retrieval/14_IndexILFlat.md)
    <!-- end id3 -->
    <!-- npu="310p" id4 -->
    - [AscendIndexILFlat](./api/01_full_retrieval/15_AscendIndexILFlat.md)
    <!-- end id4 -->
  - [近似检索]()
    <!-- npu="310p" id5 -->
    - [AscendIndexBinaryFlat](./api/02_approximate_retrieval/01_AscendIndexBinaryFlat.md)
    <!-- end id5 -->
    - [AscendIndexBinaryFlatConfig](./api/02_approximate_retrieval/02_AscendIndexBinaryFlatConfig.md)
    - [AscendIndexIVF](./api/02_approximate_retrieval/03_AscendIndexIVF.md)
    - [AscendIndexIVFConfig](./api/02_approximate_retrieval/04_AscendIndexIVFConfig.md)
    <!-- npu="310p" id1 -->
    - [AscendIndexIVFSP](./api/02_approximate_retrieval/05_AscendIndexIVFSP.md)
    <!-- end id1 -->
    - [AscendIndexIVFSPConfig](./api/02_approximate_retrieval/06_AscendIndexIVFSPConfig.md)
    - [AscendIndexIVFSQ](./api/02_approximate_retrieval/07_AscendIndexIVFSQ.md)
    - [AscendIndexIVFSQConfig](./api/02_approximate_retrieval/08_AscendIndexIVFSQConfig.md)
    - [AscendIndexIVFSQT](./api/02_approximate_retrieval/09_AscendIndexIVFSQT.md)
    - [AscendIndexIVFSQTConfig](./api/02_approximate_retrieval/10_AscendIndexIVFSQTConfig.md)
    <!-- npu="310p" id7 -->
    - [AscendIndexVStar](./api/02_approximate_retrieval/11_AscendIndexVStar.md)
    <!-- end id7 -->
    <!-- npu="310p" id8 -->
    - [AscendIndexGreat](./api/02_approximate_retrieval/12_AscendIndexGreat.md)
    <!-- end id8 -->
    - [AscendIndexMixSearchParams](./api/02_approximate_retrieval/13_AscendIndexMixSearchParams.md)
    <!-- npu="950,A3,910b" id9 -->
    - [AscendIndexIVFFlat](./api/02_approximate_retrieval/14_AscendIndexIVFFlat.md)
    <!-- end id9 -->
    <!-- npu="950" id10 -->
    - [AscendIndexIVFPQ](./api/02_approximate_retrieval/15_AscendIndexIVFPQ.md)
    <!-- end id10 -->
    <!-- npu="950,A3,910b" id11 -->
    - [AscendIndexIVFRaBitQ](./api/02_approximate_retrieval/16_AscendIndexIVFRaBitQ.md)
    <!-- end id11 -->
    <!-- npu="950,A3,910b" id13 -->
    - [AscendIndexIVFRaBitQConfig](./api/02_approximate_retrieval/17_AscendIndexIVFRaBitQConfig.md)
    <!-- end id13 -->
    <!-- npu="950" id12 -->
    - [AscendIndexCagra](./api/02_approximate_retrieval/18_AscendIndexCagra.md)
    <!-- end id12 -->
  - [属性过滤检索]()
    - [AscendIndexTS](./api/03_attribute_filtering-based_retrieval/01_AscendIndexTS.md)
    - [AttrFilter](./api/03_attribute_filtering-based_retrieval/02_AttrFilter.md)
    - [ExtraValAttr](./api/03_attribute_filtering-based_retrieval/03_ExtraValAttr.md)
    - [ExtraValFilter](./api/03_attribute_filtering-based_retrieval/04_ExtraValFilter.md)
    - [FeatureAttr](./api/03_attribute_filtering-based_retrieval/05_FeatureAttr.md)
  - [多 Index 批量检索](./api/04_multi-index_batch_retrieval/01_multi-index_batch_retrieval.md)
  - [其他功能]()
    - [IReduction](./api/05_more_functions/01_IReduction.md)
    - [AscendNNInference](./api/05_more_functions/02_AscendNNInference.md)
    - [AscendClonerOptions](./api/05_more_functions/03_AscendClonerOptions.md)
    - [AscendCloner](./api/05_more_functions/04_AscendCloner.md)
    - [DiskPQ](./api/05_more_functions/05_DiskPQ.md)
    - [GetVersionInfo](./api/05_more_functions/06_GetVersionInfo.md)
  - [未使用接口](./api/06_unused_apis/01_unused_apis.md)
  - [接口调用返回值参考](./api/07_return_code_reference/01_return_code_reference.md)
- [常用操作](./08_common_operations.md)
- [安全加固](./06_security_hardening.md)
- [FAQ](./07_faq.md)
- [附录](./09_appendix.md)
<!-- end id6 -->
