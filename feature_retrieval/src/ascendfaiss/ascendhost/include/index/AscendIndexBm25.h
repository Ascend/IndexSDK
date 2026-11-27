/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#ifndef ASCEND_INDEX_BM25_H
#define ASCEND_INDEX_BM25_H
#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ErrorCode.h"

namespace ascend
{
class AscendResourcesProxy;

namespace bm25
{
struct Index
{
    std::vector<int64_t> offsets;
    std::vector<int32_t> docs;
    std::vector<float> tf, norm, idf;
    Index(std::vector<int64_t> termOffsets, std::vector<int32_t> postingIds, std::vector<float> termFrequencies,
          std::vector<float> documentNorms, std::vector<float> inverseFrequencies);
};

struct ErrorInfo
{
    int64_t aclCode = 0;
    std::array<char, 1024> message{};

    void Set(const char* errorMessage, int64_t aclStatusCode = 0) noexcept;
};
}  // namespace bm25
}  // namespace ascend

namespace bm25search
{
using Queries = std::vector<std::vector<int32_t>>;
}

namespace faiss
{
namespace ascend
{
struct Bm25Config
{
    int deviceId = 0;
    uint64_t queryCount = 8;
    uint64_t maxQueryWords = 24;
    uint64_t topK = 10;
    std::string vendorDirectory;  // Compatibility field: must be empty. OM uses MX_INDEX_MODELPATH.
    aclrtContext context = nullptr;
    std::shared_ptr<::ascend::AscendResourcesProxy> resources;
    // Capacity limits for a managed corpus, including its committed additions.
    uint64_t maxDocuments = 65536;
    uint64_t maxVocabulary = 262144;
    uint64_t maxPostings = 8388608;
    // Number of documents in the delta that triggers background compaction. Zero disables it.
    uint64_t compactionThreshold = 4096;
};

struct Bm25Document
{
    int64_t id;
    std::string text;
};

struct Bm25DocumentFile
{
    int64_t id;
    std::string path;
};

struct Bm25FileCorpus
{
    std::vector<Bm25DocumentFile> documents;
};

struct Bm25SearchResult
{
    std::vector<float> scores;
    std::vector<int64_t> labels;
};

// Public construction failure; does not expose internal runtime exception types.
class Bm25Exception : public std::runtime_error
{
   public:
    Bm25Exception(::ascend::APP_ERROR code, const ::ascend::bm25::ErrorInfo& error)
        : std::runtime_error(error.message.data()), code_(code), error_(error)
    {
    }
    ::ascend::APP_ERROR GetCode() const noexcept { return code_; }
    const ::ascend::bm25::ErrorInfo& GetErrorInfo() const noexcept { return error_; }

   private:
    ::ascend::APP_ERROR code_;
    ::ascend::bm25::ErrorInfo error_;
};

// BM25 over a read-only pre-tokenized index or a managed text corpus.
// Activate the bound ACL context for construction, Add, Search, SearchText and destruction.
// A null config.context uses the current context. The SDK retains config.resources through cleanup.
// Wait for threads using the object to exit before destroying it. Release caller-owned
// streams and the context after destruction. Use a nested scope to release BM25 early.
// Configure the SDK model directory before the first offline operator is initialized.
class AscendIndexBm25
{
   public:
    // Copies and validates the index, matches offline OM handles, prepares resources and verifies execution.
    // On failure throws Bm25Exception (std::bad_alloc may propagate if creating the exception fails).
    AscendIndexBm25(const ::ascend::bm25::Index& index, const Bm25Config& config);
    // Create a managed index in a directory without an existing index. Text and file contents are UTF-8.
    AscendIndexBm25(const std::vector<Bm25Document>& documents, const std::string& indexDirectory,
                    const Bm25Config& config);
    AscendIndexBm25(const Bm25FileCorpus& corpus, const std::string& indexDirectory, const Bm25Config& config);
    // Open a managed index and replay all committed batches from its directory.
    explicit AscendIndexBm25(const std::string& indexDirectory);
    AscendIndexBm25(const std::string& indexDirectory, const Bm25Config& config);
    // Synchronizes the dedicated stream before releasing SDK resources. Cleanup failures are logged.
    ~AscendIndexBm25();
    AscendIndexBm25(const AscendIndexBm25&) = delete;
    AscendIndexBm25& operator=(const AscendIndexBm25&) = delete;
    AscendIndexBm25(AscendIndexBm25&&) = delete;
    AscendIndexBm25& operator=(AscendIndexBm25&&) = delete;

    // Commit one batch of managed documents. Repeating an ID with identical text is idempotent.
    ::ascend::APP_ERROR Add(const std::vector<Bm25Document>& documents);
    // Encode 1..queryCount text queries with the managed corpus vocabulary, omitting unknown words.
    ::ascend::APP_ERROR EncodeQueries(const std::vector<std::string>& queries, ::bm25search::Queries& encoded);
    // Encode and search 1..queryCount text queries against the same committed corpus version.
    ::ascend::APP_ERROR SearchText(const std::vector<std::string>& queries, Bm25SearchResult& result);
    // Input has exactly queryCount rows. Managed results use the supplied document IDs.
    // Outputs are replaced only on success. Input errors permit a subsequent valid request.
    ::ascend::APP_ERROR Search(const ::bm25search::Queries& queries, Bm25SearchResult& result);
    // Returns diagnostic details from the most recent public operation.
    ::ascend::bm25::ErrorInfo GetLastError() const;

   private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace ascend
}  // namespace faiss
#endif  // ASCEND_INDEX_BM25_H
