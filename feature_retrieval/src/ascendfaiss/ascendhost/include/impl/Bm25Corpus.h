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

#ifndef ASCEND_BM25_CORPUS_H
#define ASCEND_BM25_CORPUS_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ascendhost/include/index/AscendIndexBm25.h"

namespace ascend::bm25
{
class CorpusStore;

// Term IDs and the internal document order are append-only. Copies share immutable documents.
class Corpus
{
   public:
    static Corpus Build(const std::vector<faiss::ascend::Bm25Document>& documents);
    Corpus Append(const std::vector<faiss::ascend::Bm25Document>& documents) const;
    Index BuildIndex(size_t begin, size_t end) const;
    Index TakeSnapshotIndex();
    std::vector<float> BuildNorms(size_t begin, size_t end) const;
    std::vector<float> BuildIdfs(size_t vocabularySize) const;
    bm25search::Queries Encode(const std::vector<std::string>& queries) const;
    size_t DocumentCount() const { return documents_.size(); }
    size_t VocabularySize() const { return vocabulary_.size(); }
    size_t PostingCount() const { return postingCount_; }
    int64_t DocumentId(size_t position) const { return documents_.at(position)->id; }

   private:
    friend class CorpusStore;

    struct Document
    {
        int64_t id;
        std::string text;
        std::vector<std::pair<int32_t, uint32_t>> terms;
        uint64_t length = 0;
    };

    void AppendRaw(const std::vector<faiss::ascend::Bm25Document>& documents);
    int32_t AddWord(const std::string& word);
    void AddDocument(Document document);

    std::vector<std::shared_ptr<const Document>> documents_;
    std::unordered_map<int64_t, size_t> documentPositions_;
    std::vector<std::string> vocabulary_;
    std::unordered_map<std::string, int32_t> wordIds_;
    std::vector<uint32_t> documentFrequencies_;
    size_t postingCount_ = 0;
    uint64_t totalLength_ = 0;
    std::shared_ptr<Index> persistedIndex_;
};

// Owns the index directory lock for the lifetime of the managed index.
class CorpusStore
{
   public:
    // Keep the store alive until a staged snapshot has been published or discarded.
    class PendingSnapshot
    {
       public:
        ~PendingSnapshot();
        PendingSnapshot(PendingSnapshot&& other) noexcept;
        PendingSnapshot(const PendingSnapshot&) = delete;
        PendingSnapshot& operator=(const PendingSnapshot&) = delete;

       private:
        friend class CorpusStore;
        PendingSnapshot(int directoryFd, std::string finalName, std::vector<std::string> obsoleteFiles);

        int directoryFd_;
        std::string finalName_;
        std::string temporaryName_;
        std::vector<std::string> obsoleteFiles_;
        bool published_ = false;
    };

    CorpusStore(const std::string& directory, bool create);
    ~CorpusStore();
    CorpusStore(const CorpusStore&) = delete;
    CorpusStore& operator=(const CorpusStore&) = delete;

    std::pair<Corpus, size_t> Load(faiss::ascend::Bm25Config* config = nullptr) const;
    void SaveSnapshot(const Corpus& corpus, const Index* index = nullptr,
                      const faiss::ascend::Bm25Config& config = {}) const;
    PendingSnapshot StageSnapshot(const Corpus& corpus, const Index* index = nullptr,
                                  const faiss::ascend::Bm25Config& config = {}) const;
    void PublishSnapshot(PendingSnapshot&& snapshot) const;
    void SaveBatch(const Corpus& corpus, size_t oldDocumentCount, size_t oldVocabularySize) const;

   private:
    Corpus LoadSnapshot(size_t documentCount, faiss::ascend::Bm25Config* config) const;
    void ReplayBatch(Corpus& corpus, size_t begin) const;
    int directoryFd_ = -1;
};
}  // namespace ascend::bm25

#endif  // ASCEND_BM25_CORPUS_H
