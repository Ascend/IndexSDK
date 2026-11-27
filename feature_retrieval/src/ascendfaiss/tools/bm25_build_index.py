#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. See the repository LICENSE for details.
"""Build the IndexSDK BM25 inverted index from JSONL arrays of term IDs."""

import argparse
from array import array
from collections import Counter
import json
import math
import mmap
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory


INT32_MAX = (1 << 31) - 1
MAX_TERM_FREQUENCY = 1 << 24
MAX_DOCUMENTS = 65536 * 8192

# Match Corpus::BuildNorms and the k1 + 1 TF multiplier in bm25_fused_score_top_k.cpp.
BM25_K1 = 1.2
BM25_B = 0.75

DOC_ID = struct.Struct('<i')
TERM_FREQUENCY = struct.Struct('<f')


def read_documents(path, vocabulary_size):
    with path.open(encoding='utf-8-sig') as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: {error.msg}') from error
            if not isinstance(row, list) or any(
                not isinstance(term, int) or isinstance(term, bool) or not 0 <= term < vocabulary_size for term in row
            ):
                raise ValueError(f'{path}:{line_number}: expected term IDs in [0, {vocabulary_size})')
            counts = Counter(row)
            if any(count > MAX_TERM_FREQUENCY for count in counts.values()):
                raise ValueError(f'{path}:{line_number}: term frequency exceeds the FP32 integer limit')
            yield len(row), counts


def collect_statistics(corpus, vocabulary_size):
    frequencies = array('q', [0]) * vocabulary_size
    lengths = array('q')
    posting_count = 0
    for length, counts in read_documents(corpus, vocabulary_size):
        lengths.append(length)
        posting_count += len(counts)
        if len(lengths) > MAX_DOCUMENTS or posting_count > INT32_MAX:
            raise ValueError('Document or posting count exceeds the IndexSDK limit')
        for term in counts:
            frequencies[term] += 1
    if posting_count == 0:
        raise ValueError('The corpus must contain at least one indexed term')
    return lengths, frequencies


def write_array(path, values):
    with path.open('wb') as stream:
        if sys.byteorder != 'little':
            values = values[:]
            values.byteswap()
        values.tofile(stream)


def write_postings(index, corpus, lengths, offsets):
    cursors = offsets[:-1]
    documents_written = 0
    with (
        (index / 'posting_doc_ids.bin').open('w+b') as doc_file,
        (index / 'posting_tf.bin').open('w+b') as tf_file,
    ):
        doc_file.truncate(offsets[-1] * DOC_ID.size)
        tf_file.truncate(offsets[-1] * TERM_FREQUENCY.size)
        with mmap.mmap(doc_file.fileno(), 0) as docs, mmap.mmap(tf_file.fileno(), 0) as frequencies:
            for doc_id, (length, counts) in enumerate(read_documents(corpus, len(cursors))):
                if doc_id >= len(lengths) or lengths[doc_id] != length:
                    raise ValueError('Document lengths changed while building the index')
                for term, count in counts.items():
                    position = cursors[term]
                    if position >= offsets[term + 1]:
                        raise ValueError('Term document frequencies changed while building the index')
                    DOC_ID.pack_into(docs, position * DOC_ID.size, doc_id)
                    TERM_FREQUENCY.pack_into(frequencies, position * TERM_FREQUENCY.size, count)
                    cursors[term] += 1
                documents_written += 1
    if documents_written != len(lengths) or cursors != offsets[1:]:
        raise ValueError('Corpus statistics changed while building the index')


def build_index(corpus, vocabulary_size, output):
    """Write the five little-endian arrays consumed by IndexSDK BM25."""
    if not 1 <= vocabulary_size < INT32_MAX:
        raise ValueError('Vocabulary size must be in [1, INT32_MAX)')
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output already exists: {output}')
    if (array('q').itemsize, array('f').itemsize) != (8, 4):
        raise ValueError('The Python platform must support 64-bit offsets and 32-bit floats')

    lengths, frequencies = collect_statistics(corpus, vocabulary_size)
    offsets = array('q', [0])
    for count in frequencies:
        offsets.append(offsets[-1] + count)
    document_count = len(lengths)
    average_length = sum(lengths) / document_count
    norms = array('f', (BM25_K1 * (1 - BM25_B + BM25_B * length / average_length) for length in lengths))
    idfs = array('f', (math.log1p((document_count - count + 0.5) / (count + 0.5)) for count in frequencies))

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        index = Path(temporary) / 'index'
        index.mkdir()
        write_array(index / 'term_offsets.bin', offsets)
        write_array(index / 'doc_norm.bin', norms)
        write_array(index / 'idf.bin', idfs)
        write_postings(index, corpus, lengths, offsets)
        index.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', required=True, type=Path, help='One JSON array of integer term IDs per document')
    parser.add_argument('--vocab-size', required=True, type=int, help='Number of IDs in the caller-provided vocabulary')
    parser.add_argument('--output', required=True, type=Path, help='New directory for the five index files')
    args = parser.parse_args()
    try:
        build_index(args.corpus, args.vocab_size, args.output)
    except (OSError, ValueError, OverflowError, MemoryError) as error:
        parser.exit(1, f'BM25 index build failed: {error}\n')
    print(f'Built BM25 index at {args.output}')


if __name__ == '__main__':
    main()
