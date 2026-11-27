#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. See the repository LICENSE for details.

import math
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from bm25_build_index import build_index  # noqa: E402


def read_array(path, fmt):
    return [value for (value,) in struct.iter_unpack(fmt, path.read_bytes())]


class BuildBm25IndexTest(unittest.TestCase):
    def test_format_repeated_terms_empty_document_and_unused_id(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / 'corpus.jsonl'
            corpus.write_text('[2, 0, 2]\n[]\n[0, 1]\n[1, 1]\n', encoding='utf-8-sig')
            index = root / 'index'
            build_index(corpus, 4, index)

            self.assertEqual(read_array(index / 'term_offsets.bin', '<q'), [0, 2, 4, 5, 5])
            self.assertEqual(read_array(index / 'posting_doc_ids.bin', '<i'), [0, 2, 2, 3, 0])
            self.assertEqual(read_array(index / 'posting_tf.bin', '<f'), [1.0, 1.0, 1.0, 2.0, 2.0])
            norms = read_array(index / 'doc_norm.bin', '<f')
            idfs = read_array(index / 'idf.bin', '<f')
            for norm, length in zip(norms, (3, 0, 2, 2)):
                self.assertAlmostEqual(norm, 0.3 + 0.9 * length / 1.75)
            for idf, ratio in zip(idfs, (1.0, 1.0, 3.5 / 1.5, 9.0)):
                self.assertAlmostEqual(idf, math.log1p(ratio), places=6)

    def test_invalid_input_is_not_published(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / 'corpus.jsonl'
            index = root / 'index'
            for invalid in ('[4]', '[-1]', '[true]', '[1.0]', '{"tokens": [1]}', '[]', ''):
                with self.subTest(invalid=invalid):
                    corpus.write_text(invalid, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        build_index(corpus, 4, index)
                    self.assertFalse(index.exists())

    def test_existing_index_is_preserved(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / 'corpus.jsonl'
            corpus.write_text('[0]\n', encoding='utf-8')
            index = root / 'index'
            build_index(corpus, 1, index)
            corpus.write_text('[0, 0]\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Output already exists'):
                build_index(corpus, 1, index)
            self.assertEqual(read_array(index / 'posting_tf.bin', '<f'), [1.0])


if __name__ == '__main__':
    unittest.main()
