#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. See the repository LICENSE for details.

import json
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from bm25_prepare_dataset import encode_queries, prepare_dataset, read_records  # noqa: E402


class PrepareBm25DatasetTest(unittest.TestCase):
    def test_user_document_and_query_outputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / 'corpus.jsonl'
            corpus.write_text(
                '{"_id":"doc-1","title":"Red","text":"Apple apple."}\n{"id":42,"tokens":["南京","大桥","大桥"]}\n',
                encoding='utf-8-sig',
            )
            output = root / 'dataset'
            queries = ['apple OOV', ['大桥', '大桥'], ['OOV']]
            prepare_dataset(read_records(corpus, documents=True), output, queries)

            vocabulary = output / 'vocabulary.json'
            self.assertEqual(json.loads(vocabulary.read_text(encoding='utf-8')), ['apple', 'red', '南京', '大桥'])
            self.assertEqual(
                (output / 'document_ids.jsonl').read_text(encoding='utf-8').splitlines(), ['"doc-1"', '42']
            )
            self.assertEqual(
                [json.loads(line) for line in (output / 'query_ids.jsonl').read_text(encoding='ascii').splitlines()],
                [[0], [3, 3], []],
            )
            tf = output / 'index/posting_tf.bin'
            self.assertEqual(struct.unpack('<4f', tf.read_bytes()), (2, 1, 1, 2))
            later = root / 'new_queries.jsonl'
            encode_queries(queries, vocabulary, later)
            self.assertEqual(later.read_bytes(), (output / 'query_ids.jsonl').read_bytes())

    def test_build_rejects_duplicate_document_ids_atomically(self):
        for repeated_text in ('Apple', 'Orange'):
            with self.subTest(repeated_text=repeated_text), TemporaryDirectory() as temporary:
                root = Path(temporary)
                corpus = root / 'corpus.jsonl'
                records = [{'_id': 'doc-1', 'text': 'Apple'}, {'id': 'doc-1', 'text': repeated_text}]
                corpus.write_text('\n'.join(map(json.dumps, records)) + '\n', encoding='utf-8')
                output = root / 'dataset'

                with self.assertRaisesRegex(ValueError, "Document row 2 has duplicate ID 'doc-1'"):
                    prepare_dataset(read_records(corpus, documents=True), output)
                self.assertFalse(output.exists())

    def test_build_allows_no_queries_and_rejects_oversized_query_atomically(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'dataset'
            prepare_dataset([('doc-1', ['apple'])], output)
            self.assertTrue((output / 'index/term_offsets.bin').is_file())
            self.assertFalse((output / 'query_ids.jsonl').exists())

            rejected = root / 'rejected'
            with self.assertRaisesRegex(ValueError, '64 indexed tokens'):
                prepare_dataset([('doc-1', ['apple'])], rejected, [['apple'] * 65])
            self.assertFalse(rejected.exists())


if __name__ == '__main__':
    unittest.main()
