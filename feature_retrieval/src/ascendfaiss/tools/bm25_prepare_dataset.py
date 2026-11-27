#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. See the repository LICENSE for details.
"""Prepare BM25 document indexes and query IDs using one vocabulary."""

import argparse
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from bm25_build_index import build_index


TOKEN_PATTERN = re.compile(r'\w+')
HAN_PATTERN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]')


def tokenize(text):
    text = text.lower()
    if not HAN_PATTERN.search(text):
        return TOKEN_PATTERN.findall(text)
    try:
        from jieba import cut_for_search
    except ModuleNotFoundError as error:
        if error.name != 'jieba':
            raise
        raise ValueError('Chinese text requires jieba. Install it with python3 -m pip install jieba==0.42.1') from error
    return [token for piece in cut_for_search(text) for token in TOKEN_PATTERN.findall(piece)]


def parse_record(value, source_id, documents=False):
    if isinstance(value, dict):
        source_id = value.get('_id', value.get('id', source_id))
        if 'tokens' in value:
            value = value['tokens']
            if not isinstance(value, list):
                raise ValueError('tokens must be an array')
        elif documents:
            title = value.get('title')
            body = value.get('text')
            if title is None:
                title = ''
            if not isinstance(title, str) or not isinstance(body, str):
                raise ValueError('Expected string text and an optional string title')
            value = f'{title} {body}' if title else body
        else:
            value = value.get('text', value.get('query'))
    if not isinstance(value, (str, list)):
        raise ValueError('Expected text or an array of word strings')
    return source_id, value


def read_records(path, documents=False):
    with path.open(encoding='utf-8-sig') as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = parse_record(json.loads(line), line_number - 1, documents)
            except ValueError as error:
                raise ValueError(f'{path}:{line_number}: {error}') from error
            yield record


def word_tokens(value, tokenizer):
    tokens = tokenizer(value) if isinstance(value, str) else value
    if not isinstance(tokens, list) or not all(isinstance(token, str) and token for token in tokens):
        raise ValueError('Expected text or an array of nonempty word strings')
    return tokens


def encode_documents(documents, directory, work, tokenizer):
    vocabulary = set()
    seen_ids = set()
    tokens_path = work / 'tokens.jsonl'
    with (
        (directory / 'document_ids.jsonl').open('w', encoding='utf-8', newline='\n') as ids,
        tokens_path.open('w', encoding='utf-8', newline='\n') as words,
    ):
        for line_number, (source_id, value) in enumerate(documents, 1):
            if not isinstance(source_id, (str, int)) or isinstance(source_id, bool):
                raise ValueError(f'Document row {line_number} must use a string or integer ID')
            if source_id in seen_ids:
                raise ValueError(f'Document row {line_number} has duplicate ID {source_id!r}')
            seen_ids.add(source_id)
            try:
                tokens = word_tokens(value, tokenizer)
            except ValueError as error:
                raise ValueError(f'Document row {line_number}: {error}') from error
            vocabulary.update(tokens)
            ids.write(json.dumps(source_id, ensure_ascii=False) + '\n')
            words.write(json.dumps(tokens, ensure_ascii=False) + '\n')
    del seen_ids
    ordered_words = sorted(vocabulary)
    del vocabulary
    word_ids = {word: term_id for term_id, word in enumerate(ordered_words)}
    with (directory / 'vocabulary.json').open('w', encoding='utf-8') as stream:
        json.dump(ordered_words, stream, ensure_ascii=False)
        stream.write('\n')
    corpus = work / 'corpus_ids.jsonl'
    with tokens_path.open(encoding='utf-8') as words, corpus.open('w', encoding='ascii', newline='\n') as ids:
        for line in words:
            ids.write(json.dumps([word_ids[token] for token in json.loads(line)]) + '\n')
    tokens_path.unlink()
    return corpus, word_ids


def write_query_ids(queries, word_ids, path, tokenizer=tokenize):
    with path.open('w', encoding='ascii', newline='\n') as target:
        for line_number, value in enumerate(queries, 1):
            try:
                tokens = word_tokens(value, tokenizer)
            except ValueError as error:
                raise ValueError(f'Query row {line_number}: {error}') from error
            query = [word_ids[token] for token in tokens if token in word_ids]
            # Keep in sync with MAX_QUERY_WORDS_PER_QUERY in Bm25SearchPlan.h.
            if len(query) > 64:
                raise ValueError(f'Query row {line_number} exceeds 64 indexed tokens')
            target.write(json.dumps(query) + '\n')


def prepare_dataset(documents, output, queries=None, tokenizer=tokenize):
    """Prepare from (unique source document ID, text or tokens) records and optional queries."""
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        work = Path(temporary)
        dataset = work / 'dataset'
        dataset.mkdir()
        corpus, word_ids = encode_documents(documents, dataset, work, tokenizer)
        if queries is not None:
            write_query_ids(queries, word_ids, dataset / 'query_ids.jsonl', tokenizer)
        build_index(corpus, len(word_ids), dataset / 'index')
        dataset.rename(output)


def encode_queries(queries, vocabulary, output):
    if output.exists() or output.is_symlink():
        raise ValueError(f'Output already exists: {output}')
    words = json.loads(vocabulary.read_text(encoding='utf-8'))
    if not isinstance(words, list) or not words or not all(isinstance(word, str) and word for word in words):
        raise ValueError('Expected a vocabulary array of nonempty word strings')
    word_ids = {word: term_id for term_id, word in enumerate(words)}
    if len(word_ids) != len(words):
        raise ValueError('The vocabulary contains duplicate words')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        encoded = Path(temporary) / 'query_ids.jsonl'
        write_query_ids(queries, word_ids, encoded)
        encoded.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Prepare a corpus and optionally encode queries')
    build.add_argument('--documents', required=True, type=Path)
    build.add_argument('--queries', type=Path)
    build.add_argument('--output', required=True, type=Path)
    encode = commands.add_parser('encode-queries', help='Encode queries with an existing vocabulary')
    encode.add_argument('--queries', required=True, type=Path)
    encode.add_argument('--vocabulary', required=True, type=Path)
    encode.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    queries = (value for _, value in read_records(args.queries)) if args.queries else None
    try:
        if args.command == 'build':
            prepare_dataset(read_records(args.documents, documents=True), args.output, queries)
        else:
            encode_queries(queries, args.vocabulary, args.output)
    except (OSError, ValueError, OverflowError, MemoryError) as error:
        parser.exit(1, f'BM25 dataset preparation failed: {error}\n')
    print(f'Prepared BM25 data at {args.output}')


if __name__ == '__main__':
    main()
