# Requires fasttext, tqdm, fugashi, unidic, unidic-lite, sklearn
import re
import os
import sys
from enum import Enum
from collections.abc import Iterator, Iterable, Callable
from typing import Optional, Union
from collections import defaultdict, Counter
from urllib.request import urlretrieve
from contextlib import contextmanager
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA
import gzip
import bz2
import lzma
from itertools import chain, groupby
import argparse
from tqdm import tqdm  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import linear_kernel  # type: ignore
import fasttext  # type: ignore
import fugashi
from ja_utils import LCASE_FW2HW, fugashi_tagger

# We use the smaller model from
# https://fasttext.cc/docs/en/language-identification.html
FT_LID_MODEL_PATH = 'lid.176.ftz'
FT_LID_MODEL_URL = (
    'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'
    )

JA_LABEL = ['__label__ja']
MIN_JA_FREQ = 0.95
SIMILARITY_LIMIT = 0.95
MIN_JA_CHARS_FREQ = 0.7
MIN_LINES = 3
DATA_PATH = 'jtubespeech/video/ja/txt'
SUBLIST_PATH = 'jtubespeech/data/ja/202103.csv'
DATA_SUFFIX = '.txt'
CLEAN_PATH = 'clean-ja'
UNIQUE_PATH = 'unique-ja'
DEFAULT_FREQ_PATH = 'tubelex-ja.tsv'    # See CLI arguments
DEFAULT_NORM_FREQ_PATH = 'tubelex-ja-lower.tsv'    # See CLI arguments
DEFAULT_CHANNEL_STATS_PATH = 'tubelex-ja-channels.tsv'  # See CLI arguments
DEFAULT_MIN_VIDEOS = 3                    # See CLI arguments
DEFAULT_MIN_CHANNELS = 3                # See CLI arguments
LINEAR_KERNEL_CHUNK_SIZE = 10000

# Note: \w includes accented chars, CJK, etc.
is_word_re = re.compile(r'^\w.*\w$')
# Note: \d includes decimals in many scripts, but not CJK
not_is_word_re = re.compile(r'.*\d.*')

Tokenizer = Callable[[str], list[str]]


class Storage(Enum):
    PLAIN = (None, open, '')
    DEFLATE = (ZIP_DEFLATED, gzip.open, '.gz')
    BZIP2 = (ZIP_BZIP2, bz2.open, '.bz2')
    LZMA = (ZIP_LZMA, lzma.open, '.xz')

    def __init__(self, zip_compression, open_fn, suffix):
        self.zip_compression    = zip_compression
        self.open               = open_fn
        self.suffix             = suffix

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'Storage':
        return (
            Storage.DEFLATE if args.deflate else
            Storage.BZIP2 if args.bzip2 else
            Storage.LZMA if args.lzma else
            Storage.PLAIN
            )


def linear_kernel_piecewise(x, max_size=LINEAR_KERNEL_CHUNK_SIZE, wrapper=None):
    # To support sparse matrices we split manually (not using np.array_split).
    # Note: Row slicing is efficient for CSR (Compressed Sparse Row).
    pieces = [x[i:i + max_size] for i in range(0, x.shape[0], max_size)]
    if wrapper:
        pieces = wrapper(pieces)
    return np.concatenate([linear_kernel(p, x) for p in pieces])


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Process subtitles from jtubespeech and compute word frequencies'
        )

    arch_group = parser.add_mutually_exclusive_group()
    arch_group.add_argument(
        '--deflate', '--zip', '-z', action='store_true',
        help='Store data deflated (.zip/.gz)'
        )
    arch_group.add_argument(
        '--bzip2', '-j', action='store_true',
        help='Store data using Bzip2 (.zip/.bz2)'
        )
    arch_group.add_argument(
        '--lzma', '--xz', '-x', action='store_true',
        help='Store data using LZMA (.zip/.xz)'
        )

    parser.add_argument(
        '--tf-idf-ngram-min', dest='nmin', type=int, default=1,
        help='Consider n-grams where n>=NMIN for TF-IDF.'
        )
    parser.add_argument(
        '--tf-idf-ngram-max', dest='nmax', type=int, default=1,
        help='Consider n-grams where n<=NMAX for TF-IDF.'
        )

    parser.add_argument('--clean', '-c', action='store_true', help='Clean up data')
    parser.add_argument('--unique', '-u', action='store_true', help='Deduplicate data')
    parser.add_argument(
        '--frequencies', '-f', action='store_true',
        help='Compute frequencies'
        )

    parser.add_argument(
        '--limit', type=int, default=None,
        help='Maximum number of videos to process (for testing purposes)'
        )

    parser.add_argument(
        '--min-videos', type=int, default=DEFAULT_MIN_VIDEOS,
        help='Minimum videos for the word to be counted'
        )
    parser.add_argument(
        '--min-channels', type=int, default=DEFAULT_MIN_VIDEOS,
        help='Minimum channels for the word to be counted'
        )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help='Output filename for frequencies'
        )
    parser.add_argument(
        '--output-normalized', '-O', type=str, default=None,
        help=(
            'Output filename for frequencies of normalized words '
            '(lowercased and half-width Latin characters)'
            )
        )
    parser.add_argument(
        '--channel-stats', type=str, default=None,
        help='Output filename for channel stats (computed together with frequencies)'
        )
    dic_group = parser.add_mutually_exclusive_group()
    dic_group.add_argument(
        '--dicdir', type=str, default=None,
        help='Dictionary directory for fugashi/MeCab.'
        )
    dic_group.add_argument(
        '--dictionary', '-D', choices=('unidic', 'unidic-lite'), default=None,
        help='Dictionary (installed as a Python package) for fugashi/MeCab.'
        )

    return parser.parse_args()


def tagger_from_args(args: argparse.Namespace) -> fugashi.GenericTagger:
    # We always specificy dicdir EXPLICITLY
    if args.dicdir is not None:
        dicdir = args.dicdir
    else:
        if args.dictionary == 'unidic':
            import unidic
            dicdir = unidic.DICDIR
        else:
            assert args.dictionary is None or args.dictionary == 'unidic-lite'
            import unidic_lite
            dicdir = unidic_lite.DICDIR
    return fugashi_tagger(dicdir)


@contextmanager
def get_write_file(path: str, storage: Storage):
    try:
        write_file: Callable[[str, str], None]
        if storage.zip_compression is not None:
            zf = ZipFile(path + '.zip', 'w', storage.zip_compression)
            write_file = zf.writestr    # type: ignore
        else:
            os.makedirs(path, exist_ok=True)

            def write_file(fname: str, contents: str) -> None:
                with open(os.path.join(path, fname), 'w') as f:
                    f.write(contents)

        yield write_file
    finally:
        if storage.zip_compression is not None:
            zf.close()


RE_JAPANESE_1C = (
    # Based on http://www.localizingjapan.com/blog/2012/01/20/regular-expressions-for-\
    # japanese-text/
    # We give it a little benefit of doubt by including all kanji and even radicals,
    # which still could be indicative of Japanese or Chinese text.
    r'[\u3041-\u3096'   # Hiragana
    r'\u30A0-\u30FF'    # Katakana (full-width)
    r'\u3400-\u4DB5\u4E00-\u9FCB\uF900-\uFA6A'  # {Han} (Kanji incl. Chinese-only)
    r'\u2E80-\u2FD5'    # Kanji radicals
    r'\uFF5F-\uFF9F'    # HW Katakana and pucntutation
    r'\u3000-\u303F'    # Japanese Symbols and Punctuation
    r'\u31F0-\u31FF\u3220-\u3243\u3280-\u337F'  # Misc. Japanese Symbols and Characters
    r'\uFF01-\uFF5E]'   # FW Alphanumeric and Punctuation (basically FW ASCII)
    )
RE_JAPANESE = re.compile(RE_JAPANESE_1C)
RE_JAPANESE_SEQ = re.compile(RE_JAPANESE_1C + r'+')
JA_CHARSET = {
    c for c in map(chr, range(sys.maxunicode + 1)) if RE_JAPANESE.match(c)
    }

RE_TAG = re.compile(  # <...> and &lt;font...&gt;
    r'&lt;(font|FONT) [^&]*&gt;|'
    r'&lt;/(font|FONT)&gt;|'
    r'</?[a-zA-Z0-9.]+>|'
    r'\[br\]'  # not valid formatting tag, but present in many Udacity videos
    )

RE_WHITESPACE_ONLY = re.compile(r'^\s*$')

# YouTube subtitles contain only the following entities:
ENTITY2STR = {'lt': '<', 'gt': '>', 'nbsp': ' ', 'amp': '&', 'lrm': ''}
RE_ENTITY  = re.compile(r'&(%s);' % '|'.join(ENTITY2STR))


def repl_entities(m):
    return ENTITY2STR[m.group(1)]


RE_URL = (  # based on https://urlregex.com
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|'
    r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
# based on https://emailregex.com
RE_EMAIL = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
# modeled after RE_EMAIL
RE_WWW = r'www.[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
# social network handles generally don't start with . and don't contain consecutive .
RE_HANDLE = r'@([a-zA-Z0-9_]+.)*[a-zA-Z0-9_]+'
RE_ADDRESS = re.compile(r'|'.join((RE_URL, RE_EMAIL, RE_WWW, RE_HANDLE)))


class SubCleaner:
    n_removed_tags: int
    n_removed_addresses: int
    n_lines_empty: int
    n_lines_rep: int
    n_lines_nonj: int
    n_chars: int
    n_chars_j: int

    '''
    Clean and filter lines via the __call__ method. Counts (instance variables) reflect
    the last call (non-cumulative). Counts are not set until the iterator is exhausted.
    '''

    def _subs2text(self, lines: Iterable[str]) -> Iterator[str]:
        n_removed_tags      = 0
        n_removed_addresses = 0
        for line in lines:
            text = line.rstrip().split('\t', 2)[2]
            # Remove outmost quotes (part of the vtt2txt format):
            assert text and text[0] == '"' and text[-1] == '"'
            text = text[1:-1]
            # Remove formatting tags:
            text, n_tags = RE_TAG.subn('', text)
            # Convert HTML entities:
            text = RE_ENTITY.sub(repl_entities, text)
            # Replace addresses with space for better tokenization
            text, n_addresses = RE_ADDRESS.subn(' ', text)
            n_removed_tags += n_tags
            n_removed_addresses += n_addresses
            yield text
        self.n_removed_tags = n_removed_tags
        self.n_removed_addresses = n_removed_addresses

    def _filter_rep_empty(self, xs: Iterable[str]) -> Iterator[str]:
        n_lines_empty    = 0
        n_lines_rep      = 0
        prev: Optional[str] = None
        for x in xs:
            # In the previous step we have replaced addresses with ' ', so we consider
            # whitespace-only lines empty (which would be reasonable anyway).
            if RE_WHITESPACE_ONLY.match(x):
                n_lines_empty += 1
                continue
            if x == prev:
                n_lines_rep += 1
                continue
            prev = x
            yield x
        self.n_lines_empty = n_lines_empty
        self.n_lines_rep = n_lines_rep

    def __call__(self, lines: Iterable[str]) -> Iterator[str]:
        # Reset the counts, and set them only if the iterator is exhausted
        # (safety measure)
        if hasattr(self, 'n_removed_tags'):
            del self.n_removed_tags
            del self.n_removed_addresses
            del self.n_lines_empty
            del self.n_lines_rep
            del self.n_lines_nonj
            del self.n_chars
            del self.n_chars_j

        n_lines_nonj = 0
        n_chars = 0
        n_chars_j = 0

        for line in self._filter_rep_empty(self._subs2text(lines)):
            line_n_chars  = len(line)
            # The fastest way to count (non-)Japanese characters in a mostly
            # Japanase text is to remove sequences of Japanese characters using
            # a (compiled) regex:
            line_n_chars_j = line_n_chars - len(RE_JAPANESE_SEQ.sub('', line))
            if not line_n_chars_j:
                n_lines_nonj += 1
                continue
            n_chars += line_n_chars
            n_chars_j += line_n_chars_j
            yield line

        self.n_lines_nonj   = n_lines_nonj
        self.n_chars        = n_chars
        self.n_chars_j      = n_chars_j

    @property
    def japanese_char_frequency_in_filtered_lines(self):
        n = self.n_chars
        return (self.n_chars_j / n) if n else 0


def dir_files(path: str) -> Iterator[tuple[str, str]]:
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file.endswith(DATA_SUFFIX):  # skip garbage, e.g. .DS_Store on macOS
                yield root, file


@contextmanager
def get_files_contents(path: str, storage: Storage):
    try:
        if storage.zip_compression is not None:
            zf = ZipFile(path + '.zip', 'r')
            files = zf.namelist()

            def iter_contents(max_files=None):
                for file in files[:max_files]:
                    yield zf.read(file).decode('utf-8')

        else:
            dfs = list(dir_files(path))
            files = [f for _d, f in dfs]

            def iter_contents(max_files=None):
                for directory, file in dfs[:max_files]:
                    with open(os.path.join(directory, file)) as f:
                        yield f.read()

        yield (files, iter_contents)
    finally:
        if storage.zip_compression is not None:
            zf.close()


def do_clean(storage: Storage, limit: Optional[int]) -> None:

    if not os.path.exists(FT_LID_MODEL_PATH):
        urlretrieve(FT_LID_MODEL_URL, filename=FT_LID_MODEL_PATH)

    lid_model = fasttext.load_model(FT_LID_MODEL_PATH)

    dfs = list(dir_files(DATA_PATH))[:limit]

    n_total = len(dfs)
    n_short = 0
    n_nonjc = 0
    n_lid   = 0
    n_valid = 0
    n_valid_removed_tags = 0
    n_valid_removed_addresses = 0
    n_valid_lines_empty = 0
    n_valid_lines_rep = 0
    n_valid_lines_nonj = 0
    n_valid_lines_valid = 0

    cleaner = SubCleaner()

    with get_write_file(CLEAN_PATH, storage) as write_file:
        for directory, file in tqdm(
            desc='Cleaning',
            iterable=dfs
            ):
            # clean each line, and remove repeated or empty lines
            with open(os.path.join(directory, file)) as f:
                lines = list(cleaner(f))

            n_lines = len(lines)

            # Exclude short files:
            if n_lines < MIN_LINES:
                n_short += 1
                continue

            # Filter by Japanese char frequency:
            if cleaner.japanese_char_frequency_in_filtered_lines < MIN_JA_CHARS_FREQ:
                n_nonjc += 1
                continue

            # Filter by Japanese line frequency:
            # Deciding on Japanese-labeled line frequency is actually stricter than
            # deciding based on the whole document (which could get a high Japanese
            # probability even though many lines are not Japanese.
            #
            # Additionally, we are ignoring the probabilities.
            # labels, probs = lid_model.predict(lines)
            # ja_probs = [p for l, p in zip(labels, probs) if l == JA_LABEL]
            # ja_prob  = np.concatenate(ja_probs).mean() if ja_probs else 0

            labels, __probs = lid_model.predict(lines)
            ja_labels = [l for l in labels if l == JA_LABEL]

            ja_freq = len(ja_labels) / len(labels)

            if ja_freq < MIN_JA_FREQ:
                n_lid += 1
                continue

            n_valid_removed_tags        += cleaner.n_removed_tags
            n_valid_removed_addresses   += cleaner.n_removed_addresses
            n_valid_lines_empty         += cleaner.n_lines_empty
            n_valid_lines_rep           += cleaner.n_lines_rep
            n_valid_lines_nonj          += cleaner.n_lines_nonj
            n_valid_lines_valid         += n_lines

            text = '\n'.join(chain(lines, ('',)))   # adds trailing \n
            write_file(file, text)
            n_valid += 1

    n_valid_lines_total = (
        n_valid_lines_empty + n_valid_lines_rep +
        n_valid_lines_nonj + n_valid_lines_valid
        )

    print('Cleaning stats:')
    print('* files (determined after sequence removal and line filtering):')
    print(f'  {n_total} total')
    print(f'  {n_short} too short (<{MIN_LINES} lines)')
    print(f'  {n_nonjc} not enough J. characters (<{MIN_JA_CHARS_FREQ} characters)')
    print(f'  {n_lid} not enough J. lines (<{MIN_JA_FREQ} identified as Japanese)')
    print(f'  {n_valid} valid files after cleaning')
    print('* sequences removed from valid files:')
    print(f'  {n_valid_removed_tags} tags')
    print(f'  {n_valid_removed_addresses} addresses')
    print('* lines in valid files:')
    print(f'  {n_valid_lines_total} total lines')
    print(f'  {n_valid_lines_empty} whitespace-only lines')
    print(f'  {n_valid_lines_rep} repeated lines')
    print(f'  {n_valid_lines_nonj} lines composed of non-Japanese characters')
    print(f'  {n_valid_lines_valid} valid lines after cleaning')
    print()


def do_unique(
    storage: Storage,
    limit: Optional[int],
    tokenize: Tokenizer,
    ngram_range: tuple[int, int],
    max_matching: bool = False
    ) -> None:
    with \
        get_write_file(UNIQUE_PATH, storage) as write_file, \
        get_files_contents(CLEAN_PATH, storage) as files_contents:
        files, iter_contents = files_contents

        tfidf = TfidfVectorizer(
            tokenizer=tokenize,
            ngram_range=ngram_range
            ).fit_transform(
            tqdm(
                desc='Building TF-IDF',
                iterable=iter_contents(limit),
                total=len(files[:limit])
                )
            )

        # Cosine similarity:
        # tf-idf is normalized so we only need to compute the mutual dot-product
        # of the vectors, i.e. linear kernel:
        sim = linear_kernel_piecewise(
            tfidf,
            wrapper=lambda pcs: tqdm(desc='Computing similarity', iterable=pcs)
            )
        sim = np.tril(sim, -1)  # under diagonal only

        # print('Similarity histogram:')
        # print(np.histogram(sim, bins=20))
        # print()

        dup = np.tril(sim, -1) >= SIMILARITY_LIMIT
        dup_is, dup_lower_js = np.where(dup)
        dup_indices = set()

        # In practice, it is rarely the case, but is generally not necessary to
        # remove all the duplicates.
        # E.g. assume the pairs (1, 2) and (2, 3), but not (1, 3) are duplicates.
        # After removing 2, it is no longer necessary to remove 3.)
        # We have such "removed" duplicates in `dup_indices`.
        for i, ijs in groupby(
            zip(dup_is, dup_lower_js),
            lambda ij: ij[0]
            ):
            # Note: all(j<i for j in js) because of np.tril()
            if any((j not in dup_indices) for _, j in ijs):
                # Similar to a j<i that hasn't been already removed
                dup_indices.add(i)
            # else:
            # We have already removed all j<i to which i is similar.
            # Thus we can keep i.

        # Creating a minimum list of duplicates = Minimum Vortex Cover (NP hard).
        # https://en.wikipedia.org/wiki/Vertex_cover
        #
        # In theory going vertex-by-vertex as we do, could result in a larger cover than
        # the well-known maximum matching 2-approximation, but for our data the cover is
        # actually smaller. More importantly vertex-by-vertex approach guarantees that
        # we keep (i.e. do not add into cover/`dup_indices`) at least one node from
        # each conneted subgraph.
        #
        # For our data, the maximmum matching 2-approximation (code below) removes
        # 2318 documents instead of just 1686.
        #
        # TODO use something that combines the virtues of both?
        #
        # for i, ijs in groupby(
        #     zip(dup_is, dup_lower_js),
        #     lambda ij: ij[0]
        #     ):
        #         for _, j in ijs:
        #             if j not in dup_indices:
        #                 dup_indices.add(i)
        #                 dup_indices.add(j)
        #                 break

        print(f'Duplicate (similarity >= {SIMILARITY_LIMIT}) stats:')
        print(f'  {len(files[:limit])} total')
        print(f'  {len(dup_indices)} duplicates removed')
        print(f'  {len(files[:limit])-len(dup_indices)} valid files')
        print()

        sorted_dup_indices = sorted(dup_indices)

        # To list duplicate pairs and their similarities:
        # for i, j in zip(dup_is, dup_lower_js):
        #     print(files[i], files[j], sim[i,j])

        # To check that there are no duplicates now:
        # no_dups = np.delete(
        #     np.delete(dup, sorted_dup_indices, axis=0),
        #     sorted_dup_indices, axis=1
        #     )
        # test_dup_i = np.where(no_dups)[0]
        # assert len(test_dup_i)==0, test_dup_i

        iter_dup = iter(sorted_dup_indices)
        next_dup = next(iter_dup, None)
        for i, (file,  text) in tqdm(
            desc='Copying valid files',
            iterable=enumerate(zip(files[:limit], iter_contents(limit))),
            total=len(files[:limit])
            ):
            if i == next_dup:
                next_dup = next(iter_dup, None)
            else:
                write_file(file, text)


def do_frequencies(
    storage: Storage,
    limit: Optional[int],
    tokenize: Tokenizer,
    path: Optional[str],
    norm_path: Optional[str],
    channel_stats_path: Optional[str],
    min_videos: int,
    min_channels: int
    ) -> None:

    all_subtitles = pd.read_csv(
        SUBLIST_PATH,
        index_col='videoid',
        na_filter=False  # keep empty channelids as empty strings
        )
    # Keep manual only:
    subtitles = all_subtitles[all_subtitles['sub']]
    # Remove duplicates (pairs where ['auto']==True, ['auto']==False -- we don't care):
    subtitles = subtitles[~subtitles.index.duplicated()]

    channel_ids = subtitles['channelid']
    ch2n = Counter(channel_ids)
    n_no_channel = ch2n.pop('')
    n2chn = Counter(ch2n.values())
    n_channels_and_no_channels = len(ch2n) + n_no_channel

    with open(channel_stats_path or DEFAULT_CHANNEL_STATS_PATH, 'wt') as f:
        f.write('videos_in_channel\tchannels\n')
        for n, chn in sorted(n2chn.items()):
            f.write(f'{n}\t{chn}\n')
        f.write(f'NO_CHANNEL_ID\t{n_no_channel}\n')

    # Originally based on gather_wordfreq.py

    word_count: Counter[str, int] = Counter()
    word_videos: dict[str, set[int]] = defaultdict(set)
    word_channels: dict[str, set[Union[int, str]]] = defaultdict(set)

    norm_word_count: Counter[str, int] = Counter()
    norm_word_videos: dict[str, set[int]] = defaultdict(set)
    norm_word_channels: dict[str, set[Union[int, str]]] = defaultdict(set)

    n_words = 0

    with get_files_contents(UNIQUE_PATH, storage) as files_contents:
        files, iter_contents = files_contents
        n_videos = len(files[:limit])
        for video_no, (file, text) in tqdm(
            desc='Computing frequencies',
            iterable=enumerate(zip(files[:limit], iter_contents(limit))),
            total=n_videos
            ):
            video_id = file.removesuffix(DATA_SUFFIX)
            # Videos without a channel id are counted as unique 1-video channels:
            channel_id = channel_ids.loc[video_id] or video_no
            words = tokenize(text)
            for w in words:
                word_count[w] += 1
                word_videos[w].add(video_no)
                word_channels[w].add(channel_id)

                nw = w.lower().translate(LCASE_FW2HW)    # normalize
                norm_word_count[nw] += 1
                norm_word_videos[nw].add(video_no)
                norm_word_channels[nw].add(channel_id)

                n_words += 1  # count before removing low-freq words

        for word, videos in word_videos.items():
            if len(videos) < min_videos:
                del word_count[word]

        for word, videos in norm_word_videos.items():
            if len(videos) < min_videos:
                del norm_word_count[word]

    line_format = '%s\t%d\t%d\t%d\n'
    with storage.open(path or (DEFAULT_FREQ_PATH + storage.suffix), 'wt') as uf, \
        storage.open(
            norm_path or (DEFAULT_NORM_FREQ_PATH + storage.suffix), 'wt'
            ) as nf:

        for (f, w_count, w_videos, w_channels) in (
            (uf, word_count, word_videos, word_channels),
            (nf, norm_word_count, norm_word_videos, norm_word_channels)
            ):

            words = sorted(w_count, key=w_count.__getitem__, reverse=True)

            f.write('word\tcount\tvideos\tchannels\n')
            for word in words:
                wc = w_count[word]
                f.write(
                    line_format % (
                        word, wc, len(w_videos[word]), len(w_channels[word])
                        )
                    )

            f.write(
                line_format % ('[TOTAL]', n_words, n_videos, n_channels_and_no_channels)
                )


def main() -> None:
    args = parse()
    storage = Storage.from_args(args)
    limit = args.limit
    clean = args.clean
    unique = args.unique
    frequencies = args.frequencies
    if not (clean or unique or frequencies):
        clean = True
        unique = True
        frequencies = True

    if clean:
        do_clean(storage, limit=limit)

    if unique or frequencies:
        tagger = tagger_from_args(args)

        def tokenize(s: str) -> list[str]:
            # TfidfVectorizer requires a list of strings:
            return [
                word
                for word in tagger.parse(s).split(' ')
                if is_word_re.match(word) and not not_is_word_re.match(word)
                ]

        if unique:
            nmin: int = args.nmin
            nmax: int = args.nmax
            do_unique(
                storage,
                tokenize=tokenize,
                limit=limit,
                ngram_range=(nmin, nmax)
                )
        if frequencies:
            min_videos: int     = args.min_videos
            min_channels: int   = args.min_channels

            do_frequencies(
                storage,
                tokenize=tokenize,
                limit=limit,
                path=args.output,
                norm_path=args.output_normalized,
                channel_stats_path=args.channel_stats,
                min_videos=min_videos,
                min_channels=min_channels
                )


if __name__ == '__main__':
    main()
