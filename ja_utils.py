'''
Japanese language processing for `tubelex` and `wikipedia-word-frequency-clean`.
'''

import fugashi
import os
from typing import Optional
import argparse
import re

# Word matching (not just) for Japanese:
#
# Match words of len>=1. No decimal digits (\d) at any position.
# First and last character must be word-forming (\w), i.e. alphabet, CJK, etc.
#
# Note: \w includes accented chars, CJK, etc.
# \d are decimals in many scripts, but not CJK.

RE_WORD = re.compile(r'^(?!\d)\w([^\d]*\w)?(?<!\d)$')

# Examples (test):
assert all(RE_WORD.match(w) for w in ['a', '亀', 'コアラ', 'Pú', 'A/B', 'bla-bla'])
assert not any(RE_WORD.match(w) for w in ['', '1', 'a1', '1a', 'C3PIO', '/', '-'])

LCASE_FW2HW: dict[int, int] = dict(zip(
    range(ord('ａ'), ord('ｚ') + 1),      # full-width
    range(ord('a'), ord('z') + 1)       # half-width
    ))  # Used with str.translate() after lowercasing


def fugashi_tagger(dicdir: Optional[str]) -> fugashi.GenericTagger:
    if dicdir is None:
        return fugashi.Tagger('-O wakati')  # -d/-r supplied automatically
    # GenericTagger: we do not supply wrapper (not needed wor -O wakati)
    mecabrc = os.path.join(dicdir, 'mecabrc')
    return fugashi.GenericTagger(f'-O wakati -d {dicdir} -r {mecabrc}')


def tagger_from_args(args: argparse.Namespace) -> fugashi.GenericTagger:
    # We always specify dicdir EXPLICITLY
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

