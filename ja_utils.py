'''
Japanese language processing for `tubelex` and `wikipedia-word-frequency`.
'''

import fugashi
import os
from typing import Optional


LCASE_FW2HW: dict[int, int] = dict(zip(
    range(ord('ａ'), ord('ｚ') + 1),
    range(ord('a'), ord('z') + 1)
    ))  # Used with str.translate() after lowercasing


def fugashi_tagger(dicdir: Optional[str]) -> fugashi.GenericTagger:
    if dicdir is None:
        return fugashi.Tagger('-O wakati')  # -d/-r supplied automatically
    # GenericTagger: we do not supply wrapper (not needed wor -O wakati)
    mecabrc = os.path.join(dicdir, 'mecabrc')
    return fugashi.GenericTagger(f'-O wakati -d {dicdir} -r {mecabrc}')
