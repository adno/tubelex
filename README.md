# About TUBELEX-JA


Inspired by the SUBTLEX word lists, TUBELEX-JA is a large word list based on Japanese subtitles for YouTube videos (93M tokens from 72k subtitle files).

The project consists mainly of:

- [tubelex.py](tubelex.py): a script to create the word list,
- [tubelex-ja.tsv.xz](results/tubelex-ja.tsv.xz): the word list (Unidic Lite, not normalized),
- [tubelex-ja-lower.tsv.xz](results/tubelex-ja-lower.tsv.xz): normalized version of the word list above.
- [tubelex-ja-310.tsv.xz](results/tubelex-ja-310.tsv.xz): the word list (Unidic 3.1.0, not normalized),
- [tubelex-ja-310-lower.tsv.xz](results/tubelex-ja-310-lower.tsv.xz): normalized version of the word list above.

The word list has four versions differing in segmentation and normalization. Words are segmented with MeCab using [Unidic Lite](https://github.com/polm/unidic-lite), or alternatively with Unidic 3.1.0. Words that contain decimal digits (except kanji characters for numbers) and words that start or end with a non-word character (e.g. punctuation) are ignored in both cases.

Note that Unidic Lite is often used for Japanese segmentation in Python due to its ease of installation (see  [Fugashi](https://pypi.org/project/fugashi/) for more info). Unidic Lite is also used for tokenization of [a commonly used Japanese BERT model](https://huggingface.co/cl-tohoku/bert-base-japanese-v2). That said, Unidic 3.1.0 is of course larger and more up to date.

In the raw version, letters may be upper- or lower-case, full-width or ordinary, i.e. half-width. In the normalized version, letters are lower-case and half-width. The case normalization concerns not only the letters A-Z but also accented characters and letters of any cased alphabet (e.g. Ω is normalized to ω).

For each word, we count:
- number of occurrences,
- number of videos,
- number of channels.

For a small number of videos, there is no channel information, so we count them as separate single-video channels. Words occurring in less than 3 videos are not included. The list is sorted by the number of occurrences and contains totals on the last row labeled `[TOTAL]`. Note that totals are not sums of the previous rows' values. The data is tab-separated with a header, and the file is compressed with LZMA2 (`xz`).

As a basis for the corpus, we used manual subtitles listed in the file `data/ja/202103.csv` from the [JTubeSpeech](https://github.com/sarulab-speech/jtubespeech) repository that were still available as of 30 November 2022. (The script for downloading is also part of that repository.) The download subtitles were then processed using the [tubelex.py](tubelex.py) script according to the following steps:

1. Extract lines of subtitles and convert HTML (e.g. `&amp;`) entities to characters.

2. Remove the following text sequences:
  - formatting tags,
  - addresses (`http`(`s`), e-mail, domain names starting with `www.`, social network handles starting with `@`).

3. Remove the following lines:
  - empty lines,
  - lines repeating the previous line,
  - lines composed entirely of non-Japanese characters,

4. Remove the following files:
  - < 3 lines,
  - < 70 % Japanese characters,
  - < 95 % lines identified as Japanese language using a FastText model,

5. Remove near-duplicates of other files.

6. Create the word list (both raw and normalized) as described initially.
  
Near-duplicates are files with cosine similarity >= 0.95 between their 1-gram TF-IDF vectors. We make a reasonable effort to minimize the number of duplicates removed. See the source code for more details on this and other points. For consistency, we have used Unidic Lite for building TF-IDF regardless of the final segmentation used for frequency counting.

Note that the script saves intermediate files after cleaning and removing duplicates, and has various options we do not describe here (see `python tubelex.py --help`).

# Usage

Note that the output of the script is already included in the repository. You can, however, reproduce it by following the steps below. Results will vary based on the YouTube videos/subtitles still available for download.

1. Install the Git submodule for JTubeSpeech:

    ```git submodule init && git submodule update```
    
2. Install requirements for both tubelex (see [requirements.txt](requirements.txt)) and JTubeSpeech. The `unidic` package (as opposed to `unidic-lite`) requires an additional installation step:
	
	```python -m unidic download```

3. Optionally, modify JTubeSpeech's download script to download only subtitles without videos and/or adjust the delay between individual downloads. (TODO fork and publish.)

4. Download manual subtitles using the download script:

    ```cd jtubespeech; python scripts/download_video.py ja data/ja/202103.csv; cd ..```

5. Clean, remove duplicates and compute frequencies saving output with LZMA compression in the current directory:
    
    ```
    python tubelex.py -x --clean --unique
    python tubelex.py -x --frequencies -o tubelex-ja.tsv.xz -O tubelex-ja-lower.tsv.xz
    python tubelex.py -x --frequencies -D unidic -o tubelex-ja-310.tsv.xz -O tubelex-ja-310-lower.tsv.xz
    ```

6. Alternatively, consult the help and process the files as you see fit:

    ```python tubelex.py --help```
    
7. Optionally remove the language identification model, intermediate files, and the downloaded subtitles to save disk space:

    ```rm *.ftz *.zip; rm -r jtubespeech/video```

# Results

After cleaning and duplicate removal, there are **93,161,375 tokens**. The word list consists of **127,681 words** (124,449 normalized words) occurring in at least 3 videos. (The numbers differ slightly for the Unidic 3.1.0 version.)

## Cleaning statistics (steps 2-4 above):

* files (determined after sequence removal and line filtering):
  - 103887 total
  - 7689 too short (<3 lines)
  - 4925 not enough J. characters (<0.7 characters)
  - 16941 not enough J. lines (<0.95 identified as Japanese using a FastText model)
  - 74332 valid files after cleaning
* sequences removed from valid files:
  - 129280 tags
  - 1659 addresses
* lines in valid files:
  - 8108028 total lines
  - 2004 whitespace-only lines
  - 41200 repeated lines
  - 61780 lines composed of non-Japanese characters
  - 8003044 valid lines after cleaning

## Duplicate removal statistics (step 5 above):
  - 74332 total
  - 1767 duplicates removed
  - 72565 valid files

# Further work and similar lists

The word lists contains only the surface forms of the words (segments). For many purposes, lemmas, POS and other information would be more more useful. We plan to add further processing later.

We have not attempted yet to analyze the corpus/word list, or compare it with word lists based on smaller but more carefully curated corpora of spoken Japanese. The largest corpus of such kind would be [CSJ](https://clrd.ninjal.ac.jp/csj/index.html) (7M tokens, with publicly available [word lists](https://clrd.ninjal.ac.jp/csj/chunagon.html#data)), other, smaller corpora include [CEJC](https://www2.ninjal.ac.jp/conversation/corpus.html), [NUCC](https://mmsrv.ninjal.ac.jp/nucc/), [J-TOCC](http://nakamata.info/database/), and [BTSJ](https://ninjal-usamilab.info/btsj_corpus/).

Note that there is also a large corpus based on TV subtitles [LaboroTVSpeech](https://laboro.ai/activity/column/engineer/eg-laboro-tv-corpus-jp/) (22M tokens), which can be used for free for academic purposes (application necessary).
