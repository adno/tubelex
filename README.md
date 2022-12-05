# About TUBELEX-JA


Inspired by the SUBTLEX word lists, TUBELEX-JA is a large word list based on Japanese subtitles for YouTube videos.

The project consists mainly of:

- [tubelex.py](tubelex.py): a script to create the word list,
- [tubelex-ja.tsv.xz](results/tubelex-ja.tsv.xz): the word list (not normalized),
- [tubelex-ja-lower.tsv.xz](results/tubelex-ja-lower.tsv.xz): normalized version of the word list (alphabet is normalized to lower case and half-width).

The words were segmented with MeCab using Unidic 3.1.0. Words that contain decimal digits (except kanji characters for numbers), words that start or end with a non-word character (e.g. punctuation) were ignored.

In the raw version, alphabet may be upper- or lower-case, full-width or ordinary, i.e. half-width. In the normalized version alphabet is always lower-case and half-width. This concerns not only letters A-Z, but also accented characters and letters of any cased alphabet (e.g. Ω is normalize to ω).

For each word we count:
- number of occurrences,
- number of videos, and
- number of channels.

For a small number of videos there is no channel information, so we count them as separate single-video channels. Words occurring in less than 3 videos are not included.

The list is sorted by number of occurrences and contains totals on the last row labeled `TOTAL`. Note that totals are not sums of the previous rows' values.

The form is thus similar, yet slightly different from [wikipedia-word-frequency](https://github.com/notani/wikipedia-word-frequency). Notable differences:
- TUBELEX-JA counts also videos and channels that words appear in (WWF counts only occurrences),
- TUBELEX-JA has both a raw version and a normalized version (WWF lowercases words),
- TUBELEX-JA includes totals on the trailing row (WWF does not),
- TUBELEX-JA data is tab-separated with header (WWF data is space-separated without header),
- TUBELEX-JA does proper segmentation of Japanese.

As a basis for the corpus we used manual subtitles listed in the file `data/ja/202103.csv` from the [JTubeSpeech](https://github.com/sarulab-speech/jtubespeech) repository that were still available as of as of 30 November 2022. (The script for downloading is also part of that repository.) The download subtitles were then processed using the [tubelex.py](tubelex.py) according to the following steps:

1. Extract lines of subtitles and convert HTML (e.g. &amp;) entities to characters.

2. Remove the following text sequences:
  - formatting tags,
  - addresses (http(s), e-mail, domain names starting with `www.`, social network handles starting with `@`).

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
  
Near duplicates are files with cosine similarity >= 0.95 between their 1-gram TF-IDF vectors. We make a reasonable effort to minimize the number of duplicates removed. (Minimum vertex cover is NP-hard.) See the source code for more details on this and other points.

Note that the script saves intermediate files after cleaning and removing duplicates, and has several options (see `python tubelex.py --help`).

## Cleaning statistics (steps 2-4):

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

## Duplicate removal statistics (step 5):
  - 74332 total
  - 1686 duplicates to remove
  - 72646 unique files

# Usage

Note that the output of the script is already included in the repository. But you can reproduce it by following the steps below. Results will vary based on the YouTube videos/subtitles still available for download.

1. Install the Git submodule for JTubeSpeech:

    ```git submodule init && git submodule update```
    
2. Install requirements for both tubelex (see [requirements.txt](requirements.txt)) and JTubeSpeech.

3. Optionally, modify JTubeSpeech's download script to download only subtitles without video, and/or adjust the delay between individual downloads. (TODO fork and publish.)

4. Download manual subtitles using the download script:

    ```cd jtubespeech; python scripts/download_video.py ja data/ja/202103.csv; cd ..```

5. Clean, remove duplicates and compute frequencies saving output with LZMA compression in the current directory:
    
    ```python tubelex.py -x```

6. Alternatively consult the help and process the files as you see fit:

    ```python tubelex.py --help```
    
7. Optionally remove the language identification model, intermediate files and the downloaded subtitles to save disk space:

    ```rm *.ftz *.zip; rm -r jtubespeech/video```

# Results

