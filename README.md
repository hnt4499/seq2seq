# seq2seq
Implementation of RNN Encoder-Decoder Networks from (almost) scratch. The following papers are (nicely) implemented:
1. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks*. https://arxiv.org/abs/1409.3215
2. Bahdanau, D., Cho, K., & Bengio, Y. (2016). *Neural Machine Translation by Jointly Learning to Align and Translate*. http://arxiv.org/abs/1409.0473
3. Luong, M.-T., Pham, H., & Manning, C. D. (2015). *Effective Approaches to Attention-based Neural Machine Translation*. http://arxiv.org/abs/1508.04025

## Overview
The package was developed as a side project when I was working at NTU research group at NTU ([NTU@NLP](https://ntunlpsg.github.io/)) during my winter break as a way to get familiar with machine translation.
Therefore, `seq2seq` was chosen as a simple approach (and somewhat "quick", too, although it turned out to be more time-consuming than state-of-the-art transformer models).

## Features
The package includes:
1. Configuration file: highly flexible, configurable training and evaluation settings. Default yaml config files are included in the package as well.
2. Three implemented models: `seq2seq` architectures with/without attention mechanisms.
3. Beam search (adapted from `pytorch/fairseq`).
4. Tokenized/detokenized BLEU evaluations.
5. Pre-processing, train and evaluation scripts.
And more features to come...

## Dependencies
```
torch==1.5.1
loguru==0.5.3
pandas==1.1.5
pyyaml==5.3.1
sacrebleu==1.4.14
sacremoses==0.0.43
tqdm==4.54.1
```

**Note**: 
1. Using other versions of `torch` than `1.5.1` is not recommended as (underlying) errors might occur. Other packages can be installed with different versions as stated.
2. The above-mentioned packages have been included in `requirements.txt`. You can execute `pip install -r requirements.txt` prior to installing the `seq2seq` package instead of installing one by one manually.

## Installation

Build the project from source in development mode and allow the changes in source code to take effect immediately:
```
git clone https://github.com/hnt4499/seq2seq
cd seq2seq/
pip install -e .
```

## Usage
1. First, pre-process the data, including normalizing punctuations, removing non-printing characters, tokenizing and removing sentence pairs exceeding some source/target ratio. See also https://github.com/hnt4499/fairseq/blob/master/examples/translation/prepare-wmt14en2fr.sh. Note that the final bitexts should be a csv-type file with two columns: `source` for source langauge and `target` for target langauge.
2. Next, learn a full vocabulary using the script from `scripts/build_vocab.py`. The resulting csv files consist of words and their counts in bitexts, sorted descendingly. Example usage:
```
python scripts/build_vocab.py -f /path/to/bitext1 /path/to/bitext2 -s /path/to/save/source/vocab.csv -r /path/to/save/target/vocab.csv --show-progress
```
3. Learn a truncated vocabulary from the full vocabulary obtained from step 2 using the script from `scripts/build_truncated_vocab.py`. The script will truncate the `vocab_size` most frequently occured words and create dictionaries mapping tokens (words) to their respective index (and vice versa). Note that the four special tokens, `<pad>`, `<unk>`, `<sos>`, and `<eos>`, are included in the final vocab as well. Example usage:
```
python scripts/build_truncated_vocab.py -f /path/to/full/source/vocab.csv -s /path/to/save/truncated/source/vocab.json --vocab-size 30000
python scripts/build_truncated_vocab.py -f /path/to/full/target/vocab.csv -s /path/to/save/truncated/target/vocab.json --vocab-size 30000
```
4. Train and evaluate a model using the script `scripts/train_and_evaluate.py`. We will need to specify the path to a config file. Example config files have been provided in `work_dirs`:
```
python scripts/train_and_evaluate.py -c work_dirs/config.yaml
```
Additionally, we can use the argument `--load-from` to load state dict of a pretrained model, or even use the argument `--resume-from` to resume from an existing working directory (in case the training was interrupted).
Refer to the example config file `work_dirs/config.yaml` for more info on available train/eval settings.
