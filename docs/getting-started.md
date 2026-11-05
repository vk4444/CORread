# Getting started

## Install

```bash
pip install "corread @ git+https://github.com/vk4444/CORread"
```

Optional extras: `hub` (sharing through the Hugging Face Hub), `imagenet` (reading the ImageNet devkit), `dev` (tests, linting and these docs). CORread needs Python ≥ 3.10 and PyTorch ≥ 2.6; install the PyTorch build for your GPU first if the default one from PyPI doesn't suit it.

## Configure: `paths.toml`

CORread reads its folders from a `paths.toml` file: the one named by the `CORREAD_PATHS` environment variable, or else the first one found in the working directory or its parents. Relative paths are relative to the file.

```toml
fonts     = "data/fonts"                # <name>.ttf
wordlists = "data/wordlists"            # <name>.txt, one word per line
imagesets = "data/imagenet"             # ImageNet-style train/<class>/ and val/<class>/
wordsets  = "data/wordsets"             # generated word sets
models    = "models"                    # trained models

# optional: sharing through the Hugging Face Hub (see "Sharing")
# [hub]
# org = "my-lab"
# private = true

# optional: live status reports to your own API (see "Reporting API")
# [report]
# url = "https://example.org/api/runs"
# every = 60
```

[`examples/paths.example.toml`](https://github.com/vk4444/CORread/blob/main/examples/paths.example.toml) is a template. Keep your `paths.toml` out of version control; tokens never go in it.

## Provide the data

**Fonts.** Put `.ttf` files in the fonts folder and refer to them by file name without `.ttf`. CORread does not distribute fonts, since most are licensed; any TrueType font works, for example the open [DejaVu](https://dejavu-fonts.github.io), [Liberation](https://github.com/liberationfonts) or [Noto](https://fonts.google.com/noto) fonts.

**A wordlist.** A text file in the wordlists folder with one word per line; a word set uses its first `num_words` words. Every word becomes a folder name, so words can't contain `/` or `\`.

**ImageNet.** Download `ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar` and `ILSVRC2012_devkit_t12.tar.gz` from <https://image-net.org> (under its terms of access) into the imagesets folder, then:

```python
import corread
corread.prepare_imagenet()   # needs corread[imagenet]; about 150 GB, used up from the archives as it unpacks
```

It unpacks the archives into `train/<wnid>/` and `val/<wnid>/`, deleting each archive as it goes so the disk never holds two copies (`keep_archives=True` keeps them), and continues where it stopped if interrupted.

## Run

```python
import logging
import corread

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')

corread.generate_training_set('english', 'english', train_fonts=['DejaVuSans'], val_fonts=['DejaVuSerif'],
                              workers=16)
corread.FamilyTrainingInstance(model_choice='z', wordset='english').train()
```

CORread logs through Python's `logging` module and shows progress bars for long loops; configure logging as above to see its messages. Training uses every GPU it finds unless you pass `gpus` (a number, a list of indices, or `0` for the CPU).

See the [complete examples](https://github.com/vk4444/CORread/tree/main/examples) for scripts that generate a word set, train a family, and run a lesion experiment.
