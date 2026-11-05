# Stimuli

## Single images

```python
from corread import generate_image

img = generate_image('house', 'DejaVuSans', size=60, xshift=-20, upper=True)   # a PIL image
img.save('HOUSE.png')
```

The text is drawn in black on a white canvas (500 × 500 by default), centred and then shifted by `(xshift, yshift)` pixels. `mirror` flips the image left to right and `invert` reverses the order of the letters.

## Word sets

A word set holds `num_train` training and `num_val` validation images of each word, in ImageNet-style folders `train/<word>/` and `val/<word>/`:

```python
import corread

corread.generate_training_set(
    'english',                       # its name: the folder in the wordsets folder
    'english',                       # the wordlist, wordlists/english.txt
    train_fonts=['DejaVuSans', 'LiberationSerif'],
    val_fonts=['NotoSans'],          # a font the training images never use
    num_words=1000, num_train=1000, num_val=50,
    seed=1, workers=16,
)
```

Each image draws, uniformly at random, its font (from the fonts of its split), its size (40, 50, 60, 70 or 80), whether it is uppercase, and a shift of up to ±50 px horizontally and ±30 px vertically.

### Reproducibility

All random choices are drawn up front, from one seed, in a fixed order, so a set depends only on its seed, never on the number of worker processes. Next to the images, the set records its **recipe**:

| File | Holds |
|---|---|
| `metadata.json` | the arguments, the seed used, the rendering settings, the words, the sha256 of the wordlist and of every font, and the versions of CORread, Python and Pillow |
| `images.csv` | the font, size, case and shift drawn for every image |
| `checksums.json` | the sha256 digest of every word folder, and one over the whole set |

`regenerate_training_set` draws a set again from its recipe. It checks the fonts and wordlist against the recorded hashes, the draws against `images.csv` before drawing anything, and every word folder against `checksums.json` afterwards:

```python
corread.regenerate_training_set('english', recipe='path/to/english', wordsetsdir='elsewhere')
```

The images are byte-identical as long as Pillow is the same version: Pillow bundles the FreeType and libjpeg that render the text and encode the JPEGs. With another version the check names the word folders that differ; `allow_mismatch=True` keeps the images anyway and records `verified: false` in `regeneration.json`.

Trained models record the sha256 of their word set's `metadata.json` and the digest of its images, so every model can be traced back to the exact images it saw. Recipes can be shared and regenerated through the [Hugging Face Hub](hub.md).
