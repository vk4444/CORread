# Lesions

Which dense-layer units does a literate network read with, and can it learn to read again without them? `LesionInstance` measures how word-selective each unit is, lesions chosen units, and lets the network recover while tracking every unit. Lesions are implemented for CORnet-Z.

## Word selectivity

For each of the 512 dense-layer units *u*:

> selectivity[*u*] = (mean activation to words − mean activation to images) / SD of its activations to images

The words are all validation word images of the model's word set; the images are the *selectivity sample*, `selectivity_images` ImageNet training images drawn with `selectivity_seed`, all through the fixed validation view. A unit is word-selective when its selectivity is at least `threshold` (3 by default).

```python
from corread import LesionInstance

s = LesionInstance.word_selectivity('english/lit_bias_z_79')
s['selective_units']      # the word-selective units, most selective first
s['selectivity']          # the value of every unit
s['word_channel']         # (463, 511): the units the biased network's word outputs read
```

## Lesion and recovery

```python
LesionInstance('english/lit_bias_z_79', units=s['selective_units'], max_epochs=10).train()
```

A lesion re-initialises each unit's incoming weights (its IT convolution filter and bias) once, as PyTorch initialises an untrained layer; afterwards they train like every other weight, so a unit can learn a new role. Options:

- `reset_outgoing=True` also re-initialises the unit's weights in the readout.
- `silent_epochs=K` first silences the units for *K* epochs (their output is multiplied by 0, so they neither drive the readout nor learn) and re-initialises them when they are released. `silent_epochs=max_epochs` makes the lesion permanent.
- `units=[]` trains the same way without a lesion: the control for the extra training.

The recovery trains the way the model's literate training did: same data, transforms, validation and SGD settings (unless you override `batch_size`, `lr`, `momentum` or `weight_decay`). The run goes into a folder next to the lesioned model, `lesion_<mode>_z_<n>u_<last epoch>`, and its epochs continue the model's numbering.

## What is measured

Before the first recovery epoch the run validates the intact model, lesions it, and validates the lesioned model; every epoch then trains and validates. Each of these validations also *measures* every unit: its mean activation to the validation words and images, and its mean, SD and word selectivity on the selectivity sample.

- `metadata.json`: `intact_validation`, `initial_validation` (the lesioned model) and `history`, each with the usual loss and accuracy and a summary of the units: the mean word activation of the lesioned and of the other units, and how many of each are word-selective. The `lesion` section records the units, the method, and the norm of the lesioned filters before and after.
- `unit_tracking.npz`: one row per measurement (intact, lesioned, then each epoch) and one column per unit, for `mean_words`, `mean_val_images`, `mean_images`, `sd_images`, `selectivity` and `word_selective`.

Together they show whether the lesioned units recover their selectivity, or whether other units take over reading.
