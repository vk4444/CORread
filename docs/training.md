# Training

## The modes

Training follows Hannagan et al. (2021). A CORnet is first trained to recognize the 1000 ImageNet classes, then either keeps training on images alone or learns to read: it trains on the images and on a word set together, with one output per word added after the image outputs.

| Mode | Starts from | Trains on | Network |
|---|---|---|---|
| `pre` | random weights | images | `CORnetZ` / `CORnetS` |
| `illit` | the `pre` model | images, `max_epochs_lit` more epochs | the same network |
| `lit_bias` | the `pre` model | images and words | `CORnet*BiasedWords`: the word outputs read only the last 49 of the 512 dense-layer units |
| `lit_no_bias` | the `pre` model | images and words | `CORnet*NonbiasedWords`: the word outputs read every unit |

The dense layer is IT, the last area, averaged over space: 512 units that the readout reads. The literate networks take over the pre-literate network's areas and its image readout, so they start as the pre-literate model plus untrained word outputs. `illit` is the control for the extra training time.

```python
from corread import TrainingInstance

TrainingInstance(model_choice='z', mode='pre').train()
TrainingInstance(model_choice='z', mode='lit_bias', wordset='english').train()
```

`FamilyTrainingInstance` trains all four in order, skipping any model already trained with the same arguments, so it can be run again after an interruption, or for another word set, which reuses the pre-literate and illiterate models:

```python
from corread import FamilyTrainingInstance

FamilyTrainingInstance(model_choice='s', wordset='english', gpus=[0, 1]).train()
```

By default a model trains 50 pre-literate and 30 later epochs, SGD with learning rate 0.01, momentum 0.9, weight decay 1e-4 and batches of 100, with no learning-rate schedule. Training images are random resized crops; validation uses the standard ImageNet evaluation view (resize to 256, central 224-pixel crop), and word images are shown whole.

## Where models go

```
models/
  pre/pre_z_49/                  # pre: <name or pre_<model>_<last epoch>>
  illiterate/illit_z_79/         # illit
  english/lit_bias_z_79/         # literate: <name or word set>/<mode>_<model>_<last epoch>
  english/lit_no_bias_z_79/
```

Epochs are counted from 0 over the pre-literate epochs and then the later ones, so a literate model after 50 pre-literate epochs trains epochs 50 to 79. Each folder holds:

| File | Holds |
|---|---|
| `<name>.pth.tar` | the finished model: `epoch`, `state_dict`, `optimizer`, scores and losses |
| `checkpoints/epoch_NN.pth.tar` | a checkpoint after every epoch, with the random state |
| `metadata.json` | everything about the run (below) |
| `cat_scores.npy` | accuracy per epoch and validation batch, continuing the pre-literate model's |
| `trainloss.npy`, `valloss.npy` | the loss of every batch |
| `initial_cat_scores.npy`, `initial_valloss.npy` | the same for the starting weights |
| `training_eta_log.txt` | time per epoch and estimates of the time left |

Load a finished model with:

```python
import torch
from corread.models import CORnetZBiasedWords

net = CORnetZBiasedWords()
net.load_state_dict(torch.load('models/english/lit_bias_z_79/lit_bias_z_79.pth.tar', weights_only=True)['state_dict'])
v1, v2, v4, it, dense, outputs = net(images)   # every network returns all its activations
```

## What `metadata.json` records

- the arguments, the seed used, and how to reproduce the run;
- the model it started from, with the sha256 of its file and metadata;
- the data: size, classes and transforms of the image and word sets, and for a word set its recipe hash and image digest;
- the optimizer and loss;
- the code (CORread version, a hash of its source, the git commit and whether it had uncommitted changes) and the environment (Python, PyTorch, CUDA, GPUs);
- the validation of the starting weights (`initial_validation`) and, for every epoch, its training loss and accuracy and the validation after it (`history`), apart for images and words;
- a description of every file and value, so the file explains itself.

The starting weights are validated once before the first epoch: chance for `pre`, and the pre-literate model for the others. Every epoch then trains and validates the resulting weights, so the last history entry is the finished model.

## Interrupting and resuming

Ctrl+C stops a run cleanly. If no epoch has finished, the folder is removed; otherwise it is marked `interrupted`, and the same call with `resume=True` continues from the last checkpoint, with the random state restored. Arguments that change the result can't change on resume.

On the CPU, with `torch.use_deterministic_algorithms(True)`, the same seed gives the same run, interrupted and resumed or not. GPU runs are not bit-for-bit repeatable, as cuDNN picks the fastest algorithms.

## Optional: export and reporting

- `export=True` uploads the run to your Hugging Face organisation as it trains: see [Sharing](hub.md).
- `report=True` sends its status to your own API: see [Reporting API](reporting.md). `report_status()` sends one snapshot on demand.
