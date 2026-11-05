# Sharing through the Hugging Face Hub

Trained models, and the word sets they were trained on, can be shared through a Hugging Face organisation of your own (or your user account). Nothing is shared unless you call these functions or pass `export=True`, and everything goes to the organisation you configure.

ImageNet is never shared; everyone prepares their own copy with `corread.prepare_imagenet()`.

## Setup

1. `pip install "corread[hub]"`.
2. Sign in once with `hf auth login`, using a **write** token from <https://huggingface.co/settings/tokens>, or set `HF_TOKEN`. The token never goes in `paths.toml`.
3. In `paths.toml`:

   ```toml
   [hub]
   org = "my-lab"     # a Hugging Face organisation, or your user name
   private = true     # visibility of the model and word set repos created from here on
   ```

Every call checks the sign-in (and, for uploads, the write role in the organisation) before it starts anything long. To share with a team, create an organisation and give its members the **write** role if they train, or **read** if they only download.

## What is on the Hub

| Repo | Type | Holds |
|---|---|---|
| `<org>/<group>.<name>` | model | The model in `models/<group>/<name>`: `metadata.json` from the first epoch on; once complete also `<name>.pth.tar`, the `.npy` scores and `training_eta_log.txt`. Never the checkpoints. |
| `<org>/<word set>` | dataset | The word set's **recipe**: `metadata.json` (unchanged), `images.csv`, `checksums.json`, `sources.json`. Never the images. |
| `<org>/assets` | dataset | `fonts/<name>.ttf` and `wordlists/<name>.txt`, the inputs of the recipes. **Always private**: fonts are usually licensed for your own use only. |

`<group>` is `pre`, `illiterate` or the word set, so `models/english/lit_bias_z_79` becomes `<org>/english.lit_bias_z_79`.

Every repo gets a generated card (its README). A model's card names the pre-literate model it started from as its `base_model`, and its word set and ImageNet as its `datasets`, so the Hub shows the lineage of every model, and a word set's page lists the models trained on it. Cards show the initial and final validation and the training history, and warn when the code had uncommitted changes when a run started.

`metadata.json` records the absolute paths of your data folders, as on your machine. Keep this in mind before making a model repo public.

## Models

```python
from corread import TrainingInstance, hub

TrainingInstance(mode='pre', model_choice='z', export=True).train()   # uploads as it trains
hub.export_model('pre/pre_z_49')     # upload a model trained without export=True
hub.import_model('pre/pre_z_49')     # download a finished model into models/pre/pre_z_49
hub.list_models()                    # every model in the organisation, with its status and accuracy
```

With `export=True`:

- **Start:** the run creates its repo before the datasets load. That claims the name, so a second run of the same model, here or on another machine, is refused instead of mixing into it. A resumed run continues its own repo.
- **After every epoch:** one commit with `metadata.json` and the card.
- **On completion:** one commit with the model file, scores and log.
- **Network problems:** a failed upload only warns; the next one carries the full metadata.
- **Ctrl+C or an error:** the status is recorded on the Hub. If no epoch finished, the repo is deleted again, as the local folder is.

`import_model` only downloads complete models, and checks the model file against `model_file_sha256` in its metadata. It writes `.hub.json` (repo and revision) into the folder, and a model trained from it records where it came from (`initialised_from.hub`).

Checkpoints are not uploaded, so an unfinished run can only be resumed on the machine that ran it.

## Word sets: recipes, not images

A word set is fully determined by its recipe: the wordlist, the fonts, the seed and the rendering settings. Drawing a million images again takes about a minute on 24 cores; downloading them would mean about 9 GB.

```python
import corread
from corread import hub

corread.generate_training_set('english', 'english', train_fonts=['sans'], val_fonts=['serif'], workers=24,
                              export=True)
hub.export_wordset('english')                # or upload the recipe of an existing set
hub.import_wordset('english', workers=24)    # draw it again from the recipe
hub.list_wordsets()
```

**Exporting** uploads the fonts and wordlist the set was made with to `<org>/assets` (checked against the sha256 in its `metadata.json`), pins that revision of `<org>/assets` in `sources.json`, so a font replaced there later does not change what the recipe regenerates, and uploads the recipe.

**Importing** downloads the recipe and the pinned fonts and wordlist, then calls `corread.regenerate_training_set`, which:

1. draws the random settings again from the seed and compares them with `images.csv` before drawing any image;
2. draws every image into `<name>.part/` and compares each word folder with `checksums.json`;
3. copies the recipe files in unchanged, so models trained on the regenerated set record the same `metadata_sha256` as on the original;
4. writes `regeneration.json` (`verified`, Pillow version, time) and `.hub.json`, then renames the folder into place.

`regenerate_training_set(name, recipe=<folder>)` also works offline, from any folder with a recipe, for example to check that a set on disk can still be reproduced.

### Pillow version

Text rendering and JPEG encoding come from the FreeType and libjpeg builds that Pillow bundles, so byte-identical images need the Pillow version in the recipe (`generator.pillow`, also shown on the card). With another version, the import names the word folders that differ and stops; `allow_mismatch=True` keeps the images anyway, recorded as `verified: false`. They are still valid stimuli, just not byte-identical to the original's.

## Fonts and wordlists

```python
hub.export_assets()    # every fonts/*.ttf and wordlists/*.txt to the private <org>/assets
hub.import_assets()    # all of them into your fonts and wordlists folders
```

Both refuse to replace a file whose content differs unless `overwrite=True`. Word set recipes keep pointing to the revision they were made with.

## Security

- Downloaded recipes are checked: every file a recipe's `sources.json` names must be a plain `fonts/*.ttf` or `wordlists/*.txt`, at a pinned commit, with the recorded sha256.
- Model files are loaded with `torch.load(..., weights_only=True)`, which cannot run code from a downloaded file.
- Imported model files must match the sha256 in their metadata.
