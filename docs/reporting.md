# Reporting API

CORread can send the live status of its long jobs (training runs, lesion runs and word set generation) to a web API you run yourself, so you can follow them on a dashboard. CORread ships only the client. This page is the contract a server has to implement. [`reporting.schema.json`](reporting.schema.json) is the same contract as a JSON Schema (draft 2020-12), so a server can validate every report it receives.

This is schema version **2**.

## 1. Turning it on

In `paths.toml`:

```toml
[report]
url = "https://example.org/api/runs"   # your endpoint; https:// required (http:// only for localhost)
every = 60                              # seconds between "progress" reports
```

The token goes in an environment variable, never in a file: `export CORREAD_REPORT_TOKEN=<token>`.

| Python call | What it reports |
|---|---|
| `TrainingInstance(..., report=True).train()` | the training run, from start to end |
| `FamilyTrainingInstance(..., report=True).train()` | each run of the family, one after another, as separate runs |
| `LesionInstance(..., report=True).train()` | the lesion run, from start to end |
| `generate_training_set(..., report=True)` | the word set generation, from start to end |
| `TrainingInstance(...).report_status()` | one `snapshot` of that model's run, now: finished, interrupted, or training in another process |
| `LesionInstance(...).report_status()` | the same for a lesion run |
| `report_wordset_status(name)` | one `snapshot` of the word set `name` |

With `report=True`, a missing or unsafe setting raises immediately, before any work starts. After that, reporting never stops or slows a run (§3). `report_status()` is called on purpose, so it raises if the report doesn't get through.

## 2. Transport

| | |
|---|---|
| Method and URL | `POST <url>`, exactly as configured; no path or query is added |
| Headers | `Content-Type: application/json`<br>`Authorization: Bearer <token>`<br>`User-Agent: CORread-reporter/2` |
| Body | one JSON object (§5), UTF-8. Usually 2–30 KB; the history of an 80-epoch run keeps it under 70 KB. Accepting up to 256 KB is safe |
| TLS | certificates are verified; plain `http://` is refused unless the host is `localhost` or a loopback IP |
| Redirects | **not followed**, since following one would resend the token elsewhere. Point `url` at the final address |
| Client timeout | 10 s |

The server should compare the bearer token in constant time, and answer:

| Status | Meaning to the client |
|---|---|
| any `2xx` | accepted. The body is ignored; `{"ok": true}` is fine |
| `401` / `403` | bad or missing token |
| `400` / `422` | invalid payload |
| other `4xx`, `5xx`, timeout | failed |

The client does **not** retry a failed report. Each report carries the full state of the run, and a running job sends a newer one soon (the next `progress`, `epoch_end` or final event), so only the stale report is lost. On failure the job logs one warning and stays quiet until a report gets through again. `report_status()` raises an error with the status code and the first 300 characters of the response body.

Answer **200** to an old or duplicate report (§4) as well.

## 3. When reports are sent

| `event` | `status` sent | When | Training, lesion | Word set |
|---|---|---|---|---|
| `started` | `in progress` | the job's `metadata.json` was written and work starts. Also sent on every resume of an interrupted run, with the same `run_id` | ✓ | ✓ |
| `progress` | `in progress` | at most once every `every` seconds while batches or words are processed | ✓ | ✓ |
| `epoch_end` | `in progress` | an epoch finished training **and** validation, and its checkpoint is saved; `history` has the new entry | ✓ | |
| `complete` | `complete` | the job finished | ✓ | ✓ |
| `interrupted` | `interrupted` or `discarded` | Ctrl+C. `discarded`: a training run stopped before its first epoch finished, so its folder was deleted and it can't be resumed; its next start is a new run with a new `run_id` | ✓ | ✓ |
| `failed` | `failed` | an exception ended the job; `error` says which | ✓ | ✓ |
| `snapshot` | any (§6) | `report_status()` was called | ✓ | ✓ |

- Reports are sent from a background thread that keeps only the **newest** unsent report. If the API is slow, intermediate `progress` reports are skipped, and an `epoch_end` immediately followed by `complete` may arrive only as `complete`. Don't rely on receiving every event: treat each report as the current full state of the run.
- There is no event for the end of the initial validation. Its result (`initial_validation`) is in every report after it, starting with the next `progress` report.
- A process killed hard (`kill -9`, power loss, out of memory) sends **nothing**. Treat a run as stale when its status is `in progress` and no report has arrived for a while (3 × `every` plus a few minutes is a good rule). Show it as "no contact" rather than changing its stored status, since the run may still be alive behind a network problem.

## 4. Identity and ordering

- **`run_id`** (UUID) identifies one run: one model folder or one word set. It is stored in the run's `metadata.json` and **stays the same across resumes**. Use it as the primary key.
- **`(group, name)`** is the run's folder (`models/<group>/<name>`, or `wordsets/<name>`), unique *at any one time*. A folder can be deleted and a new run started under the same name, which gets a new `run_id`, so use these as display names, not keys.
- **`seq`** (integer) increases strictly with every report about a run, across resumes and processes (it is the send time in milliseconds; treat it as an opaque number to compare). Reports can arrive out of order: **store a report only if its `seq` is greater than the stored one for its `run_id`**, and answer 200 either way.

## 5. The payload

Every report has all the fields below; there are no optional fields, and no others (`additionalProperties: false` throughout the schema). A field without a value is `null`. Times are ISO 8601 strings with the machine's UTC offset, to the second (`"2026-10-05T21:48:05-04:00"`). Durations are in seconds.

### 5.1 Common fields

| Field | Type | Meaning |
|---|---|---|
| `schema` | `2` | payload format version; a breaking change increases it |
| `seq` | integer | ordering number (§4) |
| `sent_at` | time | when this report was built |
| `run_id` | UUID string | §4 |
| `kind` | `"training"` \| `"lesion"` \| `"wordset"` | the kind of job; decides the shape of `config`, `progress`, `initial_validation` and `history` |
| `name` | string | the run's folder name, e.g. `"pre_s_49"`, `"lit_bias_z_79"`, `"lesion_lit_bias_z_49u_89"`, `"english"` |
| `group` | string | the folder it is in. Training: `"pre"`, `"illiterate"`, or the word set (or name) a literate model is grouped under. Lesion: the group of the model it lesions. Word sets: always `"wordsets"` |
| `host` | string | name of the machine running the job |
| `event` | string | why this report was sent (§3) |
| `status` | string | the run's state (§6) |
| `started_at` | time \| null | when the run was first started (not the latest resume) |
| `finished_at` | time \| null | when it completed; `null` until `status` is `complete` |
| `config` | object | how the run is set up (§5.2–5.4) |
| `progress` | object | how far it is (§5.2–5.4) |
| `eta_seconds` | integer ≥ 0 \| null | estimated seconds until it finishes; `null` when not running or not yet estimable |
| `initial_validation` | object \| null | training: the validation of the starting weights (§5.2). Lesion: those of the intact and the lesioned model (§5.4). Word sets: `null` |
| `history` | array \| null | training and lesion: one entry per finished epoch. Word sets: `null` |
| `error` | object \| null | `{"type": "<exception class>", "message": "<≤ 500 characters>"}` when `status` is `failed`, else `null`. The home folder in the message is shortened to `~` |

### 5.2 `kind: "training"`

**How a run measures itself.** Before its first epoch a run validates the weights it starts from (`initial_validation`). Every epoch then trains on one shuffled pass over the training set, recording loss and accuracy as it goes, and afterwards validates the resulting weights on the whole validation set. So in `history[n]`:

- `train_*` describe **epoch *n*'s training**: running means over its batches, taken while the weights were changing, in training mode and on randomly cropped views;
- `val_*` describe **the weights after epoch *n***, in evaluation mode, on fixed views of the validation images. The last entry is the finished model.

`train_*` and `val_*` are therefore not a clean measure of overfitting: early in a run, validation accuracy can exceed training accuracy, since training accuracy averages over the whole epoch and uses harder, random crops.

Every loss is the mean cross-entropy per image. Every accuracy is **in percent (0–100)**: the share of images whose largest output is their own class, out of all the model's outputs. `*_images` counts the image classes and `*_words` the word classes; `*_words` is `null` for `pre` and `illit`, which have no words.

**`config`** (from the run's `metadata.json`; constant during the run)

| Field | Type | Meaning |
|---|---|---|
| `mode` | `"pre"` \| `"illit"` \| `"lit_bias"` \| `"lit_no_bias"` | pre-literate (images only), illiterate, or literate with or without the bias |
| `model_choice` | string \| null | `"z"` (CORnet-Z) or `"s"` (CORnet-S) |
| `model_class` | string \| null | Python class of the network, e.g. `"CORnetZBiasedWords"` |
| `parameters` | integer \| null | number of parameters |
| `outputs` | integer \| null | number of output classes |
| `pre_name` | string \| null | the pre-literate model a literate or illiterate run starts from (its run has `group: "pre"`, `name: pre_name`); `null` for `pre` |
| `wordset` | string \| null | word set name, when one is used |
| `batch_size` | integer \| null | training batch size |
| `lr`, `momentum`, `weight_decay` | number \| null | SGD settings; every epoch trains with `lr` |
| `img_classes`, `wrd_classes` | integer \| null | number of image and word classes |
| `train_images`, `val_images` | integer \| null | size of the training and validation sets (images + word images) |
| `device_name` | string \| null | GPU model, or the CPU |
| `gpus_used` | integer \| null | number of GPUs |

**`progress`**

| Field | Type | Meaning |
|---|---|---|
| `epochs_first` | integer \| null | number of the run's first epoch. Epochs are counted from 0 over the pre-literate epochs, then the later ones, so a literate run after 50 pre-literate epochs starts at `50` |
| `epochs_total` | integer \| null | number of epochs in this run |
| `epochs_completed` | integer | epochs finished so far (= `history.length`) |
| `epoch` | integer \| null | the epoch running now (`epochs_first + epochs_completed`); `null` when not running |
| `phase` | `"initial_validation"` \| `"training"` \| `"validation"` \| null | what the run is doing; known only inside the training process while it runs (in practice: `progress` events) |
| `batch`, `batches` | integer \| null | index of the batch being processed in that phase (0-based), and the number of batches |
| `fraction` | number 0–1 \| null | share of the run done. Training counts as the first 95 % of an epoch and validation as the last 5 %; the initial validation counts as nothing |

**`initial_validation`**: the validation of the starting weights. For `pre` it is an untrained network (about 0.1 % accuracy and a loss of about 6.9 for 1000 classes); for the other modes it is the pre-literate model they continue. `null` until it has run, and for a run started from a checkpoint.

| Field | Type | Meaning |
|---|---|---|
| `val_loss` | number \| null | mean cross-entropy over all validation images and words |
| `val_loss_images`, `val_loss_words` | number \| null | the same over the images or the words |
| `val_accuracy_images`, `val_accuracy_words` | number \| null | accuracy on the images or the words, in percent |
| `duration_seconds` | number \| null | wall time of the validation |
| `finished_at` | time \| null | when it finished |

**`history[]`**: one entry per finished epoch, oldest first: `epoch`, `lr`, `train_loss`, `train_loss_images`, `train_loss_words`, `train_accuracy_images`, `train_accuracy_words`, then the fields of `initial_validation` for the weights after the epoch (`duration_seconds` covers the epoch's training and validation).

**Plotting.** Plot `history[n]` at x = `epoch`, and `initial_validation` at x = `epochs_first − 1`. For a literate or illiterate run that is the pre-literate model's last epoch, so a chart can chain the runs of a family: the pre-literate run (`group: "pre"`, `name: config.pre_name`), then each run that continues it.

`eta_seconds` = (epochs left − share of the current epoch done) × mean epoch duration; `null` until the run's first epoch has finished.

### 5.3 `kind: "wordset"`

**`config`**: `wordlist`, `num_words` (number of words, = classes), `num_train` and `num_val` (images per word in each split), `train_fonts` and `val_fonts` (string arrays), `width` and `height` (image size in pixels).

**`progress`**: `words_done` (words whose images are all drawn; `null` in a `snapshot` of an unfinished set), `words_total`, and `fraction`.

`eta_seconds` comes from the average time per word so far; `initial_validation` and `history` are always `null`.

### 5.4 `kind: "lesion"`

A lesion run starts from a finished literate CORnet-Z model and lesions units of its dense layer (the 512 units the readout reads), then lets the model recover by training on as in its literate training. Its epochs continue the lesioned model's numbering, so after `lit_bias_z_79` they are 80, 81, ….

A lesion re-initialises a unit's incoming weights once, as in an untrained network; with `config.reset_outgoing` also its weights in the readout. With `config.silent_epochs` *K* > 0 the units are first silenced (their output is 0) for *K* epochs, and re-initialised when they are released at the start of epoch `epochs_first + K`; with *K* = `progress.epochs_total` they are never released.

**Word selectivity** of a unit = (mean activation to the words − mean activation to the images) ÷ SD of its activations to the images, over all validation word images and a fixed sample of ImageNet training images. A unit is *word-selective* when its selectivity is at least `config.selectivity_threshold`.

Before its first epoch the run validates the intact model, lesions it, and validates the lesioned model; every epoch then trains and validates as a training run does. Each validation also measures the units:

| Field | Type | Meaning |
|---|---|---|
| `mean_word_activation_lesioned` | number \| null | mean, over the lesioned units, of each unit's mean activation to the validation words; `null` without lesioned units |
| `mean_word_activation_other` | number \| null | the same over the other units |
| `word_selective_lesioned`, `word_selective_other` | integer | how many of the lesioned and of the other units are word-selective |

These unit fields are in both `initial_validation` entries and in every `history` entry, in addition to the fields of §5.2.

**`config`**

| Field | Type | Meaning |
|---|---|---|
| `source_name` | string | the lesioned model; its run has the same `group` and `name: source_name` |
| `source_mode` | `"lit_bias"` \| `"lit_no_bias"` | whether the lesioned model is biased or unbiased |
| `model_choice` … `gpus_used` | | as in §5.2 (without `mode` and `pre_name`); the SGD settings are the recovery's |
| `units_total` | integer \| null | units in the dense layer (512) |
| `lesioned_units` | integer[] | the lesioned units, ascending; empty for a control run without a lesion |
| `word_channel` | object \| null | `{"first": 463, "last": 511}`: the units the biased model's word outputs read; `null` for the unbiased model |
| `reset_outgoing` | boolean | whether the units' readout weights were re-initialised too |
| `silent_epochs` | integer ≥ 0 | epochs the units stay silent before they are released |
| `selectivity_threshold` | number \| null | selectivity from which a unit counts as word-selective |
| `selectivity_images` | integer \| null | ImageNet images in the selectivity sample |
| `selectivity_words` | integer \| null | validation word images the word activations are averaged over |

**`progress`**: as in §5.2, with more phases: `intact_validation` and `intact_selectivity`, then `initial_validation` and `initial_selectivity` (the lesioned model) before the first epoch, then `training`, `validation` and `selectivity` in every epoch. Training counts as 95 % of an epoch, validation 4 % and the selectivity measurement 1 %.

**`initial_validation`** is `null` until the intact model is validated, then `{"intact": {...}, "lesioned": null}`, then `{"intact": {...}, "lesioned": {...}}`.

**Plotting.** Plot both `initial_validation` entries at x = `epochs_first − 1`, the lesioned model's last epoch: `intact`, then a vertical drop to `lesioned`. Chain the run after the lesioned model's own run to show the whole story, and plot the unit fields in a chart of their own.

## 6. Status

| `status` | Meaning | Final? |
|---|---|---|
| `in progress` | running (or, in a `snapshot` from another machine, recorded as running) | no |
| `complete` | finished | yes |
| `interrupted` | stopped with Ctrl+C; can be resumed (`resume=True`), which sends `started` again with the same `run_id` | no |
| `discarded` | stopped before anything worth keeping was saved, and deleted | yes |
| `failed` | ended by an error (`error`); a training run can still be resumed from its last checkpoint | no |
| `stopped` | only in a `snapshot`: recorded as running on this machine, but its process no longer exists | no |

Any status may follow any other for the same `run_id` (for example, a `snapshot` of an old run), so always store the latest by `seq`.

## 7. What is never sent

The token itself, absolute paths, user names (home folders in error messages become `~`), process ids, seeds, git details, file hashes, word lists, per-class scores, and the per-unit values of lesion runs (only their summaries in §5.4 and the list of lesioned units). Everything else in a run's `metadata.json` stays on the machine.

## 8. Examples

### Training, `progress` (history shortened to 2 of 3 epochs)

```json
{
  "schema": 2,
  "seq": 1791252574351,
  "sent_at": "2026-10-05T22:09:34-04:00",
  "run_id": "95c70a15-e5f2-505b-8982-64f41a828363",
  "kind": "training",
  "name": "pre_s_49",
  "group": "pre",
  "host": "lab-gpu-01",
  "event": "progress",
  "status": "in progress",
  "started_at": "2026-10-05T21:48:05-04:00",
  "finished_at": null,
  "config": {
    "mode": "pre", "model_choice": "s", "model_class": "CORnetS",
    "parameters": 53416616, "outputs": 1000, "pre_name": null, "wordset": null,
    "batch_size": 100, "lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001,
    "img_classes": 1000, "wrd_classes": 1000, "train_images": 1281167, "val_images": 50000,
    "device_name": "NVIDIA A100 80GB PCIe", "gpus_used": 1
  },
  "progress": {
    "epochs_first": 0, "epochs_total": 50, "epochs_completed": 3, "epoch": 3,
    "phase": "training", "batch": 6400, "batches": 12812, "fraction": 0.0695
  },
  "eta_seconds": 139576,
  "initial_validation": {
    "val_loss": 6.908, "val_loss_images": 6.908, "val_loss_words": null,
    "val_accuracy_images": 0.1, "val_accuracy_words": null,
    "duration_seconds": 150.0, "finished_at": "2026-10-05T21:50:40-04:00"
  },
  "history": [
    { "epoch": 0, "lr": 0.01,
      "train_loss": 5.9, "train_loss_images": 5.9, "train_loss_words": null,
      "train_accuracy_images": 2.1, "train_accuracy_words": null,
      "val_loss": 5.2, "val_loss_images": 5.2, "val_loss_words": null,
      "val_accuracy_images": 6.3, "val_accuracy_words": null,
      "duration_seconds": 3000.0, "finished_at": "2026-10-05T22:40:40-04:00" },
    { "epoch": 1, "lr": 0.01,
      "train_loss": 4.6, "train_loss_images": 4.6, "train_loss_words": null,
      "train_accuracy_images": 11.8, "train_accuracy_words": null,
      "val_loss": 4.1, "val_loss_images": 4.1, "val_loss_words": null,
      "val_accuracy_images": 17.0, "val_accuracy_words": null,
      "duration_seconds": 3000.0, "finished_at": "2026-10-05T23:30:40-04:00" }
  ],
  "error": null
}
```

### Training, `failed` (only the fields that differ from the example above)

```json
{
  "event": "failed",
  "status": "failed",
  "progress": { "epochs_first": 0, "epochs_total": 50, "epochs_completed": 3, "epoch": null,
                "phase": null, "batch": null, "batches": null, "fraction": 0.06 },
  "eta_seconds": null,
  "error": { "type": "OutOfMemoryError", "message": "CUDA out of memory. Tried to allocate 2.00 GiB ..." }
}
```

### Lesion, one `history` entry

```json
{ "epoch": 80, "lr": 0.01,
  "train_loss": 1.75, "train_loss_images": 3.05, "train_loss_words": 0.11,
  "train_accuracy_images": 37.4, "train_accuracy_words": 97.1,
  "val_loss": 1.93, "val_loss_images": 2.73, "val_loss_words": 1.13,
  "val_accuracy_images": 42.2, "val_accuracy_words": 74.9,
  "duration_seconds": 790.0, "finished_at": "2026-10-29T13:37:50-04:00",
  "mean_word_activation_lesioned": 0.85, "mean_word_activation_other": 0.43,
  "word_selective_lesioned": 2, "word_selective_other": 33 }
```

### Word set, `progress`

```json
{
  "schema": 2,
  "seq": 1791252557812,
  "sent_at": "2026-10-05T22:09:17-04:00",
  "run_id": "12270431-eafa-4e1f-af1d-da0fe84d5dea",
  "kind": "wordset",
  "name": "english",
  "group": "wordsets",
  "host": "lab-gpu-01",
  "event": "progress",
  "status": "in progress",
  "started_at": "2026-10-05T22:09:17-04:00",
  "finished_at": null,
  "config": {
    "wordlist": "english", "num_words": 1000, "num_train": 1000, "num_val": 50,
    "train_fonts": ["sans"], "val_fonts": ["serif"], "width": 500, "height": 500
  },
  "progress": { "words_done": 300, "words_total": 1000, "fraction": 0.3 },
  "eta_seconds": 1640,
  "initial_validation": null,
  "history": null,
  "error": null
}
```

The numbers in these examples only show the shape.
