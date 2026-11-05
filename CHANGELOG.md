# Changelog

All notable changes are listed here. The project follows [semantic versioning](https://semver.org).

## 0.1.0

First public release.

- Word stimuli and byte-reproducible word sets, with recipes and checksums (`corread.stimuli`).
- Training modes `pre`, `illit`, `lit_bias` and `lit_no_bias` for CORnet-Z and CORnet-S, single runs and
  whole families, resumable and recorded in `metadata.json` (`corread.training`).
- Word selectivity of the dense-layer units, lesions and recovery runs for CORnet-Z (`corread.lesion`).
- Sharing of models and word set recipes through a Hugging Face organisation (`corread.hub`).
- Optional status reports to a web API, specified in `docs/reporting.md` and `docs/reporting.schema.json`.
- ImageNet preparation from the ILSVRC 2012 archives (`corread.prepare_imagenet`).
