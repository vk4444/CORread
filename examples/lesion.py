"""Lesion the word-selective units of a literate CORnet-Z, let it relearn to read, and show the recovery.

    python lesion.py english/lit_bias_z_79 --epochs 20 --reset-outgoing --gpus 0

The units are chosen by their word selectivity in the intact model: how many SDs of its responses to
images a unit's mean response to words lies above its mean response to images.
"""

import argparse
import json
import logging
import os

import numpy as np

from corread import LesionInstance
from corread.paths import get_path


def choose_units(model: str, threshold: float, num_units: int | None, gpus, workers: int) -> list[int]:
    sel = LesionInstance.word_selectivity(model, threshold=threshold, gpus=gpus, num_workers=workers)
    selective, channel = sel['selective_units'], sel['word_channel']
    print(len(selective), 'of 512 units are word-selective (selectivity >=', threshold, 'SD)')
    if channel:
        inside = sum(channel[0] <= u <= channel[1] for u in selective)
        print(inside, 'of them are in the word channel, units', channel[0], 'to', channel[1])
    print(
        'most selective units (unit: SDs):',
        ', '.join('%d: %.1f' % (u, sel['selectivity'][u]) for u in sel['ranking'][:20]),
    )
    return selective[:num_units]


def show_recovery(folder: str) -> None:
    with open(os.path.join(folder, 'metadata.json'), encoding='utf-8') as f:
        meta = json.load(f)
    stages = np.load(os.path.join(folder, 'unit_tracking.npz'))['stage']
    entries = [meta['intact_validation'], meta['initial_validation'], *meta['history']]
    print('stage         word acc  image acc  word act lesioned  word act other  selective lesioned/other')
    # an unfinished run has fewer entries than stages
    for stage, e in zip(stages, entries, strict=False):
        lesioned = e['mean_word_activation_lesioned']
        print(
            '%-12s %8.1f%% %9.1f%% %18s %15.3f  %d/%d'
            % (
                stage,
                e['val_accuracy_words'],
                e['val_accuracy_images'],
                '-' if lesioned is None else '%.3f' % lesioned,
                e['mean_word_activation_other'],
                e['word_selective_lesioned'],
                e['word_selective_other'],
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('model', help="the literate model, '<group>/<name>' in the models folder")
    parser.add_argument('--name', default=None, help='folder name of the lesion run')
    parser.add_argument(
        '--units', type=int, nargs='*', default=None, help='units to lesion (0-511); by default the word-selective ones'
    )
    parser.add_argument('--num-units', type=int, default=None, help='lesion only the N most selective units')
    parser.add_argument('--threshold', type=float, default=3)
    parser.add_argument('--reset-outgoing', action='store_true', help="also re-initialise the units' readout weights")
    parser.add_argument('--silent-epochs', type=int, default=0, help='silence the units this many epochs first')
    parser.add_argument('--epochs', type=int, default=10, help='recovery epochs')
    parser.add_argument('--gpus', type=int, nargs='*', default=None)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    gpus = None if args.gpus is None else (args.gpus or 0)
    units = args.units
    if units is None and args.name:
        # resuming a named run: keep the units chosen when it started, since measuring again could move a
        # unit near the threshold across it
        meta_path = os.path.join(get_path('models'), args.model.split('/')[0], args.name, 'metadata.json')
        if os.path.isfile(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                units = json.load(f)['arguments']['units']
    if units is None:
        units = choose_units(args.model, args.threshold, args.num_units, gpus, args.workers)
        if not units:
            raise SystemExit('No unit passes the selectivity test; pass --units or lower --threshold')

    run = LesionInstance(
        args.model,
        units=units,
        name=args.name,
        reset_outgoing=args.reset_outgoing,
        max_epochs=args.epochs,
        silent_epochs=args.silent_epochs,
        threshold=args.threshold,
        gpus=gpus,
        num_workers=args.workers,
        resume=True,
    )
    run.train()
    show_recovery(run.model_dir)


if __name__ == '__main__':
    main()
