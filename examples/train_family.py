"""Train a family of models on a word set: pre-literate, illiterate, and literate without and with the bias.

    python train_family.py english --model z --gpus 0 --workers 12

Run it again after an interruption: finished models are skipped and the interrupted one resumes from
its last checkpoint. Run it with another word set to reuse the pre-literate and illiterate models.
"""

import argparse
import logging

from corread import FamilyTrainingInstance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('wordset', help='name of the word set in the wordsets folder')
    parser.add_argument('--model', choices=['z', 's'], default='z', help='CORnet-Z or CORnet-S')
    parser.add_argument(
        '--gpus', type=int, nargs='*', default=None, help='GPU indices to train on (default: all; none given: the CPU)'
    )
    parser.add_argument('--workers', type=int, default=8, help='DataLoader worker processes')
    parser.add_argument('--export', action='store_true', help='upload to the Hugging Face organisation in paths.toml')
    parser.add_argument('--report', action='store_true', help='send status reports to the API in paths.toml')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    gpus = None if args.gpus is None else (args.gpus or 0)
    family = FamilyTrainingInstance(
        model_choice=args.model,
        wordset=args.wordset,
        gpus=gpus,
        num_workers=args.workers,
        resume=True,
        export=args.export,
        report=args.report,
    )
    family.train()
    for mode, instance in family.instances.items():
        print(mode.ljust(12), instance.model_dir)


if __name__ == '__main__':
    main()
