"""Generate a word set whose validation images use a font the training images never do.

    python generate_wordset.py english --wordlist english \
        --train-fonts DejaVuSans LiberationSans --val-fonts DejaVuSerif --workers 16

The fonts are .ttf files in the fonts folder of paths.toml, named without .ttf.
"""

import argparse
import logging

import corread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('name', help='name of the word set (its folder in the wordsets folder)')
    parser.add_argument('--wordlist', required=True, help='wordlist file name, without .txt')
    parser.add_argument('--train-fonts', nargs='+', required=True)
    parser.add_argument('--val-fonts', nargs='+', required=True)
    parser.add_argument('--num-words', type=int, default=1000)
    parser.add_argument('--num-train', type=int, default=1000, help='training images per word')
    parser.add_argument('--num-val', type=int, default=50, help='validation images per word')
    parser.add_argument('--seed', type=int, default=None, help='drawn at random and recorded when not given')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s')
    folder = corread.generate_training_set(
        args.name,
        args.wordlist,
        train_fonts=args.train_fonts,
        val_fonts=args.val_fonts,
        num_words=args.num_words,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
        workers=args.workers,
    )
    print('Word set in', folder)


if __name__ == '__main__':
    main()
