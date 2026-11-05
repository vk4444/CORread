# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""The generated cards (READMEs) of model, word set and assets repos."""

from __future__ import annotations

from .._io import now_iso
from ..models import NETWORKS
from ._client import (
    ASSETS_REPO,
    CARD_KEY,
    IMAGENET_DATASET,
    PRE_GROUP,
    TAG,
    exists,
    model_repo,
    url,
    wordset_repo,
    yaml_card,
)

LITERATE = ('lit_bias', 'lit_no_bias')


def _pct(value: float | None) -> str:
    return '–' if value is None else '%.1f %%' % value


def _num(value: float | None, digits: int = 3) -> str:
    return '–' if value is None else ('%.' + str(digits) + 'f') % value


def _table(rows: list[tuple[str, str]]) -> list[str]:
    return ['| | |', '|---|---|'] + ['| ' + k + ' | ' + v + ' |' for k, v in rows] + ['']


def _base_model(api, org: str, meta: dict) -> str | None:
    """The repo a model started from, when it is on the Hub."""
    init = meta.get('initialised_from') or {}
    if meta.get('mode') == 'pre' or not init.get('name'):
        return None
    repo_id = (init.get('hub') or {}).get('repo_id') or model_repo(org, PRE_GROUP, init['name'])
    return repo_id if exists(api, repo_id, 'model') else None


def _model_datasets(api, org: str, meta: dict) -> list[str]:
    """The dataset repos a model was trained on: its word set's recipe, when on the Hub, and ImageNet."""
    wordset = (meta.get('data') or {}).get('wordset') or {}
    name = (wordset.get('generation') or {}).get('name') or (meta.get('arguments') or {}).get('wordset')
    repo_id = (wordset.get('hub') or {}).get('repo_id')
    if repo_id is None and name and name != ASSETS_REPO:
        repo_id = wordset_repo(org, name)
    return ([repo_id] if repo_id and exists(api, repo_id, 'dataset') else []) + [IMAGENET_DATASET]


def claim_card(group: str, name: str, run_id: str) -> str:
    """The card a new run's repo starts with; it claims the repo until the run's first metadata replaces it."""
    return yaml_card(
        '# ' + group + '/' + name + '\n\nA training run is starting; its metadata.json follows once its '
        'datasets have loaded.\n',
        library_name='pytorch',
        tags=[TAG],
        **{CARD_KEY: {'kind': 'model', 'model': group + '/' + name, 'status': 'starting', 'run_id': run_id}},
    )


def _usage(meta: dict, model: str, repo_id: str, name: str) -> list[str]:
    args = meta.get('arguments') or {}
    mode, choice = meta.get('mode'), args.get('model_choice')
    cls = NETWORKS.get(choice, {}).get(mode)
    if meta.get('status') != 'complete' or cls is None:
        return []
    sizes = 'out_img=' + str(args.get('img_classes'))
    if mode in LITERATE:
        sizes += ', out_wrd=' + str(args.get('wrd_classes'))
    return [
        '## Use',
        '',
        'Into the models folder of your paths.toml, with its metadata, ready to continue training from:',
        '',
        '```python',
        'from corread import hub',
        "hub.import_model('" + model + "')",
        '```',
        '',
        'Or load the weights directly:',
        '',
        '```python',
        'import torch',
        'from huggingface_hub import hf_hub_download',
        'from corread.models import ' + cls.__name__,
        '',
        "ckpt = torch.load(hf_hub_download('" + repo_id + "', '" + name + ".pth.tar'), map_location='cpu', "
        'weights_only=True)',
        'net = ' + cls.__name__ + '(' + sizes + ')',
        "net.load_state_dict(ckpt['state_dict'])",
        '```',
        '',
    ]


def model_card(api, org: str, group: str, meta: dict, status: str | None = None) -> str:
    """The card of a model repo, from the model's ``metadata.json`` contents."""
    name = meta.get('name')
    model = group + '/' + name
    repo_id = model_repo(org, group, name)
    mode, choice = meta.get('mode'), (meta.get('arguments') or {}).get('model_choice')
    status = status or meta.get('status')
    epochs = meta.get('epochs') or {}
    history = meta.get('history') or []
    # each epoch validates after its training, so the last one's is the finished model's
    final = history[-1] if history and meta.get('status') == 'complete' else {}
    initial = meta.get('initial_validation') or {}
    init = meta.get('initialised_from') or {}
    env = meta.get('environment') or {}
    git = env.get('git') or {}
    data = meta.get('data') or {}
    literate = mode in LITERATE
    base_model = _base_model(api, org, meta)
    datasets = _model_datasets(api, org, meta)

    lines = [
        '# ' + model,
        '',
        'CORnet-' + str(choice).upper() + ', mode `' + str(mode) + '`, trained with '
        '[CORread](https://github.com/vk4444/CORread) ' + str(env.get('corread', '')) + '.',
        '',
        '**Status: '
        + str(status)
        + '**, '
        + str(meta.get('epochs_completed'))
        + ' of '
        + str(epochs.get('count'))
        + ' epochs (epochs '
        + str(epochs.get('first'))
        + '–'
        + str(epochs.get('last'))
        + '). '
        + (
            'The model file is here.'
            if meta.get('status') == 'complete'
            else 'Only metadata.json is uploaded until training completes.'
        ),
        '',
    ]
    if git.get('source_modified_since_commit'):
        lines += [
            "> ⚠ The library's source had uncommitted changes when this run started, so commit `"
            + str(git.get('commit'))
            + '` does not fully identify the code that trained it.',
            '',
        ]

    rows = [
        ('Seed', '`' + str(meta.get('seed_used')) + '`'),
        ('Code', '`' + str(git.get('commit') or env.get('code_sha256')) + '`'),
    ]
    if init.get('name'):
        start = '[' + init['name'] + '](' + url(base_model, 'model') + ')' if base_model else init['name']
        rows.append(
            (
                'Started from',
                start + ', epoch ' + str(init.get('epoch')) + ', file sha256 `' + str(init.get('file_sha256')) + '`',
            )
        )
    elif init.get('checkpoint'):
        rows.append(('Started from', 'checkpoint with sha256 `' + str(init.get('file_sha256')) + '`'))
    wordset = data.get('wordset') or {}
    if wordset:
        ws_name = (wordset.get('generation') or {}).get('name') or (meta.get('arguments') or {}).get('wordset')
        ws_repo = next((d for d in datasets if d != IMAGENET_DATASET), None)
        rows.append(
            (
                'Word set',
                ('[' + ws_name + '](' + url(ws_repo, 'dataset') + ')' if ws_repo else str(ws_name))
                + ', '
                + str(wordset.get('classes'))
                + ' words'
                + (', images digest `' + wordset['images_digest'] + '`' if wordset.get('images_digest') else ''),
            )
        )
    imageset = data.get('imageset') or {}
    if imageset:
        rows.append(
            (
                'Images',
                str(imageset.get('classes'))
                + ' classes, '
                + str(imageset.get('train_images'))
                + ' train / '
                + str(imageset.get('val_images'))
                + ' val',
            )
        )
    rows.append(
        (
            'Environment',
            ', '.join(
                p
                for p in (
                    env.get('device_name'),
                    env.get('torch') and 'torch ' + env['torch'],
                    env.get('python') and 'Python ' + env['python'],
                )
                if p
            )
            or '–',
        )
    )
    for label, scores in (('Initial validation', initial), ('Final validation', final)):
        if scores:
            rows.append(
                (
                    label,
                    'images '
                    + _pct(scores.get('val_accuracy_images'))
                    + (', words ' + _pct(scores.get('val_accuracy_words')) if literate else '')
                    + ', loss '
                    + _num(scores.get('val_loss')),
                )
            )
    if meta.get('model_file_sha256'):
        rows.append(('Model file sha256', '`' + meta['model_file_sha256'] + '`'))
    lines += _table(rows)

    if history:
        lines += [
            '## Training history',
            '',
            'Each epoch trains, then validates the weights after its training, so the last row is the finished '
            'model; Initial validation is the starting weights. Training values are running means over the '
            'epoch, taken while the weights change, in train mode and on randomly cropped views.',
            '',
            '| epoch | lr | train loss | train acc images |'
            + (' train acc words |' if literate else '')
            + ' val loss | val acc images |'
            + (' val acc words |' if literate else ''),
            '|---:|---:|---:|---:|---:|---:|' + ('---:|---:|' if literate else ''),
        ]
        for h in history:
            lines.append(
                '| '
                + str(h.get('epoch'))
                + ' | '
                + str(h.get('lr'))
                + ' | '
                + _num(h.get('train_loss'))
                + ' | '
                + _pct(h.get('train_accuracy_images'))
                + ' |'
                + (' ' + _pct(h.get('train_accuracy_words')) + ' |' if literate else '')
                + ' '
                + _num(h.get('val_loss'))
                + ' | '
                + _pct(h.get('val_accuracy_images'))
                + ' |'
                + (' ' + _pct(h.get('val_accuracy_words')) + ' |' if literate else '')
            )
        lines.append('')

    lines += [
        '## Files',
        '',
        '- `metadata.json`: everything about the run: arguments, seed, data, code and environment, and the '
        'per-epoch history. Uploaded after every epoch.',
    ]
    if meta.get('status') == 'complete':
        lines += [
            '- `' + name + ".pth.tar`: the finished model (the last epoch's checkpoint: `epoch`, `state_dict`, "
            '`optimizer`, scores and losses).',
            '- `cat_scores.npy`, `trainloss.npy`, `valloss.npy`, `initial_cat_scores.npy`, `initial_valloss.npy`, '
            '`training_eta_log.txt`: see `files` in metadata.json.',
        ]
    lines += ['', 'Per-epoch checkpoints are not uploaded.', '']
    lines += _usage(meta, model, repo_id, name)
    if meta.get('reproduce'):
        lines += ['## Reproduce', '', meta['reproduce'] + '.', '']

    card = {
        'kind': 'model',
        'model': model,
        'mode': mode,
        'model_choice': choice,
        'status': status,
        'epochs_completed': meta.get('epochs_completed'),
        'epochs_total': epochs.get('count'),
        'last_epoch': epochs.get('last'),
        'final_val_accuracy_images': final.get('val_accuracy_images'),
        'final_val_accuracy_words': final.get('val_accuracy_words'),
        'run_id': meta.get('run_id'),
        'updated_at': now_iso(),
    }
    extra = {'base_model_relation': 'finetune'} if base_model else {}
    return yaml_card(
        '\n'.join(lines),
        library_name='pytorch',
        tags=[TAG, 'cornet-' + str(choice), str(mode)],
        base_model=base_model,
        datasets=datasets,
        **{CARD_KEY: card},
        **extra,
    )


def assets_card(org: str) -> str:
    return yaml_card(
        '# Fonts and wordlists\n\nThe inputs of [CORread](https://github.com/vk4444/CORread) word sets: '
        '`fonts/<name>.ttf` and `wordlists/<name>.txt` (one word per line). Each word set recipe (`'
        + org
        + '/<word set>`) pins the revision of this repo that holds its files, so files replaced here later do not '
        'change what it regenerates.\n\nFonts are often licensed for your own use only, so this repo is private.\n\n'
        '```python\nfrom corread import hub\nhub.import_assets()   # into the fonts and wordlists folders of your '
        'paths.toml\n```\n',
        dataset=True,
        viewer=False,
        tags=[TAG],
        **{CARD_KEY: {'kind': 'assets'}},
    )


def wordset_card(meta: dict, checksums: dict, sources: dict) -> str:
    args = meta.get('arguments') or {}
    rendering = meta.get('rendering') or {}
    generator = meta.get('generator') or {}
    git = generator.get('git') or {}
    name = meta.get('name')
    assets_url = url(sources['assets']['repo_id'], 'dataset') + '/tree/' + sources['assets']['revision']
    splits = meta.get('splits') or {}
    rows = [
        (
            'Wordlist',
            '`'
            + str(args.get('wordlist'))
            + '`, first '
            + str(args.get('num_words'))
            + ' words, sha256 `'
            + str((meta.get('wordlist_file') or {}).get('sha256'))
            + '`',
        ),
        (
            'Train fonts',
            ', '.join('`' + f + '`' for f in args.get('train_fonts', []))
            + ' ('
            + str(args.get('num_train'))
            + ' images per word)',
        ),
        (
            'Val fonts',
            ', '.join('`' + f + '`' for f in args.get('val_fonts', []))
            + ' ('
            + str(args.get('num_val'))
            + ' images per word)',
        ),
        ('Images', str(checksums.get('images')) + ', digest `' + str(checksums.get('total')) + '`'),
        (
            'Canvas',
            str(args.get('width'))
            + ' × '
            + str(args.get('height'))
            + ' px, font sizes '
            + str(rendering.get('font_sizes'))
            + ', x shifts '
            + str(rendering.get('xshift_values'))
            + ', y shifts '
            + str(rendering.get('yshift_values')),
        ),
        ('Seed', '`' + str(meta.get('seed_used')) + '`'),
        (
            'Made with',
            'CORread '
            + str(generator.get('corread'))
            + ', Pillow '
            + str(generator.get('pillow'))
            + ', Python '
            + str(generator.get('python'))
            + (', commit `' + str(git.get('commit')) + '`' if git else ''),
        ),
        (
            'Fonts and wordlist',
            '[' + sources['assets']['repo_id'] + ' at ' + sources['assets']['revision'][:10] + '](' + assets_url + ')',
        ),
    ]
    body = '\n'.join(
        [
            '# Word set ' + str(name),
            '',
            'The recipe of a [CORread](https://github.com/vk4444/CORread) word set: everything needed to draw its '
            + str(checksums.get('images'))
            + ' images again, byte for byte. The images themselves are not uploaded.',
            '',
        ]
        + _table(rows)
        + [
            '## Regenerate the images',
            '',
            '```python',
            'from corread import hub',
            "hub.import_wordset('" + str(name) + "', workers=24)   # into the wordsets folder of your paths.toml",
            '```',
            '',
            'This downloads the recipe and the exact fonts and wordlist, draws every image again from the seed, and '
            "checks each word folder against `checksums.json`. Text rendering and JPEG encoding come from Pillow's "
            'bundled FreeType and libjpeg, so use Pillow ' + str(generator.get('pillow')) + ' to get identical files; '
            'with another build the check names the words that differ.',
            '',
            '## Files',
            '',
            '- `metadata.json`: the recipe: arguments, seed, rendering settings, words, and the sha256 of the wordlist '
            'and fonts. Unchanged from when the set was made, so the `metadata_sha256` that trained models record '
            'still matches.',
            '- `images.csv`: the font, size, case and shift drawn for every image (shown in the viewer above).',
            '- `checksums.json`: sha256 digest of every word folder, to check regenerated images.',
            '- `sources.json`: the revision of ' + sources['assets']['repo_id'] + ' that holds the fonts and wordlist.',
            '',
        ]
    )
    card = {
        'kind': 'wordset',
        'name': name,
        'wordlist': args.get('wordlist'),
        'num_words': args.get('num_words'),
        'train_fonts': args.get('train_fonts'),
        'val_fonts': args.get('val_fonts'),
        'images': checksums.get('images'),
        'images_digest': checksums.get('total'),
        'pillow': generator.get('pillow'),
        'train_images': (splits.get('train') or {}).get('total_images'),
        'val_images': (splits.get('val') or {}).get('total_images'),
    }
    return yaml_card(
        body,
        dataset=True,
        pretty_name='Word set ' + str(name),
        task_categories=['image-classification'],
        tags=[TAG, 'synthetic'],
        configs=[{'config_name': 'default', 'data_files': 'images.csv'}],
        **{CARD_KEY: card},
    )
