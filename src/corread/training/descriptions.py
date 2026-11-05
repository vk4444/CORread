# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Vojtěch Kubala
"""The explanations ``metadata.json`` carries next to its values, so each file documents itself."""

from __future__ import annotations

EPOCH_NUMBERING = 'counted from 0 over the pre-literate epochs, then the literate/illiterate ones'

REPRODUCE = (
    'create TrainingInstance with these arguments and seed=seed_used and call train(), with the same data '
    'and code (checked by the hashes below); GPU runs are not bit-for-bit repeatable'
)

SCHEDULER = 'not implemented: there is no learning-rate schedule, every epoch trains with lr'
LR_USED = 'history[].lr is the learning rate each epoch actually trained with'

EPOCH_PROCEDURE = (
    'the starting weights are validated once before the first epoch (see initial_validation); each '
    'epoch then trains on one shuffled pass over the train set and validates (no gradients) on the '
    'whole val set, so history val values are for the weights after that epoch, and the last entry '
    'is the finished model'
)

METRICS = (
    "history[].train_* are running means over the epoch's training batches, taken while the weights change, "
    'in train mode and on the randomly cropped training views; history[].val_* are for the weights after the '
    'epoch. *_loss is the mean cross-entropy per image, *_accuracy the % of images whose largest output is '
    'their class (argmax over all outputs); *_images and *_words split them by class (labels below '
    'img_classes are images), and *_words are None without word classes'
)


def cat_scores(literate: bool) -> str:
    return (
        "cat_scores[epoch, k] = accuracy (%) on val batch k after that epoch's training; val is not shuffled and "
        'has num_val_items images per class, so batch k is class k'
        + (' (word classes follow the image classes)' if literate else '')
    )


def files(model_name: str, continues_pre: bool) -> dict[str, str]:
    """What every file in a model's folder holds."""
    continued = ", continuing the pre-literate model's list" if continues_pre else ''
    return {
        'model': model_name + '.pth.tar, written when training completes; same contents as the last checkpoint',
        'checkpoints': 'checkpoints/epoch_<NN>.pth.tar after every epoch: dict of epoch, state_dict, optimizer, '
        'cat_scores, trainloss, valloss, rng, history',
        'cat_scores': 'cat_scores.npy',
        'trainloss': 'trainloss.npy, one loss per training batch' + continued,
        'valloss': 'valloss.npy, one loss per validation batch of the validation after every epoch' + continued,
        'initial_cat_scores': 'initial_cat_scores.npy, initial_cat_scores[k] = accuracy (%) of the starting weights '
        'on val batch k, from initial_validation; written with the first checkpoint',
        'initial_valloss': 'initial_valloss.npy, one loss per validation batch of that validation',
        'log': 'training_eta_log.txt',
    }


# --- lesion runs ----------------------------------------------------------------------------------

LESION_EPOCH_NUMBERING = (
    'counted from 0 over the pre-literate epochs, then the literate ones, then the recovery epochs after the lesion'
)

LESION_REPRODUCE = (
    'create LesionInstance with these arguments and seed=seed_used and call train(), with the same '
    'lesioned model, data and code (checked by the hashes below); GPU runs are not bit-for-bit '
    'repeatable'
)

LESION_LAYER = (
    'the dense layer: the {units} units the readout reads, i.e. IT averaged over space (avgpool), '
    "numbered 0..{last} like IT's channels"
)

WORD_CHANNEL_MEANING = (
    "the first and last unit the biased model's word readout reads, the only input of its word "
    'outputs; None for the unbiased model, whose readout reads every unit for every output'
)

OUTGOING_KEPT = "the lesioned units' weights in the readout were left as trained"


def outgoing_reset(word_channel_first: int | None) -> str:
    columns = (
        'bilinear.lin_img.weight[:, u] and, for units in the word channel, bilinear.lin_wrd.weight[:, u - '
        + str(word_channel_first)
        + ']'
        if word_channel_first is not None
        else 'linear.weight[:, u]'
    )
    return (
        "the lesioned units' weights in the readout, " + columns + ', were re-initialised as an untrained '
        "Linear's: U(-1/sqrt(in_features), 1/sqrt(in_features)); the output biases were left as trained"
    )


def lesion_method(silence: str, silent_epochs: int) -> str:
    return (
        silence + 'each lesioned unit u had its incoming weights, IT.conv.weight[u] and IT.conv.bias[u], '
        're-initialised once '
        + ('when it was released' if silent_epochs else 'before the first recovery epoch')
        + ' as PyTorch initialises an untrained Conv2d (the state the network started from): '
        'U(-1/sqrt(fan_in), 1/sqrt(fan_in)) with fan_in = 256*3*3; afterwards they train like every other weight'
    )


def silence(silent_epochs: int, max_epochs: int) -> str:
    if not silent_epochs:
        return ''
    if silent_epochs == max_epochs:
        return (
            "each lesioned unit u was silenced for the whole run (IT's output channel u multiplied by 0 in every "
            'forward pass, so it neither drives the readout nor learns) and never released: a permanent lesion. '
            'Had it been released, '
        )
    return (
        'each lesioned unit u was silenced for the first ' + str(silent_epochs) + " recovery epochs (IT's output "
        'channel u multiplied by 0 in every forward pass, so it neither drives the readout nor learns; only weight '
        'decay changes its weights), and released at the start of epoch epochs.first + '
        + str(silent_epochs)
        + ' (released_at): '
    )


LESION_RANDOM_NUMBERS = 'drawn from a torch.Generator seeded with seed_used, apart from the global RNG'

FILTER_NORM = "mean L2 norm of the lesioned units' IT conv filters before and after their re-initialisation"

SELECTIVITY_DEFINITION = (
    'selectivity[u] = (mean activation of unit u to the val words - its mean activation to the '
    'selectivity sample) / SD of its activations to the selectivity sample (ddof=1); u is '
    'word-selective when selectivity[u] >= threshold. An SD of 0 gives +-inf, or nan (not '
    'selective) when the two means are equal'
)
SELECTIVITY_WORDS = 'all val word images of the word set (data.wordset), through its val transform'
SELECTIVITY_IMAGES = 'the selectivity sample, data.selectivity_images'

LESION_EPOCH_PROCEDURE = (
    'before the first epoch the intact model is validated and its units measured '
    '(intact_validation), then it is lesioned (lesion), then the lesioned weights are validated '
    'and measured (initial_validation); each epoch then trains on one shuffled pass over the '
    'train set, validates (no gradients) on the whole val set and measures the units, so history '
    'val values are for the weights after that epoch, and the last entry is the finished model. '
    "To measure the units is to record, in eval mode, each dense-layer unit's mean activation "
    'to the val words and val images during the validation, then its activations to the '
    'selectivity sample (see unit_tracking in files)'
)

LESION_METRICS = (
    '. history[].mean_word_activation_lesioned and _other are the mean over the lesioned / the other '
    "units of each unit's mean activation to the val words (None when there are none), "
    'word_selective_lesioned and _other how many of them are word-selective (see lesion.selectivity); '
    'intact_validation and initial_validation have them too'
)

LESION_CAT_SCORES = (
    "cat_scores[epoch, k] = accuracy (%) on val batch k after that epoch's training; val is not "
    'shuffled and has num_val_items images per class, so batch k is class k (word classes follow the '
    "image classes). The rows before the recovery epochs are the lesioned model's"
)

LESION_FILES = {
    'trainloss': "trainloss.npy, one loss per training batch, continuing the lesioned model's list",
    'valloss': (
        'valloss.npy, one loss per validation batch of the validation after every epoch, continuing the '
        "lesioned model's list"
    ),
    'initial_cat_scores': (
        'initial_cat_scores.npy, initial_cat_scores[k] = accuracy (%) of the lesioned weights, '
        'before the first recovery epoch, on val batch k, from initial_validation; written with the '
        'first checkpoint'
    ),
    'intact_cat_scores': (
        'intact_cat_scores.npy, the same for the intact model, from intact_validation; written with '
        'the first checkpoint'
    ),
    'intact_valloss': 'intact_valloss.npy, one loss per validation batch of that validation',
    'unit_tracking': (
        'unit_tracking.npz, rewritten with every checkpoint: arrays with one row per measurement and one '
        'column per dense-layer unit: mean_words (mean activation to the val words), mean_val_images '
        '(to the val images), mean_images and sd_images (mean and SD over the selectivity sample), '
        'selectivity and word_selective (lesion.selectivity). stage names the rows (intact, lesioned, '
        "then epoch <n>), epoch gives the epoch whose weights were measured (the lesioned model's last "
        'for the first two rows); rows not measured yet are nan (word_selective False). Also units (the '
        'lesioned ones) and threshold'
    ),
}
