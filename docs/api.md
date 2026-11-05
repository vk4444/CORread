# API reference

## Stimuli

::: corread.stimuli
    options:
      members: [generate_image, generate_training_set, regenerate_training_set, report_wordset_status]

## Training

::: corread.training.TrainingInstance
    options:
      members: [train, is_complete, report_status]

::: corread.training.FamilyTrainingInstance

## Lesions

::: corread.lesion.LesionInstance
    options:
      members: [word_selectivity, train, report_status]

## Networks

::: corread.models
    options:
      members: [network_class, build_network, NETWORKS]

::: corread.models.CORnet

::: corread.models.ImageClassifier

::: corread.models.BiasedWordClassifier

::: corread.models.UnbiasedWordClassifier

::: corread.models.cornet_z

::: corread.models.cornet_s

::: corread.models.layers

## Sharing through the Hugging Face Hub

::: corread.hub

## ImageNet

::: corread.imagenet
    options:
      members: [prepare_imagenet]

## Configuration

::: corread.paths
    options:
      members: [get_path, get_section, find_config, clear_cache]
