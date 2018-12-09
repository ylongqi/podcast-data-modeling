# Podcast data modeling

This repository contains a podcast dataset and an implementation of the **A**dversarial **L**earning-based **P**odcast **R**epresentation (**ALPR**) introduced in the following paper:

**Longqi Yang, Yu Wang, Drew Dunne, Michael Sobolev, Mor Naaman and Deborah Estrin. 2018. [*More than Just Words: Modeling Non-textual Characteristics of Podcasts*](http://www.cs.cornell.edu/~ylongqi/paper/YangWDSNE19.pdf). In Proceedings of [WSDM’19](http://www.wsdm-conference.org/2019/).**

A pretrained model is also included. Please direct any questions to [Longqi Yang](http://www.cs.cornell.edu/~ylongqi/).

#### If you use this data or algorithm, please cite:

```
@inproceedings{yang2019podcast,
  title={More than Just Words: Modeling Non-textual Characteristics of Podcasts},
  author={Yang, Longqi and Wang, Yu and Dunne, Drew and Sobolev, Michael and Naaman, Mor and Estrin, Deborah},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  year={2019},
  organization={ACM}
}
```

## Data descriptions

### Raw podcast audio URLs
* Set $S_a$ (used for ALPR training and attributes prediction): [audio_links/podcast_episodes_sa.txt](audio_links/podcast_episodes_sa.txt).
* Set $S_b$ (used for popularity prediction): [audio_links/podcast_episodes_sb.txt](audio_links/podcast_episodes_sa.txt).

Each line of these files contains an podcast episode represented by a JSON object with the following fields:
```json
{
    "url": the URL to download the raw audio,
    "itunes_channel_id": the iTunes channel that the episode belongs to,
    "id": a unique epsiode ID,
    "title": the title of the episode
}
```

### Prediction labels
* Energy prediction:
    * Training data: [attributes_labels/energy_train.txt](attributes_labels/energy_train.txt).
    * Testing data: [attributes_labels/energy_test.txt](attributes_labels/energy_test.txt).
    * Format: each line contains a snippet (~12s) represented by a JSON object:
    ```json
    {
        "label": 1 denotes energetic and 0 denotes calm,
        "id": the episode ID,
        "offset": the temporal position of the snippet (starting timestamp=12*offset seconds)
    }
    ```
* Seriousness prediction:
    * Training data: [attributes_labels/seriousness_train.txt](attributes_labels/seriousness_train.txt).
    * Testing data: [attributes_labels/seriousness_test.txt](attributes_labels/seriousness_test.txt).
    * Format: follows the energy prediction task except for the labels --- 1 denotes serious and 0 denotes humorous.
* Popularity prediction:
    * Training data: [popularity_labels/popularity_train.txt](popularity_labels/popularity_train.txt).
    * Testing data: [popularity_labels/popularity_test.txt](popularity_labels/popularity_test.txt).
    * Format: each line contains an episode represented by a JSON object:
    ```json
    {
        "label": 1 denotes popular and 0 denotes long tail,
        "id": the episode ID,
        "length": the number of leading audio snippets used for the prediction (each snippet is 12s in length)
    }
    ```

### Prediction features and raw audio (caveats: files are large)
* Energy and seriousness predictions:
    * Spectrograms: 
        * Download using the script [download_spectrograms_attributes.sh](download_spectrograms_attributes.sh).
        * Spectrograms are stored in the *.npy (numpy array)* format and are named following the rule:
        ```
        data/attributes_prediction_spectrograms/e_[episode id]_[offset].npy
        ```
    * Raw audio:
        * Download using the script [download_audio_attributes.sh](download_audio_attributes.sh).
        * Audio is stored in the *.wav* format and is named following the rule:
        ```
        data/attributes_prediction_raw_audio/e_[episode id]_[offset].wav
        ```
* Popularity prediction:
    * Spectrograms:
        * Download using the script [download_spectrograms_popularity.sh](download_spectrograms_popularity.sh).
        * Spectrograms are stored in the *.npy (numpy array)* format and are named following the rule:
        ```
        data/popularity_prediction_spectrograms/e_[episode id]_[0 -- length-1].npy
        ```
    * Transcriptions:
        * Download using the script [download_transcriptions_popularity.sh](download_transcriptions_popularity.sh).
        * Transcriptions are stored in the *.txt* format and are named following the rule:
        ```
        data/popularity_prediction_transcriptions/e_[episode id].txt
        ```
        * A transcription file lists transcribed words with the following format (a word per line):
        
        **a spoken word** \t **starting time (ms)** \t **end time (ms)**
    * Raw audio:
        * Download using the script [download_audio_popularity.sh](download_audio_popularity.sh).
        * Audio is stored in the *.wav* format and is named following the rule:
        ```
        data/popularity_prediction_raw_audio/e_[episode id]_[0 -- length-1].wav
        ```

## Reproducing experimental results using the pretrained model

* Download pretrained **ALPR** using the script [download_pretrained_model.sh](download_pretrained_model.sh).
* Load the pretrained model and extract **ALPR** features as follows:

```python
from alpr_extractor import ALPRExtractor

extractor = ALPRExtractor()
extractor.load_model(path='pretrained_model/alpr')
features = extractor.forward((spectrograms + 2) / 2)
```

* An example notebook to reproduce experimental results: [energy_prediction.ipynb](energy_prediction.ipynb).