This repository contains the code for Parallel-CRNN, CRNN and LSTM models trained separately to identify 182 different Brazilian bird genera by their respective songs.

## Packages
* Python 3.7.6
* Tensorflow - 1.15.0
* Numpy, Pandas
* Librosa - 0.7.2

## Raw data:
Download high quality bird songs of brazilian birds from https://www.xeno-canto.org/ using their API.

## Preprocessed data
Raw audio converted to mel-spectograms and pickled. 2 files, one for testing and validation and the other for training.

## Notebooks
01. download_bird_song_meta.ipynb - Download meta-data on bird songs of brazilian birds, filter and write to a csv.
02. download_bird_songs.ipynb - Download audio in .mp3 format, convert to .wav and trim to 20 seconds. Process audio, convert to mel-spectrograms and pickle them.
03. model.ipynb - Parallel CRNN, CRNN and LSTM models.

## Best Model
Out of the three, CRNN model had the best performance with a testing accuracy of 41%. Considering the number of classes (182) and the size of the training dataset (~8500), 41% testing accuracy is pretty good.
