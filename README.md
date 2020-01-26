This repository contains the code for a Convolutional Recurrent Neural Network to identify 141 genuses of Brazilian birds by their respective songs.

## Packages
* Python 3.7.6
* Tensorflow - 1.15.0
* Numpy, Pandas
* Librosa - 0.7.2

## Raw data:
Download high quality bird songs of brazilian birds from https://www.xeno-canto.org/ using their API
6.2 GB raw data containing 6207 recordings of songs of 141 different bird genuses.

## Preprocessed data
Raw audio converted to mel-spectograms and pickled. 2 files, one for testing and validation and the other for training.

## Notebooks
01. download_bird_song_meta.ipynb - Download meta-data on bird songs of brazilian birds, filter and write to a csv.
02. download_bird_songs.ipynb - Download audio in .mp3 format, convert to .wav and trim to 20 seconds.
03. convert_to_npz.ipynb - Process audio, convert to mel-spectrograms and pickle them.
04. model.ipynb - Parallel CRNN, CRNN and LSTM models
