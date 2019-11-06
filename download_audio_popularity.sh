#!/bin/bash

mkdir -p data/popularity_prediction_raw_audio;

wget https://storage.googleapis.com/cornell-tech-sdl-podcast-dataset/popularity_prediction_raw_audio.tar.gz -O data/popularity_prediction_raw_audio.tar.gz;
tar -xzf data/popularity_prediction_raw_audio.tar.gz -C data/popularity_prediction_raw_audio --verbose;
