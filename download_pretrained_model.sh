#!/bin/bash

mkdir pretrained_model;

wget https://storage.googleapis.com/cornell-tech-sdl-podcast-dataset/alpr.data-00000-of-00001 -O pretrained_model/alpr.data-00000-of-00001;
wget https://storage.googleapis.com/cornell-tech-sdl-podcast-dataset/alpr.index -O pretrained_model/alpr.index;
wget https://storage.googleapis.com/cornell-tech-sdl-podcast-dataset/alpr.meta -O pretrained_model/alpr.meta;
