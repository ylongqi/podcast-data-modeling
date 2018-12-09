#!/bin/bash

mkdir pretrained_model;

wget https://www.dropbox.com/s/zn8x1p4e7knq9tb/alpr.data-00000-of-00001?dl=1 -O pretrained_model/alpr.data-00000-of-00001;
wget https://www.dropbox.com/s/di6j8dd9hnr5vyn/alpr.index?dl=1 -O pretrained_model/alpr.index;
wget https://www.dropbox.com/s/ll3uw4taovinvv4/alpr.meta?dl=1 -O pretrained_model/alpr.meta;