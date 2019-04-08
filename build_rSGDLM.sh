#!/bin/bash

rm -R build
mkdir build

cp rSGDLM build -R

cd build/rSGDLM

autoconf

R CMD build . 
