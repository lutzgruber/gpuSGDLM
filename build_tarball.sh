#!/bin/bash

rm -R build
mkdir build

cp rSGDLM/src build -R
cp rSGDLM/DESCRIPTION build
cp rSGDLM/NAMESPACE build
cp rSGDLM/configure.ac build

cd build

autoconf

R CMD build . 

cp *.tar.gz ../
