#!/usr/bin/env bash

./build.sh

docker save pat | gzip -c > PAT.tar.gz
