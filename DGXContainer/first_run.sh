#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd  )"

docker build -t nvcr.io/eavise/adversarial_yolo:v0 .

