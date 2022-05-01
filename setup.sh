#! /usr/bin/bash

if [ -z "$SAVE_DIR" ]; then
    export SAVE_DIR=$PWD
fi

if [ -d "./conf" ]; then
    rm -rf "./conf"
fi

mkdir "./conf"

directories="data logs images models"

for value in $directories
do
    path="$SAVE_DIR/$value"
    echo "${value}_dir: \"$path\"">> ./conf/config.yaml

    if [ -d $path ]; then
        rm -rf "$path"
    fi

    mkdir -m 770 "$path"
done

if [ -d "./venv" ]; then
    rm -rf "./venv"
fi

python3.8 -m virtualenv venv
source $PWD/venv/bin/activate
python -m pip install -r requirements.txt