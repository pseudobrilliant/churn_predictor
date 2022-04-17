#! /usr/bin/bash

if [ -z "$SAVE_DIR" ]
then
    export SAVE_DIR=$PWD
fi

rm .env
directories="data logs images models"
for value in $directories
do
    path="$SAVE_DIR/$value"
    export ${value^^}_DIR=$path
    echo "${value^^}_DIR=\"$path\"">>./.env
    rm -rf "$path"
    mkdir -m 770 "$path"
done

python3.8 -m virtualenv venv
source venv/bin/activate
python -m pip install -r requirements.txt