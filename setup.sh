#! /usr/bin/bash

directories="./conf ./venv ./data"

for dir in $directories
do

if [ -d $dir ]; then
    rm -rf $dir
fi

mkdir $dir

done

deactivate
python -m virtualenv venv
. $PWD/venv/bin/activate
python -m pip install -r requirements.txt