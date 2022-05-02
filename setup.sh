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
python3.8 -m virtualenv venv
source $PWD/venv/bin/activate
python -m pip install -r requirements.txt