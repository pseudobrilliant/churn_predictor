#! /usr/bin/bash

directories="./conf ./venv"

for dir in $directories
do

if [ -d $dir ]; then
    rm -rf $dir
fi

done


python3.8 -m virtualenv venv
source $PWD/venv/bin/activate
python -m pip install -r requirements.txt