#! /usr/bin/bash

directories="logs images models"

for value in $directories
do
    rm -rf $value
    mkdir -m 770 $value
done

python3.8 -m virtualenv venv
source venv/bin/activate
python -m pip install -r requirements.txt