#!/bin/sh

echo "Preprocess Raw Text Input: "
python3.5 text_preprocessing/preprocess.py  -i '../data_extracted/dataset_20201122_161404' -o '../data_preprocessed/' -a 'tokenize,ssplit,pos'
echo "Done! Generated Preprocessed Data"


