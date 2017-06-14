# encoding-of-phonology

This repository contains code to reproduce the results from:

- Afra Alishahi, Marie Barking and Grzegorz Chrupała. 2017. 
  Encoding of phonology in a recurrent neural model of grounded speech, In proceedings of CoNLL
  
## Installation

First, download and install funktional version 0.6: https://github.com/gchrupala/funktional/releases/tag/0.6

Second, install the code in the current repo:

    python setup.py develop

You also need to download and unpack the files `data.tgz` and `models.tgz` from http://doi.org/10.5281/zenodo.804392.
The files in `models.tgz` contain the pre-trained models used for the analyses in the paper.

After unpacking these files you should have the directories `data` and `models`.


## Usage

See [src/README.md](src/README.md)
