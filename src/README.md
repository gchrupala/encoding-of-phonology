# Experiments

## 5.1 Phoneme decoding

### Figure 1

Compute results:
```
python2.7 analyze.py decoding
```

Generate plot `../figures/decode.pdf`:
```
Rscript phoneme_decode.R
```

## 5.2 Phoneme discrimination

### Table 3

Compute results:
```
python2.7 analyze.py abx_all
```
Results will be written in the file `abx-all.csv`.


### Figure 2

Compute results:
```
python2.7 analyze.py abx_classes
```

Generate plot `../figures/abx_cv_same.pdf`.
```
Rscript abx_classes.R
```


## 5.3 Organization of phonemes

### Figure 3

Results:
```
python2.7 analyze.py correlation
```
Figure `../figures/correlation_mfcc.pdf`.
```
Rscript correlation.R
```

### Figure 4 and Figure 5

Results
```
python2.7 analyze.py clustering
```
Figure `../figures/hier_ari.pdf` and `../figures/dendro.pdf`.
```
Rscript randindex.R
python2.7 analyze.py dendro
```

## 5.4 Synonym discrimination

### Figure 5

Results:

```
python2.7 analyze.py synonyms
```
Figure `../figures/synonym.pdf`.
```
Rscript synonyms.R
```




