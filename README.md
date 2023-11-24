# Learning Location-guided Time-series Shapelets

This repository contains the Pytorch implementation of "Learning Location-guided Time-series Shapelets".

## Dependencies
- pytorch 1.13.1 and above
- scikit-learn 1.2.1 and above
- numba 0.57.0 and above

## How to use

1. Download the UCR archives in tsv format.

The 2018 version of the UCR archive can be found [here](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/).
The following assumes that the directory is `c:/hoge/UCRArchive_2018`.

2. Run the code.

For example, you can run the command on ItalyPowerDemand dataset
```shell
python main.py --dataset_dir 'C:/hoge/UCRArchive_2018/ItalyPowerDemand' --K 30 --dropout 0.25
```
to test the model.
See Tables II and III in the paper for other datasets.
