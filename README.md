# Game of Life GPU Optimazation
ECE1782 Project Fall 2020
## Steps
To build project
```
cd src
make
```
To run Game of Life in `src/`
```
./initboard <# of rows> <# of cols> <generated file .pbm>
./gol <# of iterations> inputs/test.pbm <output file .pbm>
```
To check timing
```
${PATH}/time -f "%e real" ./gol <# of iterations> <input file .pbm> <output file .pbm>
```
For testing and plotting time results
```
./measure.py
```
## Authors
* shirwang
* xuwen9
* zandent
