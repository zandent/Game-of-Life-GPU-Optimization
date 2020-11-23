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
./initboard <# of rows> <# of cols>
./gol <# of iterations> inputs/test.pbm <output file .pbm>
```
To check timing
```
${PATH}/time -f "%e real" ./gol 10000 inputs/1k.pbm outputs/1k.pbm
```
## Authors
* shirwang
* xuwen9
* zandent
