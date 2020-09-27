#!/bin/bash
#conda init bash
eval "$(conda shell.bash hook)"
conda activate comp135_2020f_env
cd comp135-20f-assignments/labs/
jupyter notebook
