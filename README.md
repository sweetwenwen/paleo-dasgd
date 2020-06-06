# PALEO DaSGD

This is code for paper "DaSGD: Squeezing SGD Parallelization Performance in Distributed Training Using Delayed Averaging"

## Installation PALEO

reference [PALEO](https://github.com/TalwalkarLab/paleo)

Paleo uses the following dependencies:

- numpy
- click
- six
- cuDNN (Optional. Use `--use_only_gemm` to disable cuDNN heuristics)

Use pip to install the depenencies with the pinned versions:

```
pip install -r requirements.txt
```
Tested with Python 2.7, cuDNN v4 on Ubuntu 14.04.

To install Paleo, run the following command in the cloned directory:

    python setup.py install
    
## Use Our modified PALEO

Copy our file in [PALEO](https://github.com/TalwalkarLab/paleo)/paleo

And, install the PALEO again

    python setup.py install

## Usage

Paleo provides programmatic APIs to retrieve runtime estimations.

The following is an example of V100 under weak scaling.

    python paleo/profiler.py simulate ./nets/nin.json --use_only_gemm --use_pipeline --batch_size 128 --num_workers 256 --scaling weak --device_name V100 --network_name infiniband 

    
    
