<meta name="robots" content="noindex">


## This codebase is for the paper: On Billion-Scale Graph Embedding--Towards Bridging the Performance Gap between DRAM and Persistent Memory

# Prerequisites

- Ubuntu 16.04
- Linux kernel 4.15.0
- g++ 9.4.0
- [PMDK](https://github.com/pmem/pmdk/)
- [Eigen](http://eigen.tuxfamily.org)
- [redsvd](https://code.google.com/p/redsvd/)
- [Openmp](https://www.openmp.org/)
- [gflags](https://github.com/gflags/gflags)
- [MKL 2022.0.2](https://software.intel.com/en-us/mkl)

# Datasets

The evaluated dataset soc-LiveJournal are prepraed in the "data" directory.

Since the the space limited of the repository, the other datasets [Twitter-2010](https://law.di.unimi.it/datasets.php), [Com-Orkut](https://snap.stanford.edu/) and [Twittwer](http://datasets.syr.edu/pages/datasets.html) can be found in their open resource.

# Setup

First create a PMDK pool to allocate the Persistent memory space

```

pmempool create --layout OMaGE --size 500G obj [persistent memory mount path]

```


# Graph Embedding

To start the embedding, we fist need to complie application executable files

```bash
g++ -O3 -m64 -I/opt/intel/oneapi/mkl/2022.0.2/include -Iinclude ProNE.cpp  ./csdb.cpp -Wl,--start-group /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -lgflags -lpmemobj -lredsvd -lnuma -o OMaGE
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin

```

Then run the OMaGE executed file

```
./OMaGE
```

We also probvide a script file to incorperate the complied and exectution procedure.

```
./run.sh
```


