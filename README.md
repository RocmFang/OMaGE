<meta name="robots" content="noindex">


## This codebase is for the paper: On Billion-Scale Graph Embedding–Towards Bridging the Performance Gap between DRAM and Persistent Memory

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

### Run in Single-machine Environment
```
mpiexec -n 8 ./bin/huge_walk -g ../dataset/LJ-8.data-r -p ../dataset/LJ-8.part -v 2238731 -w 2238731 --make-undirected -o ./out/walks.txt -eoutput ./out/LJ-r_emb.txt -size 128 -iter 1 -threads 72 -window 10 -negative 5  -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -debug 2
```

### Run in Distributed Environment
- Copy the train dataset to the same path of each machine, or simply place it to a shared file system, such as NFS
- Touch a host file to write each machine's IP address, such as ./hosts
- Invoke the application with MPI 

```
mpiexec -hostfile ./hosts -n 8 ./bin/huge_walk -g ../dataset/LJ-8.data-r -p ../dataset/LJ-8.part -v 2238731 -w 2238731 --make-undirected -o ./out/walks.txt -eoutput ./out/LJ-r_emb.txt -size 128 -iter 1 -threads 72 -window 10 -negative 5  -batch-size 21 -min-count 0 -sample 1e-3 -alpha 0.01 -debug 2
```

### Check the output files in "out" directory
