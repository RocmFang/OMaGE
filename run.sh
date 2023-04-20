#!/bin/bash
g++ -O3 -m64 -I/opt/intel/oneapi/mkl/2022.0.2/include -Iinclude OMaGE.cpp  ./csdb.cpp -Wl,--start-group /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2022.0.2/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -lgflags -lpmemobj -lredsvd -lnuma -o OMaGE
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/compiler/2022.0.2/linux/compiler/lib/intel64_lin
./OMaGE