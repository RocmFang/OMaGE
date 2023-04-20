#ifndef CSDB_H_
#define CSDB_H_

#include <libpmemobj++/p.hpp>
#include <libpmemobj++/pool.hpp>
#include <libpmemobj++/transaction.hpp>
#include <libpmemobj++/persistent_ptr.hpp>
#include <libpmemobj++/container/vector.hpp>
#include <thread>
#include <vector>
#include <iostream>
#include <fstream>
#include <numeric>
#include <Eigen/Dense>
#include <string>
#include "constants.h"
#include "numa.h"
#include "timer.hpp"
namespace CSDB
{
    typedef pmem::obj::vector<float> pvf;
    typedef pmem::obj::vector<int> pvi;
    typedef pmem::obj::vector<long long> pvl;
    class Smatrix
    {
        friend void Calculate(const Smatrix&,const Eigen::MatrixXf &,long long,int,int,Eigen::MatrixXf &, int, std::vector<double> &);
        friend void Calculate1(int *, int *,float *,const Eigen::MatrixXf &, long long, long long, int, int, Eigen::MatrixXf &);    
    public:
        Smatrix()=default;
        Smatrix(uint64_t num):numV(num){}
        void setFromFile(pmem::obj::pool_base&, std::string,std::string);
        Eigen::MatrixXf operator*(const Eigen::MatrixXf&) const;
        void add_I(pmem::obj::pool_base&);
        void parallel_mul(int *,int *,float *,int *,int *,float *,const Eigen::MatrixXf &,const Eigen::MatrixXf &, Eigen::MatrixXf &, Eigen::MatrixXf &);
        
        pmem::obj::p<uint64_t> numV=0;
        pmem::obj::p<uint64_t> numE=0;
        pmem::obj::persistent_ptr<pvi> col=nullptr;
        pmem::obj::persistent_ptr<pvi> deg_list=nullptr;
        pmem::obj::persistent_ptr<pvf> data=nullptr;
        pmem::obj::persistent_ptr<pvl> add_aux=nullptr;
        // pmem::obj::persistent_ptr<pmem::obj::vector<pvi>> trans_aux=nullptr;

        std::vector<std::vector<long long>> trans_aux;
        
        double Hm_init;
        double Hm_initial(int);

        // int num_thread1;
        // int num_thread2;

        std::vector<std::pair<long long, long long>> thread_allocate(int) const;
        std::vector<std::pair<long long, long long>> thread_allocate_entropy(int) const;

    };
    void Calculate(const Smatrix&,const Eigen::MatrixXf &,long long,int,int,Eigen::MatrixXf &, int, std::vector<double> &);
    void Calculate1(int *,int *,float *,const Eigen::MatrixXf &, long long, long long, int, int, Eigen::MatrixXf &);
    void setNbThreads(int);
    void setNbThreads1(int);
    // void setNbThreads_numa(int);
}



#endif