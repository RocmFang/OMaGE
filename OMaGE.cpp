#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <ctime>
#include <cmath>
#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <boost/math/special_functions/bessel.hpp>
#include "redsvd.h"
#include "numa.h"

using namespace Eigen;
using namespace boost;
using namespace CSDB;
using namespace pmem::obj;
using namespace REDSVD;

const float EPS = 0.00000000001f;


DEFINE_string(filename1, "./data/Wiki-Vote_sort_f.txt", "Filename for edgelist file.");
DEFINE_string(filename2, "./data/Wiki-Vote_deg_f.txt", "Filename for edgelist file.");
DEFINE_string(pool_path, "/mnt/pmem0/pmem_pool", "PMEM POOL PATH");
DEFINE_string(layout, "OMaGE", "POOL layout");
DEFINE_int64(pool_size, uint64_t(1073741824) * uint64_t(500), "POOL layout"); // 512 pool size
DEFINE_string(emb1, "sparse.emb", "Filename for svd results.");
DEFINE_string(emb2, "spectral.emb", "Filename for svd results.");
DEFINE_int32(num_node, 7115, "Number of node in the graph.");
DEFINE_int32(num_rank, 128, "Embedding dimension.");
DEFINE_int32(num_step, 10, "Number of order for recursion.");
DEFINE_int32(num_iter, 5, "Number of iter in randomized svd.");
DEFINE_int32(num_thread, 30, "Number of threads.");
DEFINE_int32(num_thread1, 30, "Number of threads for *.");
// DEFINE_int32(num_thread2, 15, "Number of threads for NUMA optimized *.");
DEFINE_double(theta, 0.5, "Parameter of ProNE");
DEFINE_double(mu, 0.1, "Parameter of ProNE");


MatrixXf &l2Normalize(MatrixXf &mat)
{
    for (int i = 0; i < mat.rows(); ++i)
    {
        float ssn = sqrt(mat.row(i).squaredNorm());
        if (ssn < EPS)
            ssn = EPS;
        mat.row(i) = mat.row(i) / ssn;
    }
    return mat;
}
MatrixXf getSparseEmbedding(pool_base &pop, Smatrix &A, int rank, int num_thread1)
{
    int num = A.numV; 
                                                                   
    persistent_ptr<Smatrix> pB = nullptr, pC = nullptr, pE = nullptr, pF = nullptr; // pB-F are pointers to Smatrix

    transaction::run(pop, [&]()
                     {
        pB=make_persistent<Smatrix>(num);
        pB->numE=A.numE;
        pB->deg_list=make_persistent<pvi>(num);
       
        pB->data=make_persistent<pvf>(A.numE);
        
        pB->col=make_persistent<pvi>(A.numE);
        
        // pB->add_aux=A.add_aux;
        // pB->trans_aux=A.trans_aux; 
        }); // allocate pmem space for pB

    *(pB->deg_list) = *(A.deg_list);
    *(pB->data) = *(A.data);
    *(pB->col) = *(A.col);


    float sum = 0.0f;
    long long start = 0, end;
    auto &deg = *(A.deg_list);
    auto &dat = *(pB->data);

    for (int i = 0; i < num; i++)
    {
        end = start + deg[i];
        for (long long j = start; j < end; j++)
            dat[j] /= deg[i] * 2;
        start = end;
    }              // assignment for pB
    auto &B = *pB; // B=l1normallize(A)(48-74)

    transaction::run(pop, [&]()
                     {
                         pC = make_persistent<Smatrix>(num);
                         pC->numE = B.numE;
                         pC->deg_list = make_persistent<pvi>(num);
                         pC->data = make_persistent<pvf>(B.numE);
                         pC->col = make_persistent<pvi>(B.numE);
                        //  pC->add_aux = B.add_aux;
                        //  pC->trans_aux = B.trans_aux; 
                         }); // allocate pmem space for pC

    *(pC->deg_list) = *(B.deg_list);
    *(pC->col) = *(B.col);


    long long ind = 0;
    for (int i = 0; i < num; i++)
    {
        // for (auto it : (*(B.trans_aux))[i])
        // for (auto it : B.trans_aux[i])
        for (auto it : A.trans_aux[i])
        {
            (*(pC->data))[ind++] = (*(B.data))[it];
        }

    } // assignment for pC


    auto &C = *pC;                      // C=l1normalize(B)(75-92)
    persistent_ptr<pvf> ptmp = nullptr; // ptmp points to a float array
    transaction::run(pop, [&]()
                     { ptmp = make_persistent<pvf>(num); }); // allocate pmem space for ptmp
    start = 0;
    sum = 0.0f;
    for (int i = 0; i < num; i++)
    {
        float t = 0.0f;
        end = start + (*C.deg_list)[i];
        for (long long j = start; j < end; j++)
            t += (*C.data)[j];
        t = pow(t, 0.75);
        (*ptmp)[i] = t;
        sum += t;
        start = end;
    }

    for (auto &it : *ptmp)
        it /= sum;     // assignment for ptmp
    auto &tmp = *ptmp; // tmp[i]=pow(C.row(i).sum(),0.75);(94-109)  tmp[i]/=sum(tmp);(110-111)
    transaction::run(pop, [&]()
                     {
                         pE = make_persistent<Smatrix>(num);
                         pE->numE = A.numE;
                         pE->deg_list = make_persistent<pvi>(num);

                         pE->data = make_persistent<pvf>(A.numE);
                         pE->col = make_persistent<pvi>(A.numE);

                        //  pE->add_aux = A.add_aux; 
                        //  pE->trans_aux = A.trans_aux; 
                         }); // allocate pmem space for pE
    *(pE->deg_list) = *(A.deg_list);
    *(pE->col) = *(A.col);

    for (long long i = 0; i < A.numE; i++)
        (*pE->data)[i] = (*(A.data))[i] * tmp[(*A.col)[i]]; // assignment for pE
    auto &E = *pE;                                          // E=A*tmp  (113-127) //fake_multiply
    for (auto &it : *(B.data))
    {
        if (it <= 0)
            it = 1.0f;
        it = log(it);
    } // B=validate(B);B=smfLog(B);(129-134)
    for (auto &it: *(E.data))
    {
        if (it <= 0)
            it = 1.0f;
        it = log(it);
    } // E=validate(E);B=smfLog(E);(135-140)

    transaction::run(pop, [&]()
                     {
                         pF = make_persistent<Smatrix>(num);
                         pF->numE = B.numE;
                         pF->deg_list = make_persistent<pvi>(num);

                         pF->data = make_persistent<pvf>(B.numE);
                         pF->col = make_persistent<pvi>(B.numE);

                        //  pF->add_aux = B.add_aux;
                        //  pF->trans_aux = A.trans_aux;
                          }); // allocate pmem space for pF
    *(pF->deg_list) = *(B.deg_list);
    *(pF->col) = *(B.col);
    

    for (long long i = 0; i < B.numE; i++)
        (*pF->data)[i] = (*pB->data)[i] - (*pE->data)[i]; // assignment for pF
    auto &F = *pF;                                        // F=B-E(141-155)

    F.Hm_initial(num_thread1);

    RedSVD redsvd;
    redsvd.run(pop, F, rank, A);                                                           // redsvd of F
    MatrixXf emb = redsvd.matrixU() * redsvd.singularValues().cwiseSqrt().asDiagonal(); // get sparse embedding

    return l2Normalize(emb);
}

float bessel(int a, float b)
{
    return boost::math::cyl_bessel_i(a, b);
}

MatrixXf getSpectralEmbedding(pool_base &pop, Smatrix &A, MatrixXf &a, int step, float theta, float mu, int num_thread1, int num_thread2)
{
    std::cout << "SpectralEmbedding starting" << std::endl;
    time_t t1 = time(NULL);
    int num_node = a.rows(), rank = a.cols(); // 

    A.add_I(pop);                             // A=A+I

    persistent_ptr<Smatrix> pB = nullptr;     // pB points to a Smatrix
    transaction::run(pop, [&]()
                     {
        pB=make_persistent<Smatrix>(num_node);
        pB->numE=A.numE;
        pB->deg_list=make_persistent<pvi>(num_node);
       
        pB->data=make_persistent<pvf>(A.numE);
        
        pB->col=make_persistent<pvi>(A.numE);
        
        // pB->add_aux=A.add_aux;
        // pB->trans_aux=A.trans_aux; 
        }); // allocate pmem space for pB

    *(pB->deg_list) = *(A.deg_list);
    *(pB->data) = *(A.data);
    *(pB->col) = *(A.col);
    float sum = 0.0f;
    long long start = 0, end;
    auto &deg = *(A.deg_list);
    auto &dat = *(pB->data);


    for (int i = 0; i < num_node; i++)
    {
        end = start + deg[i];
        for (long long j = start; j < end; j++)
            dat[j] /= -deg[i] * 2;
        start = end;
    }              // assignment for pB
    auto &B = *pB; // B=l1normalize(A)(172-198)
    for (int i = 0; i < num_node; i++)
            (*B.data)[(*A.add_aux)[i]]+= 1 - mu; // B=(1-mu)*I-B (200-201)
        // (*B.data)[(*B.add_aux)[i]] += 1 - mu; // B=(1-mu)*I-B (200-201)


    (*A.add_aux).clear();
    (*A.add_aux).shrink_to_fit();
    (A.trans_aux).clear();
    (A.trans_aux).shrink_to_fit();


    // MatrixXf Lx0 = a;

    B.Hm_initial(num_thread1);

    // /*
    // With NUMA
    // */
    
    time_t t2 = time(NULL);

    int *col0 =  (int *)numa_alloc_onnode(B.numE * sizeof(int), 0);
    // int *col1 =  (int *)numa_alloc_onnode(M.col.size()*sizeof(int), 1);
    // float *data0 = (float *)numa_alloc_onnode(M.data.size()*sizeof(float), 0);
    float *data1 = (float *)numa_alloc_onnode(B.numE * sizeof(float), 1);
    int *deg_list0 = (int *)numa_alloc_onnode(B.numV * sizeof(int), 0); 
    int *deg_list1 = (int *)numa_alloc_onnode(B.numV * sizeof(int), 1);

    
    memcpy(&(*deg_list0), &(*B.deg_list)[0], B.numV * sizeof(int));
    memcpy(&(*deg_list1), &(*B.deg_list)[0], B.numV * sizeof(int));
    memcpy(&(*col0), &(*B.col)[0], B.numE * sizeof(int));
    memcpy(&(*data1), &(*B.data)[0], B.numE * sizeof(float));

    int size0 = a.size();
    MatrixXf *Lx01 = (MatrixXf *)numa_alloc_onnode(size0/2, 0);//
    MatrixXf *Lx02 = (MatrixXf *)numa_alloc_onnode(size0/2, 1);
    if((a).cols() % 2 == 0 ){//
        *Lx01 = (a).block(0,0,(a).rows(),(a).cols()/2);
        *Lx02 = (a).block(0,(a).cols()/2,(a).rows(),(a).cols()/2);
    }
    else if((a).cols() % 2 == 1){
        *Lx01 = (a).block(0,0,(a).rows(),(a).cols()/2);
        *Lx02 = (a).block(0,(a).cols()/2 ,(a).rows(),(a).cols()/2+1);
    }


    time_t t3 = time(NULL);

    MatrixXf *Lx1 = (MatrixXf *)numa_alloc_onnode(size0, 0);//
    // MatrixXf *Lx2 = (MatrixXf *)numa_alloc_onnode(size0, 1);


    *Lx1 = B * a;
    *Lx1 = 0.5 * (B * (*Lx1)) - a;

    time_t t4 = time(NULL);


    MatrixXf conv = bessel(0, theta) * a;
    conv -= 2 * bessel(1, theta) * (*Lx1);


    time_t t5 = time(NULL);

    MatrixXf *Lx11 = (MatrixXf *)numa_alloc_onnode(size0/2, 0);
    MatrixXf *Lx12 = (MatrixXf *)numa_alloc_onnode(size0/2, 1);
    MatrixXf *Lx21 = (MatrixXf *)numa_alloc_onnode(size0/2, 0);
    MatrixXf *Lx22 = (MatrixXf *)numa_alloc_onnode(size0/2, 1);

    if((*Lx1).cols() % 2 == 0 ){
        *Lx11 = (*Lx1).block(0,0,(*Lx1).rows(),(*Lx1).cols()/2);
        *Lx12 = (*Lx1).block(0,(*Lx1).cols()/2,(*Lx1).rows(),(*Lx1).cols()/2);
    }
    else if((*Lx1).cols() % 2 == 1){
        *Lx11 = (*Lx1).block(0,0,(*Lx1).rows(),(*Lx1).cols()/2);
        *Lx12 = (*Lx1).block(0,(*Lx1).cols()/2 ,(*Lx1).rows(),(*Lx1).cols()/2+1);
    }
    

    // numa_free(Lx1, size0);
    (*Lx1).resize(0,0);


    int sizec = conv.size();
    MatrixXf *conv1 = (MatrixXf *)numa_alloc_onnode(conv.size() / 2, 0);
    MatrixXf *conv2 = (MatrixXf *)numa_alloc_onnode(conv.size() / 2, 1);
    

    if(conv.cols() % 2 == 0 ){
        *conv1 = conv.block(0,0,conv.rows(),conv.cols()/2);
        *conv2 = conv.block(0,conv.cols()/2,conv.rows(),conv.cols()/2);
    }
    else if(conv.cols() % 2 == 1){
        *conv1 = conv.block(0,0,conv.rows(),conv.cols()/2);
        *conv2 = conv.block(0,conv.cols()/2 ,conv.rows(),conv.cols()/2+1);
    }
    

    conv.resize(0,0);


    (*Lx21).resize((*Lx11).rows(),(*Lx11).cols());
    (*Lx22).resize((*Lx12).rows(),(*Lx12).cols());

    time_t t6 = time(NULL);
    

    float t7 = 0.0;

    B.Hm_initial(num_thread2);

    for (int i = 2; i < step; i++)
    {
        
        time_t t8 = time(NULL);
 
        (B).parallel_mul(col0,deg_list0,data1,col0,deg_list1,data1,*Lx11,*Lx12,*Lx21,*Lx22);
        (B).parallel_mul(col0,deg_list0,data1,col0,deg_list1,data1,*Lx21,*Lx22,*Lx21,*Lx22);

        t7 += time(NULL) - t8 + 0.0;
    
        *Lx21 = ((*Lx21) - 2 * (*Lx11)) - (*Lx01);
        *Lx22 = ((*Lx22) - 2 * (*Lx12)) - (*Lx02);

        if (i % 2 == 0){
            *conv1 += 2 * bessel(i, theta) * (*Lx21);
            *conv2 += 2 * bessel(i, theta) * (*Lx22);
        }
        else{
            *conv1 -= 2 * bessel(i, theta) * (*Lx21);
            *conv2 -= 2 * bessel(i, theta) * (*Lx22);
        }
        *Lx01 = *Lx11;  
        *Lx02 = *Lx12;
        *Lx11 = *Lx21;  
        *Lx12 = *Lx22;
        std::cout << "Bessell time: " << i <<"\t"<< (time(NULL) - t8 + 0.0) << std::endl;
    }

    std::cout << "NUMA Sparse and dense multiply time: " << (t7 + 0.0) << std::endl;

    conv.resize((*conv1).rows(), (*conv1).cols()+(*conv2).cols());
    conv << (*conv1),(*conv2);

    // numa_free(Lx01, size0/2);
    // numa_free(Lx02, size0/2);
    // numa_free(Lx11, size0/2);
    // numa_free(Lx12, size0/2);
    // numa_free(Lx21, size0/2);
    // numa_free(Lx22, size0/2);
    // numa_free(conv1, conv.size()/2);
    // numa_free(conv2, conv.size()/2);

    // numa_free(col0, B.numE * sizeof(int));
    // numa_free(data1, B.numE * sizeof(float));

    (*Lx01).resize(0,0);
    (*Lx02).resize(0,0);
    (*Lx11).resize(0,0);
    (*Lx12).resize(0,0);
    (*Lx21).resize(0,0);
    (*Lx22).resize(0,0);
    (*conv1).resize(0,0);
    (*conv2).resize(0,0);

    numa_free(col0, B.numE * sizeof(int));
    numa_free(data1, B.numE * sizeof(float));

    
    A.Hm_initial(num_thread1);

    MatrixXf emb = A * (a - conv);
    std::cout << "Chebyshev time: " << (time(NULL) - t1 + 0.0) << std::endl;
    std::cout << "Sparse and dense multiply time: " << (t7 + (t4 - t3) + 0.0) << std::endl;
    
    return l2Normalize(emb);



}
void saveEmbedding(MatrixXf &data, std::string output)
{
    int m = data.rows(), d = data.cols();
    FILE *emb = fopen(output.c_str(), "wb");
    fprintf(emb, "%d %d\n", m, d);
    for (int i = 0; i < m; i++)
    {
        fprintf(emb, "%d", i);
        for (int j = 0; j < d; j++)
            fprintf(emb, " %f", data(i, j));
        fprintf(emb, "\n");
    }
    fclose(emb);
}
int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    Eigen::setNbThreads(FLAGS_num_thread); 
    CSDB::setNbThreads1(FLAGS_num_thread1); // set sparse multiplicaton thread num
    // CSDB::setNbThreads_numa(FLAGS_num_thread2); // set sparse multiplicaton thread num
    std::cout << "Hi OMaGE!!!" << std::endl;
    time_t t1 = time(NULL);
    // auto pop = pool_base::open(FLAGS_pool_path, FLAGS_layout); // open pmem pool
    pmem::obj::pool_base pop;
    const char *path = "/mnt/pmem0/pmem_pool"; // where the PMEM pool is
    if (access(path, F_OK) == 0)
    {
        
        pop = pool_base::open(FLAGS_pool_path, FLAGS_layout);
    }
    else
    {
        pop = pool_base::create(FLAGS_pool_path, FLAGS_layout, FLAGS_pool_size, S_IWUSR | S_IRUSR);
    }
    persistent_ptr<Smatrix> psmat = nullptr; // psmat points to input sparse matrix
    transaction::run(pop, [&]()
                     { psmat = make_persistent<Smatrix>(FLAGS_num_node); }); // allocate space for psmat
    Smatrix &A = *psmat;                                                     // A is a ref to input sparse matrix
    A.setFromFile(pop, FLAGS_filename1, FLAGS_filename2);                    // initialization for A
    time_t t2 = time(NULL);
    std::cout << "Running time of Reading graph:" << t2 - t1 << std::endl;
    MatrixXf feature = getSparseEmbedding(pop, A, FLAGS_num_rank, FLAGS_num_thread1); // get sparse emb
    time_t t3 = time(NULL);
    std::cout << "Running time of getting sparse embedding:" << t3 - t2 << std::endl;
    MatrixXf embedding = getSpectralEmbedding(pop, A, feature, FLAGS_num_step, FLAGS_theta, FLAGS_mu, FLAGS_num_thread1, FLAGS_num_thread1/2); // get spectral emb
    std::cout << "Running time of getting spectral embedding:" << time(NULL) - t3 << std::endl;
    std::cout << "Running time of ProNE:" << time(NULL) - t1 << std::endl;
    // saveEmbedding(feature,FLAGS_emb1);
    // saveEmbedding(embedding,FLAGS_emb2);
    pop.close();
    return 0;
}
