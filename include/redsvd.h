
#ifndef REDSVD_H__
#define REDSVD_H__

#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include "util.h"
#include "csdb.h"
#include <ctime>

namespace REDSVD
{
    class RedSVD
    {
    public:

    
        void run(pmem::obj::pool_base &pop, CSDB::Smatrix &A, const int rank, CSDB::Smatrix &AA)
        {
            // Gaussian Random Matrix for A^T
            char mem[1024] = {0};
            Eigen::MatrixXf O(A.numV, rank);

            Util::sampleGaussianMat(O);

            pmem::obj::persistent_ptr<CSDB::Smatrix> pAT = nullptr;
            pmem::obj::transaction::run(pop, [&]()
                                        {
                         pAT = pmem::obj::make_persistent<CSDB::Smatrix>(A.numV);
                         pAT->numE = A.numE;
                         pAT->deg_list = pmem::obj::make_persistent<CSDB::pvi>(A.numV);

                         pAT->data = pmem::obj::make_persistent<CSDB::pvf>(A.numE);
                         pAT->col = pmem::obj::make_persistent<CSDB::pvi>(A.numE);
                        //  pAT->add_aux = A.add_aux;
                        //  pAT->trans_aux = A.trans_aux; 
                        });
            *(pAT->deg_list) = *(A.deg_list);
            *(pAT->col) = *(A.col);
            long long ind = 0;
            for (int i = 0; i < A.numV; i++)
            {
                // for (auto it : (*(A.trans_aux))[i])
                for (auto it : ((AA.trans_aux))[i])
                    (*(pAT->data))[ind++] = (*(A.data))[it];
            }
            auto &AT = *pAT;     //AT=transpose(A)
            // Compute Sample Matrix of A^T
            
            AT.Hm_initial(30);

            std::cout << "Sparse Multipy start" << std::endl;
            time_t t1 = time(NULL);
            Eigen::MatrixXf Y = AT * O;  
            time_t t2 = time(NULL);           

            O.resize(0, 0);

            // Orthonormalize Y
            Util::processGramSchmidt(Y);

            // Range(B) = Range(A^T)
            std::cout << "Sparse Multipy start" << std::endl;
            time_t t3 = time(NULL);
            Eigen::MatrixXf B = A * Y;
            std::cout << "Sparse and dense multiply time: " << (time(NULL) - t3 + (t2 - t1) + 0.0) << std::endl;



            // Gaussian Random Matrix
            Eigen::MatrixXf P(B.cols(), rank);
            Util::sampleGaussianMat(P);

            // Compute Sample Matrix of B
            std::cout << "Dense Multipy start" << std::endl;
            Eigen::MatrixXf Z = B * P;
            std::cout << "Dense Multipy end" << std::endl;
            // Orthonormalize Z
            Util::processGramSchmidt(Z);

                        
            P.resize(0, 0);

            // Range(C) = Range(B)
            Eigen::MatrixXf C = Z.transpose() * B;


            B.resize(0, 0);

            Eigen::JacobiSVD<Eigen::MatrixXf> svdOfC(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
            printf("jacobiSVD \n");

            // C = USV^T
            // A = Z * U * S * V^T * Y^T()
            matU_ = Z * svdOfC.matrixU();
            matS_ = svdOfC.singularValues();
            matV_ = Y * svdOfC.matrixV();
            printf("jacobiSVD done!\n");
        }

        const Eigen::MatrixXf &matrixU() const
        {
            return matU_;
        }

        const Eigen::VectorXf &singularValues() const
        {
            return matS_;
        }

        const Eigen::MatrixXf &matrixV() const
        {
            return matV_;
        }

    private:
        Eigen::MatrixXf matU_;
        Eigen::VectorXf matS_;
        Eigen::MatrixXf matV_;
    };
}
#endif