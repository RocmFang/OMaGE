#include "csdb.h"
// #include <ctime>

using namespace std;
using namespace CSDB;

void CSDB::setNbThreads(int num)
{
    MULTITHREAD_ON = true;
    thread_num = num;
}

void CSDB::setNbThreads1(int num)
{
    MULTITHREAD_ON = true;
    num_thread = num;

}

// void CSDB::setNbThreads_numa(int num)
// {
//     MULTITHREAD_ON = true;
//     num_thread2 = num;

// }


void CSDB::Smatrix::setFromFile(pmem::obj::pool_base &pop, std::string graph_file, std::string deg_file)
{

    std::vector<int> deg_list1, col1;
    std::vector<long long> add_aux1;
    // std::vector<std::vector<int>> trans_aux1(numV);

    deg_list1.resize(numV); // deg_list的size为numV
    trans_aux.resize(numV); // trans_aux的size为numV
    add_aux1.resize(numV);

    std::ifstream fin(deg_file.c_str());

    while (1)
    {
        std::string x, y;
        if (!(fin >> x >> y))
            break;
        int index = atoi(x.c_str()), deg = atoi(y.c_str());
        numE += deg;
        deg_list1[index] = deg;
        // (*trans_aux)[index].reserve(deg);
    }
    fin.close();

    col1.resize(numE);

    std::ifstream fin2(graph_file.c_str());
    bool flag = false;
    int ind = 0;
    long long col_ind = 0;
    for (int i = 0; i < numV; i++)
    {
        int num = deg_list1[i];
        flag = true;
        while (num--)
        {
            std::string x, y;
            fin2 >> x >> y;
            int a = atoi(x.c_str()), b = atoi(y.c_str());
            if (flag && i < b)
            {
                flag = false;
                add_aux1[ind++] = i + col_ind;
            }
            col1[col_ind++] = b;
            trans_aux[b].push_back(col_ind - 1);
        }
        if (flag)
            add_aux1[ind++] = i + col_ind;
    }
    fin2.close();

    pmem::obj::transaction::run(pop, [&]()
                                {
        deg_list=pmem::obj::make_persistent<pvi>(numV);
        col=pmem::obj::make_persistent<pvi>(numE);
        // trans_aux=pmem::obj::make_persistent<pmem::obj::vector<pvi>>(numV);
        add_aux=pmem::obj::make_persistent<pvl>(numV);
        data=pmem::obj::make_persistent<pvf>(numE,2.0f); });

    pop.memcpy_persist(&(*deg_list)[0], &deg_list1[0], numV * sizeof(int));

    pop.memcpy_persist(&(*col)[0], &col1[0], numE * sizeof(int));

    pop.memcpy_persist(&(*add_aux)[0], &add_aux1[0], numV * sizeof(long long));


    printf("Copy data done!\n");
}


vector<pair<long long, long long>> CSDB::Smatrix::thread_allocate(int num_thread) const
{
    // const int AVE_ELEM = numE / thread_num;
    const int AVE_ELEM = numE / num_thread;
    std::vector<std::pair<long long, long long>> thread_pos;

    long long i = 0, tmp = 0, total = 0;
    thread_pos.push_back({0, 0});
    while (i < numV)
    {
        while (i < numV && tmp < AVE_ELEM)
        {
            total += (*deg_list)[i];
            tmp += (*deg_list)[i++];
        }
        thread_pos.push_back({i, total});
        tmp = 0;
    }
    return thread_pos;
}

double Smatrix::Hm_initial(int num_thread)
{

    double Hj;
    long double pi = 0;

    for (int j; j < numV; j++)
    {
        // pi = (float)deg_list[j] / numV;
        pi = (float)(*deg_list)[j] / (numE / num_thread);
        Hj -= pi * log(pi);
    }

    Hm_init = Hj;
    printf("Hm_func: %f \n", Hm_init);

    return Hm_init;
}

vector<pair<long long, long long>> Smatrix::thread_allocate_entropy(int thread_num1) const
{
    if (!MULTITHREAD_ON)
    {
        cerr << "Plz assign the thread num first" << endl;
        return {};
    }

    long long AVE_ELEM_init = numE / thread_num1, in_AVE_ELEM, un_AVE_ELEM = AVE_ELEM_init, AVE_ELEM = AVE_ELEM_init;

    vector<pair<long long, long long>> thread_pos;

    int i = 0, TA_ID = thread_num1, tag = 0, ID = thread_num1;
    long long total = 0, elem_total = 0, tmp_sps = 0;
    long long rsps_total = 0, rst = 0;
    double beta, tmp = 0, Hi = 0, Hj = 0, Hm = 0, Hmin = 0, Hm_total = 0;
    long double pi = 0;

    thread_pos.push_back({0, 0});


    Hm = Hm_init / thread_num1;

    while (i < numV)
    {
        while (i < numV && tmp < AVE_ELEM)
        {
            total += (*deg_list)[i];

            // pi = (float)deg_list[i] / numV;
            pi = (float)(*deg_list)[i] / AVE_ELEM;
            Hi -= pi * log(pi);

            tmp += (*deg_list)[i++];
        }
        // printf("Hi: %f\n", Hi);

        // printf("Hm: %f \n", Hm);

        int _numV = numV;

        double Z_Hm = (Hm) / (log(_numV));
        double Z_Hi = (Hi) / (log(_numV));

        beta = (Hm * (1 - Z_Hm + 0.75 * Z_Hm)) / (Hi * (1 - Z_Hi + 0.75 * Z_Hi));

        tmp_sps = tmp * (beta > 1 ? min<double>({beta, 2}) : max<double>({beta, 0.7}));

     
        if (tmp < tmp_sps)
        {
            while (i < numV && tmp < tmp_sps && total < numE) //
            {

                total += (*deg_list)[i];
                tmp += (*deg_list)[i];

                if (i != numV)
                {
                    // pi = (float)deg_list[i] / numV;
                    pi = (float)(*deg_list)[i] / tmp_sps;
                    Hi += -pi * log(pi);
                    // printf("NaN: %d pi:  %lf\n", i, pi);
                }
                i++;
            }
        }
        else
        {

            while (i < numV + 1 && tmp > tmp_sps && total < numE + 1)
            {
                total -= (*deg_list)[i - 1];
                tmp -= (*deg_list)[i - 1];

                // pi = (float)deg_list[i-1] / numV;
                pi = (float)(*deg_list)[i - 1] / tmp_sps;
                Hi -= -pi * log(pi);
                i--;
            }
        }

        rsps_total += (i - rst);
        elem_total += tmp;

        thread_pos.push_back({i, total});

        if (TA_ID > 1)
        {
            un_AVE_ELEM = (numE - elem_total) / (--TA_ID);
            in_AVE_ELEM = (elem_total) / (thread_num1 - TA_ID);
          
            if (un_AVE_ELEM > AVE_ELEM_init * 0.55 && tag == 0) //(AVE_ELEM_init < tmp_sps)
            {
                AVE_ELEM = in_AVE_ELEM;
                Hm_total += Hi;
                Hm = (Hm_total) / (thread_num1 - TA_ID);
            }
            else
            {
                AVE_ELEM = un_AVE_ELEM;
                Hm_total += Hi;
                // Hm = (Hm_init - Hm_total)/(TA_ID);
                Hm = (Hm_total) / (thread_num1 - TA_ID);
                tag = 1;
            }
        }
        else
        {

            in_AVE_ELEM = (elem_total) / (thread_num1 - TA_ID);
            AVE_ELEM = in_AVE_ELEM;
            Hm_total += Hi;
            Hm = Hm_total / (thread_num1 - TA_ID);

            ID++;
        }

        // Hm_total += Hi;
        // Hm = Hm_total / (thread_num - TA_ID);

        // Hm_total += Hi;
        // Hm = (Hm_init-Hm_total)/TA_ID;


        tmp = 0;
        rst = i;
        Hi = 0;
    }

    return thread_pos;
}

void CSDB::Calculate(const Smatrix &A, const Eigen::MatrixXf &B, long long block_start, int rst, int red, Eigen::MatrixXf &res, int th_ind, std::vector<double> &tt)
{

    int bb, cc;
    long long ind;

    // Timer t_thread;

    for (int j = 0; j < B.cols(); j++)
    {
        ind = block_start;
        for (int i = rst; i < red; i++)
        {

            auto aa = (*A.deg_list)[i];
            long long start = ind, end = ind + aa;
            float tmp = 0.0f;
            while (start != end)
            {
                bb = (*A.data)[start];        // sparse nnz
                cc = B((*A.col)[start++], j); // dense nnz

                tmp += cc * bb; // tem_result
            }

            res(i, j) = tmp; // result

            ind += (*A.deg_list)[i];
        }
    }

}

void CSDB::Calculate1(int *col, int *deg_list, float *data, const Eigen::MatrixXf &B, long long block_start, long long block_end, int rst, int red, Eigen::MatrixXf &res)
{
    int bb, cc;

    long long ind;
    for (int j = 0; j < B.cols(); j++)
    {
        ind = block_start;
        for (int i = rst; i < red; i++)
        {
            // Timer t1;
            int aa = deg_list[i];
            long long start = ind, end = ind + aa;
            float tmp = 0.0f;
            // Timer t2, t3;
            while (start != end)
            {
                bb = data[start];        // sparse nnz
                cc = B(col[start++], j); // dense nnz
                tmp += cc * bb;          // tem_result
            }
            res(i, j) = tmp; // result
            ind += deg_list[i];
        }
    }
}

Eigen::MatrixXf CSDB::Smatrix::operator*(const Eigen::MatrixXf &other) const
{
    Eigen::MatrixXf res(numV, other.cols());

    Timer t1;
    // int num_thread1 = 30;
    // vector<pair<long long, long long>> allocate = thread_allocate(num_thread1);
    // printf("thread1:%d\n", num_thread);
    vector<pair<long long, long long>> allocate = thread_allocate_entropy(num_thread);

    // std::cout << "Coculate start " << std::endl;
    // printf("allocate szie: %d\n", allocate.size());

    Timer t2;

    std::vector<std::thread> threads(allocate.size() - 1);
    std::vector<double> tt(allocate.size() - 1);

    // for (int i = 0; i < threads.size(); i++)
    //     threads[i] = std::thread(Calculate, std::ref(*this), std::ref(other), allocate[i].second, allocate[i].first, allocate[i + 1].first, std::ref(res));

    // for (auto &entry : threads)
    //     entry.join();
    int n = allocate.size() / 2; // 15

    std::thread *threads1 = (std::thread *)numa_alloc_onnode(n, 0); // thread1和2完成一组操作
    std::thread *threads2 = (std::thread *)numa_alloc_onnode(n, 1);
    for (int i = 0, j = i + n; i < n; i++, j++)
    { 
        threads1[i] = std::thread(Calculate, ref(*this), ref(other), allocate[i].second, allocate[i].first, allocate[i + 1].first, std::ref(res), i, std::ref(tt));
        threads2[i] = std::thread(Calculate, ref(*this), ref(other), allocate[j].second, allocate[j].first, allocate[j + 1].first, std::ref(res), j, std::ref(tt));
    }
    for (int i = 0, j = i + n; i < n; i++, j++)
    {
        threads1[i].join();
        threads2[i].join();
    }

    numa_free(threads1, n); 
    numa_free(threads2, n);

    return res;
}

void CSDB::Smatrix::add_I(pmem::obj::pool_base &pop)
{

    pmem::obj::persistent_ptr<CSDB::pvi> ptmp = nullptr;
    pmem::obj::transaction::run(pop, [&]()
                                {
        numE+=numV;
        ptmp=pmem::obj::make_persistent<CSDB::pvi>(numE);
        pmem::obj::delete_persistent<CSDB::pvf>(data);
        data=pmem::obj::make_persistent<CSDB::pvf>(numE,2.0f); });

    auto &tmp = *ptmp;
    auto &cl = *col;
    long long indtmp = 0, indcl = 0;

    for (int i = 0; i < numV; i++)
    {
        while (indtmp != (*add_aux)[i])
            tmp[indtmp++] = cl[indcl++];
        tmp[indtmp++] = i;
    }
    pmem::obj::transaction::run(pop, [&]()
                                {
        pmem::obj::delete_persistent<CSDB::pvi>(col);
        col=ptmp; });
    for (auto &it : *deg_list)
        it++;
}

void CSDB::Smatrix::parallel_mul(int *col0, int *deg_list0, float *data0, int *col1, int *deg_list1, float *data1, const Eigen::MatrixXf &B1, const Eigen::MatrixXf &B2, Eigen::MatrixXf &res1, Eigen::MatrixXf &res2)
{
    // time_t t1 = time(NULL);

    // int num_thread2 = 15;
    // vector<pair<long long, long long>> allocate = thread_allocate(num_thread2);
    // printf("thread2:%d\n", num_thread/2);

    vector<pair<long long, long long>> allocate = thread_allocate_entropy(num_thread/2);
    // printf("allocate szie: %d\n", allocate.size());

    std::vector<std::thread> threads(allocate.size() - 1);
    std::vector<double> t11(allocate.size() - 1), t22(allocate.size() - 1), t33(allocate.size() - 1);
    // vector<thread> threads(allocate.size() - 1);

    // int n = allocate.size()/2;
    std::thread *threads1 = (std::thread *)numa_alloc_onnode(allocate.size() - 1, 0); // thread1和2完成一组操作
    std::thread *threads2 = (std::thread *)numa_alloc_onnode(allocate.size() - 1, 1);
    for (int i = 0; i < t11.size(); i++)
    { // 
        threads1[i] = std::thread(Calculate1, col0, deg_list0, data0, std::ref(B1), allocate[i].second, allocate[i + 1].second, allocate[i].first, allocate[i + 1].first, std::ref(res1));
        threads2[i] = std::thread(Calculate1, col1, deg_list1, data1, std::ref(B2), allocate[i].second, allocate[i + 1].first, allocate[i].first, allocate[i + 1].first, std::ref(res2));
    }
    // Calculate1(col0,*deg_list0,);
    for (int i = 0; i < t11.size(); i++)
    {
        threads1[i].join();
        threads2[i].join();
    }
    numa_free(threads1, allocate.size() - 1);
    numa_free(threads2, allocate.size() - 1);

}