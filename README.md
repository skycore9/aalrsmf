LIBMF is a library for large-scale sparse matrix factorization. For the
optimization problem it solves, please refer to [2].



Table of Contents
=================

- Installation
- Data Format
- Model Format
- Command Line Usage
- Examples
- Library Usage
- SSE, AVX, and OpenMP
- Building Windows and Mac Binaries
- References



Installation
============

- Unix & Cygwin

  Type `make' to build `mf-train' and `mf-precict.'

- Windows & Mac
    
  See `Building Windows and Mac Binaries' to compile. For Windows, pre-built
  binaries are available in the directory `windows.'



Data Format
===========

The data format is:

    <row_idx> <col_idx> <value>

See an example `bigdata.tr.txt.'

Note: If the values in the test set are unknown, please write dummy zeros.



Model Format
============

    LIBMF factorizes a training matrix `R' into a k-by-m matrix `P' and a
    k-by-n matrix `Q' such that `R' is approximated by P'Q. After the training
    process, the two factor matrices `P' and `Q' are stored into a model file.
    The file starts with a header including the following parameters:

        `m': the number of rows in the training matrix,
        `n': the number of columns in the training matrix,
        `k': the number of latent factors,
        `b': the average of all elements in the training matrix.

    From the 5th line, the columns of `P' and `Q' are stored line by line.
    For each line, there are two leading tokens followed by the values of a
    column.  The first token is the name of the stored column, and the second
    word indicates the type of values. If the second word is `T', the column is
    a real-valued vector. Otherwise, all values in the column are NaN. For
    example, if

        P = [1 NaN 2; 3 NaN 4; 5 NaN 6], Q = [-1, -2; -3, -4; -5; -6],

    and the value `b' is 0.5, the content of the model file is:

        ========================model file========================
        m 3
        n 2
        k 3
        b 0.5
        p1 T 1 3 5
        p2 F 0 0 0
        p3 T 2 4 6
        q1 T -1 -3 -5
        q2 T -2 -4 -6
        =========================================================



Command Line Usage
==================

-   `mf-train'

    usage: mf-train [options] training_set_file [model_file]

    options:
    -l <lambda>: set regularization cost (default 0.1)
    -k <factor>: set number of latent factors (default 8)
    -t <iter>: set number of iterations (default 20)
    -r <eta>: set learning rate (default 0.1)
    -s <threads>: set number of threads (default 12)
    -p <path>: set path to validation set
    -v <fold>: set number of folds for cross validation
    --quiet: quiet mode (no outputs)
    --nmf: perform non-negative matrix factorization

    In the training process, the following information is printed on the
    screen:

        - iter: the index of iteration
        - tr_rmse: RMSE in the training set
        - va_rmse: RMSE in the validation set if `-p' is specified
        - obj: objective function value

    Here `tr_rmse' and `obj' are estimation because calculating true values can
    be time-consuming. In the end of training process the true tr_rmse is
    printed.
    
-   `mf-predict'

    usage: mf-predict test_file model_file output_file



Examples
========

> mf-train bigdata.tr.txt model

train a model using the default parameters

> mf-train -l 0.5 -k 16 -t 30 -r 0.05 -s 4 bigdata.tr.txt model

train a model using the following parameters:

    regularization cost = 0.5
    latent factors = 16
    iterations = 30
    learning rate = 0.05
    threads = 4

> mf-train -p bigdata.te.txt bigdata.tr.txt model

use bigdata.te.txt as validation set

> mf-train -v 5 bigdata.tr.txt

do five fold cross validation

> mf-train --nmf bigdata.tr.txt

do non-negative matrix factorization

> mf-train --quiet bigdata.tr.txt

do not print message to screen

> mf-predict bigdata.te.txt model output

do prediction



Library Usage
=============

These structures and functions are declared in the header file `mf.h.' You need
to #include `mf.h' in your C/C++ source files and link your program with
`mf.cpp.' You can see `mf-train.c' and `mf-predict.c' for examples showing how
to use them.

Before you predict test data, you need to construct a model (`mf_model') using
training data. A model can also be saved in a file for later use. Once a model
is available, you can use it to predict new data.


There are four public data structures in LIBMF.

-   struct mf_node
    {
        mf_int u;
        mf_int v;
        mf_float r;
    };

    `mf_node' represents an element in a sparse matrix. `u' represents the row
    index, `v' represents the column index, and `r' represents the value.


-   struct mf_problem
    {
        mf_int m;
        mf_int n;
        mf_long nnz;
        struct mf_node *R;
    };

    `mf_problem' represents a sparse matrix. Each element is represented by
    `mf_node.' `m' represents the number of rows, `n' represents the number of
    columns, `nnz' represents the number of non-zero elements, and `R' is an
    array of `mf_node' whose length is `nnz.'


-   struct mf_parameter
    {
        mf_int k; 
        mf_int nr_threads;
        mf_int nr_bins;
        mf_int nr_iters;
        mf_float lambda; 
        mf_float eta;
        bool do_nmf;
        bool quiet; 
        bool copy_data;
    };

    `mf_parameter' represents the parameters used for training. The meaning of
    each variable is:

    variable      meaning                             default
    =========================================================
    k             number of latent factors                  8
    nr_threads    number of threads used                   12
    nr_bins       number of blocks                         20
    nr_iters      number of iterations                     20
    lambda        regularization cost                     0.1
    eta           learning rate                           0.1
    do_nmf        perform NMF                           false
    quiet         no outputs to stdout                  false
    copy_data     copy data in training procedure        true

    In LIBMF, we parallelize the computation by gridding the data matrix into
    blocks. `nr_bins' is used to set the number of blocks. According to our
    experiments, this parameter is insensitive to both effectiveness and
    efficiency. In most cases the default value should work well.

    By default, at the beginning of the training procedure, the data matrix is
    copied because it is modified in the training process. To save memory,
    `copy_data' can be set to false with the following effects.
    
        (1) The raw data is directly used without being copied.
        (2) The order of nodes may be changed.
        (3) The value in each node may become slightly different.

    To obtain a parameter with default values, use the function
    `get_default_parameter.'

-   struct mf_model
    {
        mf_int m;
        mf_int n;
        mf_int k;
        mf_float b;
        mf_float *P;
        mf_float *Q;
    };

    `mf_model' is used to store models in LIBMF. `m' represents the number of
    rows, `n' represents the number of columns, `k' represents the number of
    latent factors, and `b' is the average value of all elements in the training
    matrix. `P' is used to store a kxm matrix in column oriented format.  For
    example, if `P' stores a 3x4 matrix, then the content of `P' is:
        
        P11 P21 P31 P12 P22 P32 P13 P23 P33 P14 P24 P34

    `Q' is used to store a kxn matrix in the same manner.


Functions available in LIBMF include:


-   mf_parameter mf_get_default_param();

    Get default parameters.

-   mf_int mf_save_model(struct mf_model const *model, char const *path);
    
    Save a model. It returns 0 on sucess and 1 on failure.

-   struct mf_model* mf_load_model(char const *path);

    Load a model. If the model could not be loaded, a nullptr is returned.

-   void mf_destroy_model(struct mf_model **model);
    
    Destroy a model.

-   struct mf_model* mf_train(
        struct mf_problem const *prob, 
        mf_parameter param);

    Train a model.

-   struct mf_model* mf_train_with_validation(
        struct mf_problem const *Tr, 
        struct mf_problem const *Va, 
        mf_parameter param);

    Train a model with training set `Tr' and validation set `Va.' The RMSE of
    the validation set is printed at each iteration.
    
-   mf_float mf_cross_validation(
        struct mf_problem const *prob, 
        mf_int nr_folds, 
        mf_parameter param);

    Do cross validation with `nr_folds' folds.

-   mf_float mf_predict(struct mf_model const *model, mf_int p_idx, mf_int q_idx);

    Predict the value at the position (p_idx, q_idx). If `p_idx' or `q_idx' can not
    be found in the training data, the function returns the average of all values
    in the training matrix.



SSE, AVX, and OpenMP
====================

LIBMF utilizes SSE instructions to accelerate the computation. If you cannot
use SSE on your platform, then please comment out

    DFLAG = -DUSESSE

in Makefile to disable SSE.

Some modern CPUs support AVX, which is more powerful than SSE. To enable
AVX, please uncomment the following lines in Makefile.

    DFLAG = -DUSEAVX
    CFLAGS += -mavx

If OpenMP is not available on your platform, then please comment out the
following lines in Makefile.

    DFLAG += -DUSEOMP
    CXXFLAGS += -fopenmp

Note: Please always run `make clean all' if these flags are changed.



Building Windows and Mac and Binaries
=====================================

-   Windows

    Windows binaries are in the directory `windows.' To build them via
    command-line tools of Microsoft Visual Studio, use the following steps:

    1. Open a DOS command box (or Developer Command Prompt for Visual Studio)
    and go to libmf directory. If environment variables of VC++ have not been
    set, type

    "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin\amd64\vcvars64.bat"

    You may have to modify the above command according which version of VC++
    or where it is installed.

    2. Type

    nmake -f Makefile.win clean all

    3. (optional) To build shared library mf_c.dll, type

    nmake -f Makefile.win lib

-   Mac
    
    To complie LIBMF on Mac, a GCC complier is required, and users need to
    slightly modify the Makefile. The following instructions are tested with
    GCC 4.9.

    1. Set the complier path to your GCC complier. For example, the first
       line in the Makefile can be
   
       CXX = g++-4.9

    2. Remove `-march=native' from `CXXFLAGS.' The second line in the Makefile
       Should be

       CXXFLAGS = -O3 -pthread -std=c++0x

    3. If AVX is enabled, we add `-Wa,-q' to the `CXXFLAGS,' so the previous
       `CXXFLAGS' becomes

       CXXFLAGS = -O3 -pthread -std=c++0x -Wa,-q
  


References
==========

[1] W.-S. Chin, Y. Zhuang, Y.-C. Juan, and C.-J. Lin. A Fast Parallel
Stochastic Gradient Method for Matrix Factorization in Shared Memory Systems.
ACM TIST, 2015. (www.csie.ntu.edu.tw/~cjlin/papers/libmf/libmf_journal.pdf)

[2] W.-S. Chin, Y. Zhuang, Y.-C. Juan, and C.-J. Lin. A learning-rate schedule
for stochastic gradient methods to matrix factorization. PAKDD, 2015.


For any questions and comments, please email:

    cjlin@csie.ntu.edu.tw
