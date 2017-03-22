LIBMF++ is a library for large-scale sparse matrix factorization. For the
optimization problem it solves, please refer to [1]. This library is based on LIBMF.

Table of Contents
=================

- Installation
- Data Format
- Command Line Usage
- Examples
- References

Installation
============

- Unix & Cygwin

  Type `make' to build `mf-train' and `mf-precict.'


Data Format
===========

The data format is:

    <row_idx> <col_idx> <value>

Note: If the values in the test set are unknown, please write dummy zeros.

Command Line Usage
==================

-   `mf-train'

    usage: mf-train [options] training_set_file [model_file]

    options:
    -l <lambda>: set regularization parameter (default 0.1)
    -k <features>: set number of latent features (default 8)
    -t <iter>: set number of iterations (default 20)
    -s <threads>: set number of threads (default 12)
    -r <rho>: set rho parameter(default 0.1)
    -e <epsilon>: set epsilon parameter(default 0.001)
    -p <path>: set path to the validation set
    -v <fold>: set number of folds for cross validation
    --quiet: quiet mode (no outputs)

    In the training process, the following information is printed on the
    screen:

        - iter: the index of iteration
        - time: time cost of iteration 
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

> mf-train -l 0.5 -k 16 -t 30 -r 0.05 -e 0.00001 -s 4 bigdata.tr.txt model

train a model using the following parameters:

    regularization cost = 0.5
    latent factors = 16
    iterations = 30
    rho = 0.05
    epsilon = 0.00001
    threads = 4

> mf-train -p bigdata.te.txt bigdata.tr.txt model

use bigdata.te.txt as validation set

> mf-train -v 5 bigdata.tr.txt

do five fold cross validation

> mf-train --quiet bigdata.tr.txt

do not print message to screen

> mf-predict bigdata.te.txt model output

do prediction


References
==========

[1] Wei F, Guo H, Cheng S, et al. AALRSMF: An Adaptive Learning Rate Schedule for Matrix Factorization[C]//Asia-Pacific Web Conference. Springer International Publishing, 2016: 410-413.


For any questions and comments, please email:

    enable@mail.ustc.edu.cn
