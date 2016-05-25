#include <iostream>
#include <stdio.h>
#include "cusparse.h"
using namespace std;

class icc{
public:
  /*
    Constructor: build the sparse mobility matrix M
    and compute the Cholesky factorization M=L*L.T
    where L is a lower triangular matrix.
  */
  icc(const double blob_radius, 
      const double eta, 
      const double cutoff,
      const int number_of_blobs,
      const double *x);

  /*
    Destructor: free memory on the GPU and CPU.
  */
  ~icc();

  /*
    Build sparse mobility matrix M.
  */
  int buildSparseMobilityMatrix();

  /*
    Muliply by Cholesky factorization L.
  */
  int multL();

  /*
    Muliply by inverse of the Cholesky factorization L^{-1}.
  */
  int multInvL();

private:
  // CPU variables
  int d_icc_is_initialized;
  double d_blob_radius, d_eta, d_cutoff;
  int d_number_of_blobs;
  const double *d_x;
  int d_threads_per_block, d_num_blocks;
  unsigned long long int d_nnz;
  cusparseHandle_t d_cusp_handle;
  cusparseIndexBase_t d_base;
  cusparseStatus_t d_cusp_status;
  double *d_cooVal;
  int *d_cooRowInd, *d_cooColInd, *d_csrRowPtr;
  cusparseMatDescr_t d_descr_M;  
  // csric02Info_t d_info_M; for version cuda 7.5
  cusparseSolveAnalysisInfo_t d_info_M;

  // GPU variables
  double *d_x_gpu;
  unsigned long long int *d_nnz_gpu;
  double *d_cooVal_gpu;
  int *d_cooRowInd_gpu, *d_cooColInd_gpu, *d_csrRowPtr_gpu, *d_cooVal_sorted_gpu;
  unsigned long long int *d_index_gpu;
};


