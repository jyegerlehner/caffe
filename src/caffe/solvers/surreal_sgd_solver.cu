#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
__global__ void SurrealSGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
    Dtype g_val = g[i];
    Dtype h_val = h[i];
//    if ( h_val == 0)
//    {
//      g[i] = h[i] = local_rate*g_val;
//    }
//    else
    if ( g_val * h_val < 0)
    {
      g[i] = h[i] = local_rate*g_val;
    } else {
      g[i] = h[i] = momentum*h_val + local_rate*g_val;
    }
  }
}

template <typename Dtype>
void surreal_sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  SurrealSGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
template void surreal_sgd_update_gpu<float>(int, float*, float*, float, float);
template void surreal_sgd_update_gpu<double>(int, double*, double*, double, double);

}  // namespace caffe
