#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template<typename Dtype>
__global__ void ComputeMeans( Dtype* means,
                              const Dtype* in_buffer,
                              int channels,
                              int num,
                              int height,
                              int width )
{
  // Loop for each mean that is to be computed.
  CUDA_KERNEL_LOOP(channel, channels) {
    float mean = 0.0f;
    float ctr = 0.0f;
    for( int n = 0; n < num; ++n )
    {
      int term1 = (n * channels + channel)*height;
      for( int h = 0; h < height; ++h )
      {
        int term2 = (term1 + h) * width;
        for( int w = 0; w < width; ++w )
        {
//          int offset = ((n * channels + channel) * height + h) * width + w;
          int offset = term2 + w;
          mean += in_buffer[offset];
          ctr += 1.0;
        }
      }
    }
    means[channel] = mean / ctr;
  }
}

template<typename Dtype>
__global__ void ComputeXminusMean(
    const Dtype* in_buffer,
    const Dtype* means_buffer,
    int count,
    int channels,
    int num,
    int height,
    int width,
    Dtype* result_buffer)
{
  // Subtract the mean from each item in the in_buffer.
  CUDA_KERNEL_LOOP(index, count) {
    int hw = width*height;
    int chw = channels*hw;
    int n = index / chw;
    int c = index / hw - n*channels;
    int h = index / width - (n*channels + c)*height;
    int w = index - ((n*channels+c)*height+h)*width;

    int result_index = channels*( (n*height*width) +
                                      (h*width) + w) + c;

    // Place in temp blob, where it is layed out as n*h*w in the zeroeth
    // dimension, and channels in the second dimension.
    result_buffer[result_index] = in_buffer[index] - means_buffer[c];
  }
}

template<typename Dtype>
__global__ void ZeroDiagonal(Dtype* cov_data, int dim)
{
  CUDA_KERNEL_LOOP(index,dim) {
    int offset = index + index*dim;
    cov_data[offset] = 0.0;
  }
}

template <typename Dtype>
void CovLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1);
  int canonical_axis = this->layer_param_.cov_loss_param().axis();

  std::vector<int> bottom_shape = bottom[0]->shape();

  Dtype* temp_data = temp_.mutable_gpu_data();

  ComputeMeans<Dtype><<<CAFFE_GET_BLOCKS(axis_dim_),
                        CAFFE_CUDA_NUM_THREADS >>>(
              mean_.mutable_gpu_data(),
              bottom[0]->gpu_data(),
              axis_dim_,
              bottom[0]->num(),
              bottom[0]->height(),
              bottom[0]->width());

  // Subtract the corresponding channel mean from each element of the
  // bottom blob.
  ComputeXminusMean<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),
      CAFFE_CUDA_NUM_THREADS >>>(
    bottom[0]->gpu_data(),
    mean_.gpu_data(),
    bottom[0]->count(),
    axis_dim_,
    bottom[0]->num(),
    bottom[0]->height(),
    bottom[0]->width(),
    temp_data);

  int num = ComputeNewOuterDim( bottom[0]->shape(),canonical_axis );
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, axis_dim_, axis_dim_, num,
      1./num,
      temp_.gpu_data(),
      temp_.gpu_data(),
      0.,
      cov_.mutable_gpu_data());

  // We want to penalize covariance, but not variance. So zero out the
  // diagonal terms.
  ZeroDiagonal<Dtype><<<CAFFE_GET_BLOCKS(axis_dim_),
      CAFFE_CUDA_NUM_THREADS >>>(cov_.mutable_gpu_data(), axis_dim_);

  // square terms in xcov
  Dtype dot;
  caffe_gpu_dot<Dtype>(cov_.count(), cov_.gpu_data(), cov_.gpu_data(), &dot);

  Dtype loss = dot / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CovLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();
  const Dtype top_diff = top[0]->cpu_diff()[0];

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  int num = ComputeNewOuterDim( bottom[0]->shape(),canonical_axis );

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, axis_dim_, axis_dim_,
      2.0*top_diff/num,
      temp_.gpu_data(),
      cov_.gpu_data(),
      0.,
      bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(CovLossLayer);

}  // namespace caffe
