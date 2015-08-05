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

//__device__ __force_inline__ void NchwFromIndex(
//                    int index,
//                    int channels,
//                    int num,
//                    int height,
//                    int width,
//                    int& n,
//                    int& c,
//                    int& h,
//                    int& w)
//{

//}


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

template <typename Dtype>
void XCovLoss2Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 2);
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();

  for (int i = 0 ; i < bottom.size() ; i++) {
    int axis_dim = 0;
    Dtype* temp_data = 0;
    std::vector<int> bottom_shape = bottom[i]->shape();

    if ( i == 0 )
    {
      axis_dim = axis_dim0_;
      temp_data = temp_0_.mutable_gpu_data();
    } else if ( i == 1 ) {
      axis_dim = axis_dim1_;
      temp_data = temp_1_.mutable_gpu_data();
    } else {
      LOG(FATAL) << "Bad bottom index in XCovLoss2Layer::ForwardGPU"
                 << std::endl;
    }

    ComputeMeans<Dtype><<<CAFFE_GET_BLOCKS(axis_dim),
                          CAFFE_CUDA_NUM_THREADS >>>(
                mean_vec_[i]->mutable_gpu_data(),
                bottom[i]->gpu_data(),
                axis_dim,
                bottom[i]->num(),
                bottom[i]->height(),
                bottom[i]->width());

    // Subtract the corresponding channel mean from each element of the
    // bottom blob.
    ComputeXminusMean<Dtype><<<CAFFE_GET_BLOCKS(bottom[i]->count()),
        CAFFE_CUDA_NUM_THREADS >>>(
      bottom[i]->gpu_data(),
      mean_vec_[i]->gpu_data(),
      bottom[i]->count(),
      axis_dim,
      bottom[i]->num(),
      bottom[i]->height(),
      bottom[i]->width(),
      temp_data);
  }

  int num = ComputeNewOuterDim( bottom[0]->shape(),canonical_axis );
  int dim0 = axis_dim0_;
  int dim1 = axis_dim1_;

  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim0, dim1, num, 1./num,
      temp_vec_[0]->gpu_data(),
      temp_vec_[1]->gpu_data(),
      0.,
      xcov_.mutable_gpu_data());

  // square terms in xcov
  Dtype dot;
  caffe_gpu_dot<Dtype>(xcov_.count(), xcov_.gpu_data(),
      xcov_.gpu_data(), &dot);

  Dtype loss = dot / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void XCovLoss2Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();
  const Dtype top_diff = top[0]->cpu_diff()[0];

  Dtype* bottom_diff_0 = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff();

  int num = ComputeNewOuterDim( bottom[0]->shape(),canonical_axis );
  int dim0 = axis_dim0_;
  int dim1 = axis_dim1_;

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim0, dim1,
      top_diff/num,
      temp_vec_[1]->gpu_data(),
      xcov_.gpu_data(),
      0.,
      bottom_diff_0);

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim1, dim0,
      top_diff/num,
      temp_vec_[0]->gpu_data(),
      xcov_.gpu_data(),
      0.,
      bottom_diff_1);
}


INSTANTIATE_LAYER_GPU_FUNCS(XCovLoss2Layer);

}  // namespace caffe
