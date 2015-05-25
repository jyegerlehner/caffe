#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_channel_max_sma(const int num,
                                   const int channels,
                                   const int spatial_dim,
                                   const int softmax_size,
                                   const int num_softmaxes,
                                   const Dtype* data,
                                   Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    // For each softmax along the canonical axis.
    for( int smi = 0; smi < num_softmaxes; ++smi) {
      Dtype maxval = -FLT_MAX;
      // For each channel within this softmax.
      for (int c_off = 0; c_off < softmax_size; ++c_off) {
        int c = smi * softmax_size + c_off;
        int data_index = (n * channels + c) * spatial_dim + s;
        maxval = max(data[data_index], maxval);
      }
      //int out_index = index*num_softmaxes + smi;
      int out_index = s + (n * num_softmaxes + smi) * spatial_dim ; //index*num_softmaxes + smi;
      out[out_index] = maxval;
    }
  }

}

template <typename Dtype>
__global__ void kernel_channel_subtract_sma(const int count,
                                        const int softmax_size,
                                        const int spatial_dim,
                                        const Dtype* channel_max,
                                        Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / softmax_size / spatial_dim;
    int s = index % spatial_dim;

    int softmax_max_index = n * spatial_dim + s;
    //int softmax_index = chanset*num_softmaxes
    data[index] -= channel_max[softmax_max_index];
  }
}

template <typename Dtype>
__global__ void kernel_exp_sma(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum_sma(const int num,
                                       const int channels,
                                       const int spatial_dim,
                                       const int softmax_size,
                                       const int num_softmaxes,
                                       const Dtype* data,
                                       Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    // For each softmax along the canonical axis.
    for( int smi = 0; smi < num_softmaxes; ++smi) {
      Dtype sum = 0;
      // For each channel within this softmax.
      for (int c_off = 0; c_off < softmax_size; ++c_off) {
        int c = smi * softmax_size + c_off;
        int data_index = (n * channels + c) * spatial_dim + s;
        sum += data[data_index];
      }
      //int out_index = index*num_softmaxes + smi;
      int out_index = s + (n * num_softmaxes + smi) * spatial_dim ;
      channel_sum[out_index] = sum;
    }
  }
}

template <typename Dtype>
__global__ void kernel_channel_div_sma( const int num,
                                    const int channels,
                                    const int spatial_dim,
                                    const int softmax_size,
                                    const int num_softmaxes,
                                    const Dtype* sums,
                                    Dtype* out) {
  CUDA_KERNEL_LOOP(index, num*spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    // For each softmax along the canonical axis.
    for( int smi = 0; smi < num_softmaxes; ++smi) {
      int sum_index = s + (n * num_softmaxes + smi) * spatial_dim ; //index*num_softmaxes + smi;
      Dtype sum = sums[sum_index];
      // For each channel within this softmax.
      for (int c_off = 0; c_off < softmax_size; ++c_off) {
        int c = smi * softmax_size + c_off;
        int data_index = (n * channels + c) * spatial_dim + s;
        out[data_index] = out[data_index] / sum;
      }
      //int out_index = index*num_softmaxes + smi;
    }
  }
}

template <typename Dtype>
__global__ void kernel_softmax_dot(const int num,
                                   const int channels,
                                   const int spatial_dim,
                                   const int softmax_size,
                                   const int num_softmaxes,
                                   const Dtype* data_1,
                                   const Dtype* data_2,
                                   Dtype* softmax_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    // For each softmax along the canonical axis.
    for( int smi = 0; smi < num_softmaxes; ++smi) {
      Dtype dot = 0;
      // For each channel within this softmax.
      for (int c_off = 0; c_off < softmax_size; ++c_off) {
        int c = smi * softmax_size + c_off;
        int data_index = (n * channels + c) * spatial_dim + s;
        dot += data_1[data_index] * data_2[data_index];
      }
      //int out_index = index*num_softmaxes + smi;
      int out_index = s + (n * num_softmaxes + smi) * spatial_dim ; //index*num_softmaxes + smi;
      softmax_dot[out_index] = dot;
    }
  }
}

//template<typename Dtype>
//void InitBlob( Blob<Dtype>& blob )
//{
//  int cardinality = blob.num();
//  int channels = blob.channels();
//  int height = blob.height();
//  int width = blob.width();
//  Dtype* ptr = blob.mutable_cpu_data();
//  for(int n = 0; n < cardinality; ++n) {
//    for(int c = 0; c < channels; ++c ) {
//      for(int h = 0; h < height; ++h ) {
//        for(int w = 0; w < width; ++w ) {
//          ptr[blob.offset(n,c,h,w)] = (Dtype) -333333.0;
//        }
//      }
//    }
//  }
//}

template <typename Dtype>
void SoftmaximaLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int input_count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int count = bottom[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, bottom_data, top_data);

//  int height = bottom[0]->height();
//  int width = bottom[0]->width();
//  maxes_.Reshape(outer_num_, num_softmaxes_, height, width);
//  bot_minus_maxes_.Reshape(outer_num_, channels, height, width);
//  bot_exponentiated_.Reshape(outer_num_, channels, height, width);
//  denom_sums_.Reshape(outer_num_, num_softmaxes_, height, width);

//  InitBlob(maxes_);
//  InitBlob(bot_minus_maxes_);
//  InitBlob(bot_exponentiated_);
//  InitBlob(denom_sums_);

  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
//  scale_data = scale_.mutable_gpu_data();
//  top_data = top[0]->mutable_gpu_data();
  kernel_channel_max_sma<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(  outer_num_,
                                  channels,
                                  inner_num_,
                                  softmax_size_,
                                  num_softmaxes_,
                                  top_data,
                                  scale_data);
//  maxes_.CopyFrom(scale_,false,false);

  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
//  scale_data = scale_.mutable_gpu_data();
//  top_data = top[0]->mutable_gpu_data();
  kernel_channel_subtract_sma<Dtype><<<CAFFE_GET_BLOCKS(input_count),
      CAFFE_CUDA_NUM_THREADS>>>(input_count, softmax_size_, inner_num_,
      scale_data, top_data);

//  bot_minus_maxes_.CopyFrom(*top[0], false,false);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
//  top_data = top[0]->mutable_gpu_data();
  kernel_exp_sma<Dtype><<<CAFFE_GET_BLOCKS(input_count), CAFFE_CUDA_NUM_THREADS>>>(
      input_count, top_data, top_data);

//  bot_exponentiated_.CopyFrom(*top[0], false, false);

  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
//  scale_data = scale_.mutable_gpu_data();
//  top_data = top[0]->mutable_gpu_data();

  kernel_channel_sum_sma<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_,
                                channels,
                                inner_num_,
                                softmax_size_,
                                num_softmaxes_,
                                top_data,
                                scale_data);
//  denom_sums_.CopyFrom(scale_,false,false);

  Blob<Dtype> top_before_div;
  top_before_div.ReshapeLike(*(top[0]));
  top_before_div.CopyFrom(*(top[0]));
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
//  scale_data = scale_.mutable_gpu_data();
//  top_data = top[0]->mutable_gpu_data();

  kernel_channel_div_sma<Dtype><<<CAFFE_GET_BLOCKS(outer_num_*inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_,
                                channels,
                                inner_num_,
                                softmax_size_,
                                num_softmaxes_,
                                scale_data,
                                top_data);

  //PrintNanVals<Dtype>( *(top[0]), top_before_div, scale_ );
}

template <typename Dtype>
void SoftmaximaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int count = top[0]->count();
  int channels = top[0]->shape(softmax_axis_);
  caffe_copy(count, top_diff, bottom_diff);
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff.
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_softmax_dot<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_,
                                channels,
                                inner_num_,
                                softmax_size_,
                                num_softmaxes_,
                                top_diff,
                                top_data,
                                scale_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract_sma<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, softmax_size_, inner_num_,
      scale_data, bottom_diff);
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaximaLayer);


}  // namespace caffe
