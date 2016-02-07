#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/layers/softmaxima_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
__global__ void kernel_sample(const int num,
                              const int channels,
                              const int spatial_dim,
                              const int softmax_size,
                              const int num_softmaxes,
                              Dtype* top)
{

}

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
        if (c_off == 0)
        {
          maxval = data[data_index];
        }
        else
        {
          maxval = max(data[data_index], maxval);
        }
      }
      int out_index = s + (n * num_softmaxes + smi) * spatial_dim ;
      out[out_index] = maxval;
    }
  }
}
//CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));

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
    Dtype data_before = data[index];
    Dtype channel_max_val = channel_max[softmax_max_index];
    Dtype data_after = data_before - channel_max_val;
    data[index] = data_after;
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
__global__ void kernel_sample(const int num,
                          const int channels,
                          const int spatial_dim,
                          const int softmax_size,
                          const int num_softmaxes,
                          const Dtype* uniform_dist_buffer,
                          const Dtype* probs,
                          Dtype* out_data)
{
  CUDA_KERNEL_LOOP(index, num*spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    // For each softmax along the canonical axis.
    for( int smi = 0; smi < num_softmaxes; ++smi) {
      // Find the uniform-distributed sample for this particular softmax.
      int uni_samp_index = s + (n * num_softmaxes + smi) * spatial_dim ;
      Dtype uni_samp = uniform_dist_buffer[uni_samp_index];

      Dtype bottom = 0;
      // For each channel within this softmax.
      for (int c_off = 0; c_off < softmax_size; ++c_off) {
        int c = smi * softmax_size + c_off;
        int data_index = (n * channels + c) * spatial_dim + s;
        Dtype prob = probs[data_index];
        if( (uni_samp <= (bottom + prob)) &&
            (uni_samp > bottom))
        {
          out_data[data_index] = 1;
        }
        else
        {
          out_data[data_index] = 0;
        }
        bottom += prob;
      }
    }
  }
}

template <typename Dtype>
__global__ void kernel_winner_take_all(const int num,
                                       const int channels,
                                       const int spatial_dim,
                                       const int softmax_size,
                                       const int num_softmaxes,
                                       Dtype* out,
                                       const Dtype* probs)
{
  CUDA_KERNEL_LOOP(index, num*spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    // For each softmax along the canonical axis.
    for( int smi = 0; smi < num_softmaxes; ++smi) {
      Dtype largest_prob = -1.0;
      int largest_prob_index = -1;

      // For each channel within this softmax.
      for (int c_off = 0; c_off < softmax_size; ++c_off) {
        int c = smi * softmax_size + c_off;
        int data_index = (n * channels + c) * spatial_dim + s;
        Dtype val = probs[data_index];

        if ( val > largest_prob)
        {
          largest_prob = val;
          largest_prob_index = data_index;
        }
      }

      for (int c_off = 0; c_off < softmax_size; ++c_off)
      {
        int c = smi * softmax_size + c_off;
        int data_index = (n * channels + c) * spatial_dim + s;
        out[data_index] = ((data_index == largest_prob_index) ?
              1 : 0 );
      }
    }
  }
}


// out_probs is only assigned if winner_takes_all is true. Otherwise it is
// ignored. If winner_takes_all, then the out buffer is assigned the binarized
// result.
template <typename Dtype>
__global__ void kernel_channel_div_sma( const int num,
                                    const int channels,
                                    const int spatial_dim,
                                    const int softmax_size,
                                    const int num_softmaxes,
                                    const Dtype* sums,
                                    Dtype* out,
                                    bool store_probs,
                                    Dtype* out_probs) {
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
        Dtype val_before = out[data_index];
        Dtype val = val_before / sum;
        if (val < 0.0 ) val = 0.0;
        else if( val > 1.0) val = 1.0;

        if (store_probs) {
          out_probs[data_index] = val;
        } else {
          out[data_index] = val;
        }
      }
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

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max_sma<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(  outer_num_,
                                  channels,
                                  inner_num_,
                                  softmax_size_,
                                  num_softmaxes_,
                                  top_data,
                                  scale_data);

  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract_sma<Dtype><<<CAFFE_GET_BLOCKS(input_count),
      CAFFE_CUDA_NUM_THREADS>>>(input_count, softmax_size_, inner_num_,
      scale_data, top_data);

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp_sma<Dtype><<<CAFFE_GET_BLOCKS(input_count), CAFFE_CUDA_NUM_THREADS>>>(
      input_count, top_data, top_data);

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum_sma<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_,
                                channels,
                                inner_num_,
                                softmax_size_,
                                num_softmaxes_,
                                top_data,
                                scale_data);

  Dtype* output_probs_buffer = (WinnerTakeAll() || StrictSparsity()) ?
        this->output_probs_.mutable_gpu_data() : 0;

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div_sma<Dtype><<<CAFFE_GET_BLOCKS(outer_num_*inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_,
                                channels,
                                inner_num_,
                                softmax_size_,
                                num_softmaxes_,
                                scale_data,
                                top_data,
                                WinnerTakeAll() || StrictSparsity(),
                                output_probs_buffer);

  if (StrictSparsity())
  {
    static const int KMIN_RNG_BUFFER_SIZE = 256;
    Dtype* uniform_dist_buffer;
    int buffer_size;
    // scale_ is a temp blob and has the correct shape to hold the desired
    // uniformly-distributed values: one for each softmax in the layer.
    if (this->scale_.count() < KMIN_RNG_BUFFER_SIZE)
    {
      // Always generate at least KMIN_RNG_BUFFER_SIZE values; otherwise
      // the result doesn't end up being uniformly-distributed.
      buffer_size = KMIN_RNG_BUFFER_SIZE;
      if (small_uniform_dist_blob_.count() < KMIN_RNG_BUFFER_SIZE)
      {
        small_uniform_dist_blob_.Reshape(KMIN_RNG_BUFFER_SIZE,1,1,1);
      }
      uniform_dist_buffer = small_uniform_dist_blob_.mutable_gpu_data();
    }
    else
    {
      uniform_dist_buffer = this->scale_.mutable_gpu_data();
      buffer_size = this->scale_.count();
    }
    caffe_gpu_rng_uniform<Dtype>(buffer_size,0,1,uniform_dist_buffer);

    kernel_sample<Dtype><<<CAFFE_GET_BLOCKS(outer_num_*inner_num_),
        CAFFE_CUDA_NUM_THREADS>>>(outer_num_,
                                  channels,
                                  inner_num_,
                                  softmax_size_,
                                  num_softmaxes_,
                                  uniform_dist_buffer,
                                  output_probs_buffer,
                                  top_data);
  }
  else if (WinnerTakeAll())
  {
    kernel_winner_take_all<Dtype><<<CAFFE_GET_BLOCKS(outer_num_*inner_num_),
        CAFFE_CUDA_NUM_THREADS>>>(outer_num_,
                                  channels,
                                  inner_num_,
                                  softmax_size_,
                                  num_softmaxes_,
                                  top_data,
                                  output_probs_buffer);
  }

}

template <typename Dtype>
void SoftmaximaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();

  // Use mean-field activations for the backprop if WinnerTakeAll.
  const Dtype* top_data = (WinnerTakeAll() ||StrictSparsity()) ?
                            output_probs_.gpu_data() : top[0]->gpu_data();
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
