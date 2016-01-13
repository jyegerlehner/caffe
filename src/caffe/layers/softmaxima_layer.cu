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
                                        Dtype* data,
                                        int* debug_int,
                                        Dtype* debug_float) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / softmax_size / spatial_dim;
    int s = index % spatial_dim;

    int softmax_max_index = n * spatial_dim + s;
    Dtype data_before = data[index];
    Dtype channel_max_val = channel_max[softmax_max_index];
    Dtype data_after = data_before - channel_max_val;
    data[index] = data_after;
    if(::isnan(data_after))
    {
      debug_int[0] = softmax_max_index;
      debug_float[0] = data_before;
      debug_float[1] = channel_max_val;
      debug_float[2] = data_after;
    }
    else
    {
      data[index] = data_after;
      if(index == 0)
      {
        debug_int[1] = softmax_max_index;
        debug_float[3] = data_before;
        debug_float[4] = channel_max_val;
        debug_float[5] = data_after;
      }
    }
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
                                    bool winner_take_all,
                                    Dtype* out_probs,
                                    int* debug_int,
                                    Dtype* debug_float) {
  CUDA_KERNEL_LOOP(index, num*spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    // For each softmax along the canonical axis.
    for( int smi = 0; smi < num_softmaxes; ++smi) {
      int sum_index = s + (n * num_softmaxes + smi) * spatial_dim ; //index*num_softmaxes + smi;
      Dtype sum = sums[sum_index];

      Dtype largest_prob = -1.0;
      int largest_prob_index = -1;

      // For each channel within this softmax.
      for (int c_off = 0; c_off < softmax_size; ++c_off) {
        int c = smi * softmax_size + c_off;
        int data_index = (n * channels + c) * spatial_dim + s;
        Dtype val_before = out[data_index];
        Dtype val = val_before / sum;

        if (::isnan(val))
        {
          debug_int[0] = data_index;
          debug_float[0] = val_before;
          debug_float[1] = sum;
          debug_float[2] = val;
        }

        if ( winner_take_all) {
          if ( val > largest_prob)
          {
            largest_prob = val;
            largest_prob_index = data_index;
          }
          out_probs[data_index] = val;
        } else {
          if (val < 0.0 ) val = 0.0;
          else if( val > 1.0) val = 1.0;
  //        Dtype val = out[data_index] / sum;
          out[data_index] = val;
        }
      }

      if (winner_take_all)
      {
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

//  if( CheckForNanGPU(top[0]->count(), bottom_data) )
//  {
//    LOG(ERROR) << this->layer_param_.name() << "Softmaxima NaN in bottom, A" << std::endl;
//  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max_sma<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(  outer_num_,
                                  channels,
                                  inner_num_,
                                  softmax_size_,
                                  num_softmaxes_,
                                  top_data,
                                  scale_data);
//  if( CheckForNanGPU(scale_.count(), scale_.gpu_data()) )
//  {
//    LOG(ERROR) << this->layer_param_.name() << "Softmaxima NaN in scale, B"
//               << std::endl;
//  }

  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract_sma<Dtype><<<CAFFE_GET_BLOCKS(input_count),
      CAFFE_CUDA_NUM_THREADS>>>(input_count, softmax_size_, inner_num_,
      scale_data, top_data, debug_int_.mutable_gpu_data(),
                                debug_float_.mutable_gpu_data());

//  if( CheckForNanGPU(top[0]->count(), top[0]->gpu_data()) )
//  {
//    std::cout << "NaN index, before, max, after = " << debug_int_.cpu_data()[0]
//                 << "," << debug_float_.cpu_data()[0] << ","
//                    << debug_float_.cpu_data()[1] << ","
//                       << debug_float_.cpu_data()[2] << std::endl;
//    std::cout << "Good index, before, max, after = " << debug_int_.cpu_data()[1]
//                 << "," << debug_float_.cpu_data()[3] << ","
//                    << debug_float_.cpu_data()[4] << ","
//                       << debug_float_.cpu_data()[5] << std::endl;
//    LOG(ERROR) << this->layer_param_.name() << "Softmaxima NaN C." << std::endl;
//  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp_sma<Dtype><<<CAFFE_GET_BLOCKS(input_count), CAFFE_CUDA_NUM_THREADS>>>(
      input_count, top_data, top_data);

//  if( CheckForNanGPU(top[0]->count(), top[0]->gpu_data()) )
//  {
//    LOG(ERROR) << this->layer_param_.name() << "Softmaxima NaN D." << std::endl;
//  }

  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum_sma<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_,
                                channels,
                                inner_num_,
                                softmax_size_,
                                num_softmaxes_,
                                top_data,
                                scale_data);

//  if( CheckForNanGPU(scale_.count(), scale_.gpu_data()) )
//  {
//    LOG(ERROR) << this->layer_param_.name() << "Softmaxima NaN E." << std::endl;
//  }

  Dtype* output_probs_buffer = WinnerTakeAll() ?
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
                                WinnerTakeAll(),
                                output_probs_buffer,
                                debug_int_.mutable_gpu_data(),
                                debug_float_.mutable_gpu_data());

//  if( CheckForNanGPU(top[0]->count(), top[0]->gpu_data()))
//  {
//    std::cout << "NaN index, before, sum, after = " << debug_int_.cpu_data()[0]
//                 << "," << debug_float_.cpu_data()[0] << ","
//                    << debug_float_.cpu_data()[1] << ","
//                       << debug_float_.cpu_data()[2] << std::endl;
//    LOG(ERROR) << this->layer_param_.name() << " NaN F." << std::endl;
//  }

  Dtype test_val;
  if( CheckForOutOfRangeGPU(top[0]->count(), top[0]->gpu_data(),
                            test_val))
  {
    LOG(FATAL) << "Found softmaxima output not between 0 and 1: "
               << test_val << ", in top of layer " << this->layer_param_.name();
  }


//  CheckForNanGPU("softmaxima5", "top", *top[0]);
//  if( WinnerTakeAll())
//  {
//    CheckForNanGPU("softmaxima6", "output_probs", this->output_probs_);
//  }
}

template <typename Dtype>
void SoftmaximaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();

  // Use mean-field activations for the backprop if WinnerTakeAll.
  const Dtype* top_data = WinnerTakeAll() ? output_probs_.gpu_data() :
                                            top[0]->gpu_data();
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
      scale_data, bottom_diff, debug_int_.mutable_gpu_data(),
                                debug_float_.mutable_gpu_data());
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaximaLayer);


}  // namespace caffe
