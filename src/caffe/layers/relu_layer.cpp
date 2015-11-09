#include <algorithm>
#include <vector>

#include "caffe/neuron_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

  // Threshholded RELU (TREC) has zero output in the TRAIN phase when the
  // input does not exceed the threshold theta.
  if ( this->phase_ == TRAIN && this->layer_param_.relu_param().has_theta() ) {
    Dtype theta = this->layer_param_.relu_param().theta();
    for (int i = 0; i < count; ++i) {
      Dtype input = bottom_data[i];
      // TRec output is non-zero only when the magnitude of the input
      // exceeds the threshold theta.
      if (std::abs(input) > theta) {
        top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
      } else {
        top_data[i] = Dtype(0.0);
      }
    }
  } else {
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();

    // Threshholded RELU (TREC) has zero output in the TRAIN phase when the
    // input does not exceed the threshold theta.
    if ( this->phase_ == TRAIN &&
         this->layer_param_.relu_param().has_theta() ) {
      for (int i = 0; i < count; ++i) {
        Dtype theta = this->layer_param_.relu_param().theta();
        Dtype input = bottom_data[i];
        // TREC derivative is non-zero only when the magnitude of the input
        // exceeds the threshold theta.
        if (std::abs(input) > theta) {
          bottom_diff[i] = top_diff[i] * ((input > 0)
              + negative_slope * (input <= 0));
        } else {
          bottom_diff[i] = Dtype(0.0);
        }
      }
    } else {
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
            + negative_slope * (bottom_data[i] <= 0));
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
