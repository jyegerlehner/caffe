#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void DiffMagnitudeMonitoringLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  for(int i=0; i < bottom.size(); ++i) {
    Blob<Dtype>* bottom_blob = bottom[i];
    Blob<Dtype>* top_blob = top[i];

    int count = bottom_blob->count();
    Dtype dot = 0;
    caffe_gpu_dot(count, bottom_blob->gpu_diff(),
                  bottom_blob->gpu_diff(), &dot);
    top_blob->mutable_cpu_data()[0] = std::sqrt(dot);
  }
}

template <typename Dtype>
void DiffMagnitudeMonitoringLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), Dtype(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DiffMagnitudeMonitoringLayer);

} // namespace caffe
