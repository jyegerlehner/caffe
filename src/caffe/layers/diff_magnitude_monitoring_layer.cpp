#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void DiffMagnitudeMonitoringLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK(bottom.size() == top.size()) << this->layer_param_.name() << " requires"
                                     << " Number of top and bottom blobs must "
                                     << " be the same.";
}

template <typename Dtype>
void DiffMagnitudeMonitoringLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  typedef Blob<Dtype> BlobType;
  typedef std::vector<BlobType*> BlobVec;
  typedef typename BlobVec::const_iterator BlobIter;
  typedef std::vector<int> BlobShape;
  BlobShape shape;
  shape.push_back(1);
  for(BlobIter it = top.begin(); it != top.end(); ++it) {
    (*it)->Reshape(shape);
  }
}

template <typename Dtype>
void DiffMagnitudeMonitoringLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
  typedef Blob<Dtype> BlobType;
//  typedef std::vector<BlobType*> BlobVec;
//  typedef typename BlobVec::iterator BlobIter;
//  typedef std::vector<int> BlobShape;
  for(int i=0; i < bottom.size(); ++i) {
    BlobType* bottom_blob = bottom[i];
    BlobType* top_blob = top[i];
    Dtype l2_norm_diff = std::sqrt(bottom_blob->sumsq_diff());
    top_blob->mutable_cpu_data()[0] = l2_norm_diff;
  }
}

template <typename Dtype>
void DiffMagnitudeMonitoringLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      caffe_set(bottom[i]->count(), Dtype(0),
                bottom[i]->mutable_cpu_data());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiffMagnitudeMonitoringLayer);
#endif

INSTANTIATE_CLASS(DiffMagnitudeMonitoringLayer);
REGISTER_LAYER_CLASS(DiffMagnitudeMonitoring);

} // namespace caffe
