#ifndef CAFFE_SOFTMAXIMA_LAYER_HPP_
#define CAFFE_SOFTMAXIMA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the softmaxima function.
 *
 */
template <typename Dtype>
class SoftmaximaLayer : public Layer<Dtype> {
 public:
  explicit SoftmaximaLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Softmaxima"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool WinnerTakeAll() const
  {
    return this->layer_param_.softmaxima_param().winner_take_all();
  }
  int outer_num_;
  int inner_num_;
  int softmax_axis_;
  /// sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> sum_multiplier_;
  /// scale is an intermediate Blob to hold temporary results.
  Blob<Dtype> scale_;
  /// Blob<Dtype> softmaxima result before binarization. Unused
  /// unless winner_take_all = true is specified.
  Blob<Dtype> output_probs_;

  // The ratio of the input size along the canonical axis to the softmax size.
  int num_softmaxes_;
  // The size of each softmax.
  int softmax_size_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_LAYER_HPP_
