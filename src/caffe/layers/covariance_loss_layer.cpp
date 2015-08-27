#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void CovLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void CovLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();
  axis_dim_ = bottom[0]->shape()[canonical_axis];

  top[0]->Reshape(1, 1, 1, 1);

  cov_.Reshape(axis_dim_, axis_dim_, 1, 1);

  int new_outer_dim = ComputeNewOuterDim(bottom[0]->shape(),
                                          canonical_axis);

  shuffled_bottom_.Reshape(new_outer_dim, axis_dim_,1,1 );

  CHECK(shuffled_bottom_.count() == bottom[0]->count()) << "shuffled blob"
                                  << " size does not match bottom.";

  {
    vector<int> mean_vec_shape;
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);

    // Set the shape to be all 1's except the canonical axis which is
    // size axis_dim0.
    mean_vec_shape[canonical_axis] = axis_dim_;
    mean_.Reshape(mean_vec_shape);

    vector<int> tmp_vec_shape;
    tmp_vec_shape.push_back(new_outer_dim);
    tmp_vec_shape.push_back(1);
    tmp_vec_shape.push_back(1);
    tmp_vec_shape.push_back(1);

    tmp_vec_shape[canonical_axis] = axis_dim_;
    temp_.Reshape(tmp_vec_shape);
  }

  batch_sum_multiplier_.Reshape(new_outer_dim, 1, 1, 1);
  Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
  caffe_set(batch_sum_multiplier_.count(), Dtype(1), batch_multiplier_data);
}

template <typename Dtype>
void CovLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1);
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();
  Dtype* shuffled_buffer = 0;
  std::vector<int> bottom_shape = bottom[0]->shape();

  shuffled_buffer = shuffled_bottom_.mutable_cpu_data();

  InnerToOuter<Dtype>(bottom[0]->cpu_data(),
                      shuffled_buffer,
                      bottom[0]->num(),
                      bottom[0]->channels(),
                      bottom[0]->height(),
                      bottom[0]->width(),
                      canonical_axis);

  int num = ComputeNewOuterDim(bottom_shape,canonical_axis );
  int dim = axis_dim_;

  // calculate mean vector over batch
  caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, shuffled_buffer,
      batch_sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());

  // broadcast and negative mean vector
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      batch_sum_multiplier_.cpu_data(),
      mean_.cpu_data(),
      0.,
      temp_.mutable_cpu_data());

  // subtract mean
  caffe_add(temp_.count(), shuffled_buffer, temp_.cpu_data(),
      temp_.mutable_cpu_data());

  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, axis_dim_, axis_dim_, num,
      1./num,
      temp_.cpu_data(),
      temp_.cpu_data(),
      0.,
      cov_.mutable_cpu_data());

  // Zero out the diagonal terms. We wish to penalize covariance, but not
  // variance.
  Dtype* cov_data = cov_.mutable_cpu_data();
  for(int index = 0; index < axis_dim_; ++index)
  {
    int offset = cov_.offset(index,index,0,0);
    cov_data[offset] = 0.0;
  }

  // square terms in xcov
  Dtype dot = caffe_cpu_dot<Dtype>(cov_.count(), cov_.cpu_data(),
                                   cov_.cpu_data());

  Dtype loss = dot / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CovLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype top_diff = top[0]->cpu_diff()[0];
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();

  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num = ComputeNewOuterDim( bottom[0]->shape(), canonical_axis );

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, axis_dim_, axis_dim_,
      2.0*top_diff/num,
      temp_.cpu_data(),
      cov_.cpu_data(),
      0.,
      bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(CovLossLayer);
#endif

INSTANTIATE_CLASS(CovLossLayer);
REGISTER_LAYER_CLASS(CovLoss);

}  // namespace caffe
