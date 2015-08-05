#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

int ComputeNewOuterDim(const std::vector<int>& bottom_shape,
                       int canonical_axis )
{
  int dim = 1;
  for(int i = 0; i < bottom_shape.size(); ++i)
  {
    if (canonical_axis != i )
    {
      dim *= bottom_shape[i];
    }
  }
  return dim;
}

template <typename Dtype>
void XCovLoss2Layer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  mean_vec_.clear();
  mean_vec_.push_back(&mean_0_);
  mean_vec_.push_back(&mean_1_);

  temp_vec_.clear();
  temp_vec_.push_back(&temp_0_);
  temp_vec_.push_back(&temp_1_);
}

template <typename Dtype>
void XCovLoss2Layer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  int num = bottom[0]->num();
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();
  axis_dim0_ = bottom[0]->shape()[canonical_axis];
  axis_dim1_ = bottom[1]->shape()[canonical_axis];

  top[0]->Reshape(1, 1, 1, 1);

  xcov_.Reshape(axis_dim0_, axis_dim1_, 1, 1);

  int new_outer_dim = ComputeNewOuterDim(bottom[0]->shape(),
                                          canonical_axis);

  shuffled_bottom0_.Reshape(new_outer_dim, axis_dim0_,1,1 );
  shuffled_bottom1_.Reshape(new_outer_dim, axis_dim1_,1,1 );

  CHECK(shuffled_bottom0_.count() == bottom[0]->count()) << "shuffled blob"
                                  << " size does not match bottom.";
  CHECK(shuffled_bottom1_.count() == bottom[1]->count()) << "shuffled blob"
                                  << " size does not match bottom.";

  {
    vector<int> mean_vec_shape;
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);

    // Set the shape to be all 1's except the canonical axis which is
    // size axis_dim0.
    mean_vec_shape[canonical_axis] = axis_dim0_;
    mean_vec_[0]->Reshape(mean_vec_shape);

    vector<int> tmp_vec_shape;
    tmp_vec_shape.push_back(new_outer_dim);
    tmp_vec_shape.push_back(1);
    tmp_vec_shape.push_back(1);
    tmp_vec_shape.push_back(1);

    tmp_vec_shape[canonical_axis] = axis_dim0_;
    temp_vec_[0]->Reshape(tmp_vec_shape);
  }

  {
    vector<int> mean_vec_shape;
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);
    mean_vec_shape.push_back(1);
    // Set the shape to be all 1's except the canonical axis which is
    // size axis_dim1.
    mean_vec_shape[canonical_axis] = axis_dim1_;
    mean_vec_[1]->Reshape(mean_vec_shape);
    vector<int> tmp_vec_shape;
    tmp_vec_shape.push_back(new_outer_dim);
    tmp_vec_shape.push_back(1);
    tmp_vec_shape.push_back(1);
    tmp_vec_shape.push_back(1);
    tmp_vec_shape[canonical_axis] = axis_dim1_;
    temp_vec_[1]->Reshape(tmp_vec_shape);
  }

  batch_sum_multiplier_.Reshape(new_outer_dim, 1, 1, 1);
  Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
  caffe_set(batch_sum_multiplier_.count(), Dtype(1), batch_multiplier_data);
}

template <typename Dtype>
void XCovLoss2Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // for now, we support only two inputs
  CHECK_EQ(bottom.size(), 2);
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();

  for (int i = 0 ; i < bottom.size() ; i++) {
    Dtype* shuffled_buffer = 0;
    int axis_dim = 0;
    std::vector<int> bottom_shape = bottom[i]->shape();

    if ( i == 0 )
    {
      shuffled_buffer = shuffled_bottom0_.mutable_cpu_data();
      axis_dim = axis_dim0_;
    } else if ( i == 1 ) {
      shuffled_buffer = shuffled_bottom1_.mutable_cpu_data();
      axis_dim = axis_dim1_;
    }

    InnerToOuter<Dtype>(bottom[i]->cpu_data(),
                        shuffled_buffer,
                        bottom[i]->num(),
                        bottom[i]->channels(),
                        bottom[i]->height(),
                        bottom[i]->width(),
                        canonical_axis);

    int num = ComputeNewOuterDim(bottom_shape,canonical_axis );
    int dim = axis_dim;

    // calculate mean vector over batch
    caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, shuffled_buffer,
        batch_sum_multiplier_.cpu_data(), 0., mean_vec_[i]->mutable_cpu_data());

    // broadcast and negative mean vector
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        batch_sum_multiplier_.cpu_data(),
        mean_vec_[i]->cpu_data(),
        0.,
        temp_vec_[i]->mutable_cpu_data());

    // subtract mean
    caffe_add(temp_vec_[i]->count(), shuffled_buffer, temp_vec_[i]->cpu_data(),
        temp_vec_[i]->mutable_cpu_data());
  }

  int num = ComputeNewOuterDim( bottom[0]->shape(),canonical_axis );
  int dim0 = axis_dim0_;
  int dim1 = axis_dim1_;

  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim0, dim1, num, 1./num,
      temp_vec_[0]->cpu_data(),
      temp_vec_[1]->cpu_data(),
      0.,
      xcov_.mutable_cpu_data());

  // square terms in xcov
  Dtype dot = caffe_cpu_dot<Dtype>(xcov_.count(), xcov_.cpu_data(),
      xcov_.cpu_data());

  Dtype loss = dot / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void XCovLoss2Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype top_diff = top[0]->cpu_diff()[0];
  int canonical_axis = this->layer_param_.xcov_loss_param().axis();

  Dtype* bottom_diff_0 = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff();
//  Dtype* bottom_diff_0 = shuffled_bottom0_.mutable_cpu_diff();
//  Dtype* bottom_diff_1 = shuffled_bottom1_.mutable_cpu_diff();

//  int num = bottom[0]->num();
//  int dim0 = bottom[0]->count() / num;
//  int dim1 = bottom[1]->count() / num;

  int num = ComputeNewOuterDim( bottom[0]->shape(),canonical_axis );
  int dim0 = axis_dim0_;
  int dim1 = axis_dim1_;

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim0, dim1,
      top_diff/num,
      temp_vec_[1]->cpu_data(),
      xcov_.cpu_data(),
      0.,
      bottom_diff_0);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim1, dim0,
      top_diff/num,
      temp_vec_[0]->cpu_data(),
      xcov_.cpu_data(),
      0.,
      bottom_diff_1);

//  // Now return the shuffled result to the bottom blobs.
//  OuterToInner<Dtype>(bottom[0]->mutable_cpu_diff(),
//                      shuffled_bottom0_.cpu_data(),
//                      bottom[0]->num(),
//                      bottom[0]->channels(),
//                      bottom[0]->height(),
//                      bottom[0]->width(),
//                      1);

//  // Now return the shuffled result to the bottom blobs.
//  OuterToInner<Dtype>(bottom[1]->mutable_cpu_diff(),
//                      shuffled_bottom1_.cpu_data(),
//                      bottom[1]->num(),
//                      bottom[1]->channels(),
//                      bottom[1]->height(),
//                      bottom[1]->width(),
//                      1);
}

#ifdef CPU_ONLY
STUB_GPU(XCovLoss2Layer);
#endif

INSTANTIATE_CLASS(XCovLoss2Layer);
REGISTER_LAYER_CLASS(XCovLoss2);

}  // namespace caffe
