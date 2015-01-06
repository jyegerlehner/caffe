#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InverseMVNLayer<Dtype>::SetBlobFinder(const BlobFinder<Dtype> &blob_finder)
{
  this->blob_helper_ = MvnBlobHelper<Dtype>( this->layer_param_, blob_finder );
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::LayerSetUp(const std::vector<Blob<Dtype> *> &bottom,
                                        const std::vector<Blob<Dtype> *> &top) {
  const MVNParameter& param = this->layer_param_.mvn_param();

  // If the parameter specifies a name for the variance and or mean blob,
  // then that blob must appear in the bottom vector of blobs.
  bool specifies_var_blob = param.has_variance_blob();
  if ( specifies_var_blob)
  {
    CHECK(blob_helper_.VarianceBlob(bottom) != NULL) << "InverseMVNLayer " <<
      this->layer_param_.name() << " specifies variance blob " <<
      param.variance_blob() << " but was not found in bottom blobs.";
  }

  bool it_has_mean_blob = param.has_mean_blob();
  CHECK(it_has_mean_blob) << "InverseMVNLayer requires a mean in the "
                             << "bottom blobs.";

  CHECK(blob_helper_.MeanBlob(bottom) != NULL) << "Mean blob "
         << param.mean_blob() << " not found bottom blobs of layer "
         << this->layer_param_.name();

  CHECK(blob_helper_.DataBlob(bottom) != NULL) << "No bottom data blob found "
    << "for InverseMVNLayer " << this->layer_param_.name();
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  Blob<Dtype>* output_blob = top[0];
  Blob<Dtype>* input_blob = blob_helper_.DataBlob(bottom);

  // Reshape the data output blob, which is always in the top vector.
  output_blob->Reshape(input_blob->num(), input_blob->channels(),
                       input_blob->height(), input_blob->width());

  temp_.Reshape(input_blob->num(), input_blob->channels(),
      input_blob->height(), input_blob->width());
  sum_multiplier_.Reshape(1, 1, input_blob->height(), input_blob->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = blob_helper_.DataBlob(bottom)->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  Blob<Dtype>* mean_blob = blob_helper_.MeanBlob(bottom);
  if (this->layer_param_.mvn_param().normalize_variance()) {
    // Get the variance blob.
    Blob<Dtype>* variance_blob = blob_helper_.VarianceBlob(bottom);

    // denormalize for variance.
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_blob->cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    caffe_mul(temp_.count(), bottom_data, temp_.cpu_data(), top_data);

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_blob->cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  } else {
    // Add the mean back.
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_blob->cpu_data(), sum_multiplier_.cpu_data(), 0.,
            temp_.mutable_cpu_data());

    caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);
  }
}

template <typename Dtype>
void InverseMVNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int dim = bottom[0]->count() / num;
  Dtype eps = 1e-10;

  if (this->layer_param_.mvn_param().normalize_variance()) {
//    caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
//    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
//          sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
//    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
//          mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
//          bottom_diff);
//    caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

//    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
//            sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
//    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
//            mean_.cpu_data(), sum_multiplier_.cpu_data(), 1.,
//            bottom_diff);

//    caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
//        bottom_diff);

//    // put the squares of bottom into temp_
//    caffe_powx(temp_.count(), bottom_data, Dtype(2),
//        temp_.mutable_cpu_data());

//    // computes variance using var(X) = E(X^2) - (EX)^2
//    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
//        sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
//    caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
//        sum_multiplier_.cpu_data(), 0.,
//        variance_.mutable_cpu_data());  // E(X^2)
//    caffe_powx(mean_.count(), mean_.cpu_data(), Dtype(2),
//        temp_.mutable_cpu_data());  // (EX)^2
//    caffe_sub(mean_.count(), variance_.cpu_data(), temp_.cpu_data(),
//        variance_.mutable_cpu_data());  // variance

//    // normalize variance
//    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
//          variance_.mutable_cpu_data());

//    caffe_add_scalar(variance_.count(), eps, variance_.mutable_cpu_data());

//    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
//        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
//        temp_.mutable_cpu_data());

//    caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);


    // remove this
    caffe_copy(temp_.count(), top_diff, bottom_diff);
  } else {
    caffe_copy(temp_.count(), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(InverseMVNLayer);
#endif

INSTANTIATE_CLASS(InverseMVNLayer);
REGISTER_LAYER_CLASS(MVN, InverseMVNLayer);
}  // namespace caffe
