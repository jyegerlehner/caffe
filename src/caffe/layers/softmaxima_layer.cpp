#include <algorithm>
#include <numeric>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaximaLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);

  int num_softmaxed_inputs = bottom[0]->shape(softmax_axis_);
  if (!this->layer_param_.has_softmaxima_param()) {
    LOG(FATAL) << "Softmaxima layer must have a softmaxima_param.";
  }

  if (!this->layer_param_.softmaxima_param().has_softmax_size()) {
    LOG(FATAL) << "Must specify size of softmaxima's softmax.";
  }

  if ( num_softmaxed_inputs % this->layer_param_.softmaxima_param().softmax_size()
       != 0 ) {
    LOG(FATAL) << "Softmaxima's canonical axis size must be an "
               << "integer multiple of the softmax_size.";
  }

  softmax_size_ = this->layer_param_.softmaxima_param().softmax_size();
  num_softmaxes_ = num_softmaxed_inputs / softmax_size_;

  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = num_softmaxes_;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaximaLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Blob<Dtype>& orig_input = *(bottom[0]);
  Blob<Dtype> input;
  input.ReshapeLike(orig_input);
  Dtype* in_data = input.mutable_cpu_data();
  const Dtype* orig_in_data = orig_input.cpu_data();
  for(int i=0; i < orig_input.count(); ++i)
  {
    *(in_data++) = *(orig_in_data++);
  }

  Blob<Dtype>& result = *top[0];//( new Blob<Dtype>());
  std::vector<int> shape = bottom[0]->shape();
  int cardinality = shape[0];
  int channels = shape[1];
  int height = shape[2];
  int width = shape[3];

  // The number of canonical axis input channels must be an integer multiple
  // of the softmax size of the softmaxima layer.
  if (channels % softmax_size_ != 0) {
    LOG(FATAL) << "In a SoftmaximaLayer, the number of channels must "
                  "be integer multiple of the softmax size.";
  }

  int num_softmaxes = channels / softmax_size_;

  Blob<Dtype> maxes;
  maxes.Reshape(cardinality, num_softmaxes,height,width);

//    std::cout << "NaiveForward:" << std::endl;

  for( int instance = 0; instance < cardinality; ++instance) {
    for(int h = 0; h < height; ++h ) {
      for(int w = 0; w < width; ++w ) {
        for( int softmax_index = 0; softmax_index < num_softmaxes;
             ++softmax_index) {

          // Find the max of each channel that participates in this softmax.
          float max = -10000.0;
          // For each scalar that participates in this softmax, compute its
          // exponentiation.
          for( int inner_index = 0; inner_index < softmax_size_;
               ++inner_index ) {
            int channel = softmax_index * softmax_size_ + inner_index;
            float val = input.data_at(instance, channel, h, w);
            //expons.push_back(std::exp(val));
            if ( val > max)
            {
              max = val;
            }
          }

          (maxes.mutable_cpu_data()[maxes.offset(instance,softmax_index,
                                                h,w)]) = max;
//            std::cout << "max(" << instance << "," << softmax_index << ","
//                      << h << "," << w << ")"
//                      << "=" << max << std::endl;

          in_data = input.mutable_cpu_data();
          // Subtract the max from each of the values before exponentiating.
          for( int inner_index = 0; inner_index < softmax_size_;
               ++inner_index ) {
            int channel = softmax_index * softmax_size_ + inner_index;
            float val = input.data_at(instance, channel, h, w);
            //expons.push_back(std::exp(val));
            in_data[input.offset(instance,channel,h,w)] =
                val - max;

//              std::cout << "minusmax(" << instance << "," << channel
//                        << "," << h << "," << w << ")"
//                        << "=" << (val - max) << std::endl;

          }

          std::vector<float> expons;
          // For each scalar that participates in this softmax, compute its
          // exponentiation.
          for( int inner_index = 0; inner_index < softmax_size_;
               ++inner_index ) {
            int channel = softmax_index * softmax_size_ + inner_index;
            float val = input.data_at(instance, channel, h, w);
            expons.push_back(std::exp(val));

//              std::cout << "exp(" << instance << "," << channel
//                        << "," << h << "," << w << ")"
//                        << "=" << std::exp(val) << std::endl;
          }

          // Compute the sum.
          Dtype sum = std::accumulate(expons.begin(), expons.end(), (Dtype)0.0f);

//            std::cout << "exp(" << instance << "," << softmax_index
//                      << "," << h << "," << w << ")"
//                      << "=" << sum << std::endl;

//            std::cout << "(n,smi,h,w)=" <<
          // Compute the softmax's output.
          for( int inner_index = 0; inner_index < softmax_size_;
               ++ inner_index) {
            int channel = softmax_index * softmax_size_ + inner_index;
            Dtype val = expons[inner_index] / sum;
            Dtype* result_data = result.mutable_cpu_data();
            int offset = result.offset(instance, channel,h,w);
            result_data[offset] = val;
          }
        }
      }
    }
  }
  //PrintBlob("Naive maxes", maxes);
}

//template <typename Dtype>
//void SoftmaximaLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
//    const vector<bool>& propagate_down,
//    const vector<Blob<Dtype>*>& bottom) {

//  const Dtype* top_diff = top[0]->cpu_diff();
//  const Dtype* top_data = top[0]->cpu_data();
//  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//  Dtype* scale_data = scale_.mutable_cpu_data();
//  int channels = top[0]->shape(softmax_axis_);
//  int dim = top[0]->count() / outer_num_;
//  caffe_copy(top[0]->count(), top_diff, bottom_diff);
//  for (int i = 0; i < outer_num_; ++i) {
//    // compute dot(top_diff, top_data) and subtract them from the bottom diff
//    for (int k = 0; k < inner_num_; ++k) {
//      for(int softmax_index = 0;
//          softmax_index < num_softmaxes_;
//          ++softmax_index) {
//        int offset = i * dim + softmax_index*softmax_size_*inner_num_ + k ;
//        scale_data[softmax_index*inner_num_ + k] =
//            caffe_cpu_strided_dot<Dtype>(softmax_size_,
//            bottom_diff + offset, inner_num_,
//            top_data + offset, inner_num_);
//      }
//    }
//    // subtraction
//    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
//        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
//  }
//  // elementwise multiplication
//  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
//}

template <typename Dtype>
void SoftmaximaLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
//  Dtype* scale_data = scale_.mutable_cpu_data();
//  int channels = top[0]->shape(softmax_axis_);
//  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);

  std::vector<int> shape = bottom[0]->shape();
  int cardinality = shape[0];
  int height = shape[2];
  int width = shape[3];

  for( int instance = 0; instance < cardinality; ++instance) {
    for(int h = 0; h < height; ++h ) {
      for(int w = 0; w < width; ++w ) {
        for( int softmax_index = 0; softmax_index < num_softmaxes_;
             ++softmax_index) {

            Dtype dot_prod = 0.0;
            // Compute each softmax's dot product of diffs and activations.
            for( int index = 0; index < softmax_size_; ++index) {
              int channel = softmax_index*softmax_size_ + index;
              int offset = top[0]->offset(instance, channel, h, w);

              dot_prod += top_diff[offset] * top_data[offset];
            }

            // Compute each softmax's dot product of diffs and activations.
            for( int index = 0; index < softmax_size_; ++index) {
              int channel = softmax_index*softmax_size_ + index;
              int offset = top[0]->offset(instance, channel, h, w);
              // Subtract dot prod from each element of the softmax.
              bottom_diff[offset] -= dot_prod;

              // multiply activation times previous result.
              bottom_diff[offset] *= top_data[offset];
            }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaximaLayer);
#endif

INSTANTIATE_CLASS(SoftmaximaLayer);

}  // namespace caffe
