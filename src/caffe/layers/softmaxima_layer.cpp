#include <algorithm>
#include <numeric>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/common_layers.hpp"

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

  int softmax_size = this->layer_param_.softmaxima_param().softmax_size();
  if ( num_softmaxed_inputs % softmax_size != 0 ) {
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
  if(WinnerTakeAll())
  {
    output_probs_.ReshapeLike(*top[0]);
  }

  debug_int_.Reshape(1,1,1,1);
  debug_float_.Reshape(3,1,1,1);
}

template<typename Dtype>
void InitBlob( Blob<Dtype>& blob )
{
  int cardinality = blob.num();
  int channels = blob.channels();
  int height = blob.height();
  int width = blob.width();
  Dtype* ptr = blob.mutable_cpu_data();
  for(int n = 0; n < cardinality; ++n) {
    for(int c = 0; c < channels; ++c ) {
      for(int h = 0; h < height; ++h ) {
        for(int w = 0; w < width; ++w ) {
          ptr[blob.offset(n,c,h,w)] = (Dtype) -333333.0;
        }
      }
    }
  }
}

void AssignDims(const std::vector<int>& shape, int& num, int& channels,
                int& height, int& width)
{
  CHECK(shape.size() >= 2) << "Softmaxima bottom is does not have "
                              << "at least 2 dimensions.";
  CHECK(shape.size() <= 4) << "Softmaxima bottom has more than 4 dimensions.";
  num = shape[0];
  channels = shape[1];
  height = shape.size() >= 3 ? shape[2] : 1;
  width = shape.size() >= 4 ? shape[3] : 1;
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

  Blob<Dtype>& result = WinnerTakeAll() ? output_probs_ : *top[0];
  int cardinality;
  int channels;
  int height;
  int width;
  AssignDims(bottom[0]->shape(), cardinality, channels, height, width);
  // The number of canonical axis input channels must be an integer multiple
  // of the softmax size of the softmaxima layer.
  if (channels % softmax_size_ != 0) {
    LOG(FATAL) << "In a SoftmaximaLayer, the number of channels must "
                  "be integer multiple of the softmax size.";
  }

  int num_softmaxes = channels / softmax_size_;

  for( int instance = 0; instance < cardinality; ++instance) {
    for(int h = 0; h < height; ++h ) {
      for(int w = 0; w < width; ++w ) {
        for( int softmax_index = 0; softmax_index < num_softmaxes;
             ++softmax_index) {

          // Find the max of each channel that participates in this softmax.
          float max;
          // For each scalar that participates in this softmax, compute its
          // exponentiation.
          for( int inner_index = 0; inner_index < softmax_size_;
               ++inner_index ) {
            int channel = softmax_index * softmax_size_ + inner_index;
            float val = input.data_at(instance, channel, h, w);
            if (inner_index == 0)
            {
              max = val;
            }
            else if (val > max)
            {
              max = val;
            }
          }

//          (maxes_.mutable_cpu_data()[maxes_.offset(instance,softmax_index,
//                                                h,w)]) = max;
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
//            bot_minus_maxes_.mutable_cpu_data()[bot_minus_maxes_.offset(instance,channel,h,w)]
//                = val - max;

          }

          std::vector<float> expons;
          // For each scalar that participates in this softmax, compute its
          // exponentiation.
          for( int inner_index = 0; inner_index < softmax_size_;
               ++inner_index ) {
            int channel = softmax_index * softmax_size_ + inner_index;
            float val = input.data_at(instance, channel, h, w);
            float exp_val = std::exp(val);
            expons.push_back(exp_val);

//              std::cout << "exp(" << instance << "," << channel
//                        << "," << h << "," << w << ")"
//                        << "=" << std::exp(val) << std::endl;
//            bot_exponentiated_.mutable_cpu_data()[bot_exponentiated_.offset(instance,channel,h,w)] =
//              exp_val;
          }

          // Compute the sum.
          Dtype sum = std::accumulate(expons.begin(), expons.end(), (Dtype)0.0f);

//            std::cout << "exp(" << instance << "," << softmax_index
//                      << "," << h << "," << w << ")"
//                      << "=" << sum << std::endl;

//            std::cout << "(n,smi,h,w)=" <<

//          denom_sums_.mutable_cpu_data()[denom_sums_.offset(instance,softmax_index,h,w)] =
//              sum;

          // Compute the softmax's output.
          int highest_prob_index = -1;
          Dtype highest_prob = -1.0f;
          Dtype* result_data = result.mutable_cpu_data();
          for( int inner_index = 0; inner_index < softmax_size_;
               ++ inner_index) {
            int channel = softmax_index * softmax_size_ + inner_index;
            Dtype val = expons[inner_index] / sum;
            int offset = result.offset(instance, channel,h,w);

            if ( std::isnan(val)) {
              std::cout << "Found NaN" << std::endl;
            }
            result_data[offset] = val;
            if ( val > highest_prob )
            {
              highest_prob = val;
              highest_prob_index = inner_index;
            }
          }
          // If we are binarizing the output, the highest probability unit
          // has activation = 1, all others zero.
          if (WinnerTakeAll()) {
            for( int inner_index = 0; inner_index < softmax_size_;
                 ++inner_index)
            {
              int channel = softmax_index * softmax_size_ + inner_index;
              int offset = result.offset(instance, channel,h,w);
              top[0]->mutable_cpu_data()[offset] =
                  inner_index == highest_prob_index ? 1.0f : 0.0f;

            }
          }
        }
      }
    }
  }
  //PrintBlob("Naive maxes", maxes);
}

template <typename Dtype>
void SoftmaximaLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  bool winner_take_all = WinnerTakeAll();
  const Dtype* top_data = winner_take_all ? output_probs_.cpu_data() :
                                            top[0]->cpu_data();
  if (this->layer_param_.softmaxima_param().cross_entropy_backprop())
  {
    // Bottom diff will now hold the imputed target output =
    //  top_data - top_diff.
    caffe_sub(top[0]->count(), top_data, top_diff, bottom_diff);

    for(int i =0; i < top[0]->count(); ++i)
    {
      // Constrain the imputed target to be between 0.0 and 1.0.
      if (bottom_diff[i] < 0.0)
      {
        bottom_diff[i] = 0.0;
      }
      else if (bottom_diff[i] > 1.0)
      {
        bottom_diff[i] = 1.0;
      }
    }

    // Now compute bottom diff as actual - imputed target activations.
    caffe_sub(top[0]->count(), top_data, bottom_diff, bottom_diff);
  }
  else
  {
    caffe_copy(top[0]->count(), top_diff, bottom_diff);

    int cardinality;
    int dummy_channels;
    int height;
    int width;
    AssignDims(bottom[0]->shape(), cardinality, dummy_channels,
                    height, width);
    CHECK_EQ(dummy_channels, softmax_size_*num_softmaxes_) <<
          "Inconsistent channels and softmax size and number of softmaxes.";

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

                Dtype diff_val = top_diff[offset];
                Dtype data_val = top_data[offset];
                dot_prod += diff_val * data_val;
              }

              for( int index = 0; index < softmax_size_; ++index) {
                int channel = softmax_index*softmax_size_ + index;
                int offset = top[0]->offset(instance, channel, h, w);
                // Subtract dot prod from each element of the softmax.
                Dtype diff_val = bottom_diff[offset];
                diff_val -= dot_prod;

                Dtype data_val = top_data[offset];
                // multiply activation times previous result.
                diff_val *= data_val;
                bottom_diff[offset] = diff_val;
              }
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
REGISTER_LAYER_CLASS(Softmaxima);

}  // namespace caffe
