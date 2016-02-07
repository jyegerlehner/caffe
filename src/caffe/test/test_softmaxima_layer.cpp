#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/solver.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/softmaxima_layer.hpp"
#include "caffe/sgd_solvers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

//template <typename Dtype>
//void PrintBlob( const std::string& nam, const Blob<Dtype>& blob, bool diffs_not_data = false )
//{
//  std::cout << "Blob: " << nam << std::endl;
//  int num = blob.num();
//  int chans = blob.channels();
//  int height;
//  int width;
//  if (blob.shape().size() > 2)
//  {
//    height = blob.height();
//    width = blob.width();
//  }
//  else
//  {
//    height = 1;
//    width = 1;
//  }

//  std::cout << "shape=(" << num << "," << chans << "," << height << ","
//            << width << ")" << std::endl;
//  for( int n = 0; n < num; ++n) {
//    for( int c = 0; c < chans; ++c) {
//      for( int h = 0; h < height; ++h) {
//        for( int w=0; w < width; ++w) {
//          Dtype val = diffs_not_data ? blob.diff_at(n,c,h,w) : blob.data_at(n,c,h,w);
//          std::cout << "data(" << n << "," << c << "," << h << "," << w << ")"
//                       << "=" << val << std::endl;
//        }
//      }
//    }
//  }
//}

template <typename Dtype>
struct NaiveSoftmaximaLayer {
  NaiveSoftmaximaLayer( int softmax_size ) :
  softmax_size_(softmax_size) {
  }

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

  shared_ptr<Blob<Dtype> > Forward( const Blob<Dtype>& orig_input) {
    // Make a copy of the original input.
    Blob<Dtype> input;
    input.ReshapeLike(orig_input);
    Dtype* in_data = input.mutable_cpu_data();
    const Dtype* orig_in_data = orig_input.cpu_data();
    for(int i=0; i < orig_input.count(); ++i)
    {
      *(in_data++) = *(orig_in_data++);
    }

    shared_ptr<Blob<Dtype> > result( new Blob<Dtype>());
    result->ReshapeLike(input);
    std::vector<int> shape = input.shape();
    int cardinality = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];

    // The number of canonical axis input channels must be an integer multiple
    // of the softmax size of the softmaxima layer.
    EXPECT_EQ(channels % softmax_size_, 0);

    int num_softmaxes = channels / softmax_size_;

    maxes_.Reshape(cardinality, num_softmaxes, height, width);
    bot_minus_maxes_.Reshape(cardinality, channels, height, width);
    bot_exponentiated_.Reshape(cardinality, channels, height, width);
    denom_sums_.Reshape(cardinality, num_softmaxes, height, width);

    InitBlob(maxes_);
    InitBlob(bot_minus_maxes_);
    InitBlob(bot_exponentiated_);
    InitBlob(denom_sums_);

//    std::cout << "NaiveForward:" << std::endl;

    for( int instance = 0; instance < cardinality; ++instance) {
      for(int h = 0; h < height; ++h ) {
        for(int w = 0; w < width; ++w ) {
          for( int softmax_index = 0; softmax_index < num_softmaxes;
               ++softmax_index) {

            // Find the max of each channel that participates in this softmax.
            float max = -std::numeric_limits<float>::max();
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

            (maxes_.mutable_cpu_data()[maxes_.offset(instance,softmax_index,
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
              bot_minus_maxes_.mutable_cpu_data()[bot_minus_maxes_.offset(instance,channel,h,w)]
                  = val - max;
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

              bot_exponentiated_.mutable_cpu_data()[bot_exponentiated_.offset(instance,channel,h,w)] =
                exp_val;
            }

            // Compute the sum.
            Dtype sum = std::accumulate(expons.begin(), expons.end(), (Dtype)0.0f);

//            std::cout << "exp(" << instance << "," << softmax_index
//                      << "," << h << "," << w << ")"
//                      << "=" << sum << std::endl;

            denom_sums_.mutable_cpu_data()[denom_sums_.offset(instance,softmax_index,h,w)] =
                sum;
//            std::cout << "(n,smi,h,w)=" <<
            // Compute the softmax's output.
            for( int inner_index = 0; inner_index < softmax_size_;
                 ++ inner_index) {
              int channel = softmax_index * softmax_size_ + inner_index;
              Dtype val = expons[inner_index] / sum;
              Dtype* result_data = result->mutable_cpu_data();
              int offset = result->offset(instance, channel,h,w);
              result_data[offset] = val;
            }
          }
        }
      }
    }
    //PrintBlob("Naive maxes", maxes);
    return result;
  }

  Blob<Dtype>& GetMaxes() {
    return maxes_;
  }

  Blob<Dtype>& GetBotMinusMaxes() {
    return bot_minus_maxes_;
  }

  Blob<Dtype>& GetBotExponentiated() {
    return bot_exponentiated_;
  }

  Blob<Dtype>& GetDenomSums() {
    return denom_sums_;
  }

private:
  int softmax_size_;
  // The maxes of each softmax. N * num_softmaxes * h * w
  Blob<Dtype> maxes_;
  // The bottom, with the max for each softmax subtracted off.
  // N * chans * h * w
  Blob<Dtype> bot_minus_maxes_;
  // The bottom, after subtracting per-softmax-max, then exponentiated.
  // N * chans * h * w
  Blob<Dtype> bot_exponentiated_;
  // The sum of the exponentiated inputs that are summed for the softmax.
  // N * num_softmaxes * h * w
  Blob<Dtype> denom_sums_;
};

template <typename Dtype>
struct NaiveNonGPUSoftmaximaLayer {
  NaiveNonGPUSoftmaximaLayer(int softmax_size) :
    softmax_size_(softmax_size),
    winner_take_all_(false) {}

  void SetWinnerTakeAll() {
    winner_take_all_ = true;
  }

  Blob<Dtype>& GetMaxes() {
    return maxes_;
  }

  Blob<Dtype>& GetBotMinusMaxes() {
    return bot_minus_maxes_;
  }

  Blob<Dtype>& GetBotExponentiated() {
    return bot_exponentiated_;
  }

  Blob<Dtype>& GetDenomSums() {
    return denom_sums_;
  }

  shared_ptr<Blob<Dtype> > Forward( const Blob<Dtype>& input) {
    const Dtype* bottom_data = input.cpu_data();

    int softmax_axis = 1;
    int outer_num_ = input.count(0, softmax_axis);
    int inner_num_ = input.count(softmax_axis + 1);

    shared_ptr<Blob<Dtype> > result( new Blob<Dtype>());
    result->ReshapeLike(input);

    int num_softmaxed_inputs = input.shape(softmax_axis);

//    int num_softmaxed_inputs =
//      vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));

    vector<int> scale_dims = input.shape();
    int num_softmaxes_ = num_softmaxed_inputs / softmax_size_;
    Blob<Dtype> scale;
    scale_dims[softmax_axis] = num_softmaxes_;
    scale.Reshape(scale_dims);
    Dtype* scale_data = scale.mutable_cpu_data();

    int count = input.count();
    int channels = num_softmaxed_inputs;
    //caffe_copy(count, bottom_data, top_data);
    Dtype* top_data = result->mutable_cpu_data();
    for( int i=0; i < count; ++i)
    {
      Dtype dat = *(bottom_data++);
      *(top_data++) = dat;
    }

    top_data = result->mutable_cpu_data();
    scale_data = scale.mutable_cpu_data();
    // We need to subtract the max to avoid numerical issues, compute the exp,
    // and then normalize.
    // compute max
    // Put the max for each softmax in scale.
    // inner_num_*outer_num_
    kernel_channel_max( outer_num_,
                        channels,
                        inner_num_,
                        softmax_size_,
                        num_softmaxes_,
                        top_data,
                        scale_data);

    maxes_.CopyFrom(scale,false,true);

    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
//    kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
//        CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
//        scale_data, top_data);
    top_data = result->mutable_cpu_data();
    scale_data = scale.mutable_cpu_data();
      kernel_channel_subtract(count,
                             softmax_size_,
                             inner_num_,
                             scale_data,
                             top_data);

      bot_minus_maxes_.CopyFrom(*result,false,true);

      //Print out the blob after maxes have been subtracted.
//      PrintBlob("NonGPU top after subtract", *result);


    // exponentiate
    // NOLINT_NEXT_LINE(whitespace/operators)
//    kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
//        count, top_data, top_data);
      top_data = result->mutable_cpu_data();
      scale_data = scale.mutable_cpu_data();
    kernel_exp(count, top_data, top_data);

    bot_exponentiated_.CopyFrom(*result,false,true);

    // sum after exp
    // NOLINT_NEXT_LINE(whitespace/operators)
//    kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
//        CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
//        scale_data);

    top_data = result->mutable_cpu_data();
    scale_data = scale.mutable_cpu_data();
    kernel_channel_sum(outer_num_,
                       channels,
                       inner_num_,
                       softmax_size_,
                       num_softmaxes_,
                       top_data,
                       scale_data);

    denom_sums_.CopyFrom(scale,false,true);

//    PrintBlob("NonGPU softmax sums after exp", scale);

    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
//    kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
//        CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
//        scale_data, top_data);

    if ( winner_take_all_ ) {
      out_probs_.Reshape(result->shape());
    }
    top_data = result->mutable_cpu_data();
    scale_data = scale.mutable_cpu_data();
    Dtype* out_probs_data = winner_take_all_ ? out_probs_.mutable_cpu_data():
                                               0;
    kernel_channel_div(outer_num_,
                       channels,
                       inner_num_,
                       softmax_size_,
                       num_softmaxes_,
                       scale_data,
                       top_data,
                       winner_take_all_,
                       out_probs_data);

//    PrintBlob("NonGPU softmax div by sum", top_data);

    return result;
  }
private:
  void kernel_channel_max( const int num,
                           const int channels,
                           const int spatial_dim,
                           const int softmax_size,
                           const int num_softmaxes,
                           const Dtype* data,
                           Dtype* out) {
    for( int index = 0; index < num * spatial_dim; ++index)  {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      // For each softmax along the canonical axis.
      for( int smi = 0; smi < num_softmaxes; ++smi) {
        Dtype maxval = -std::numeric_limits<float>::max();
        // For each channel within this softmax.
        for (int c_off = 0; c_off < softmax_size; ++c_off) {
          int c = smi * softmax_size + c_off;
          int data_index = (n * channels + c) * spatial_dim + s;
          maxval = std::max(data[data_index], maxval);
        }
        //int out_index = index*num_softmaxes + smi;
        int out_index = s + (n * num_softmaxes + smi) * spatial_dim ; //index*num_softmaxes + smi;
        out[out_index] = maxval;
      }
    }
  }

  void kernel_channel_sum( const int num,
                           const int channels,
                           const int spatial_dim,
                           const int softmax_size,
                           const int num_softmaxes,
                           const Dtype* data,
                           Dtype* out) {

    for( int index = 0; index < num * spatial_dim; ++index)  {
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
        int out_index = s + (n * num_softmaxes + smi) * spatial_dim ; //index*num_softmaxes + smi;
        out[out_index] = sum;
      }
    }
  }

  void kernel_channel_div( const int num,
                           const int channels,
                           const int spatial_dim,
                           const int softmax_size,
                           const int num_softmaxes,
                           const Dtype* sums,
                           Dtype* out,
                           bool winner_take_all,
                           Dtype* out_probs) {

    for( int index = 0; index < num * spatial_dim; ++index)  {
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
          Dtype val = out[data_index] / sum;
          if ( winner_take_all) {
            if ( val > largest_prob)
            {
              largest_prob = val;
              largest_prob_index = data_index;
            }
            out_probs[data_index] = val;
          } else {
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

//        // For each channel within this softmax.
//        for (int c_off = 0; c_off < softmax_size; ++c_off) {
//          int c = smi * softmax_size + c_off;
//          int data_index = (n * channels + c) * spatial_dim + s;
//          out[data_index] = out[data_index] / sum;
//        }
//        //int out_index = index*num_softmaxes + smi;
      }
    }
  }


  void kernel_channel_subtract(const int count,
                               const int softmax_size,
                               const int spatial_dim,
                               const Dtype* channel_max,
                               Dtype* data)
  {
    for( int index = 0; index < count; ++index )
    {
      int n = index / softmax_size / spatial_dim;
      int s = index % spatial_dim;

      int softmax_max_index = n * spatial_dim + s;
      //int softmax_index = chanset*num_softmaxes
      data[index] -= channel_max[softmax_max_index];
    }
  }

  void kernel_exp(const int count, const Dtype* data, Dtype* out)
  {
    for( int index = 0; index < count; ++index )
    {
      out[index] = exp(data[index]);
    }
  }

  int softmax_size_;

  Blob<Dtype> maxes_;
  // The bottom, with the max for each softmax subtracted off.
  // N * chans * h * w
  Blob<Dtype> bot_minus_maxes_;
  // The bottom, after subtracting per-softmax-max, then exponentiated.
  // N * chans * h * w
  Blob<Dtype> bot_exponentiated_;
  // The sum of the exponentiated inputs that are summed for the softmax.
  // N * num_softmaxes * h * w
  Blob<Dtype> denom_sums_;
  // The output of the softmax as probabilities in the case where
  // winner_take_all_ == true. Otherwise ignored.
  Blob<Dtype> out_probs_;
  bool winner_take_all_;

};

template<typename Dtype>
void FillBottomBlob(Blob<Dtype>* blob, Dtype offset) {
  Dtype* bdata = blob->mutable_cpu_data();
  int num = blob->num();
  int channels = blob->channels();
  int height = blob->height();
  int width = blob->width();
  for(int n = 0; n < num; ++n ) {
    for( int c = 0; c < channels; ++c ) {
      for( int h = 0; h < height; ++h ) {
        for( int w = 0; w < width; ++w ) {
          bdata[ blob->offset(n, c,h, w)] =
              n* 2.0  - c * 1.0 - ( w * w/blob->width() * 2.0 - 1.0) * h + offset;
        }
      }
    }
  }
}

template <typename TypeParam>
class SoftmaximaLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaximaLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()),
        blob_bottom1_( new Blob<Dtype>(10,16,36,36)),
        blob_bottom2_( new Blob<Dtype>(10,16,36,36)) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
//    filler.Fill(this->blob_bottom_);
    {
      FillBottomBlob(blob_bottom_.get(), (Dtype) -1.0f);
      FillBottomBlob(blob_bottom1_.get(), (Dtype)  -1.0f);
      FillBottomBlob(blob_bottom2_.get(), (Dtype) 0.5f);
    }
    blob_bottom_vec_.push_back(blob_bottom_.get());
    blob_top_vec_.push_back(blob_top_.get());

//    filler.Fill(this->blob_bottom1_);
//    filler.Fill(this->blob_bottom2_);
  }
  virtual ~SoftmaximaLayerTest()
  {
    blob_bottom_.reset();
    blob_top_.reset();
    blob_bottom1_.reset();
    blob_bottom2_.reset();
  }
  shared_ptr<Blob<Dtype> > blob_bottom_;
  shared_ptr<Blob<Dtype> > blob_top_;

  // Multiple iteration test
  shared_ptr<Blob<Dtype> > blob_bottom1_;
  shared_ptr<Blob<Dtype> > blob_bottom2_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  shared_ptr<Blob<Dtype> > ForwardPropThroughTwoSoftmaxes( int softmax_size,
                                shared_ptr<Blob<Dtype> > bottom) {
    std::stringstream ss;
    ss <<    "name: 'TestNetwork' ";
    ss <<    "input: 'data' ";
    ss <<    "input_dim: " << bottom->num() << " ";
    ss <<    "input_dim: " << bottom->channels() << " ";
    ss <<    "input_dim: " << bottom->height() << " ";
    ss <<    "input_dim: " << bottom->width() << " ";
    ss <<    "layer { ";
    ss <<    "  name: 'slice' ";
    ss <<    "  type: 'Slice' ";
    ss <<    "  bottom: 'data' ";
    ss <<    "  top: 'slice1' ";
    ss <<    "  top: 'slice2' ";
    ss <<    "  slice_param {";
    ss <<    "     slice_point: " << softmax_size << " ";
    ss <<    "  }";
    ss <<    "} ";
    ss <<    "layer { ";
    ss <<    " name: 'softmax1' ";
    ss <<    " type: 'Softmax' ";
    ss <<    " bottom: 'slice1' ";
    ss <<    " top: 'sm1' ";
    ss <<    "} ";
    ss <<    "layer { ";
    ss <<    " name: 'softmax2' ";
    ss <<    " type: 'Softmax' ";
    ss <<    " bottom: 'slice2' ";
    ss <<    " top: 'sm2' ";
    ss <<    "} ";
    ss <<    "layer { ";
    ss <<    " name: 'concat' ";
    ss <<    " type: 'Concat' ";
    ss <<    " bottom: 'sm1' ";
    ss <<    " bottom: 'sm2' ";
    ss <<    " top: 'result' ";
    ss <<    "} ";

    std::string proto = ss.str();

    vector<Blob<Dtype>*> bottom_blob_vec;
    bottom_blob_vec.push_back(bottom.get());
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
//    BlobFinder<Dtype> blob_finder;
//    LayerFinder<Dtype> layer_finder;
    Net<Dtype> net( param );
    net.Forward(bottom_blob_vec);

    EXPECT_TRUE(net.has_blob("data"));
    EXPECT_TRUE(net.has_blob("result"));
    shared_ptr<Blob<Dtype> > result_blob;
    result_blob = net.blob_by_name("result");
    return result_blob;
  }

  std::string CreateXorPrototxt(bool strict_sparsity)
  {
    std::stringstream ss;
    ss << "base_lr: 0.1 ";
    ss << "lr_policy: 'fixed' ";
    ss << "display: 1 ";
    ss << "momentum: 0.90 ";
    ss << "solver_type: SGD ";
    ss << "solver_mode: GPU ";
    ss << "device_id: 0 ";
    ss << "max_iter: 20000 ";
//    ss << "debug_info: true ";
    ss << "net_param { ";
    ss << "name: 'XorNetwork' ";
    ss << "layer { ";
    ss << "  name: 'input_data_layer' ";
    ss << "  type: 'DummyData' ";
    ss << "  top: 'input_data' ";
    ss << "  dummy_data_param {";
    ss << "    shape { ";
    ss << "      dim: 4 ";
    ss << "      dim: 2 ";
    ss << "     } ";
    ss << "  } ";
    ss << "} ";
    ss << "layer { ";
    ss << "  name: 'output_target_layer' ";
    ss << "  type: 'DummyData' ";
    ss << "  top: 'target_data' ";
    ss << "  dummy_data_param {";
    ss << "    shape { ";
    ss << "      dim: 4 ";
    ss << "      dim: 6 ";
    ss << "     } ";
    ss << "  } ";
    ss << "} ";
    ss << "layer { ";
    ss << "  name: 'innerprod1' ";
    ss << "  type: 'InnerProduct' ";
    ss << "  inner_product_param { ";
    ss << "    num_output: 3 ";
    ss << "    weight_filler { ";
    ss << "      type: 'gaussian' ";
    ss << "      std: 0.1 ";
    ss << "    } ";
    ss << "    bias_filler { ";
    ss << "      type: 'constant' ";
    ss << "    } ";
    ss << "  } ";
    ss << "  bottom: 'input_data' ";
    ss << "  top: 'ip1' ";
    ss << "} ";
    ss << "layer { ";
    ss << "  name: 'sigmoid1' ";
    ss << "  type: 'Sigmoid' ";
    ss << "  bottom: 'ip1' ";
    ss << "  top: 'sigmoid1' ";
    ss << "} ";
    ss << "layer { ";
    ss << "  name: 'innerprod2' ";
    ss << "  type: 'InnerProduct' ";
    ss << "  inner_product_param { ";
    ss << "    num_output: 6 ";
    ss << "    weight_filler { ";
    ss << "      type: 'gaussian' ";
    ss << "      std: 0.1 ";
    ss << "    } ";
    ss << "    bias_filler { ";
    ss << "      type: 'constant' ";
    ss << "    } ";
    ss << "  } ";
    ss << "  bottom: 'sigmoid1' ";
    ss << "  top: 'ip2' ";
    ss << "} ";
    ss << "layer { ";
    ss << "  name: 'softmaxima' ";
    ss << "  type: 'Softmaxima' ";
    ss << "  bottom: 'ip2' ";
    ss << "  top: 'softmaxima' ";
    ss << "  softmaxima_param { ";
    ss << "    softmax_size: 2 ";
    if(strict_sparsity)
    {
      ss << "    mode: STRICT_SPARSITY";
    }
    ss << "  } ";
    ss << "} ";
    ss << "layer { ";
    ss << "  name: 'loss' ";
    ss << "  type: 'EuclideanLoss' ";
    ss << "  bottom: 'softmaxima' ";
    ss << "  bottom: 'target_data' ";
    ss << "  top: 'loss' ";
    ss << "} ";
    ss << "} ";
    return ss.str();
  }

  template<typename Dtype>
  struct InputOutputVal
  {
    InputOutputVal(Dtype lx0, Dtype lx1):
      x0(lx0),
      x1(lx1)
    {
    }

    Dtype x0;
    Dtype x1;
  };

  void AssignXorBlobs(Net<Dtype>& net)
  {
    typedef InputOutputVal<Dtype> IOVal;
    {
      shared_ptr<Blob<Dtype> > input = net.blob_by_name("input_data");
      ASSERT_EQ(input->shape().size(), 2);
      ASSERT_EQ(input->num(), 4);
      ASSERT_EQ(input->channels(), 2);

      std::vector<IOVal> inputs;
      inputs.push_back(IOVal(0.0, 0.0));
      inputs.push_back(IOVal(1.0, 0.0));
      inputs.push_back(IOVal(0.0, 1.0));
      inputs.push_back(IOVal(1.0, 1.0));

      for(int i = 0; i < inputs.size(); ++i)
      {
        int offset = input->offset(i,0,0,0);
        input->mutable_cpu_data()[offset] = inputs[i].x0;
        offset = input->offset(i,1,0,0);
        input->mutable_cpu_data()[offset] = inputs[i].x1;
      }
    }

    {
      shared_ptr<Blob<Dtype> > output = net.blob_by_name("target_data");
      std::vector<IOVal> xor_outputs;
      std::vector<IOVal> or_outputs;
      std::vector<IOVal> and_outputs;
      xor_outputs.push_back(IOVal(0.0, 1.0));
      xor_outputs.push_back(IOVal(1.0, 0.0));
      xor_outputs.push_back(IOVal(1.0, 0.0));
      xor_outputs.push_back(IOVal(0.0, 1.0));

      or_outputs.push_back(IOVal(0.0, 1.0));
      or_outputs.push_back(IOVal(1.0, 0.0));
      or_outputs.push_back(IOVal(1.0, 0.0));
      or_outputs.push_back(IOVal(1.0, 0.0));

      and_outputs.push_back(IOVal(0.0, 1.0));
      and_outputs.push_back(IOVal(0.0, 1.0));
      and_outputs.push_back(IOVal(0.0, 1.0));
      and_outputs.push_back(IOVal(1.0, 0.0));

      for(int i = 0; i < xor_outputs.size(); ++i)
      {
        // XOR
        int offset = output->offset(i,0,0,0);
        output->mutable_cpu_data()[offset] = xor_outputs[i].x0;
        offset = output->offset(i,1,0,0);
        output->mutable_cpu_data()[offset] = xor_outputs[i].x1;

        //OR
        offset = output->offset(i,2,0,0);
        output->mutable_cpu_data()[offset] = or_outputs[i].x0;
        offset = output->offset(i,3,0,0);
        output->mutable_cpu_data()[offset] = or_outputs[i].x1;

        //AND
        offset = output->offset(i,4,0,0);
        output->mutable_cpu_data()[offset] = and_outputs[i].x0;
        offset = output->offset(i,5,0,0);
        output->mutable_cpu_data()[offset] = and_outputs[i].x1;
      }
    }
  }

  void TrainXOR(bool strict_sparsity)
  {
    std::string proto = CreateXorPrototxt(strict_sparsity);
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    switch (Caffe::mode()) {
      case Caffe::CPU:
        param.set_solver_mode(SolverParameter_SolverMode_CPU);
        break;
      case Caffe::GPU:
        param.set_solver_mode(SolverParameter_SolverMode_GPU);
        param.set_device_id(1);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
    }

    SGDSolver<Dtype> solver(param);
    AssignXorBlobs(*solver.net());
    solver.Solve();
    ASSERT_LT(solver.net()->blob_by_name("loss")->cpu_data()[0], 0.01f);
  }
};

TYPED_TEST_CASE(SoftmaximaLayerTest, TestDtypesAndDevices);

// Forward propagating through the Softmaxima layer should produce the same
// result as forward propagating through two individual Softmax layers whose
// outputs are concatenated appropriately.
TYPED_TEST(SoftmaximaLayerTest, TestForward_NaiveImplementation) {
  typedef typename TypeParam::Dtype Dtype;

  shared_ptr<Blob<Dtype> > expected_result =
            this->ForwardPropThroughTwoSoftmaxes(5, this->blob_bottom_);

  NaiveSoftmaximaLayer<Dtype> layer(5);
  shared_ptr<Blob<Dtype> > result = layer.Forward(*(this->blob_bottom_vec_[0]));

  EXPECT_EQ((expected_result->shape()), (result->shape()));

  // Test that results are the same.
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          Dtype expected_val = expected_result->data_at(i,j,k,l);
          Dtype actual_val = result->data_at(i, j, k, l);
          ASSERT_NEAR( expected_val, actual_val, 1e-3 );
        }
      }
    }
  }
}

// Forward propagating through the Softmaxima layer should produce the same
// result as forward propagating through two individual Softmax layers whose
// outputs are concatenated appropriately.
TYPED_TEST(SoftmaximaLayerTest, TestForward_NaiveImplementationLargeBlob) {
  typedef typename TypeParam::Dtype Dtype;

  shared_ptr<Blob<Dtype> > expected_result =
            this->ForwardPropThroughTwoSoftmaxes(8, this->blob_bottom1_);

  NaiveSoftmaximaLayer<Dtype> layer(8);
  shared_ptr<Blob<Dtype> > result = layer.Forward(*(this->blob_bottom1_));

  EXPECT_EQ((expected_result->shape()), (result->shape()));

  // Test that results are the same.
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          Dtype expected_val = expected_result->data_at(i,j,k,l);
          Dtype actual_val = result->data_at(i, j, k, l);
          ASSERT_NEAR( expected_val, actual_val, 1e-3 );
        }
      }
    }
  }
}

// Forward propagating through the Softmaxima layer should produce the same
// result as forward propagating through two individual Softmax layers whose
// outputs are concatenated appropriately.
TYPED_TEST(SoftmaximaLayerTest, TestForward_NaiveNonGpuImplementation) {
  typedef typename TypeParam::Dtype Dtype;

  NaiveSoftmaximaLayer<Dtype> layer(5);
  shared_ptr<Blob<Dtype> > expected_result =
      layer.Forward(*(this->blob_bottom_vec_[0]));

  NaiveNonGPUSoftmaximaLayer<Dtype> nongpulayer(5);
  shared_ptr<Blob<Dtype> > result =
      nongpulayer.Forward(*(this->blob_bottom_vec_[0]));

  EXPECT_EQ((expected_result->shape()), (result->shape()));

  // Test that results are the same.
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          Dtype expected_val = expected_result->data_at(i,j,k,l);
          Dtype actual_val = result->data_at(i, j, k, l);
          ASSERT_NEAR( expected_val, actual_val, 1e-3 );
        }
      }
    }
  }
}

// Forward propagating through the Softmaxima layer should produce the same
// result as forward propagating through two individual Softmax layers whose
// outputs are concatenated appropriately.
TYPED_TEST(SoftmaximaLayerTest, TestForward_SoftmaximaLayer) {
  typedef typename TypeParam::Dtype Dtype;

  shared_ptr<Blob<Dtype> > expected_result =
            this->ForwardPropThroughTwoSoftmaxes(5, this->blob_bottom_ );

  LayerParameter layer_param;
  std::string proto = "softmaxima_param { softmax_size: 5 }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype>& result = *(this->blob_top_vec_[0]);
  std::vector<int> expected_shape = expected_result->shape();

  EXPECT_EQ(expected_shape, (result.shape()));
  EXPECT_EQ(expected_shape.size(), 4);
  EXPECT_EQ(result.num(), expected_shape[0]);
  EXPECT_EQ(result.channels(), expected_shape[1]);
  EXPECT_EQ(result.height(), expected_shape[2]);
  EXPECT_EQ(result.width(), expected_shape[3]);

  int bottom_blob_num = this->blob_bottom_->num();
  int bottom_blob_channels = this->blob_bottom_->channels();
  int bottom_blob_width = this->blob_bottom_->width();
  int bottom_blob_height = this->blob_bottom_->height();

  EXPECT_EQ(bottom_blob_num, expected_shape[0]);
  EXPECT_EQ(bottom_blob_channels, expected_shape[1]);
  EXPECT_EQ(bottom_blob_height, expected_shape[2]);
  EXPECT_EQ(bottom_blob_width, expected_shape[3]);

  // Test that results are the same.
  for (int i = 0; i < bottom_blob_num; ++i) {
    for (int k = 0; k < bottom_blob_height; ++k) {
      for (int l = 0; l < bottom_blob_width; ++l) {
        for (int j = 0; j < bottom_blob_channels; ++j) {
          Dtype expected_val = expected_result->data_at(i,j,k,l);
          Dtype actual_val = result.data_at(i, j, k, l);
          ASSERT_NEAR( expected_val, actual_val, 1e-3 );
        }
      }
    }
  }
}

//===========================================================================
// This section pertains to testing the behavior of the softmax with
// somewhat ill-conditioned inputs that might tend to produce bad
// softmax outputs (e.g. not in range 0.0->1.0).
//
std::vector<double> softmax_ground_truth(const std::vector<double>& vals)
{
  double max;
  // Get max val.
  {
    bool got_first = false;
    for(int i=0; i < vals.size(); ++i)
    {
      if (!got_first)
      {
        max = vals[i];
        got_first = true;
      }
      else if (vals[i] > max)
      {
        max = vals[i];
      }
    }
  }

  // subtract max.
  std::vector<double> vals_minus_max = vals;
  {
    for(int i=0; i < vals.size(); ++i)
    {
      vals_minus_max[i] = vals[i] - max;
    }
  }

  for(int i=0; i < vals_minus_max.size(); ++i)
  {
    vals_minus_max[i] = exp(vals_minus_max[i]);
  }

  double sum = 0;
  for(int i=0; i < vals_minus_max.size(); ++i)
  {
    sum += vals_minus_max[i];
  }

  std::vector<double> result;
  for(int i=0; i < vals_minus_max.size(); ++i)
  {
    result.push_back(vals_minus_max[i] / sum);
  }

  return result;
}

TYPED_TEST(SoftmaximaLayerTest, TestSoftmaxGroundTruth)
{
  std::vector<double> vals;
  vals.push_back(5.);
  vals.push_back(2.);
  vals.push_back(2.);
  vals.push_back(-5.);
  vals.push_back(1.);
  vals.push_back(4.);
  vals.push_back(4.);
  vals.push_back(-1.);

  std::vector<double> softmaxes = softmax_ground_truth(vals);

  std::vector<double> expected_vals;
  expected_vals.push_back(0.5387429417);
  expected_vals.push_back(0.0268224317);
  expected_vals.push_back(0.0268224317);
  expected_vals.push_back(2.44588917131484E-005);
  expected_vals.push_back(0.0098674212);
  expected_vals.push_back(0.1981924523);
  expected_vals.push_back(0.1981924523);
  expected_vals.push_back(0.0013354102);

  for(int i =0; i < expected_vals.size(); ++i)
  {
    ASSERT_NEAR(softmaxes[i],
                expected_vals[i], 0.001);
  }
}

std::vector<double> softmax_with_normalizer(const std::vector<double>& vals)
{
  double max;
  // Get max val.
  {
    bool got_first = false;
    for(int i=0; i < vals.size(); ++i)
    {
      if (!got_first)
      {
        max = vals[i];
        got_first = true;
      }
      else if (vals[i] > max)
      {
        max = vals[i];
      }
    }
  }

  double sum = 0;
  for(int i=0; i < vals.size(); ++i)
  {
    sum += exp(vals[i] - max);
  }

  double normalizer = log(sum);

  std::vector<double> result;
  for(int i=0; i < vals.size(); ++i)
  {
    double val = vals[i] - max - normalizer;
    result.push_back(exp(val));
  }
  return result;
}

TYPED_TEST(SoftmaximaLayerTest, TestSoftmaxNormalizer)
{
  std::vector<double> vals;
  vals.push_back(5.);
  vals.push_back(2.);
  vals.push_back(2.);
  vals.push_back(-5.);
  vals.push_back(1.);
  vals.push_back(4.);
  vals.push_back(4.);
  vals.push_back(-1.);

  std::vector<double> softmaxes_wn = softmax_with_normalizer(vals);
  std::vector<double> expected_softmaxes = softmax_ground_truth(vals);

  for(int i =0; i < expected_softmaxes.size(); ++i)
  {
    EXPECT_NEAR(softmaxes_wn[i],
                expected_softmaxes[i], 0.001);
  }
}

TYPED_TEST(SoftmaximaLayerTest, TestForward_LargeVals)
{
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype> bottom(1,8,1,1);
  {
    Dtype* data = bottom.mutable_cpu_data();
    data[0] = 9e37;
    data[1] = 5e37;
    data[2] = 5e37;
    data[3] = 5e37;
    data[4] = 5e37;
    data[5] = -5e37;
    data[6] = 1e37;
    data[7] = 1e37;

  }
  vector<Blob<Dtype>*> blob_bottom_vec;
  blob_bottom_vec.push_back(&bottom);

  LayerParameter layer_param;
  std::string proto = "softmaxima_param { softmax_size: 8 }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);
  layer.SetUp(blob_bottom_vec, this->blob_top_vec_);
  layer.Forward(blob_bottom_vec, this->blob_top_vec_);

  Blob<Dtype>& result = *(this->blob_top_vec_[0]);

  EXPECT_EQ(result.num(), 1);
  EXPECT_EQ(result.channels(), 8);
  EXPECT_EQ(result.height(), 1);
  EXPECT_EQ(result.width(), 1);

  int bottom_blob_num = bottom.num();
  int bottom_blob_channels = bottom.channels();
  int bottom_blob_width = bottom.width();
  int bottom_blob_height = bottom.height();

  EXPECT_EQ(bottom_blob_num, 1);
  EXPECT_EQ(bottom_blob_channels, 8);
  EXPECT_EQ(bottom_blob_height, 1);
  EXPECT_EQ(bottom_blob_width, 1);


  std::vector<double> bottom_as_vec;
  {
    for(int chan= 0; chan < bottom.channels(); ++chan)
    {
      bottom_as_vec.push_back(bottom.data_at(0,chan,0,0));
    }
  }
  std::vector<double> expected_result = softmax_ground_truth(bottom_as_vec);

  // Test that results are the same.
  for (int i = 0; i < bottom_blob_num; ++i) {
    for (int k = 0; k < bottom_blob_height; ++k) {
      for (int l = 0; l < bottom_blob_width; ++l) {
        for (int j = 0; j < bottom_blob_channels; ++j) {
          Dtype expected_val = expected_result[j];
          Dtype actual_val = result.data_at(i, j, k, l);
          ASSERT_NEAR( expected_val, actual_val, 1e-3 );
        }
      }
    }
  }
}

TYPED_TEST(SoftmaximaLayerTest, TestForward_SmallVals)
{
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype> bottom(1,8,1,1);
  {
    Dtype* data = bottom.mutable_cpu_data();
    data[0] = 1.0e-37;
    data[1] = -1.0e-37;
    data[2] = 1.0e-37;
    data[3] = -1.0e-37;
    data[4] = 1.0e-37;
    data[5] = -1.0e-37;
    data[6] = 1.0e-37;
    data[7] = -1.0e-37;

  }
  vector<Blob<Dtype>*> blob_bottom_vec;
  blob_bottom_vec.push_back(&bottom);

  LayerParameter layer_param;
  std::string proto = "softmaxima_param { softmax_size: 8 }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);
  layer.SetUp(blob_bottom_vec, this->blob_top_vec_);
  layer.Forward(blob_bottom_vec, this->blob_top_vec_);

  Blob<Dtype>& result = *(this->blob_top_vec_[0]);

  EXPECT_EQ(result.num(), 1);
  EXPECT_EQ(result.channels(), 8);
  EXPECT_EQ(result.height(), 1);
  EXPECT_EQ(result.width(), 1);

  int bottom_blob_num = bottom.num();
  int bottom_blob_channels = bottom.channels();
  int bottom_blob_width = bottom.width();
  int bottom_blob_height = bottom.height();

  EXPECT_EQ(bottom_blob_num, 1);
  EXPECT_EQ(bottom_blob_channels, 8);
  EXPECT_EQ(bottom_blob_height, 1);
  EXPECT_EQ(bottom_blob_width, 1);


  std::vector<double> bottom_as_vec;
  {
    for(int chan= 0; chan < bottom.channels(); ++chan)
    {
      bottom_as_vec.push_back(bottom.data_at(0,chan,0,0));
    }
  }
  std::vector<double> expected_result = softmax_ground_truth(bottom_as_vec);

  // Test that results are the same.
  for (int i = 0; i < bottom_blob_num; ++i) {
    for (int k = 0; k < bottom_blob_height; ++k) {
      for (int l = 0; l < bottom_blob_width; ++l) {
        for (int j = 0; j < bottom_blob_channels; ++j) {
          Dtype expected_val = expected_result[j];
          Dtype actual_val = result.data_at(i, j, k, l);
          ASSERT_NEAR( expected_val, actual_val, 1e-3 );
        }
      }
    }
  }
}

template<typename Dtype>
bool CheckForOutOfRange(int size, const Dtype* ptr, Dtype& val)
{
  for(int index = 0; index < size; ++index)
  {
    if (ptr[index] > static_cast<Dtype>(1))
    {
      val = ptr[index];
      return true;
    }
    else if (ptr[index] < static_cast<Dtype>(0))
    {
      val = ptr[index];
      return true;
    }
  }
  return false;
}

TYPED_TEST(SoftmaximaLayerTest, TestCheckOutOfRange)
{
  typedef typename TypeParam::Dtype Dtype;
  Blob<Dtype> blob(1,3,1,1);
  Dtype* data = blob.mutable_cpu_data();
  data[0] = 0.1;
  data[1] = 0.0;
  data[2] = 2.0;

  Dtype test_val;
  bool result = CheckForOutOfRange(blob.count(), blob.cpu_data(), test_val);
  ASSERT_EQ(result, true);
  ASSERT_NEAR(test_val, static_cast<Dtype>(2.0), static_cast<Dtype>(0.001));
}

TYPED_TEST(SoftmaximaLayerTest, TestNotCheckOutOfRange)
{
  typedef typename TypeParam::Dtype Dtype;
  Blob<Dtype> blob(1,3,1,1);
  Dtype* data = blob.mutable_cpu_data();
  data[0] = 0.1;
  data[1] = 1.0;
  data[2] = 0.5;

  Dtype test_val;
  bool result = CheckForOutOfRange(blob.count(), blob.cpu_data(), test_val);
  ASSERT_EQ(result, false);
}

//===========================================================================
// End section.
//===========================================================================

TYPED_TEST(SoftmaximaLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  std::string proto = "softmaxima_param { softmax_size: 5 }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

template<typename Dtype>
void CompareBlobs(const std::string& msg, Blob<Dtype>& b1, Blob<Dtype>& b2)
{
  ASSERT_EQ(b1.num(), b2.num());
  ASSERT_EQ(b1.channels(), b2.channels());
  ASSERT_EQ(b1.height(), b2.height());
  ASSERT_EQ(b1.width(), b2.width());

  for(int n = 0; n < b1.num(); ++n) {
    for(int c = 0; c < b1.channels(); ++c) {
      for(int h = 0; h < b1.height(); ++h) {
        for(int w = 0; w < b1.width(); ++w ) {
          ASSERT_NEAR(b1.data_at(n,c,h,w), b2.data_at(n,c,h,w), 0.0001);
        }
      }
    }
  }
}

TYPED_TEST(SoftmaximaLayerTest, TestForwardIdenticalToNaive) {
//  vector<Blob<Dtype>*> blob_bottom_cpugpu_vec_;
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  std::string proto = "softmaxima_param { softmax_size: 8 }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);

  // Forward prop once.
  {
    vector<Blob<Dtype>*> bottom_blob_vec;
    bottom_blob_vec.push_back(this->blob_bottom1_.get());

    Blob<Dtype> top_blob;
    vector<Blob<Dtype>*> top_blob_vec;
    top_blob_vec.push_back(&top_blob);
    layer.SetUp(bottom_blob_vec, top_blob_vec);
    layer.Forward(bottom_blob_vec, top_blob_vec);
    Blob<Dtype>& result = *(top_blob_vec[0]);

    NaiveSoftmaximaLayer<Dtype> naivelayer(8);
    shared_ptr<Blob<Dtype> > expected_result = naivelayer.Forward(*(bottom_blob_vec[0]));

//    CompareBlobs("maxes",
//                 naivelayer.GetMaxes(),
//                 layer.GetMaxes());
//    CompareBlobs("bottom_minus_maxes",
//                 naivelayer.GetBotMinusMaxes(),
//                 layer.GetBotMinusMaxes());
//    CompareBlobs("bottom_exponentiated",
//                 naivelayer.GetBotExponentiated(),
//                 layer.GetBotExponentiated());
//    CompareBlobs("denom_sums",
//                 naivelayer.GetDenomSums(),
//                 layer.GetDenomSums());

    // Test that results are the same.
    for (int i = 0; i < this->blob_bottom1_->num(); ++i) {
      for (int k = 0; k < this->blob_bottom1_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom1_->width(); ++l) {
          for (int j = 0; j < this->blob_bottom1_->channels(); ++j) {
            Dtype expected_val = expected_result->data_at(i,j,k,l);
            Dtype actual_val = result.data_at(i, j, k, l);
            ASSERT_NEAR( expected_val, actual_val, 1e-4 );
          }
        }
      }
    }
  }

  // Forward prop again.
  {
    vector<Blob<Dtype>*> bottom_blob_vec;
    bottom_blob_vec.push_back(this->blob_bottom2_.get());

    Blob<Dtype> top_blob;
    vector<Blob<Dtype>*> top_blob_vec;
    top_blob_vec.push_back(&top_blob);
    layer.Forward(bottom_blob_vec, top_blob_vec);
    Blob<Dtype>& result = *(top_blob_vec[0]);

    NaiveSoftmaximaLayer<Dtype> naivelayer(8);
    shared_ptr<Blob<Dtype> > expected_result = naivelayer.Forward(*(bottom_blob_vec[0]));

    // Test that results are the same.
    for (int i = 0; i < this->blob_bottom1_->num(); ++i) {
      for (int k = 0; k < this->blob_bottom1_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom1_->width(); ++l) {
          for (int j = 0; j < this->blob_bottom1_->channels(); ++j) {
            Dtype expected_val = expected_result->data_at(i,j,k,l);
            Dtype actual_val = result.data_at(i, j, k, l);
            ASSERT_NEAR( expected_val, actual_val, 1e-4 );
          }
        }
      }
    }
  }
}

TYPED_TEST(SoftmaximaLayerTest, TestForwardNonGpuIdenticalToNaive) {
//  vector<Blob<Dtype>*> blob_bottom_cpugpu_vec_;
  typedef typename TypeParam::Dtype Dtype;

//  LayerParameter layer_param;
//  std::string proto = "softmaxima_param { softmax_size: 8 }";
//  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
//  SoftmaximaLayer<Dtype> layer(layer_param);
  NaiveNonGPUSoftmaximaLayer<Dtype> nglayer(8);

  // Forward prop once.
  {
    shared_ptr<Blob<Dtype> > result_ptr = nglayer.Forward(*this->blob_bottom1_);
    Blob<Dtype>& result = *result_ptr;

    NaiveSoftmaximaLayer<Dtype> layer(8);
    shared_ptr<Blob<Dtype> > expected_result = layer.Forward(*this->blob_bottom1_);

//    CompareBlobs("maxes",
//                 nglayer.GetMaxes(),
//                 layer.GetMaxes());
//    CompareBlobs("bottom_minus_maxes",
//                 nglayer.GetBotMinusMaxes(),
//                 layer.GetBotMinusMaxes());
//    CompareBlobs("bottom_exponentiated",
//                 nglayer.GetBotExponentiated(),
//                 layer.GetBotExponentiated());
//    CompareBlobs("denom_sums",
//                 nglayer.GetDenomSums(),
//                 layer.GetDenomSums());

    int nan_ctr = 0;
    int non_nan_ctr = 0;
    // Test that results are the same.
    for (int i = 0; i < this->blob_bottom1_->num(); ++i) {
      for (int k = 0; k < this->blob_bottom1_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom1_->width(); ++l) {
          for (int j = 0; j < this->blob_bottom1_->channels(); ++j) {
            Dtype expected_val = expected_result->data_at(i,j,k,l);
            Dtype actual_val = result.data_at(i, j, k, l);
            EXPECT_NEAR( expected_val, actual_val, 1e-4 );
            if ( std::isnan(actual_val) )
            {
              nan_ctr++;
            }
            else
            {
              non_nan_ctr++;
            }
          }
        }
      }
    }
    EXPECT_EQ(nan_ctr,0);
    EXPECT_GT(non_nan_ctr, 0);
  }

  // Forward prop again.
  {
    NaiveSoftmaximaLayer<Dtype> layer(8);
    shared_ptr<Blob<Dtype> > expected_result = layer.Forward(*this->blob_bottom2_);

    shared_ptr<Blob<Dtype> > result_ptr = nglayer.Forward(*this->blob_bottom2_);
    Blob<Dtype>& result = *result_ptr;

    // Test that results are the same.
    for (int i = 0; i < this->blob_bottom1_->num(); ++i) {
      for (int k = 0; k < this->blob_bottom1_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom1_->width(); ++l) {
          for (int j = 0; j < this->blob_bottom1_->channels(); ++j) {
            Dtype expected_val = expected_result->data_at(i,j,k,l);
            Dtype actual_val = result.data_at(i, j, k, l);
            ASSERT_NEAR( expected_val, actual_val, 1e-4 );
          }
        }
      }
    }
  }
}

TYPED_TEST(SoftmaximaLayerTest, TestNaiveNonGpu_WinnerTakeAllForward) {
  typedef typename TypeParam::Dtype Dtype;
  const int SOFTMAX_SIZE = 8;
//  LayerParameter layer_param;
//  std::string proto = "softmaxima_param { softmax_size: 8 winner_take_all: true }";
//  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
//  SoftmaximaLayer<Dtype> layer(layer_param);

  NaiveNonGPUSoftmaximaLayer<Dtype> nglayer(8);
  nglayer.SetWinnerTakeAll();
  // Forward prop once.
  shared_ptr<Blob<Dtype> > result_ptr = nglayer.Forward(*this->blob_bottom1_);
  Blob<Dtype>& result = *result_ptr;

//  vector<Blob<Dtype>*> bottom_blob_vec;
//  bottom_blob_vec.push_back(this->blob_bottom1_);

//  layer.SetUp(bottom_blob_vec, this->blob_top_vec_);
//  layer.Forward(bottom_blob_vec, this->blob_top_vec_);
//  Blob<Dtype>& result = *(this->blob_top_vec_[0]);

  NaiveSoftmaximaLayer<Dtype> naivelayer(SOFTMAX_SIZE);
  shared_ptr<Blob<Dtype> > expected_probs = naivelayer.Forward(*this->blob_bottom1_);

  int num_channels = this->blob_bottom1_->channels();
  int num_softmaxes = num_channels / SOFTMAX_SIZE;

  // Test that results are the same.
  for (int n = 0; n < this->blob_bottom1_->num(); ++n) {
    for (int h = 0; h < this->blob_bottom1_->height(); ++h) {
      for (int w = 0; w < this->blob_bottom1_->width(); ++w) {
//  int n=1; {
//    int h = 1; {
//      int w = 1; {
        for(int sm_index = 0; sm_index < num_softmaxes; ++sm_index )
        {
          Dtype largest_prob = -1.0;
          int largest_prob_channel = -1;

          for(int smi = 0; smi < SOFTMAX_SIZE; ++smi)
          {
            int chan = sm_index*SOFTMAX_SIZE + smi;
            Dtype val = expected_probs->data_at(n,chan,h,w);
            if (val > largest_prob)
            {
              largest_prob = val;
              largest_prob_channel = chan;
            }
          }

          // The one with the highest prob should be 1; others all zero.
          for(int smi = 0; smi < SOFTMAX_SIZE; ++smi)
          {
            int chan = sm_index*SOFTMAX_SIZE + smi;
            Dtype expected_val = (chan == largest_prob_channel) ? 1 : 0;
            Dtype actual_val = result.data_at(n,chan,h,w);
//            std::cout << "chan=" << chan << ", expected_val=" << expected_val <<
//                         ", actual_val==" << actual_val
//                      << ", prob=" << expected_probs->data_at(n,chan,h,w) << std::endl;
            ASSERT_NEAR( expected_val, actual_val, 0.00001);
          }
        }
      }
    }
  }
}

TYPED_TEST(SoftmaximaLayerTest, TestWinnerTakeAllForward) {
  typedef typename TypeParam::Dtype Dtype;
  const int SOFTMAX_SIZE = 8;
  LayerParameter layer_param;
  std::string proto =
      "softmaxima_param { softmax_size: 8  mode: WINNER_TAKE_ALL }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);

  vector<Blob<Dtype>*> bottom_blob_vec;
  bottom_blob_vec.push_back(this->blob_bottom1_.get());

  vector<Blob<Dtype>*> top_blob_vec;
  top_blob_vec.push_back(this->blob_top_.get());

  layer.SetUp(bottom_blob_vec, top_blob_vec);
  layer.Forward(bottom_blob_vec, top_blob_vec);

  NaiveSoftmaximaLayer<Dtype> naivelayer(SOFTMAX_SIZE);
  shared_ptr<Blob<Dtype> > expected_probs = naivelayer.Forward(*this->blob_bottom1_);

  int num_channels = this->blob_bottom1_->channels();
  int num_softmaxes = num_channels / SOFTMAX_SIZE;

  // Test that results are the same.
  for (int n = 0; n < this->blob_bottom1_->num(); ++n) {
    for (int h = 0; h < this->blob_bottom1_->height(); ++h) {
      for (int w = 0; w < this->blob_bottom1_->width(); ++w) {
        for(int sm_index = 0; sm_index < num_softmaxes; ++sm_index )
        {
          Dtype largest_prob = -1.0;
          int largest_prob_channel = -1;

          for(int smi = 0; smi < SOFTMAX_SIZE; ++smi)
          {
            int chan = sm_index*SOFTMAX_SIZE + smi;
            Dtype val = expected_probs->data_at(n,chan,h,w);
            if (val > largest_prob)
            {
              largest_prob = val;
              largest_prob_channel = chan;
            }
          }

          // The one with the highest prob should be 1; others all zero.
          for(int smi = 0; smi < SOFTMAX_SIZE; ++smi)
          {
            int chan = sm_index*SOFTMAX_SIZE + smi;
            Dtype expected_val = (chan == largest_prob_channel) ? 1 : 0;
            Dtype actual_val = top_blob_vec[0]->data_at(n,chan,h,w);
            ASSERT_NEAR( expected_val, actual_val, 0.00001);
          }
        }
      }
    }
  }
}


TYPED_TEST(SoftmaximaLayerTest, TestForwardNonNan) {
//  vector<Blob<Dtype>*> blob_bottom_cpugpu_vec_;
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  std::string proto = "softmaxima_param { softmax_size: 8 }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);

  // Forward prop once.
  {
    vector<Blob<Dtype>*> bottom_blob_vec;
    bottom_blob_vec.push_back(this->blob_bottom1_.get());

    Blob<Dtype> top_blob;
    vector<Blob<Dtype>*> top_blob_vec;
    top_blob_vec.push_back(&top_blob);
    layer.SetUp(bottom_blob_vec, top_blob_vec);
    layer.Forward(bottom_blob_vec, top_blob_vec);
    Blob<Dtype>& result = *(top_blob_vec[0]);

    // Test that results are the same.
    for (int i = 0; i < this->blob_bottom1_->num(); ++i) {
      for (int k = 0; k < this->blob_bottom1_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom1_->width(); ++l) {
          for (int j = 0; j < this->blob_bottom1_->channels(); ++j) {
            Dtype actual_val = result.data_at(i, j, k, l);
            ASSERT_FALSE( std::isnan(actual_val) );
          }
        }
      }
    }
  }

  // Forward prop again.
  {
    vector<Blob<Dtype>*> bottom_blob_vec;
    bottom_blob_vec.push_back(this->blob_bottom2_.get());

    Blob<Dtype> top_blob;
    vector<Blob<Dtype>*> top_blob_vec;
    top_blob_vec.push_back(&top_blob);
    layer.Forward(bottom_blob_vec, top_blob_vec);
    Blob<Dtype>& result = *(top_blob_vec[0]);

    int nan_ctr = 0;
    int nonnan_ctr = 0;
    // Test that results are the same.
    for (int i = 0; i < this->blob_bottom1_->num(); ++i) {
      for (int k = 0; k < this->blob_bottom1_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom1_->width(); ++l) {
          for (int j = 0; j < this->blob_bottom1_->channels(); ++j) {
            Dtype actual_val = result.data_at(i, j, k, l);
            EXPECT_FALSE( std::isnan(actual_val) );
            if( std::isnan(actual_val)) {
              nan_ctr++;
            }
            else
            {
              nonnan_ctr++;
            }
          }
        }
      }
    }
    EXPECT_EQ(nan_ctr, 0);
    EXPECT_GT(nonnan_ctr, 0);
  }
}

TYPED_TEST(SoftmaximaLayerTest, TestNonGpuForwardNonNan) {
  typedef typename TypeParam::Dtype Dtype;

  NaiveNonGPUSoftmaximaLayer<Dtype> nongpulayer(8);

  // Forward prop once.
  {
    shared_ptr<Blob<Dtype> > result_ptr = nongpulayer.Forward(*this->blob_bottom1_);
    Blob<Dtype>& result = *result_ptr;

    // Test that results are the same.
    for (int i = 0; i < this->blob_bottom1_->num(); ++i) {
      for (int k = 0; k < this->blob_bottom1_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom1_->width(); ++l) {
          for (int j = 0; j < this->blob_bottom1_->channels(); ++j) {
            Dtype actual_val = result.data_at(i, j, k, l);
            ASSERT_FALSE( std::isnan(actual_val) );
          }
        }
      }
    }
  }

  // Forward prop again.
  {
    shared_ptr<Blob<Dtype> > result_ptr = nongpulayer.Forward(*this->blob_bottom2_);
    Blob<Dtype>& result = *result_ptr;

    int nan_ctr = 0;
    int nonnan_ctr = 0;
    // Test that results are the same.
    for (int i = 0; i < this->blob_bottom1_->num(); ++i) {
      for (int k = 0; k < this->blob_bottom1_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom1_->width(); ++l) {
          for (int j = 0; j < this->blob_bottom1_->channels(); ++j) {
            Dtype actual_val = result.data_at(i, j, k, l);
            if( std::isnan(actual_val)) {
              nan_ctr++;
            }
            else
            {
              nonnan_ctr++;
            }
          }
        }
      }
    }
    EXPECT_EQ(nan_ctr, 0);
    EXPECT_GT(nonnan_ctr, 0);
  }
}

// Assign some arbitrary diffs to a blob.
template <typename Dtype>
void SetTopBlobDiff(Blob<Dtype>& blob)
{
  int count = blob.count();
  Dtype val = -1.0;
  Dtype delta = 2.0/count;
  Dtype* buffer = blob.mutable_cpu_diff();
  for(int n = 0; n < blob.num(); ++n) {
    for(int c = 0; c < blob.channels(); ++c ) {
      for(int h=0; h < blob.height(); ++h ) {
        for(int w=0; w < blob.width(); ++w) {
          buffer[blob.offset(n,c,h,w)] = val;
          val+=delta;
        }
      }
    }
  }
}

template <typename Dtype>
void AssertBlobDiffsEqual(const Blob<Dtype>& b1, const Blob<Dtype>& b2) {
  ASSERT_EQ( b1.count(), b2.count());
  int count = b1.count();
  for(int index = 0; index < count; ++index) {
    Dtype val1 = b1.cpu_diff()[index];
    Dtype val2 = b2.cpu_diff()[index];
    if (val1 != 0.0 || val2 != 0.0 )
    {
      ASSERT_NEAR(val1, val2, 0.00001);
    } else {
      ASSERT_NEAR(val1, val2, 0.00001);
    }
  }
}

// When we binarize the output in forward propagation, we want the backward
// to behave the same as when we're not binarized.
TYPED_TEST(SoftmaximaLayerTest, TestBackward_WinnerTakeAllSameAsNot) {
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype> binarized_bottom_blob;

  //Forward and backward prop through the binarizing layer.
  {
    Blob<Dtype> binarized_top_blob;
    binarized_top_blob.ReshapeLike(*this->blob_top_);
    binarized_bottom_blob.ReshapeLike(*this->blob_bottom1_);
    binarized_bottom_blob.CopyFrom(*this->blob_bottom1_);

    LayerParameter layer_param;
    std::string proto = "softmaxima_param { softmax_size: 8 mode: WINNER_TAKE_ALL }";
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
    SoftmaximaLayer<Dtype> layer(layer_param);

    vector<Blob<Dtype>*> bottom_blob_vec;
    bottom_blob_vec.push_back(&binarized_bottom_blob);

    vector<Blob<Dtype>*> top_blob_vec;
    top_blob_vec.push_back(&binarized_top_blob);

    layer.SetUp(bottom_blob_vec, top_blob_vec);
    layer.Forward(bottom_blob_vec, top_blob_vec);

    //Set the top blob's diff.
    SetTopBlobDiff(binarized_top_blob);

    // prop_down is ignored by SoftmaximaLayer::Backward*. So just pass
    // empty to pacify the compiler.
    std::vector<bool> prop_down;
//    PrintBlob("top diffs", *top_blob_vec[0], true);

    layer.Backward(top_blob_vec, prop_down, bottom_blob_vec);
  }

  Blob<Dtype> nonbinarized_bottom_blob;

  // Forward and backward prop through the non-binarizing layer.
  {
    Blob<Dtype> nonbinarized_top_blob;
    nonbinarized_top_blob.ReshapeLike(*this->blob_top_);
    nonbinarized_bottom_blob.ReshapeLike(*this->blob_bottom1_);
    nonbinarized_bottom_blob.CopyFrom(*this->blob_bottom1_);

    LayerParameter layer_param;
    std::string proto = "softmaxima_param { softmax_size: 8 }";
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
    SoftmaximaLayer<Dtype> layer(layer_param);

    vector<Blob<Dtype>*> bottom_blob_vec;
    bottom_blob_vec.push_back(&nonbinarized_bottom_blob);

    vector<Blob<Dtype>*> top_blob_vec;
    top_blob_vec.push_back(&nonbinarized_top_blob);

    layer.SetUp(bottom_blob_vec, top_blob_vec);
    layer.Forward(bottom_blob_vec, top_blob_vec);

    //Set the top blob's diff.
    SetTopBlobDiff(nonbinarized_top_blob);

    // prop_down is ignored by SoftmaximaLayer::Backward*. So just pass
    // empty to pacify the compiler.
    std::vector<bool> prop_down;
    layer.Backward(top_blob_vec, prop_down, bottom_blob_vec);
  }

  // Diffs of binarized and nonbinarized softmaxima layers should be the same.
  AssertBlobDiffsEqual(nonbinarized_bottom_blob, binarized_bottom_blob);
}

TYPED_TEST(SoftmaximaLayerTest, TestXorTraining)
{
  Caffe::set_random_seed(2101150);
  this->TrainXOR(false);
}

TYPED_TEST(SoftmaximaLayerTest, TestXorTraining_StrictSparsity)
{
  Caffe::set_random_seed(2101151);
  this->TrainXOR(true);
}


// Test out our idea for sampling from the softmaxima using a
// normally-distributed random values: one random value per softmax.
std::vector<float> NaivelySample(const std::vector<float> probs,
                                 int softmax_size)
{
  int index = 0;
  int softmax_count = probs.size() / softmax_size;

  // We need as many uniform distribution samples as softmaxes.
  std::vector<float> distrib_buffer(softmax_count);
  caffe_rng_uniform<float>(softmax_count, 0.0f, 1.0f, &distrib_buffer[0]);
  std::vector<float> result;

  const float TOLERANCE = 0.0001f;
  for(int smi = 0; smi < softmax_count; ++smi)
  {
    float bottom = 0.0f;
    for(int j=0; j < softmax_size; ++j)
    {
      float top = bottom + probs[index++];
      float uniform_val = distrib_buffer[smi];
      if (uniform_val >= bottom && uniform_val < top)
      {
        result.push_back(1.0f);
      }
      else
      {
        result.push_back(0.0f);
      }
      bottom = top;
    }
    if (std::fabs(bottom - 1.0f)  > TOLERANCE)
    {
      throw std::runtime_error("Softmax probabilties didn't "
                                            "add up to 1.");
    }
  }
  if (result.size() != probs.size())
  {
    throw std::runtime_error("Result doesn't have same "
             "size as probs.");
  }
  return result;
}


TYPED_TEST(SoftmaximaLayerTest, NaiveSoftmaxSampler)
{
  typedef typename TypeParam::Dtype Dtype;
  std::vector<float> probvec;
  const int SOFTMAX_SIZE = 4;
  const int CHANNELS = 8;
  // First softmaxima probs.
  probvec.push_back(0.1);
  probvec.push_back(0.7);
  probvec.push_back(0.0);
  probvec.push_back(0.2);
  // Second softmaxima probs.
  probvec.push_back(0.0);
  probvec.push_back(0.3);
  probvec.push_back(0.5);
  probvec.push_back(0.2);
  std::vector<int> val_counts(probvec.size(), 0);

  ASSERT_EQ(probvec.size(), CHANNELS)
      << "Sanity check number of probabilities.";

  const int ITERATIONS = 1000;
  for(int i = 0; i < ITERATIONS; ++i)
  {
    std::vector<float> sampled_vals =
        NaivelySample(probvec, SOFTMAX_SIZE);

    ASSERT_EQ(sampled_vals.size(), probvec.size()) << "Wrong number of elements"
                      << " in sampled values.";

    int one_ctr = 0;
    for(int chan = 0; chan < probvec.size(); ++chan)
    {
      Dtype val = sampled_vals[chan];
      bool is_zero = (std::fabs(val) < 0.01f);
      bool is_one = (std::fabs(val - 1.0f) < 0.01f);
      ASSERT_TRUE(is_zero || is_one) << "Every sampled val should be 1 or zero.";
      ASSERT_GE(val, 0) << "Every sampled val should be between 1 and zero"
                           << " inclusive";
      ASSERT_LE(val, 1) << "Every sampled val should be between 1 and zero"
                        << " inclusive";
      if (is_one)
      {
        one_ctr++;
        val_counts[chan]++;
      }
    }
    ASSERT_EQ(one_ctr, CHANNELS / SOFTMAX_SIZE) << "Each softmax should have "
              << "exactly one non-zero activation.";
  }

  const float TOLERANCE = 0.05;
  // Each channel should have been nearly the same as it's probability of
  // activation times number of iterations.
  for(int chan = 0; chan < probvec.size(); ++chan)
  {
    float ratio = val_counts[chan] / static_cast<float>(ITERATIONS);
    ASSERT_NEAR(ratio, probvec[chan], TOLERANCE) <<
      "Wrong probability in sampled values.";
  }
}

TYPED_TEST(SoftmaximaLayerTest, ForwardWithStrictSparsity)
{
  Caffe::set_random_seed(2101150);
  typedef typename TypeParam::Dtype Dtype;
  std::vector<Dtype> probvec;
  // First softmaxima probs.
  probvec.push_back(std::log(0.1));
  probvec.push_back(std::log(0.7));
  probvec.push_back(std::log(0.0));
  probvec.push_back(std::log(0.2));
  // Second softmaxima probs.
  probvec.push_back(std::log(0.0));
  probvec.push_back(std::log(0.3));
  probvec.push_back(std::log(0.5));
  probvec.push_back(std::log(0.2));
  // Third softmaxima probs.
  probvec.push_back(std::log(0.0));
  probvec.push_back(std::log(0.1));
  probvec.push_back(std::log(0.0));
  probvec.push_back(std::log(0.9));

  const int BATCH_SIZE = 1000;
  const int CHANNELS = 12;
  const int HEIGHT = 4;
  const int WIDTH = 5;
  const int SOFTMAX_SIZE = 4;

  // Scenario: softmaxima outputting batch size 1000, 8 channels, 4x4 hxw.
  Blob<Dtype> bottom(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH);
  Blob<Dtype> top;

  Dtype* bottom_ptr = bottom.mutable_cpu_data();
  for(int num = 0; num < BATCH_SIZE; ++num)
  {
    for(int h = 0; h < HEIGHT; ++h)
    {
      for(int w = 0; w < WIDTH; ++w)
      {
        for(int chan = 0; chan < probvec.size(); ++chan)
        {
          // Use the same vector of log probs for every item in batch and
          // spatial location.
          bottom_ptr[bottom.offset(num,chan,h,w)] = probvec[chan];
        }
      }
    }
  }

  vector<Blob<Dtype>*> bottom_vec;
  bottom_vec.push_back(&bottom);
  vector<Blob<Dtype>*> top_vec;
  top_vec.push_back(&top);

  // Forward prop through the layer.
  LayerParameter layer_param;
  std::string proto =
      "softmaxima_param { softmax_size: 4 mode: STRICT_SPARSITY }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);
  layer.SetUp(bottom_vec, top_vec);
  layer.Forward(bottom_vec, top_vec);
  Blob<Dtype>& result = *(top_vec[0]);


  int one_ctr = 0;
  std::vector<Dtype> val_counts(CHANNELS, 0);
  const Dtype* result_ptr = result.cpu_data();
  for(int num = 0; num < BATCH_SIZE; ++num)
  {
    for(int h = 0; h < HEIGHT; ++h)
    {
      for(int w = 0; w < WIDTH; ++w)
      {
        for(int chan = 0; chan < CHANNELS; ++chan)
        {
          Dtype val = result_ptr[result.offset(num,chan,h,w)];
          bool is_zero = (std::fabs(val) < 0.01f);
          bool is_one = (std::fabs(val - 1.0f) < 0.01f);
          ASSERT_TRUE(is_zero || is_one);
          ASSERT_GE(val, 0);
          ASSERT_LE(val, 1);
          if (is_one)
          {
            one_ctr++;
            val_counts[chan]++;
          }
        }
      }
    }
  }

  // There should exactly one "on" activation for each softmax.
  int top_count = top.count();
  const int num_softmaxes = top_count /SOFTMAX_SIZE;
  int outer_dim = BATCH_SIZE*HEIGHT*WIDTH;
  ASSERT_EQ(one_ctr, num_softmaxes);
  ASSERT_EQ(CHANNELS, probvec.size());
  for(int chan = 0; chan < CHANNELS; ++chan)
  {
    float ratio = val_counts[chan]/ static_cast<float>(outer_dim);
    const float TOLERANCE = 0.05f;
    ASSERT_NEAR(ratio, static_cast<float>(std::exp(probvec[chan])), TOLERANCE);
  }
}

}  // namespace caffe

