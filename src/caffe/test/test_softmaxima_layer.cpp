#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename Dtype>
void PrintBlob( const std::string& nam, Blob<Dtype>& blob )
{
  std::cout << "Blob: " << nam << std::endl;
  int num = blob.num();
  int chans = blob.channels();
  int height = blob.height();
  int width = blob.width();

  std::cout << "shape=(" << num << "," << chans << "," << height << ","
            << width << ")" << std::endl;
  for( int n = 0; n < num; ++n) {
    for( int c = 0; c < chans; ++c) {
      for( int h = 0; h < height; ++h) {
        for( int w=0; w < width; ++w) {
          Dtype val = blob.data_at(n,c,h,w);
          std::cout << "data(" << n << "," << c << "," << h << "," << w << ")"
                       << "=" << val << std::endl;
        }
      }
    }
  }
}

template <typename Dtype>
struct NaiveSoftmaximaLayer {
  NaiveSoftmaximaLayer( int softmax_size ) :
  softmax_size_(softmax_size) {
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

private:
  int softmax_size_;
};

template <typename Dtype>
struct NaiveNonGPUSoftmaximaLayer {
  NaiveNonGPUSoftmaximaLayer(int softmax_size) :
    softmax_size_(softmax_size) {}

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

    // subtract
    // NOLINT_NEXT_LINE(whitespace/operators)
//    kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
//        CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
//        scale_data, top_data);
      kernel_channel_subtract(count,
                             softmax_size_,
                             inner_num_,
                             scale_data,
                             top_data);

      //Print out the blob after maxes have been subtracted.
//      PrintBlob("NonGPU top after subtract", *result);


    // exponentiate
    // NOLINT_NEXT_LINE(whitespace/operators)
//    kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
//        count, top_data, top_data);
    kernel_exp(count, top_data, top_data);

    // sum after exp
    // NOLINT_NEXT_LINE(whitespace/operators)
//    kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
//        CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
//        scale_data);
    kernel_channel_sum(outer_num_,
                       channels,
                       inner_num_,
                       softmax_size_,
                       num_softmaxes_,
                       top_data,
                       scale_data);

//    PrintBlob("NonGPU softmax sums after exp", scale);

    // divide
    // NOLINT_NEXT_LINE(whitespace/operators)
//    kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
//        CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
//        scale_data, top_data);
    kernel_channel_div(outer_num_,
                       channels,
                       inner_num_,
                       softmax_size_,
                       num_softmaxes_,
                       scale_data,
                       top_data);

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
        Dtype maxval = -1000.0;
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
                           Dtype* out) {
    for( int index = 0; index < num * spatial_dim; ++index)  {
      int n = index / spatial_dim;
      int s = index % spatial_dim;
      // For each softmax along the canonical axis.
      for( int smi = 0; smi < num_softmaxes; ++smi) {
        int sum_index = s + (n * num_softmaxes + smi) * spatial_dim ; //index*num_softmaxes + smi;
        Dtype sum = sums[sum_index];
        // For each channel within this softmax.
        for (int c_off = 0; c_off < softmax_size; ++c_off) {
          int c = smi * softmax_size + c_off;
          int data_index = (n * channels + c) * spatial_dim + s;
          out[data_index] = out[data_index] / sum;
        }
        //int out_index = index*num_softmaxes + smi;
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
};

template <typename TypeParam>
class SoftmaximaLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaximaLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SoftmaximaLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  shared_ptr<Blob<Dtype> > ForwardPropThroughTwoSoftmaxes() {
    std::string proto =
        "name: 'TestNetwork' "
        "input: 'data' "
        "input_dim: 2 "
        "input_dim: 10 "
        "input_dim: 2 "
        "input_dim: 3 "
        "layer { "
        "  name: 'slice' "
        "  type: 'Slice' "
        "  bottom: 'data' "
        "  top: 'slice1' "
        "  top: 'slice2' "
        "  slice_param {"
        "     slice_point: 5"
        "  }"
        "} "
        "layer { "
        " name: 'softmax1' "
        " type: 'Softmax' "
        " bottom: 'slice1' "
        " top: 'sm1' "
        "} "
        "layer { "
        " name: 'softmax2' "
        " type: 'Softmax' "
        " bottom: 'slice2' "
        " top: 'sm2' "
        "} "
        "layer { "
        " name: 'concat' "
        " type: 'Concat' "
        " bottom: 'sm1' "
        " bottom: 'sm2' "
        " top: 'result' "
        "} ";

    vector<Blob<Dtype>*> bottom_blob_vec;
    bottom_blob_vec.push_back(this->blob_bottom_);
    NetParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    Net<Dtype> net( param );
    net.Forward(blob_bottom_vec_);

    EXPECT_TRUE(net.has_blob("data"));
    EXPECT_TRUE(net.has_blob("result"));
    shared_ptr<Blob<Dtype> > result_blob;
    result_blob = net.blob_by_name("result");
    return result_blob;
  }
};

TYPED_TEST_CASE(SoftmaximaLayerTest, TestDtypesAndDevices);

// Forward propagating through the Softmaxima layer should produce the same
// result as forward propagating through two individual Softmax layers whose
// outputs are concatenated appropriately.
TYPED_TEST(SoftmaximaLayerTest, TestForward_NaiveImplementation) {
  typedef typename TypeParam::Dtype Dtype;

  shared_ptr<Blob<Dtype> > expected_result =
            this->ForwardPropThroughTwoSoftmaxes();

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
          EXPECT_NEAR( expected_val, actual_val, 1e-4 );
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
  shared_ptr<Blob<Dtype> > expected_result = layer.Forward(*(this->blob_bottom_vec_[0]));

  NaiveNonGPUSoftmaximaLayer<Dtype> nongpulayer(5);
  shared_ptr<Blob<Dtype> > result = nongpulayer.Forward(*(this->blob_bottom_vec_[0]));

  EXPECT_EQ((expected_result->shape()), (result->shape()));

  // Test that results are the same.
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          Dtype expected_val = expected_result->data_at(i,j,k,l);
          Dtype actual_val = result->data_at(i, j, k, l);
          EXPECT_NEAR( expected_val, actual_val, 1e-4 );
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
            this->ForwardPropThroughTwoSoftmaxes();

  LayerParameter layer_param;
  std::string proto = "softmaxima_param { softmax_size: 5 }";
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
  SoftmaximaLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Blob<Dtype>& result = *(this->blob_top_vec_[0]);

  EXPECT_EQ((expected_result->shape()), (result.shape()));

  // Test that results are the same.
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int k = 0; k < this->blob_bottom_->height(); ++k) {
      for (int l = 0; l < this->blob_bottom_->width(); ++l) {
        for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
          Dtype expected_val = expected_result->data_at(i,j,k,l);
          Dtype actual_val = result.data_at(i, j, k, l);
          EXPECT_NEAR( expected_val, actual_val, 1e-4 );
        }
      }
    }
  }
}


//TYPED_TEST(SoftmaximaLayerTest, TestGradient) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  SoftmaximaLayer<Dtype> layer(layer_param);
//  GradientChecker<Dtype> checker(1e-2, 1e-3);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_);
//}

}  // namespace caffe

