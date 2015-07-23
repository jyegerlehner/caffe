#include <cmath>
#include <cstring>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

//template<typename Dtype>
//class XCovLossLayerForTest : public XCovLossLayer<Dtype>
//{
//public:
//  explicit XCovLossLayerForTest(const LayerParameter& param) :
//    XCovLossLayer<Dtype>(param) {}
//  void ShowMeans()
//  {
//    std::cout << "XCovLossLayer mean0:" << std::endl;
//    caffe::Show(this->mean_0_);
//    std::cout << "XCovLossLayer mean1:" << std::endl;
//    caffe::Show(this->mean_1_);
//  }
//};

template<typename Dtype>
struct NaiveXCovLossLayer
{
  typedef Eigen::MatrixXf Matrix;

  void ShowMeans()
  {
    std::cout << "NaiveXCov mean0:" << std::endl;
    Show(mean_0_);
    std::cout << "NaiveXCov mean1:" << std::endl;
    Show(mean_1_);
  }

  NaiveXCovLossLayer()
  {
    mean_vec_.push_back(&mean_0_);
    mean_vec_.push_back(&mean_1_);
//    temp_vec_.push_back(&temp_0_);
//    temp_vec_.push_back(&temp_1_);
  }

  virtual ~NaiveXCovLossLayer()
  {
//    std::vector<Blob<Dtype>* > mean_vec_;
//    Blob<Dtype> mean_0_;
//    Blob<Dtype> mean_1_;
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

  // Compute the mean of a single location along the axis (channel) across all
  // other dimensions (in this case, example number, and the spatial indices).
  float ComputeMean( const Blob<Dtype>& bottom, int channel )
  {
    int num = bottom.num();
    int height = bottom.height();
    int width = bottom.width();
    const Dtype* bottom_data = bottom.cpu_data();

    float mean = 0.0f;
    float ctr = 0.0f;
    for( int n = 0; n < num; ++n )
    {
      for( int h = 0; h < height; ++h )
      {
        for( int w = 0; w < width; ++w )
        {
            int offset = bottom.offset( n, channel, h, w );
            mean += bottom_data[offset];
            ctr += 1.0;
        }
      }
    }
    return mean / ctr;
  }

//  for( int row = 0; row < bottom[0]->channels(); ++row )
//  {
//    for( int col =0; col < bottom[1]->channels(); ++col )
//    {
//      xcov(row, col) = deltas0[row] * deltas[col];
//    }
//  }


  // Returns the covariance matrix before the elements are squared.
  Matrix ComputeXcov( int dim0,
                      int dim1,
                      std::vector<Blob<Dtype>* >& mean_vec_,
                      const std::vector<Blob<Dtype>* >& bottom )
  {
    Blob<Dtype>& bottom0 = *bottom[0];
    Blob<Dtype>& bottom1 = *bottom[1];

    CHECK(bottom0.num() == bottom1.num() );
    CHECK(bottom0.height() == bottom1.height() );
    CHECK(bottom0.width() == bottom1.width() );
    CHECK(mean_vec_[0]->count() == bottom0.channels());
    CHECK(mean_vec_[1]->count() == bottom1.channels());
    CHECK(dim0 == bottom0.channels());
    CHECK(dim1 == bottom1.channels());

    int num = bottom0.num();
    int height = bottom0.height();
    int width = bottom0.width();

    Matrix xcov(dim0,dim1);

    // Number of instances of each channel.
    float N = num*height*width;
    for( int chan0 = 0; chan0 < dim0; ++chan0 )
    {
      for( int chan1 = 0; chan1 < dim1; ++chan1 )
      {
        float accumulator = 0.0f;
        // Compute (x-ux)(y-uy) across all examples and spatial locations.
        for(int n = 0; n < num; ++n )
        {
          for(int h=0; h < height; ++h)
          {
            for(int w=0; w < width; ++w)
            {
              int b0_index = bottom0.offset(n,chan0,h,w);
              int b1_index = bottom1.offset(n,chan1,h,w);
              float mean0 = mean_vec_[0]->cpu_data()[chan0];
              float mean1 = mean_vec_[1]->cpu_data()[chan1];
              float val0 = bottom0.cpu_data()[b0_index];
              float val1 = bottom1.cpu_data()[b1_index];
              accumulator += (val0 - mean0)*(val1 - mean1) / N;
            }
          }
        }
        xcov(chan0,chan1) = accumulator;
      }
    }
    return xcov;
  }

  float Forward( const std::vector<Blob<Dtype>* > bottom ) {
    int AXIS = 1;


    // xcovar dim of the first bottom blob.
    int dim0 = bottom[0]->shape()[AXIS];
    // xcovar dim of the second bottom blob.
    int dim1 = bottom[1]->shape()[AXIS];

    // Dimensions of bottoms must be same on all except the AXIS dim
    // of the shape.
    std::vector<int> shape0 = bottom[0]->shape();
    std::vector<int> shape1 = bottom[1]->shape();

//    xcov_temp_.Reshape( dim0*dim1, 1, 1, 1 );
//    xcov_.Reshape( dim0*dim1, 1, 1, 1 );

//    batch_sum_multiplier_.Reshape(bottom[0]->num(); /* *dim0*dim1? */, 1, 1, 1);
//    Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
//    caffe_set(batch_sum_multiplier_.count(), Dtype(1), batch_multiplier_data);

    // for now, we support only two inputs
    CHECK_EQ(bottom.size(), 2);

    std::vector<int> inner_dims;
    for( int b=0; b < bottom.size(); ++b )
    {
      std::vector<int> shape = bottom[b]->shape();
      int tot = 1;
      for( int inner_dim = AXIS+1;
           inner_dim < shape.size();
           ++inner_dim ) {
        tot *= shape[inner_dim];
      }
      inner_dims.push_back(tot);
    }

//      std::vector<int> shape = input.shape();
//      int cardinality = shape[0];
//      int channels = shape[1];
//      int height = shape[2];
//      int width = shape[3];

    mean_vec_[0]->Reshape(1, dim0, 1, 1 );
    {
      Blob<Dtype>& bottom0 = *bottom[0];
      Blob<Dtype>& mean_blob = *mean_vec_[0];
      // There is a mean for each channel, computed
      // across all [example index, (h,w)].
      for( int channel = 0; channel < dim0; ++channel )
      {
        int mean_index = mean_blob.offset(0,channel,0,0);
        mean_blob.mutable_cpu_data()[mean_index] = ComputeMean( bottom0, channel );
      }
    }

    mean_vec_[1]->Reshape(1, dim1, 1, 1 );
    {
      Blob<Dtype>& bottom1 = *bottom[1];
      Blob<Dtype>& mean_blob = *mean_vec_[1];
      // There is a mean for each channel, computed
      // across all [example index, (h,w)].
      for( int channel = 0; channel < dim1; ++channel )
      {
        int mean_index = mean_blob.offset(0,channel,0,0);
        mean_blob.mutable_cpu_data()[mean_index] = ComputeMean( bottom1, channel );
      }
    }

    Matrix xcov = ComputeXcov( dim0, dim1, mean_vec_, bottom );

    float error = 0.0f;
    for(int chan0 = 0; chan0 < dim0; ++chan0)
    {
      for(int chan1 = 0; chan1 < dim1; ++chan1)
      {
        float val = xcov(chan0,chan1);
        error += val*val;
      }
    }
    error *= 0.5f;
    return error;
  }

//  Blob<Dtype> batch_sum_multiplier_;

//  // The means
  std::vector<Blob<Dtype>* > mean_vec_;
  Blob<Dtype> mean_0_;
  Blob<Dtype> mean_1_;
//  std::vector<Blob<Dtype>* > temp_vec_;
};


template <typename TypeParam>
class XCovLossLayer2Test : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  XCovLossLayer2Test()
      : blob_bottom_0_(new Blob<Dtype>(5, 3, 1, 1)),
        blob_bottom_1_(new Blob<Dtype>(5, 5, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    CountFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_0_);
    filler.Fill(this->blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~XCovLossLayer2Test() {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(XCovLossLayer2Test, TestDtypesAndDevices);

// Test the naive implementation gives same results as XCovLossLayer
// for blob of nxcx1x1.
TYPED_TEST(XCovLossLayer2Test, TestNaiveXcovForwardConsistency) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  XCovLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype kErrorMargin = 1e-5;
  for (int i = 0; i < this->blob_top_->count(); i++) {
    Dtype val = *(this->blob_top_->cpu_data() + i);
    EXPECT_NEAR(val, 0.048000015318393707, kErrorMargin);
  }

  NaiveXCovLossLayer<Dtype> naive;
  float naive_value = naive.Forward(this->blob_bottom_vec_);
//  naive.ShowMeans();
//  layer.ShowMeans();

  EXPECT_EQ(this->blob_top_->count(),1);
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], naive_value, kErrorMargin);
}

template<typename Dtype>
void Reshuffle( const Blob<Dtype>& blob, Blob<Dtype>& result )
{
  // Assumes canonical axis = 1;
  int in_num = blob.num();
  int in_height = blob.height();
  int in_width = blob.width();
  int result_size = in_num*in_height*in_width;
  result.Reshape(result_size, blob.channels(),1,1);
  Dtype* result_data = result.mutable_cpu_data();
  int out_inx = 0;
  for(int n = 0; n < in_num; ++n )
  {
    for(int h = 0; h < in_height; ++h )
    {
      for( int w = 0; w < in_width; ++w )
      {
        for(int c = 0; c < blob.channels(); ++c)
        {
          int offset = blob.offset(n,c,h,w);
          Dtype in_val = blob.cpu_data()[offset];

          int result_offset = result.offset( out_inx, c, 0, 0);
          result_data[result_offset] = in_val;
        }
        out_inx++;
      }
    }
  }
}

// When given a blob with NxCxHxW, the naive implementation should produce the
// same xcov as XCovLossLayer does on an (N*H*W)xCx1x1 blob.
TYPED_TEST(XCovLossLayer2Test, TestNaiveXcovForwardConsistency_HW) {
  typedef typename TypeParam::Dtype Dtype;

  std::vector<Blob<Dtype>* > bottom;
  Blob<Dtype> bottom0(2, 3, 2, 4);
  Blob<Dtype> bottom1(2, 2, 2, 4);
  FillerParameter filler_param;
  CountFiller<Dtype> filler(filler_param);
  filler.Fill(&bottom0);
  filler.Fill(&bottom1);
  bottom.push_back(&bottom0);
  bottom.push_back(&bottom1);

  NaiveXCovLossLayer<Dtype> naive;
  float naive_value = naive.Forward(bottom);

  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Blob<Dtype> new_top_0;
  {
    std::vector<Blob<Dtype>* > new_bottom;
    Blob<Dtype> new_bot_0;
    Reshuffle(bottom0, new_bot_0 );
    Blob<Dtype> new_bot_1;
    Reshuffle(bottom1, new_bot_1 );
    new_bottom.push_back(&new_bot_0);
    new_bottom.push_back(&new_bot_1);
    std::vector<Blob<Dtype>*> new_top;
    new_top.push_back(&new_top_0);
    XCovLossLayer<Dtype> layer(layer_param);
    layer.SetUp(new_bottom, new_top);
    layer.Forward(new_bottom, new_top);
    EXPECT_EQ(new_top.size(),1);
  }

  const Dtype kErrorMargin = 1e-5;
  EXPECT_NEAR(new_top_0.cpu_data()[0], naive_value, kErrorMargin);
}


//TYPED_TEST(XCovLossLayer2Test, TestForward) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  XCovLossLayer2<Dtype> layer(layer_param);
//  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

//  for (int i = 0; i < this->blob_top_->count(); i++) {
//    Dtype val = *(this->blob_top_->cpu_data() + i);
//    const Dtype kErrorMargin = 1e-5;
//    EXPECT_NEAR(val, 0.048000015318393707, kErrorMargin);
//  }
//}

//TYPED_TEST(XCovLossLayer2Test, TestGradient) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  XCovLossLayer2<Dtype> layer(layer_param);
//  GradientChecker<Dtype> checker(1e-2, 1e-2);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_);
//}

//TYPED_TEST(XCovLossLayer2Test, TestFatShapeForward) {
//  typedef typename TypeParam::Dtype Dtype;

//  LayerParameter param;
//  std::string proto = "xcov_param { axis: 1 }";
//  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));

//  XCovLossLayer2<Dtype> layer(param);
//  GradientChecker<Dtype> checker(1e-2, 1e-2);
//  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
//      this->blob_top_vec_);
//}


}  // namespace caffe
