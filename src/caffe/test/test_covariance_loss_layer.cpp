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

template<typename Dtype>
struct NaiveCovLossLayer
{
  typedef Eigen::MatrixXf Matrix;

  void ShowMeans()
  {
    std::cout << "NaiveCov mean:" << std::endl;
    Show(mean_);
  }

  Blob<Dtype>* GetMean()
  {
    return &mean_;
  }

  NaiveCovLossLayer()
  {
  }

  virtual ~NaiveCovLossLayer()
  {
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

  // Returns the covariance matrix before the elements are squared.
  Matrix ComputeCov( int dim,
                      const std::vector<Blob<Dtype>* >& bottom )
  {
    Blob<Dtype>& bottom_blob = *bottom[0];

    CHECK(mean_.count() == bottom_blob.channels());
    CHECK(dim == bottom_blob.channels());

    int num = bottom_blob.num();
    int height = bottom_blob.height();
    int width = bottom_blob.width();

    Matrix cov(dim,dim);

    // Number of instances of each channel.
    float N = num*height*width;
    for( int chan0 = 0; chan0 < dim; ++chan0 )
    {
      for( int chan1 = 0; chan1 < dim; ++chan1 )
      {
        float accumulator = 0.0f;
        // Compute (x-ux)(y-uy) across all examples and spatial locations.
        for(int n = 0; n < num; ++n )
        {
          for(int h=0; h < height; ++h)
          {
            for(int w=0; w < width; ++w)
            {
              int b0_index = bottom_blob.offset(n,chan0,h,w);
              int b1_index = bottom_blob.offset(n,chan1,h,w);
              float mean0 = mean_.cpu_data()[chan0];
              float mean1 = mean_.cpu_data()[chan1];
              float val0 = bottom_blob.cpu_data()[b0_index];
              float val1 = bottom_blob.cpu_data()[b1_index];
              accumulator += (val0 - mean0)*(val1 - mean1) / N;
            }
          }
        }
        cov(chan0,chan1) = accumulator;
      }
    }
    return cov;
  }

  float Forward( const std::vector<Blob<Dtype>* > bottom ) {
    int AXIS = 1;

    // covar dim of the first bottom blob.
    int dim = bottom[0]->shape()[AXIS];

    // Dimensions of bottoms must be same on all except the AXIS dim
    // of the shape.
    std::vector<int> shape = bottom[0]->shape();

    CHECK_EQ(bottom.size(), 1);

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

    mean_.Reshape(1, dim, 1, 1 );
    {
      Blob<Dtype>& bottom_blob = *bottom[0];
      // There is a mean for each channel, computed
      // across all [example index, (h,w)].
      for( int channel = 0; channel < dim; ++channel )
      {
        int mean_index = mean_.offset(0,channel,0,0);
        mean_.mutable_cpu_data()[mean_index] = ComputeMean( bottom_blob,
                                                            channel );
      }
    }

    Matrix cov = ComputeCov( dim, bottom );

    float error = 0.0f;
    for(int chan0 = 0; chan0 < dim; ++chan0)
    {
      for(int chan1 = 0; chan1 < dim; ++chan1)
      {
        float val = cov(chan0,chan1);
        error += val*val;
      }
    }
    error *= 0.5f;
    return error;
  }

  Blob<Dtype> mean_;
};


template <typename TypeParam>
class CovLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CovLossLayerTest()
      : blob_bottom_(new Blob<Dtype>(5, 3, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    CountFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CovLossLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CovLossLayerTest, TestDtypesAndDevices);

// Test the naive implementation gives same results as XCovLossLayer
// for blob of nxcx1x1.
TYPED_TEST(CovLossLayerTest, TestNaiveCovForwardConsistency) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CovLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  const Dtype kErrorMargin = 1e-5;
//  for (int i = 0; i < this->blob_top_->count(); i++) {
//    Dtype val = *(this->blob_top_->cpu_data() + i);
//    EXPECT_NEAR(val, 0.048000015318393707, kErrorMargin);
//  }

  NaiveCovLossLayer<Dtype> naive;
  float naive_value = naive.Forward(this->blob_bottom_vec_);

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

template <typename Dtype>
void ExpectEqual(const Blob<Dtype>& blob1, const Blob<Dtype>& blob2 )
{
  EXPECT_EQ(blob1.count(), blob2.count());
  EXPECT_EQ(blob1.shape().size(), blob2.shape().size());
  for(int i = 0; i < blob1.shape().size(); ++i )
  {
    EXPECT_EQ(blob1.shape()[i], blob2.shape()[i]);
  }
  for(int i = 0; i < blob1.count(); ++i)
  {
    Dtype val1 = blob1.cpu_data()[i];
    Dtype val2 = blob2.cpu_data()[i];
    EXPECT_NEAR(val1, val2, 0.00001f);
  }
}

//TYPED_TEST(XCovLossLayer2Test, TestInnerToOuterIdempotence) {
//  typedef typename TypeParam::Dtype Dtype;

//  Blob<Dtype> bottom(2, 3, 2, 4);
//  FillerParameter filler_param;
//  CountFiller<Dtype> filler(filler_param);
//  filler.Fill(&bottom);

//  Blob<Dtype> orig_blob;
//  orig_blob.CopyFrom(bottom, true, true);
//  orig_blob.CopyFrom(bottom);
//  Blob<Dtype> expected_result;
//  expected_result.CopyFrom(orig_blob, true, true);
//  expected_result.CopyFrom(orig_blob);

//  Blob<Dtype> outerized_blob;
//  outerized_blob.Reshape( orig_blob.num()*
//                          orig_blob.height()*
//                          orig_blob.width(),
//                          orig_blob.channels(),
//                          1,1);

//  InnerToOuter<Dtype>(orig_blob.cpu_data(),
//                      outerized_blob.mutable_cpu_data(),
//                      orig_blob.num(),
//                      orig_blob.channels(),
//                      orig_blob.height(),
//                      orig_blob.width(),
//                      1);

//  Blob<Dtype> result_blob;
//  result_blob.ReshapeLike(orig_blob);

//  OuterToInner<Dtype>(result_blob.mutable_cpu_data(),
//               outerized_blob.cpu_data(),
//               orig_blob.num(),
//               orig_blob.channels(),
//               orig_blob.height(),
//               orig_blob.width(),
//               1);

//  ExpectEqual(result_blob, expected_result);
//}

//TYPED_TEST(CovLossLayerTest, TestForwardSimple) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  CovLossLayer<Dtype> layer(layer_param);

//  ShowBlob(*this->blob_bottom_);
//  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
//  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

//  for (int i = 0; i < this->blob_top_->count(); i++) {
//    Dtype val = *(this->blob_top_->cpu_data() + i);
//    const Dtype kErrorMargin = 1e-5;
//    EXPECT_NEAR(val, 0.048000015318393707, kErrorMargin);
//  }
//}

TYPED_TEST(CovLossLayerTest, TestForwardVsNaive) {
  typedef typename TypeParam::Dtype Dtype;

  std::vector<Blob<Dtype>* > bottom;
  Blob<Dtype> bottom0(2, 3, 2, 4);
  FillerParameter filler_param;
  CountFiller<Dtype> filler(filler_param);
  filler.Fill(&bottom0);
  bottom.push_back(&bottom0);

  Blob<Dtype> top_blob;
  top_blob.Reshape(1,1,1,1);
  std::vector<Blob<Dtype>*> top;
  top.push_back(&top_blob);

  // Compute forward using naive implementation.
  NaiveCovLossLayer<Dtype> naive;
  float naive_value = naive.Forward(bottom);

  // Compute forward using XCovLoss2Layer.
  LayerParameter layer_param;
  CovLossLayer<Dtype> layer(layer_param);
  layer.SetUp(bottom, top);
  layer.Forward(bottom, top);

  // Compare the means.
  ExpectEqual(*naive.GetMean(), *layer.GetMean());

  // Error should be the same.
  const Dtype kErrorMargin = 1e-5;
  EXPECT_NEAR(top_blob.cpu_data()[0], naive_value, kErrorMargin);
}

//TYPED_TEST(XCovLossLayer2Test, TestNchwFromIndex) {
//  typedef typename TypeParam::Dtype Dtype;
//  Blob<Dtype> bottom(2, 3, 5, 4);
//  FillerParameter filler_param;
//  CountFiller<Dtype> filler(filler_param);
//  filler.Fill(&bottom);

//  for(int expected_n = 0; expected_n < bottom.num(); ++expected_n)
//    for(int expected_c = 0; expected_c < bottom.channels(); ++expected_c)
//      for(int expected_h = 0; expected_h < bottom.height(); ++expected_h)
//        for(int expected_w = 0; expected_w < bottom.width(); ++expected_w)
//        {
//          int offset = bottom.offset(expected_n,
//                                     expected_c,
//                                     expected_h,
//                                     expected_w);
//          int n = -1;
//          int c = -1;
//          int h = -1;
//          int w = -1;

//          NchwFromIndex( offset,
//                            bottom.channels(),
//                            bottom.num(),
//                            bottom.height(),
//                            bottom.width(),
//                            n, c, h, w);
//          ASSERT_EQ(n, expected_n);
//          ASSERT_EQ(c, expected_c);
//          ASSERT_EQ(h, expected_h);
//          ASSERT_EQ(w, expected_w);
//        }

//}

// Test the case where the canonical axis is 1 (channels), and the inner axis
// dimensions (height and width) are greater than 1.
TYPED_TEST(CovLossLayerTest, TestGradient_HW_GT_1) {
  typedef typename TypeParam::Dtype Dtype;

  std::vector<Blob<Dtype>* > bottom;
  Blob<Dtype> bottom_blob(2, 3, 2, 4);
  FillerParameter filler_param;
  CountFiller<Dtype> filler(filler_param);
  filler.Fill(&bottom_blob);
  bottom.push_back(&bottom_blob);

  LayerParameter layer_param;
  CovLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, bottom, this->blob_top_vec_);
}

// Test the case where the canonical axis is 1 (channels), and the inner axis
// dimensions (height and width) are exactly equal to 1.
TYPED_TEST(CovLossLayerTest, TestGradientSimple) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  CovLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
