#include <cmath>
#include <cstring>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "gtest/gtest.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace {
  const int INPUT_NUM = 2;
  const int INPUT_CHANNELS = 3;
  const int INPUT_HEIGHT = 4;
  const int INPUT_WIDTH = 5;
}

namespace caffe {

template <typename TypeParam>
class InverseMVNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  InverseMVNLayerTest()
      : mvn_blob_bottom_(new Blob<Dtype>(INPUT_NUM, INPUT_CHANNELS,
                                         INPUT_HEIGHT, INPUT_WIDTH)),
        mvn_mean_blob_(new Blob<Dtype>()),
        mvn_scale_blob_(new Blob<Dtype>()),
        mvn_result_blob_(new Blob<Dtype>()),
        inverse_mvn_blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);

    mvn_blob_bottom_vec_.push_back(mvn_blob_bottom_);
    mvn_blob_top_vec_.push_back(mvn_mean_blob_);
    mvn_blob_top_vec_.push_back(mvn_scale_blob_);
    mvn_blob_top_vec_.push_back(mvn_result_blob_);

    // The blob that contains the means computed by the mvn layer.
    inverse_mvn_blob_bottom_vec_.push_back(mvn_mean_blob_);
    // The blob that contains the scales computed by the mvn layer.
    inverse_mvn_blob_bottom_vec_.push_back(mvn_scale_blob_);
    // The blob that has the scaled, mean-subtracted output of the mvn layer.
    inverse_mvn_blob_bottom_vec_.push_back(mvn_result_blob_);
    // The inverse mvn layer's output blob.
    inverse_mvn_blob_top_vec_.push_back(inverse_mvn_blob_top_);
  }
  virtual ~InverseMVNLayerTest() {
    delete mvn_mean_blob_;
    delete mvn_scale_blob_;
    delete mvn_result_blob_;
    delete inverse_mvn_blob_top_;
  }
  Blob<Dtype>* const mvn_mean_blob_;
  Blob<Dtype>* const mvn_scale_blob_;
  Blob<Dtype>* const mvn_result_blob_;
  Blob<Dtype>* const inverse_mvn_blob_top_;

  vector<Blob<Dtype>*> mvn_blob_bottom_vec_;
  vector<Blob<Dtype>*> mvn_blob_top_vec_;
  vector<Blob<Dtype>*> inverse_mvn_blob_bottom_vec_;
  vector<Blob<Dtype>*> inverse_mvn_blob_top_vec_;
};

TYPED_TEST_CASE(InverseMVNLayerTest, TestDtypesAndDevices);

TYPED_TEST(InverseMVNLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  // Default mvn layer does both mean and variance normalization.
  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" scale_blob: \"scale_a\"}",
          &mvn_layer_param));
  shared_ptr<MVNLayer<Dtype> > mvn_layer(new MVNLayer<Dtype>(mvn_layer_param));
  mvn_layer->SetUp(this->mvn_blob_bottom_vec_, this->mvn_blob_top_vec_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" scale_blob: \"scale_a\" }"));
  shared_ptr<InverseMVNLayer<Dtype> >
      inverse_mvn_layer(new InverseMVNLayer<Dtype>( inverse_mvn_layer_param ));
  inverse_mvn_layer->SetUp(this->inverse_mvn_blob_bottom_vec_,
                           this->inverse_mvn_blob_top_vec_);

  EXPECT_EQ(this->mvn_blob_top_vec_.size(), 3);
  EXPECT_EQ(this->inverse_mvn_blob_top_vec_.size(), 1);

  const int num_channels = 5;
  Blob<Dtype>* blob0 = this->mvn_blob_top_vec_[0];
  EXPECT_EQ(blob0->num(), INPUT_NUM);
  EXPECT_EQ(blob0->height(), INPUT_HEIGHT);
  EXPECT_EQ(blob0->width(), INPUT_WIDTH);
  EXPECT_EQ(blob0->channels(), INPUT_CHANNELS);

  Blob<Dtype>* blob1 = this->mvn_blob_top_vec_[1];
  EXPECT_EQ(blob1->num(), INPUT_NUM);
  EXPECT_EQ(blob1->height(), 1);
  EXPECT_EQ(blob1->width(), 1);
  EXPECT_EQ(blob1->channels(), INPUT_CHANNELS);

  Blob<Dtype>* blob2 = this->inverse_mvn_blob_top_vec_[0];
  EXPECT_EQ(blob2->num(), INPUT_NUM);
  EXPECT_EQ(blob2->height(), INPUT_HEIGHT);
  EXPECT_EQ(blob2->width(), INPUT_WIDTH);
  EXPECT_EQ(blob2->channels(), INPUT_CHANNELS);
}

// We rely on MVNLayer working correctly (as it is independently tested).
// Then we test that the mvn layer composed with the inverse mvn layer
// yields the identity function (i.e. returns the original data).
TYPED_TEST(InverseMVNLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" scale_blob: \"scale_a\"}",
          &mvn_layer_param));
  MVNLayer<Dtype> mvn_layer(mvn_layer_param);
  // Create an MVN layer.
  mvn_layer.SetUp(this->mvn_blob_bottom_vec_,
                  this->mvn_blob_top_vec_);


  // Create an InverseMVN layer to "undo" or invert the effect of the MVN
  // layer.
  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" scale_blob: \"scale_a\"}",
          &inverse_mvn_layer_param));
  InverseMVNLayer<Dtype>  inverse_mvn_layer(inverse_mvn_layer_param);
  inverse_mvn_layer.SetUp(this->inverse_mvn_blob_bottom_vec_,
                          this->inverse_mvn_blob_top_vec_);

  // Run the blob forward through the MVN layer.
  mvn_layer.Forward(this->mvn_blob_bottom_vec_,
                this->mvn_blob_top_vec_);
  // Run the output of the MVN layer forward through the Inverse MVN layer.
  inverse_mvn_layer.Forward(this->inverse_mvn_blob_bottom_vec_,
                this->inverse_mvn_blob_top_vec_);


  int num = this->mvn_blob_bottom_->num();
  int channels = this->mvn_blob_bottom_->channels();
  int height = this->mvn_blob_bottom_->height();
  int width = this->mvn_blob_bottom_->width();

  // Expect that the dimensions of the blob coming out of the InverseMVN layer
  // are the same as what went in to the MVN layer.
  EXPECT_EQ(num, this->inverse_mvn_blob_top_->num());
  EXPECT_EQ(channels, this->inverse_mvn_blob_top_->channels());
  EXPECT_EQ(height, this->inverse_mvn_blob_top_->height());
  EXPECT_EQ(width, this->inverse_mvn_blob_top_->width());

  EXPECT_EQ(num, INPUT_NUM);
  EXPECT_EQ(channels, INPUT_CHANNELS);
  EXPECT_EQ(height, INPUT_HEIGHT);
  EXPECT_EQ(width, INPUT_WIDTH);

  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < channels; ++j) {
      for (int k = 0; k < height; ++k) {
        for (int l = 0; l < width; ++l) {
          // Since the InverseMVNLayer is the inverse operation of the
          // MVNLayer, expect the input to the latter to be the output of the
          // former.
          Dtype expected_data = this->mvn_blob_bottom_->data_at(i, j, k, l);
          Dtype actual_data = this->inverse_mvn_blob_top_->data_at(i, j, k, l);
          const Dtype kErrorBound = 0.0001;
          EXPECT_NEAR(expected_data, actual_data, kErrorBound);
        }
      }
    }
  }
}

TYPED_TEST(InverseMVNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  // Default mvn layer does both mean and variance normalization.
  LayerParameter mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" scale_blob: \"scale_a\"}",
          &mvn_layer_param));
  shared_ptr<MVNLayer<Dtype> > mvn_layer(new MVNLayer<Dtype>(mvn_layer_param));
  mvn_layer->SetUp(this->mvn_blob_bottom_vec_, this->mvn_blob_top_vec_);

  LayerParameter inverse_mvn_layer_param;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      "mvn_param { mean_blob: \"mean_a\" scale_blob: \"scale_a\" }"));
  shared_ptr<InverseMVNLayer<Dtype> >
      inverse_mvn_layer(new InverseMVNLayer<Dtype>( inverse_mvn_layer_param ));
  inverse_mvn_layer->SetUp(this->inverse_mvn_blob_bottom_vec_,
                           this->inverse_mvn_blob_top_vec_);

  EXPECT_EQ(this->mvn_blob_top_vec_.size(), 3);
  EXPECT_EQ(this->inverse_mvn_blob_top_vec_.size(), 1);

  mvn_layer.Forward(this->mvn_blob_bottom_vec_,
                this->mvn_blob_top_vec_);

  inverse_mvn_layer.Forward(this->inverse_mvn_blob_bottom_vec_,
                            this->inverse_mvn_blob_top_vec_);

  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&inverse_mvn_layer,
      this->inverse_mvn_blob_bottom_vec_,
      this->inverse_mvn_blob_top_vec_);
}

} // namespace caffe
