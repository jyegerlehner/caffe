#include <cstring>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
  ConstantFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_value(10.);
    filler_.reset(new ConstantFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~ConstantFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<ConstantFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], this->filler_param_.value());
  }
}


template <typename Dtype>
class UniformFillerTest : public ::testing::Test {
 protected:
  UniformFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_min(1.);
    filler_param_.set_max(2.);
    filler_.reset(new UniformFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~UniformFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<UniformFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

TYPED_TEST(UniformFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], this->filler_param_.min());
    EXPECT_LE(data[i], this->filler_param_.max());
  }
}

template <typename Dtype>
class PositiveUnitballFillerTest : public ::testing::Test {
 protected:
  PositiveUnitballFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_.reset(new PositiveUnitballFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~PositiveUnitballFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<PositiveUnitballFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(PositiveUnitballFillerTest, TestDtypes);

TYPED_TEST(PositiveUnitballFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int num = this->blob_->num();
  const int count = this->blob_->count();
  const int dim = count / num;
  const TypeParam* data = this->blob_->cpu_data();
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 0);
    EXPECT_LE(data[i], 1);
  }
  for (int i = 0; i < num; ++i) {
    TypeParam sum = 0;
    for (int j = 0; j < dim; ++j) {
      sum += data[i * dim + j];
    }
    EXPECT_GE(sum, 0.999);
    EXPECT_LE(sum, 1.001);
  }
}

template <typename Dtype>
class GaussianFillerTest : public ::testing::Test {
 protected:
  GaussianFillerTest()
      : blob_(new Blob<Dtype>(2, 3, 4, 5)),
        filler_param_() {
    filler_param_.set_mean(10.);
    filler_param_.set_std(0.1);
    filler_.reset(new GaussianFiller<Dtype>(filler_param_));
    filler_->Fill(blob_);
  }
  virtual ~GaussianFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<GaussianFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);

TYPED_TEST(GaussianFillerTest, TestFill) {
  EXPECT_TRUE(this->blob_);
  const int count = this->blob_->count();
  const TypeParam* data = this->blob_->cpu_data();
  TypeParam mean = 0.;
  TypeParam var = 0.;
  for (int i = 0; i < count; ++i) {
    mean += data[i];
    var += (data[i] - this->filler_param_.mean()) *
        (data[i] - this->filler_param_.mean());
  }
  mean /= count;
  var /= count;
  // Very loose test.
  EXPECT_GE(mean, this->filler_param_.mean() - this->filler_param_.std() * 5);
  EXPECT_LE(mean, this->filler_param_.mean() + this->filler_param_.std() * 5);
  TypeParam target_var = this->filler_param_.std() * this->filler_param_.std();
  EXPECT_GE(var, target_var / 5.);
  EXPECT_LE(var, target_var * 5.);
}

template <typename Dtype>
class ScharrFillerTest : public ::testing::Test {
 protected:
  ScharrFillerTest()
      : filler_param_() {
    filler_.reset(new ScharrFiller<Dtype>(filler_param_));
  }
  virtual ~ScharrFillerTest() {}
  FillerParameter filler_param_;
  shared_ptr<ScharrFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(ScharrFillerTest, TestDtypes);

TYPED_TEST(ScharrFillerTest, TestFill) {
  Blob<TypeParam>* const blob_ = new Blob<TypeParam>(2, 3, 3, 3);
  this->filler_->Fill(blob_);
  // The Scharr filter always has two output channels, vertical and horiz.
  EXPECT_EQ(blob_->num(), 2);
  //EXPECT_EQ(this->blob_->channels(), 3);
  // The Scharr filter is always a 3x3 filter.
  EXPECT_EQ(blob_->height(), 3);
  EXPECT_EQ(blob_->width(), 3);

  for(int output_chan = 0; output_chan < 2; ++output_chan ) {
    for(int input_chan = 0; input_chan < blob_->channels();++input_chan) {
          const TypeParam* data = blob_->cpu_data();
          if(output_chan == 0) {
            // horiz filter.
            EXPECT_EQ(data[blob_->offset(0, input_chan, 0, 0)],3.0f );
            EXPECT_EQ(data[blob_->offset(0, input_chan, 0, 1)],10.0f);
            EXPECT_EQ(data[blob_->offset(0, input_chan, 0, 2)], 3.0f);
            EXPECT_EQ(data[blob_->offset(0, input_chan, 1, 0)],0.0f);
            EXPECT_EQ(data[blob_->offset(0, input_chan, 1, 1)],0.0f);
            EXPECT_EQ(data[blob_->offset(0, input_chan, 1, 2)],0.0f);
            EXPECT_EQ(data[blob_->offset(0, input_chan, 2, 0)],-3.0f);
            EXPECT_EQ(data[blob_->offset(0, input_chan, 2, 1)],-10.0f);
            EXPECT_EQ(data[blob_->offset(0, input_chan, 2, 2)],-3.0f);
          } else if ( output_chan == 1) {
            EXPECT_EQ(data[blob_->offset(1, input_chan, 0, 0)],3.0f);
            EXPECT_EQ(data[blob_->offset(1, input_chan, 0, 1)],0.0f);
            EXPECT_EQ(data[blob_->offset(1, input_chan, 0, 2)],-3.0f);
            EXPECT_EQ(data[blob_->offset(1, input_chan, 1, 0)],10.0f);
            EXPECT_EQ(data[blob_->offset(1, input_chan, 1, 1)],0.0f);
            EXPECT_EQ(data[blob_->offset(1, input_chan, 1, 2)],-10.0f);
            EXPECT_EQ(data[blob_->offset(1, input_chan, 2, 0)],3.0f);
            EXPECT_EQ(data[blob_->offset(1, input_chan, 2, 1)],0.0f);
            EXPECT_EQ(data[blob_->offset(1, input_chan, 2, 2)],-3.0f);
          } else {
            EXPECT_EQ(true, false );
          };
    }
  }
  delete blob_;
}

// Test the effect of a ScharrFiller-filled convolution filter on an image.
// It should act as an vertical and horizontal edge detectors.
TYPED_TEST(ScharrFillerTest, TestConvolution) {
  // Two 5x5 grayscale images.
  Blob<TypeParam> image_blob(2, 1, 5, 5);

  TypeParam* img_data = image_blob.mutable_cpu_data();
  for(int h = 0; h < 5; ++h) {
    for(int w = 0; w < 5; ++w) {
      // One image will have a solid horizontal bar
      int horiz_index = image_blob.offset(0,0,h,w);
      // second image will have a solid vertical bar
      int vert_index = image_blob.offset(1,0,h,w);

      // Horizontal edge between solid white and black in this image.
      img_data[horiz_index] = h >= 2 ? 1.0 : 0.0;

      // Vertical edge between solid white and black in this image.
      img_data[vert_index] = w >= 2 ? 1.0 : 0.0;
    }
  }

  vector<Blob<TypeParam>*> blob_bottom_vec;
  blob_bottom_vec.push_back(&image_blob);
  vector<Blob<TypeParam>*> blob_top_vec;
  blob_top_vec.push_back(new Blob<TypeParam>());

  // Run the convolution filter on the image.
  LayerParameter layer_param;
  ConvolutionParameter* convolution_param =
      layer_param.mutable_convolution_param();
  convolution_param->set_bias_term(false);
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(1);
  convolution_param->set_num_output(2);
  convolution_param->mutable_weight_filler()->set_type("scharr");
  shared_ptr<Layer<TypeParam> > layer( new ConvolutionLayer<TypeParam>(layer_param));
  layer->SetUp(blob_bottom_vec, blob_top_vec);
  layer->Forward(blob_bottom_vec, blob_top_vec);

  const TypeParam* data = blob_top_vec[0]->cpu_data();

  ASSERT_EQ(blob_top_vec.size(), 1);
  ASSERT_EQ(blob_top_vec[0]->num(), 2 );
  ASSERT_EQ(blob_top_vec[0]->channels(), 2 );
  ASSERT_EQ(blob_top_vec[0]->height(), 3 );
  ASSERT_EQ(blob_top_vec[0]->width(), 3 );

  Blob<TypeParam>& top_blob = *blob_top_vec[0];

  // horiz filter on first image with horizontal image
  EXPECT_EQ(data[top_blob.offset(0, 0, 0, 0)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(0, 0, 0, 1)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(0, 0, 0, 2)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(0, 0, 1, 0)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(0, 0, 1, 1)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(0, 0, 1, 2)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(0, 0, 2, 0)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 0, 2, 1)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 0, 2, 2)], 0.0f);

  // horiz filter on second image with vertical image
  EXPECT_EQ(data[top_blob.offset(1, 0, 0, 0)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 0, 0, 1)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 0, 0, 2)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 0, 1, 0)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 0, 1, 1)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 0, 1, 2)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 0, 2, 0)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 0, 2, 1)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 0, 2, 2)], 0.0f);

  //vertical filter on first image with horiz image
  EXPECT_EQ(data[top_blob.offset(0, 1, 0, 0)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 1, 0, 1)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 1, 0, 2)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 1, 1, 0)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 1, 1, 1)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 1, 1, 2)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 1, 2, 0)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 1, 2, 1)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(0, 1, 2, 2)], 0.0f);

  //vertical filter on second image with vertical image
  EXPECT_EQ(data[top_blob.offset(1, 1, 0, 0)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(1, 1, 0, 1)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(1, 1, 0, 2)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 1, 1, 0)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(1, 1, 1, 1)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(1, 1, 1, 2)], 0.0f);
  EXPECT_EQ(data[top_blob.offset(1, 1, 2, 0)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(1, 1, 2, 1)], -16.0f);
  EXPECT_EQ(data[top_blob.offset(1, 1, 2, 2)], 0.0f);

class XavierFillerTest : public ::testing::Test {
 protected:
  XavierFillerTest()
      : blob_(new Blob<Dtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new XavierFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = 0.;
    Dtype ex2 = 0.;
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = sqrt(ex2 - mean*mean);
    Dtype target_std = sqrt(2.0 / n);
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std, target_std, 0.1);
  }
  virtual ~XavierFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<XavierFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(XavierFillerTest, TestDtypes);

TYPED_TEST(XavierFillerTest, TestFillFanIn) {
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(XavierFillerTest, TestFillFanOut) {
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(XavierFillerTest, TestFillAverage) {
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

template <typename Dtype>
class MSRAFillerTest : public ::testing::Test {
 protected:
  MSRAFillerTest()
      : blob_(new Blob<Dtype>(1000, 2, 4, 5)),
        filler_param_() {
  }
  virtual void test_params(FillerParameter_VarianceNorm variance_norm,
      Dtype n) {
    this->filler_param_.set_variance_norm(variance_norm);
    this->filler_.reset(new MSRAFiller<Dtype>(this->filler_param_));
    this->filler_->Fill(blob_);
    EXPECT_TRUE(this->blob_);
    const int count = this->blob_->count();
    const Dtype* data = this->blob_->cpu_data();
    Dtype mean = 0.;
    Dtype ex2 = 0.;
    for (int i = 0; i < count; ++i) {
      mean += data[i];
      ex2 += data[i] * data[i];
    }
    mean /= count;
    ex2 /= count;
    Dtype std = sqrt(ex2 - mean*mean);
    Dtype target_std = sqrt(2.0 / n);
    EXPECT_NEAR(mean, 0.0, 0.1);
    EXPECT_NEAR(std, target_std, 0.1);
  }
  virtual ~MSRAFillerTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  FillerParameter filler_param_;
  shared_ptr<MSRAFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(MSRAFillerTest, TestDtypes);

TYPED_TEST(MSRAFillerTest, TestFillFanIn) {
  TypeParam n = 2*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_IN, n);
}
TYPED_TEST(MSRAFillerTest, TestFillFanOut) {
  TypeParam n = 1000*4*5;
  this->test_params(FillerParameter_VarianceNorm_FAN_OUT, n);
}
TYPED_TEST(MSRAFillerTest, TestFillAverage) {
  TypeParam n = (2*4*5 + 1000*4*5) / 2.0;
  this->test_params(FillerParameter_VarianceNorm_AVERAGE, n);
}

}  // namespace caffe
