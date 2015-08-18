#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class DiffMagMonitorLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DiffMagMonitorLayerTest()
      : blob_bottom0_(new Blob<Dtype>(2, 3, 1, 2)),
        blob_bottom1_(new Blob<Dtype>(2, 3, 1, 2)) {
  }
  virtual ~DiffMagMonitorLayerTest() {
    delete blob_bottom0_;
    delete blob_bottom1_;
  }

  void CountFill(Blob<Dtype>& blob, Dtype offset) {
    Dtype val = offset;
    int index = 0;
    for(int n=0; n < blob.num(); ++n)
      for(int c=0; c < blob.channels(); ++c)
        for(int h=0; h < blob.height(); ++h)
          for(int w=0; w < blob.width(); ++w)
          {
            blob.mutable_cpu_diff()[index++] = val;
            val += 0.5;
          }
  }

  // Compute the expected value of the diff length.
  Dtype ComputeL2Mag(const Blob<Dtype>& blob) {
    Dtype val = 0.0;
    for(int index = 0; index < blob.count(); ++index) {
      Dtype diff = blob.cpu_diff()[index];
      val += diff*diff;
    }
    val = std::sqrt(val);
    return val;
  }

  void TestForward() {
    // Fill the bottom blob diff data.
    CountFill(*blob_bottom0_, 1.0);
    CountFill(*blob_bottom1_, -2.0);

    std::vector<Blob<Dtype>* > bottom_blobs;
    bottom_blobs.push_back(this->blob_bottom0_);
    bottom_blobs.push_back(this->blob_bottom1_);

    std::vector<Blob<Dtype>* > top_blobs;
    Blob<Dtype> top0;
    Blob<Dtype> top1;
    top_blobs.push_back(&top0);
    top_blobs.push_back(&top1);

    LayerParameter layer_param;
    layer_param.add_propagate_down(false);
    layer_param.add_propagate_down(false);
    DiffMagnitudeMonitoringLayer<Dtype> layer(layer_param);

    layer.SetUp(bottom_blobs,top_blobs);
    layer.Forward(bottom_blobs,top_blobs);

    {
      Dtype expected_l2mag = ComputeL2Mag(*this->blob_bottom0_);
      EXPECT_NEAR(top_blobs[0]->cpu_data()[0], expected_l2mag, 0.00001);
    }

    {
      Dtype expected_l2mag = ComputeL2Mag(*this->blob_bottom1_);
      EXPECT_NEAR(top_blobs[1]->cpu_data()[0], expected_l2mag, 0.00001);
    }
  }

  Blob<Dtype>* const blob_bottom0_;
  Blob<Dtype>* const blob_bottom1_;
};

TYPED_TEST_CASE(DiffMagMonitorLayerTest, TestDtypesAndDevices);

TYPED_TEST(DiffMagMonitorLayerTest, TestForward) {
  this->TestForward();
}

}  // namespace caffe
