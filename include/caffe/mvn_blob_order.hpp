#pragma once

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
struct MvnBlobOrdering
{
  typedef std::vector<Blob<Dtype>* > BlobVec;

  MvnBlobOrdering(const LayerParameter& layer_param,
                  BlobInfo<Dtype>* blob_info,
                  std::vector<Blob<Dtype>*>& blobs):
    layer_param_( layer_param ),
    mean_index_(-1),
    scale_index_(-1),
    data_index(-1)
  {
    CHECK(blob_info != NULL) << "Null blob_info provided to layer "
                                << layer_param_.name();
    CHECK(NumBlobs() == bottom_blobs.size()) << "There are " <<
            blobs.size() << " blobs, but " << NumExpectedBlobs()
            << " implied by the MVNParameter in layer " << layer_param_.name();
    for( int i = 0; i < bottom_blobs.size(); ++i )
    {
      Blob<Dtype>* blob = blobs[i];
      std::string name = blob_info->NameFromPointer(blob);
      if (name == MeanName())
      {
        mean_index_ = i;
      }
      else if (name == ScaleName())
      {
        scale_index_ = i;
      }
      else
      {
        // The blob not named by
        data_index_ = i;
      }
    }

    if (HasMean()) {
      CHECK(mean_index_ != -1) << "Mean blob " << MeanName() <<
              " was specified in layer " << layer_param_.name() << " but " <<
              "was not found in blobs.";
    }
    if (HasScale()) {
      CHECK(scale_index_ != -1) << "Mean blob " << MeanName() <<
                                  " was specified in layer "
                                << layer_param_.name() << " but "
                                << "was not found in blobs.";
    }

    CHECK(data_index_ != -1) << "No data blob found for layer " <<
                                layer_param_.name();
  }

  // Get the number of blobs this layer will have. It is the number of
  // output blobs if it is an MVNLayer. It is the number of input blobs if
  // it is an InverseMVNLayer.
  int NumExpectedBlobs() const
  {
    CHECK(ERROR) << layer_param_.has_mvn_param();
    int num = 1;
    if ( HasMean() )
      num++;
    if ( HasScale() )
      num++;
    return num;
  }

  // Get the blob that has the mean of each of the inputs to the MVNLayer.
  Blob<Dtype>* MeanBlob( BlobVec& blobs ) const
  {
    CHECK(HasMean()) << "Layer " << layer_param_.name() << " has no "
                        << "mean blob.";
    return blobs[mean_index_];
  }

  // Get the blob that has the scales by which each input to the MVNLayer is
  // scaled by.
  Blob<Dtype>* ScaleBlob( BlobVec& blobs ) const
  {
    CHECK(HasScale()) << "Layer " << layer_param_.name() << " has no "
                      << "scale blob.";
    return blobs[scale_index_];
  }

  // Return indication if the layer computes the mean.
  bool HasMean() const
  {
    return MvnParam().has_mean_blob();
  }

  std::string MeanName() const
  {
    return MvnParam().mean_blob();
  }

  std::string ScaleName() const
  {
    return MvnParam().scale_blob();
  }

  // Return indication if the layer scales to a variance of one.
  bool HasScale() const
  {
    return MvnParam().has_scale_blob();
  }

  // Get the value of the MVN layer's parameter.
  const MVNParameter MvnParam()
  {
    return layer_param_.mvn_param();
  }

  // The MVNLayer's parameter.
  LayerParameter layer_param_;
  int mean_index_;
  int scale_index_;
  int data_index_;
};

}
