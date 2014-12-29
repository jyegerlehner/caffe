#pragma once

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/blob_finder.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
struct MvnBlobOrdering
{
  typedef std::vector<Blob<Dtype>* > BlobVec;

  MvnBlobOrdering():
    layer_param_(),
    mean_index_(-1),
    variance_index_(-1),
    data_index_(-1),
    blob_finder_(),
    initialized_(false)
  {
  }

  MvnBlobOrdering(const LayerParameter& layer_param,
                  const BlobFinder<Dtype>& blob_finder):
    layer_param_( layer_param ),
    mean_index_(-1),
    variance_index_(-1),
    data_index_(-1),
    blob_finder_(blob_finder),
    initialized_(false)
  {
  }

  void LazyInit(const vector<Blob<Dtype>*>& blobs)
  {
    if (!initialized_) {
      SetUp(blobs);
      initialized_ = true;
    }
  }

  void SetUp( const vector<Blob<Dtype>*>& blobs )
  {
    CHECK(NumBlobs() == blobs.size()) << "There are " <<
            blobs.size() << " blobs, but " << NumBlobs()
            << " implied by the MVNParameter in layer " << layer_param_.name();
    for( int i = 0; i < blobs.size(); ++i )
    {
      Blob<Dtype>* blob = blobs[i];
      std::string name = blob_finder_.NameFromPointer(blob);
      if (name == MeanName())
      {
        mean_index_ = i;
      }
      else if (name == VarianceName())
      {
        variance_index_ = i;
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
    if (HasVariance()) {
      CHECK(variance_index_ != -1) << "Mean blob " << VarianceName() <<
                                  " was specified in layer "
                                << layer_param_.name() << " but "
                                << "was not found in blobs.";
    }

    CHECK(data_index_ != -1) << "No data blob found for layer " <<
                                layer_param_.name();
  }

  // Get the number of top blobs this layer will have. It is the number of
  // output blobs if it is an MVNLayer. It is the number of input blobs if
  // it is an InverseMVNLayer.
  int NumBlobs() const
  {
//    CHECK(layer_param_.has_mvn_param()) << "MVNLayer has no parameter.";
    int num = 1;
    if ( HasMean() )
      num++;
    if ( HasVariance() )
      num++;
    return num;
  }

  // Get the blob that has the mean of each of the inputs to the MVNLayer.
  Blob<Dtype>* MeanBlob( const BlobVec& blobs )
  {
    LazyInit(blobs);
    CHECK(HasMean()) << "Layer " << layer_param_.name() << " has no "
                        << "mean blob.";
    return blobs[mean_index_];
  }

  // Get the blob that has the scales by which each input to the MVNLayer is
  // scaled by.
  Blob<Dtype>* VarianceBlob( const BlobVec& blobs )
  {
    LazyInit(blobs);
    CHECK(HasVariance()) << "Layer " << layer_param_.name() << " has no "
                      << "scale blob.";
    return blobs[variance_index_];
  }

  //
  Blob<Dtype>* DataBlob( const BlobVec& blobs )
  {
    LazyInit(blobs);
    CHECK(data_index_ < (int) blobs.size()) << "Invalid data blob index in MVNLayer "
                                    << this->layer_param_.name();
    return blobs[data_index_];
  }

  // Return indication if the layer exports the mean in the top blobs.
  bool HasMean() const
  {
    return MvnParam().has_mean_blob();
  }

  std::string MeanName() const
  {
    return MvnParam().mean_blob();
  }

  std::string VarianceName() const
  {
    return MvnParam().variance_blob();
  }

  // Return indication if the layer scales to a variance of one.
  bool HasVariance() const
  {
    return MvnParam().has_variance_blob();
  }

  // Get the value of the MVN layer's parameter.
  MVNParameter MvnParam() const
  {
    return layer_param_.mvn_param();
  }

  // The MVNLayer's parameter.
  LayerParameter layer_param_;
  int mean_index_;
  int variance_index_;
  int data_index_;
  BlobFinder<Dtype> blob_finder_;
  bool initialized_;
};

}
