#pragma once
#include <map>
#include <string>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"

namespace caffe {

typedef std::vector<string> ParamNames;

template <typename Dtype>
struct BlobFinder {
  typedef shared_ptr< Blob<Dtype> > SharedBlobPtr;
  typedef std::map<SharedBlobPtr, std::string> BlobToNameMap;
  typedef std::map<std::string, SharedBlobPtr> NameToBlobMap;
  typedef boost::shared_ptr<BlobFinder<Dtype> > Ptr;
  typedef std::map<std::string, bool> ParamActivationMap;

  void AddParamBlob(const std::string& name, SharedBlobPtr blob_ptr);
  void AddActivationBlob( const std::string& name, SharedBlobPtr blob_ptr);
  SharedBlobPtr PointerFromName(const std::string& name);
  std::string NameFromPointer(SharedBlobPtr blob_pointer) const;
  std::string NameFromPointer(Blob<Dtype>* blob_pointer) const;
  ParamNames GetParamNames() const;

  bool Exists(const std::string& name) const;
 private:
  BlobToNameMap blob_to_name_;
  NameToBlobMap name_to_blob_;
  ParamActivationMap is_param_map_;
};

}  // namespace caffe
