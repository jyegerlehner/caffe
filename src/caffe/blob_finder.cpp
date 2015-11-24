#include <string>
#include <stdexcept>

#include "caffe/blob_finder.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
void BlobFinder<Dtype>::AddBlob(
    const std::string& name,SharedBlobPtr blob_ptr ) {
  blob_to_name_[blob_ptr] = name;
  name_to_blob_[name] = blob_ptr;
}

template <typename Dtype>
typename BlobFinder<Dtype>::SharedBlobPtr BlobFinder<Dtype>::PointerFromName(
                    const std::string& name) {
  return name_to_blob_[name];
}

template <typename Dtype>
std::string BlobFinder<Dtype>::NameFromPointer(
                      SharedBlobPtr blob_pointer) const {
  typename BlobToNameMap::const_iterator it = blob_to_name_.find(blob_pointer);
  if (it == blob_to_name_.end()) {
    CHECK(it != blob_to_name_.end()) << "Blob pointer not found in pointer-to-"
                                     << "name map.";
  }

  return it->second;
}

template <typename Dtype>
std::string BlobFinder<Dtype>::NameFromPointer(
                      Blob<Dtype>* blob_pointer) const {
  for(typename BlobToNameMap::const_iterator it = blob_to_name_.begin();
      it != blob_to_name_.end();
      ++it) {
    if (it->first.get() == blob_pointer)
    {
      return it->second;
    }
  }
  LOG(FATAL) << "Failed to find blob." << std::endl;
}


template <typename Dtype>
bool BlobFinder<Dtype>::Exists(const std::string& name) const {
  if ( name == std::string("") || name.size() == 0)
  {
    throw std::runtime_error("Blob with no name.");
  }
  return name_to_blob_.find(name) != name_to_blob_.end();
}

INSTANTIATE_CLASS(BlobFinder);

}  // namespace caffe
