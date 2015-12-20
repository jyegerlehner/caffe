#include <string>
#include <stdexcept>

#include "caffe/blob_finder.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"

namespace caffe {

template <typename Dtype>
void BlobFinder<Dtype>::AddParamBlob(
    const std::string& name,SharedBlobPtr blob_ptr ) {
  if( name == Net<Dtype>::AUTOMATIC_BLOB_NAME )
  {
    throw std::runtime_error("Adding automatic blo      b to blob finder.");
  }
  blob_to_name_[blob_ptr] = name;
  name_to_blob_[name] = blob_ptr;
  is_param_map_[name] = true;
}

template <typename Dtype>
void BlobFinder<Dtype>::AddActivationBlob( const std::string& name,
                                           SharedBlobPtr blob_ptr) {
  if( name == Net<Dtype>::AUTOMATIC_BLOB_NAME )
  {
    throw std::runtime_error("Adding automatic blob to blob finder.");
  }
  blob_to_name_[blob_ptr] = name;
  name_to_blob_[name] = blob_ptr;
  is_param_map_[name] = false;
}


template <typename Dtype>
typename BlobFinder<Dtype>::SharedBlobPtr BlobFinder<Dtype>::PointerFromName(
                    const std::string& name) {
  std::cout << "BlobFinder::PointerFromName: " << name << std::endl;
  if ( name == Net<Dtype>::AUTOMATIC_BLOB_NAME)
  {
    throw std::runtime_error("tried to retrieve automatic blob by name.");
  }
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
  if ( name == Net<Dtype>::AUTOMATIC_BLOB_NAME)
  {
    return false;
  }
  return name_to_blob_.find(name) != name_to_blob_.end();
}

template <typename Dtype>
ParamNames BlobFinder<Dtype>::GetParamNames() const
{
  ParamNames result;
  for(typename NameToBlobMap::const_iterator it = name_to_blob_.begin();
      it != name_to_blob_.end();
      ++it)
  {
    std::string name = it->first;
    ParamActivationMap::const_iterator pit = is_param_map_.find(name);
    if (pit != is_param_map_.end())
    {
      bool is_param = pit->second;
      if (is_param)
      {
        result.push_back(it->first);
      }
    }
  }
  return result;
}

INSTANTIATE_CLASS(BlobFinder);

}  // namespace caffe
