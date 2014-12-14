#include "blob_info.hpp"

namespace caffe {

void BlobInfo::AddBlob( const std::string& name, Blob<Dtype>* blob_ptr )
{
  blob_to_name_[blob_ptr] = name;
  name_to_blob_[name] = blob_ptr;
}

Blob<Dtype>* BlobInfo::PointerFromName( const std::string& name ) {
  return name_to_blob_[name];
}

std::string BlobInfo::NameFromPointer( const Blob<Dtype>* blob_pointer ) {
  return blob_to_name_[blob_pointer];
}

bool BlobInfo::Exists(const std::string& name) {
  return name_to_blob_.find(name) != name_to_blob_.end();
}

INSTANTIATE_CLASS(BlobInfo);

} // namespace caffe
