#pragma once
#include <boost/shared_ptr.hpp>
#include <map>
#include <string>

#include "caffe/layer.hpp"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
class LayerFinder
{
public:
  typedef shared_ptr<Layer<Dtype> > SharedLayerPtr;
  typedef std::map<std::string, SharedLayerPtr > NameToBlobMap;

  LayerFinder();
  void AddLayer( const std::string& name, SharedLayerPtr layer_ptr);
  SharedLayerPtr PointerFromName(const std::string& name);
  bool Exists(const std::string& name);
  void Clear();
private:
  NameToBlobMap name_to_layer_map_;
  bool enabled_;
};
}
