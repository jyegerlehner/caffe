#include <stdexcept>

#include "caffe/common.hpp"
#include "caffe/layer_finder.hpp"

namespace caffe
{

template<typename Dtype>
bool LayerFinder<Dtype>::IsInitialized(Layer<Dtype>* layer_ptr) const
{
  return initialized_layers_.find(layer_ptr) != initialized_layers_.end();
}

template<typename Dtype>
void LayerFinder<Dtype>::LayerInitialized(Layer<Dtype>* layer_ptr)
{
  initialized_layers_.insert(layer_ptr);
}

template <typename Dtype>
void LayerFinder<Dtype>::AddLayer(
    const std::string& name, SharedLayerPtr layer_ptr ) {
  name_to_layer_map_[name] = layer_ptr;
}

template <typename Dtype>
typename LayerFinder<Dtype>::SharedLayerPtr LayerFinder<Dtype>::PointerFromName(const std::string& name) {
  if ( name == std::string("") || name.size() == 0)
  {
    throw std::runtime_error("Layer with no name: PointerFromName.");
  }
  return name_to_layer_map_[name];
}

template <typename Dtype>
bool LayerFinder<Dtype>::Exists(const std::string& name) {
  if ( name == std::string("") || name.size() == 0)
  {
    throw std::runtime_error("Layer with no name.");
  }
  return name_to_layer_map_.find(name) != name_to_layer_map_.end();
}

template <typename Dtype>
void LayerFinder<Dtype>::Clear() {
  name_to_layer_map_.clear();
  initialized_layers_.clear();
}

template <typename Dtype>
LayerFinder<Dtype>::LayerFinder():
  enabled_(false)
{
}

INSTANTIATE_CLASS(LayerFinder);

}
