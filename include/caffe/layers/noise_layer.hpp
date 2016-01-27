#ifndef NOISE_LAYER_HPP
#define NOISE_LAYER_HPP

#include "caffe/layers/neuron_layer.hpp"

namespace caffe
{

/**
 * @brief NoiseLayer for adding noise to a blob.
 */
template <typename Dtype>
class NoiseLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides NoiseParameter noise_param.
   *
   */
  explicit NoiseLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param), inplace_(false) {}
  virtual inline const char* type() const { return "Noise"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the noised output.
   *        y = \max(0, x)
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with the top diff.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  const NoiseParameter& NoiseParam() const {
    return this->layer_param_.noise_param();
  }

  const FillerParameter& FillerParam() const {
    return NoiseParam().filler_param();
  }

  std::string NoiseType() const {
    const FillerParameter& filler_param = this->NoiseParam().filler_param();
    return filler_param.type();
  }

  // Returns true if we are noising in-place. I.e. the top and bottom blob are
  // the same.
  bool Inplace() const {
    return inplace_;
  }

  // Buffer for noise used when top and bottom blob are the same.
  Blob<Dtype> inplace_noise_;
  // True iff bottom and top blob are the same.
  bool inplace_;

  static const std::string GAUSSIAN;
  static const std::string UNIFORM;
};

}

#endif // NOISE_LAYER_HPP
