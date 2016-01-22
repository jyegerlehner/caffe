#include <vector>

#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
//  if (CheckForNanGPU<Dtype>(top[0]->count(), top[0]->gpu_data()))
//  {
//    std::stringstream ss;
//    ss << "Found nan in conv layer top: " << this->layer_param_.name() << std::endl;
//    for(int i=0; i < this->blobs_.size(); ++i)
//    {
//      Blob<Dtype>* weight_blob = this->blobs_[i].get();
//      if (CheckForNanGPU(weight_blob->count(), weight_blob->gpu_data()))
//      {
//        ss << "Param blob had nan" << std::endl;
//      } else {
//        ss << "Param blob did not have nan" << std::endl;
//      }
//    }

//    {
//      std::fstream of;
//      of.open("nan_blob_o.csv");
//      PrintBlob(of, "BlobWithNan",*top[0]);
//    }

//    {
//      std::ofstream of;
//      of.open("nan_blob_input.csv");
//      PrintBlob(of, "BlobWithNan",*bottom[0]);
//    }

//    {
//      std::ofstream of;
//      of.open("nan_blob_weights.csv");
//      PrintBlob(of, "BlobWithNan",*(this->blobs_[0]));
//    }

//    {
//      std::ofstream of;
//      of.open("nan_blob_biases.csv");
//      PrintBlob(of, "BlobWithNan",*(this->blobs_[1]));
//    }

//    LOG(ERROR) << ss.str();
//  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
