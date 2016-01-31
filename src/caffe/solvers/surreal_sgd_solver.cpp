#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
void UpdateMomentum(int count,
                    Dtype local_rate,
                    const Dtype* instantaneous_diffs,
                    Dtype momentum,
                    Dtype* accumulated_diffs)
{
  for(int i = 0; i < count; ++i)
  {
    Dtype accumulated_diff = accumulated_diffs[i];
    Dtype instaneous_diff = instantaneous_diffs[i];

//    if (accumulated_diff == 0)
//    {
//      Dtype instantaneous_contribution = instantaneous_diffs[i]*local_rate;
//      accumulated_diffs[i] = instantaneous_contribution;
//    }
//    else
    if ( instaneous_diff*accumulated_diff < 0 )
    {
      // Components along this axis of the instantaneous and accumulated
      // gradients point in opposite directions. So just zero it out.
      Dtype instantaneous_contribution = instantaneous_diffs[i]*local_rate;
      accumulated_diffs[i] = instantaneous_contribution;
    } else {
      // Accumulate in the usual way SGD+momentum way.
      Dtype instantaneous_contribution = instantaneous_diffs[i]*local_rate;
      accumulated_diffs[i] = instantaneous_contribution +
          momentum*accumulated_diff;
    }
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void surreal_sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

template <typename Dtype>
void SurrealSGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate)
{
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    UpdateMomentum<Dtype>(net_params[param_id]->count(), local_rate,
                   net_params[param_id]->cpu_diff(), momentum,
                   this->history_[param_id]->mutable_cpu_data());

    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());

    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    surreal_sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        this->history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

INSTANTIATE_CLASS(SurrealSGDSolver);
REGISTER_SOLVER_CLASS(SurrealSGD);
}
