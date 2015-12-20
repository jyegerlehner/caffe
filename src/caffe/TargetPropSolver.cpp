#include "caffe/TargetPropSolver.h"

#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe
{

template<typename Dtype>
TargetPropSolver<Dtype>::TargetPropSolver(
    const caffe::SolverParameter& solver_param ):
  param_(solver_param)
{
  //  solver(caffe::SolverRegistry<float>::CreateSolver(solver_param,
  //                                                    blob_finder,
  //                                                    layer_finder));
}

template<typename Dtype>
void TargetPropSolver<Dtype>::SetResumeFile( const std::string& filename)
{
  resume_file_ = filename;
}

template<typename Dtype>
void TargetPropSolver<Dtype>::SetWeightsFile(const std::string& filename)
{
  weights_file_ = filename;
}

template<typename Dtype>
void TargetPropSolver<Dtype>::SetActionFunction(ActionCallback action_callback)
{
  action_callback_ = action_callback;
}

template<typename Dtype>
void TargetPropSolver<Dtype>::InitLongLoopNet()
{
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating long loop net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating long loop net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating long loop net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating long loop net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    long_loop_net_.reset(new Net<Dtype>(net_param, blob_finder_, layer_finder_));
  } else {
    //    long_loop_net_.reset(new Net<Dtype>(net_param, blob_finder_, layer_finder_,
    //                              root_solver_->net_.get()));
    throw std::runtime_error("Parallel training not supported for target prop.");
  }

  // Parse/Create the dependent nets and their solvers:
  // for each encoder loop
  // {
  //   Copy the original solver parameter.
  for(int i = 0; i < net_param.encoder_loop_size(); ++i)
  {
    loop_solvers_.push_back(CreateDependentNetSolver(net_param.encoder_loop(i)));
  }

  for(int i = 0; i < net_param.decoder_loop_size(); ++i)
  {
    loop_solvers_.push_back(CreateDependentNetSolver(net_param.decoder_loop(i)));
  }

}

template<typename Dtype>
shared_ptr<Solver<Dtype> > TargetPropSolver<Dtype>::CreateDependentNetSolver(
    const NetParameter& loop_net_param )
{
  SolverParameter loop_solver_param = param_;
  // Clear out the old net that was set on param_ from the loop_solver_param
  loop_solver_param.clear_train_net_param();
  loop_solver_param.clear_train_net();
  loop_solver_param.clear_net_param();
  loop_solver_param.clear_net();
  loop_solver_param.clear_test_net();
  loop_solver_param.clear_test_net_param();

  // Copy the dependent net's param into the solver param.
  loop_solver_param.mutable_net_param()->CopyFrom(loop_net_param);
  shared_ptr<Solver<Dtype> > solver_ptr(
        SolverRegistry<Dtype>::CreateSolver(loop_solver_param,
          blob_finder_,
          layer_finder_));
  return solver_ptr;
}

//template<typename Dtype>
//void TargetPropSolver<Dtype>::ParseAndCreateLoopSolvers()
//{
//}

template<typename Dtype>
void TargetPropSolver<Dtype>::ToHDF5(const std::string& filename)
{
  hid_t file_hid = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << filename << " to save weights.";
  hid_t data_hid = H5Gcreate2(file_hid, "data", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error saving weights to " << filename << ".";

  // Save all the parameter blobs into the file.
  typedef std::vector<std::string> ParamNames;
  ParamNames param_names = blob_finder_.GetParamNames();
  for(ParamNames::const_iterator it = param_names.begin();
      it != param_names.end();
      ++it)
  {
    typename BlobFinder<Dtype>::SharedBlobPtr param_blob =
        blob_finder_.PointerFromName(*it);
    hdf5_save_nd_dataset<Dtype>(data_hid, (*it).c_str(), *param_blob);
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template<typename Dtype>
void TargetPropSolver<Dtype>::CopyTrainedLayersFromHDF5(
    const std::string& trained_filename)
{
  hid_t file_hid = H5Fopen(trained_filename.c_str(), H5F_ACC_RDONLY,
                           H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open " << trained_filename;
  hid_t data_hid = H5Gopen2(file_hid, "data", H5P_DEFAULT);
  CHECK_GE(data_hid, 0) << "Error reading weights from " << trained_filename;
  int num_param_blobs = hdf5_get_num_links(data_hid);

  for(int i =0; i < num_param_blobs; ++i)
  {
    string source_param_name = hdf5_get_name_by_idx(data_hid, i);
    if(!blob_finder_.Exists(source_param_name))
    {
      DLOG(INFO) << "Ignoring source param " << source_param_name;
      continue;
    }
    DLOG(INFO) << "Copying source param " << source_param_name;

    shared_ptr<Blob<Dtype> > target_blob =
        blob_finder_.PointerFromName(source_param_name);

    hdf5_load_nd_dataset(data_hid, source_param_name.c_str(), 0, kMaxBlobAxes,
        target_blob.get());
  }
  H5Gclose(data_hid);
  H5Fclose(file_hid);
}

template<typename Dtype>
void TargetPropSolver<Dtype>::LoadWeights()
{
  if (weights_file_.size() >= 3 &&
      weights_file_.compare(weights_file_.size() - 5, 5, ".tph5") == 0) {
    CopyTrainedLayersFromHDF5(weights_file_);
  } else {
    //CopyTrainedLayersFromBinaryProto(weights_file_);
    throw std::runtime_error("Only support loading target prop models from hdf5");
  }
}

template<typename Dtype>
void TargetPropSolver<Dtype>::Run(const std::vector<int>& gpus)
{
  // Load the long loop net
  InitLongLoopNet();

  CHECK(resume_file_.empty()) << "Resuming target prop solver not supported.";

  // Load layers from weights.
  if(weights_file_.size() > 0)
  {
    LoadWeights();
  }


  // Forward prop the long loop net


  //while more iterations
  //{
  //    for each encoder loop solver:
  //      Step iters
  //
  //    for each decoder loop solver:
  //      Step iters
  //}
}

INSTANTIATE_CLASS(TargetPropSolver);

}
