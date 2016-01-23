#include "caffe/TargetPropSolver.h"

#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe
{

template<typename Dtype>
TargetPropSolver<Dtype>::TargetPropSolver(
    const caffe::SolverParameter& solver_param ):
  param_(solver_param),
  requested_early_exit_(false)
{
  // Load the long loop net
  InitLongLoopNet();
  // Create the dependent nets and their solvers.
  InitDependentNets();
}

template<typename Dtype>
void TargetPropSolver<Dtype>::SetResumeFile( const std::string& filename)
{
  resume_file_ = filename;
}

//template<typename Dtype>
//void TargetPropSolver<Dtype>::SetWeightsFile(const std::string& filename)
//{
//  weights_file_ = filename;
//}

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

  bool has_net_file = param_.has_net();
  bool has_net_param = param_.has_net_param();
  CHECK(has_net_file || has_net_param)
      << "Solver does not specify a net prototxt file "
                            << " in its net parameter.";
  LOG_IF(INFO, Caffe::root_solver())
      << "Creating long loop net from net file: " << param_.net();
  if (has_net_file)
    ReadNetParamsFromTextFileOrDie(param_.net(), &top_net_param_);
  else if (has_net_param)
    top_net_param_.CopyFrom(param_.net_param());
  else
    throw std::runtime_error("Need either net param or net file.");

  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(top_net_param_.state());
  net_state.MergeFrom(param_.train_state());
  top_net_param_.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    this->long_loop_solver_ =
        boost::shared_ptr<Solver<Dtype> >(
                caffe::SolverRegistry<Dtype>::CreateSolver(param_,
                                                      blob_finder_,
                                                      layer_finder_));
  } else {
    //    long_loop_net_.reset(new Net<Dtype>(net_param, blob_finder_, layer_finder_,
    //                              root_solver_->net_.get()));
    throw std::runtime_error("Parallel training not supported for target prop.");
  }
}

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

template <typename Dtype>
string TargetPropSolver<Dtype>::SnapshotFilename(const string extension) {
  string filename(param_.snapshot_prefix());
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", GetIter());
  return filename + iter_str_buffer + extension;
}

template<typename Dtype>
void TargetPropSolver<Dtype>::Snapshot()
{
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  this->ToHDF5(model_filename);
}

template<typename Dtype>
void TargetPropSolver<Dtype>::LoadWeights(const std::string& weights_file)
{
  if (weights_file.size() >= 3)
    CopyTrainedLayersFromHDF5(weights_file);
  else
    //CopyTrainedLayersFromBinaryProto(weights_file_);
    throw std::runtime_error("Only support loading target prop models from hdf5");
}

// Take the original solver, and make one of
template<typename Dtype>
void TargetPropSolver<Dtype>::FillDependentSolverParam(
    SolverParameter& solver_param,
    const NetParameter& dependent_net_param)
{
  solver_param.CopyFrom(param_);
  solver_param.clear_net();
  solver_param.clear_net_param();
  solver_param.clear_test_net();
  solver_param.clear_test_net_param();
  solver_param.clear_train_net_param();
  solver_param.clear_test_interval();
  solver_param.clear_test_iter();
  solver_param.mutable_train_net_param()->CopyFrom(dependent_net_param);
}

template<typename Dtype>
void TargetPropSolver<Dtype>::InitDependentNets()
{
  for (int i = 0; i < top_net_param_.dependent_net_size(); ++i)
  {
    SolverParameter solver_param;
    FillDependentSolverParam(solver_param, top_net_param_.dependent_net(i));

    // Now we've created a solver param whose net is this encoder loop's
    // net. Instantiate the solver and add to the list.
    shared_ptr<caffe::Solver<Dtype> >
        solver(caffe::SolverRegistry<Dtype>::CreateSolver( solver_param,
                                                          blob_finder_,
                                                          layer_finder_));
    loop_solvers_.push_back(solver);
  }

//  for(int i = 0; i < top_net_param_.dependent_net_size(); ++i)
//  {
//    loop_solvers_.push_back(CreateDependentNetSolver(
//                              top_net_param_.dependent_net(i)));
//  }


}

template<typename Dtype>
SolverAction::Enum TargetPropSolver<Dtype>::GetRequestedAction()
{
  if (action_callback_)
  {
    return action_callback_();
  }
  return SolverAction::NONE;
}

template<typename Dtype>
void TargetPropSolver<Dtype>::TestLongLoopNet(int iter, Dtype& loss)
{
  Dtype last_iter_loss = 0;
  CHECK(Caffe::root_solver());
//  LOG(INFO) << "Iteration " << iter
//            << ", Testing long loop net " << long_loop_net_->name();
//  CHECK_NOTNULL(test_nets_[test_net_id].get())->
//      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = LongLoopNet();
  loss = 0;
  for (int i = 0; i < param_.test_iter(0); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
    if (last_iter_loss == 0 )
      last_iter_loss = iter_loss;
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(0);
    LOG(INFO) << "Long loop test loss: " << loss;
  }

  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(0);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
  if (loss == 0)
    loss = last_iter_loss;
}

template<typename Dtype>
void TargetPropSolver<Dtype>::Run(const std::vector<int>& gpus, Dtype& loss)
{
  CHECK(resume_file_.empty()) << "Resuming target prop solver not supported.";

//  // Load layers from weights.
//  if(weights_file_.size() > 0)
//  {
//    LoadWeights();
//  }

  {
    // Forward prop the long loop net to generate loss.
    Dtype loss;
    LongLoopNet()->ForwardPrefilled(&loss);
    LOG(INFO) << "Long loop loss: " << loss;
  }

  
  // Run each solver on a minibatch.
  while (GetIter() < param_.max_iter() && !requested_early_exit_)
  {
    // Forward prop the first layer in the long-loop net since that's the one
    // that gets the data from the training database.
    //LongLoopNet()->ForwardFromTo(0,0);
    long_loop_solver_->Step(1,true);

    // Run each solver 1 iter.
    for(int solver_index = 0;
        solver_index < loop_solvers_.size();
        ++solver_index)
    {
      loop_solvers_[solver_index]->Step(1, true);
    }

    SolverAction::Enum request = GetRequestedAction();
    if (request == SolverAction::STOP)
    {
      requested_early_exit_ = true;
    }

    bool do_test = (param_.test_interval() != 0);
    if( param_.has_test_interval())
    {
      if ( ((do_test = GetIter() % param_.test_interval()) == 0))
      {
        TestLongLoopNet(param_.test_iter(0), loss);
      }
    }

    int last_index = loop_solvers_.size()-1;
    if(loop_solvers_[last_index]->ShouldDisplay())
    {
      LOG(INFO) << "Iteration " << loop_solvers_[last_index]->iter();
//                << ", lr=" << loop_solvers_[last_index]->GetLearningRate();
    }
  }
  TestLongLoopNet(1, loss);

  if (param_.snapshot_after_train())
  {
    Snapshot();
  }
}

template<typename Dtype>
int TargetPropSolver<Dtype>::GetIter() const
{
  CHECK(loop_solvers_.size() > 0) << "Empty loop solvers vector.";
  return loop_solvers_[0]->iter();
}

template<typename Dtype>
typename BlobFinder<Dtype>::SharedBlobPtr TargetPropSolver<Dtype>::BlobByName(
    const std::string blob_name)
{
  return blob_finder_.PointerFromName(blob_name);
}

template<typename Dtype>
shared_ptr<Blob<Dtype> >  TargetPropSolver<Dtype>::ForwardLongLoop()
{
  std::vector<Blob<Dtype>* > bottom_vec;
  (void) LongLoopNet()->Forward(bottom_vec);
  shared_ptr<Blob<Dtype> > blob = LongLoopNet()->blob_by_name("h0_hat");
  return blob;
}

template<typename Dtype>
shared_ptr<Blob<Dtype> >  TargetPropSolver<Dtype>::ForwardLoops()
{
  Dtype loss;
  std::vector<Blob<Dtype>* > bottom_vec;
  for( int i =0; i < loop_solvers_.size(); ++i)
  {
    (void) loop_solvers_[i]->net()->Forward(bottom_vec, &loss);
  }
  int solver_index = loop_solvers_.size()-1;
  shared_ptr<Blob<Dtype> > blob = loop_solvers_[solver_index]->net()->blob_by_name("h0_hat");

  //PrintBlob("h0_hat, short loops", *blob);
  return blob;
}

template<typename Dtype>
void ShowNetsBlobPointer(const std::string& blob_name,
                         const Net<Dtype>& net )
{
  if (net.has_blob(blob_name))
  {
    std::cout << "net " << net.name()
                 << " blob " << blob_name
                 << " pointer==" << net.blob_by_name(blob_name)
                    << std::endl;
  }
  else
  {
    std::cout << "net " << net.name()
                 << " has no blob named " << blob_name << std::endl;
  }

}

template<typename Dtype>
void TargetPropSolver<Dtype>::ShowBlobPointers(const std::string& blob_name)
{
  ShowNetsBlobPointer<Dtype>(blob_name, *LongLoopNet());
  for( int i = 0; i < loop_solvers_.size(); ++i)
  {
    ShowNetsBlobPointer<Dtype>(blob_name, *loop_solvers_[i]->net());
  }
}

template<typename Dtype>
boost::shared_ptr<Net<Dtype> > TargetPropSolver<Dtype>::LongLoopNet()
{
  return this->long_loop_solver_->net();
}

INSTANTIATE_CLASS(TargetPropSolver);

}
