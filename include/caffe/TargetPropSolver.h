#ifndef TARGETPROPSOLVER_H
#define TARGETPROPSOLVER_H

#include <caffe/common.hpp>
#include <caffe/solver.hpp>
#include <caffe/solver_factory.hpp>
#include <vector>

namespace caffe
{

template<typename Dtype>
class TargetPropSolver
{
public:
  TargetPropSolver(const caffe::SolverParameter& param);
  void SetResumeFile(const std::string& filename);
//  void SetWeightsFile(const std::string& filename);
  void Run(const std::vector<int>& gpus, Dtype& loss);
  void SetActionFunction(ActionCallback action_callback);
  typename BlobFinder<Dtype>::SharedBlobPtr BlobByName(const std::string blob_name);
  shared_ptr<Blob<Dtype> > ForwardLongLoop();
  shared_ptr<Blob<Dtype> > ForwardLoops();
  void ShowBlobPointers(const std::string& blob_name);
  void LoadWeights(const std::string& weights_file);

protected:
  void InitLongLoopNet();
//  shared_ptr<Solver<Dtype> > CreateDependentNetSolver(
//      const NetParameter& loop_net_param );
  void ParseAndCreateLoopSolvers();
  void ToHDF5(const std::string& filename);
  void CopyTrainedLayersFromHDF5(const std::string& trained_filename);
  bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param);
  void InitDependentNets();
  void FillDependentSolverParam(SolverParameter& solver_param,
                         const NetParameter& dependent_net_param);
  void TestLongLoopNet(int iter, Dtype& loss);
  SolverAction::Enum GetRequestedAction();
  void Snapshot();
  string SnapshotFilename(const string extension);

  BlobFinder<Dtype> blob_finder_;
  LayerFinder<Dtype> layer_finder_;
  // Has a separate solver for each dependent net (AKA identity loop).
  std::vector<boost::shared_ptr<Solver<Dtype> > > loop_solvers_;
  ActionCallback action_callback_;
  // This will be run forward prop only as the "test" net to get
  // current autoencoding performance.
  shared_ptr<Net<Dtype> > LongLoopNet();
  NetParameter top_net_param_;
  boost::shared_ptr<Solver<Dtype> > long_loop_solver_;
  SolverParameter param_;
  std::string resume_file_;
  bool requested_early_exit_;
  int GetIter() const;
};

}

#endif // TARGETPROPSOLVER_H
