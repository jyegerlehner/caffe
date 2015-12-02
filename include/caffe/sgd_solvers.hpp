#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/blob_finder.hpp"
#include "caffe/solver.hpp"

namespace caffe {

/**
 * @brief Optimizes the parameters of a Net using
 *        stochastic gradient descent (SGD) with momentum.
 */
template <typename Dtype>
class SGDSolver : public Solver<Dtype> {
 public:
  typedef std::map<int,string> ParamNameMap;
  explicit SGDSolver(const SolverParameter& param,
                     BlobFinder<Dtype>& blob_finder)
      : Solver<Dtype>(param) { PreSolve(blob_finder); }
  explicit SGDSolver(const string& param_file,
                     BlobFinder<Dtype>& blob_finder)
      : Solver<Dtype>(param_file) { PreSolve(blob_finder); }
  virtual inline const char* type() const { return "SGD"; }

  const vector<shared_ptr<Blob<Dtype> > >& history() { return history_; }
  ParamNameMap CreateParamNameMap() const;
  static std::string Name() { return "SGD"; }
  static Solver<Dtype>* Make( const SolverParameter& param,
                              BlobFinder<Dtype>& blob_finder) {
    return new SGDSolver<Dtype>(param, blob_finder);
  }

 protected:
  void GetOrCreateBlob( BlobFinder<Dtype>& blob_finder,
                       const std::string& base_name,
                        const vector<int>& shape,
                        vector<shared_ptr<Blob<Dtype> > >& vect,
                        const std::string& suffix);
  void PreSolve(BlobFinder<Dtype>& blob_finder);
  Dtype GetLearningRate();
  virtual void ApplyUpdate();
  virtual void Normalize(int param_id);
  virtual void Regularize(int param_id);
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  virtual void ClipGradients();
  virtual void SnapshotSolverState(const string& model_filename);
  virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
  virtual void SnapshotSolverStateToHDF5(const string& model_filename);
  virtual void RestoreSolverStateFromHDF5(const string& state_file);
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
  // history maintains the historical momentum data.
  // update maintains update related data and is not needed in snapshots.
  // temp maintains other information that might be needed in computation
  //   of gradients/updates and is not needed in snapshots
  vector<shared_ptr<Blob<Dtype> > > history_, update_, temp_;

  DISABLE_COPY_AND_ASSIGN(SGDSolver);
};

template <typename Dtype>
class NesterovSolver : public SGDSolver<Dtype> {
 public:
  explicit NesterovSolver(const SolverParameter& param,
                          BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param, blob_finder) {}
  explicit NesterovSolver(const string& param_file,
                          BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param_file, blob_finder) {}
  virtual inline const char* type() const { return "Nesterov"; }
  static std::string Name() { return "Nesterov"; }
  static Solver<Dtype>* Make( const SolverParameter& param,
                              BlobFinder<Dtype>& blob_finder) {
    return new NesterovSolver<Dtype>(param, blob_finder);
  }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(NesterovSolver);
};

template <typename Dtype>
class AdaGradSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaGradSolver(const SolverParameter& param, BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param, blob_finder) { constructor_sanity_check(); }
  explicit AdaGradSolver(const string& param_file, BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param_file, blob_finder) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "AdaGrad"; }
  static std::string Name() { return "AdaGrad"; }
  static Solver<Dtype>* Make( const SolverParameter& param,
                              BlobFinder<Dtype>& blob_finder) {
    return new AdaGradSolver<Dtype>(param, blob_finder);
  }
 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with AdaGrad.";
  }

  DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
};


template <typename Dtype>
class RMSPropSolver : public SGDSolver<Dtype> {
 public:
  explicit RMSPropSolver(const SolverParameter& param,
                         BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param, blob_finder) { constructor_sanity_check(); }
  explicit RMSPropSolver(const string& param_file,
                         BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param_file, blob_finder) { constructor_sanity_check(); }
  virtual inline const char* type() const { return "RMSProp"; }
  static std::string Name() { return "RMSProp"; }
  static Solver<Dtype>* Make( const SolverParameter& param,
                              BlobFinder<Dtype>& blob_finder) {
    return new RMSPropSolver<Dtype>(param, blob_finder);
  }

 protected:
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  void constructor_sanity_check() {
    CHECK_EQ(0, this->param_.momentum())
        << "Momentum cannot be used with RMSProp.";
    CHECK_GE(this->param_.rms_decay(), 0)
        << "rms_decay should lie between 0 and 1.";
    CHECK_LT(this->param_.rms_decay(), 1)
        << "rms_decay should lie between 0 and 1.";
  }

  DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
};

template <typename Dtype>
class AdaDeltaSolver : public SGDSolver<Dtype> {
 public:
  explicit AdaDeltaSolver(const SolverParameter& param,
                          BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param, blob_finder ) { AdaDeltaPreSolve(); }
  explicit AdaDeltaSolver(const string& param_file,
                          BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param_file, blob_finder) { AdaDeltaPreSolve(); }
  virtual inline const char* type() const { return "AdaDelta"; }

  static std::string Name() { return "AdaDelta"; }
  static Solver<Dtype>* Make( const SolverParameter& param,
                              BlobFinder<Dtype>& blob_finder) {
    return new AdaDeltaSolver<Dtype>(param, blob_finder);
  }
 protected:
  void AdaDeltaPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
};

/**
 * @brief AdamSolver, an algorithm for first-order gradient-based optimization
 *        of stochastic objective functions, based on adaptive estimates of
 *        lower-order moments. Described in [1].
 *
 * [1] D. P. Kingma and J. L. Ba, "ADAM: A Method for Stochastic Optimization."
 *     arXiv preprint arXiv:1412.6980v8 (2014).
 */
template <typename Dtype>
class AdamSolver : public SGDSolver<Dtype> {
 public:
  explicit AdamSolver(const SolverParameter& param,
                      BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param, blob_finder) { AdamPreSolve();}
  explicit AdamSolver(const string& param_file,
                      BlobFinder<Dtype>& blob_finder)
      : SGDSolver<Dtype>(param_file, blob_finder) { AdamPreSolve(); }
  virtual inline const char* type() const { return "Adam"; }
  static std::string Name() { return "Adam"; }
  static Solver<Dtype>* Make( const SolverParameter& param,
                              BlobFinder<Dtype>& blob_finder) {
    return new AdamSolver<Dtype>(param, blob_finder);
  }

 protected:
  void AdamPreSolve();
  virtual void ComputeUpdateValue(int param_id, Dtype rate);

  DISABLE_COPY_AND_ASSIGN(AdamSolver);
};

}  // namespace caffe

#endif  // CAFFE_SGD_SOLVERS_HPP_
