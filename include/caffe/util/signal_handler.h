#pragma once
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

namespace caffe {

class SignalHandler
{
public:
  // Contructor. Specify what action to take when a signal is received.
  SignalHandler(SolverParameter_Action SIGINT_action,
                SolverParameter_Action SIGHUP_action );
  ActionCallback GetActionFunction();
private:
  SignalHandler(); // Not implemented.
  SolverParameter_Action CheckForSignals() const;
  SolverParameter_Action SIGINT_action_;
  SolverParameter_Action SIGHUP_action_;
};

} // namespace caffe
