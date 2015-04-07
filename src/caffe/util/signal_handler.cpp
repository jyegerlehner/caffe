#include <boost/bind.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/thread.hpp>
#include <glog/logging.h>

#include <signal.h>

#include "caffe/util/signal_handler.h"

namespace {
  boost::mutex signal_mutex;
  bool got_sigint = false;
  bool got_sighup = false;
  bool already_hooked_up = false;

  void handle_signal(int signal) {
    boost::mutex::scoped_lock scoped_lock(signal_mutex);
    switch (signal) {
    case SIGHUP:
      got_sighup = true;
      break;
    case SIGINT:
      got_sigint = true;
      break;
    default:
      LOG(FATAL) << "Caught wrong signal.";
    }
  }

  void HookupHandler() {
    boost::mutex::scoped_lock scoped_lock(signal_mutex);

    if (already_hooked_up) {
      LOG(FATAL) << "Tried to hookup signal handlers more than once.";
    }
    already_hooked_up = true;

    struct sigaction sa;
    // Setup the sighub handler
    sa.sa_handler = &handle_signal;
    // Restart the system call, if at all possible
    sa.sa_flags = SA_RESTART;
    // Block every signal during the handler
    sigfillset(&sa.sa_mask);
    // Intercept SIGHUP and SIGINT
    if (sigaction(SIGHUP, &sa, NULL) == -1) {
      LOG(FATAL) << "Cannot install SIGHUP handler.";
    }
    if (sigaction(SIGINT, &sa, NULL) == -1) {
      LOG(FATAL) << "Cannot install SIGINT handler.";
    }
  }

  // Return true iff a SIGINT has been received since the last time this
  // function was called.
  bool GotSIGINT() {
    boost::mutex::scoped_lock scoped_lock(signal_mutex);
    bool result = got_sigint;
    got_sigint = false;
    return result;
  }

  // Return true iff a SIGHUP has been received since the last time this
  // function was called.
  bool GotSIGHUP() {
    boost::mutex::scoped_lock scoped_lock(signal_mutex);
    bool result = got_sighup;
    got_sighup = false;
    return result;
  }
}  // namespace

namespace caffe {

SignalHandler::SignalHandler(SolverParameter_Action SIGINT_action,
                             SolverParameter_Action SIGHUP_action):
  SIGINT_action_(SIGINT_action),
  SIGHUP_action_(SIGHUP_action) {
  HookupHandler();
}

SolverParameter_Action SignalHandler::CheckForSignals() const {
  if (GotSIGHUP()) {
    return SIGHUP_action_;
  }
  if (GotSIGINT()) {
    return SIGINT_action_;
  }
  return SolverParameter_Action_NONE;
}

// Return the function that the solver can use to find out if a snapshot or
// early exit is being requested.
ActionCallback SignalHandler::GetActionFunction() {
  return boost::bind(&SignalHandler::CheckForSignals, this);
}

}  // namespace caffe
