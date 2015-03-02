#pragma once

#include "boost/function.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Type of a function that accepts training statistics from the Solver.
typedef boost::function<void(const caffe::TrainStats& stats)>
                                                            TrainStatsListener;
// Type of a function that accepts testing statistics from the Solver.
typedef boost::function<void(const caffe::TestStats& stats)> TestStatsListener;

}
