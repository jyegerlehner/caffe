#pragma once

#include "caffe/proto/caffe.pb.h"

namespace caffe
{

class TrainStatsDisplay
{
public:
  void ReceiveTrainStats(const TrainStats& stats);
  void ReceiveTestStats(const TestStats& stats);
};

}
