#pragma once
#include <fstream>
#include "caffe/util/stats_listener.hpp"

// Listens for training statistics
class TrainStatsLog
{
public:
  TrainStatsLog( bool create_log, const std::string& path_prefix );
  void ReceiveTrainStats(const TrainStats& stats);
  void ReceiveTestStats(const TestStats& stats);
private:
  // File stream to write to.
  std::ofstream stream_;
  bool create_log_;
  // Prefix where to place the log.
  std::string path_prefix_;
};
