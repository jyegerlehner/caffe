#include "TrainStatsDisplay.h"

namespace caffe {

void TrainStatsDisplay::ReceiveTrainStats(const TrainStats& stats) {
  LOG(INFO) << "Iteration " << stats.get_iter()
                << ", loss = " << stats.get_loss();

  for(int i = 0; i < stats.stat_size(); ++i )
  {
    ostringstream loss_msg_stream;
    TrainStat stat = stats.stat(i);
    for(int k = 0; k < stat.loss.size(); ++k)
    {
      if (loss_weight) {
        loss_msg_stream << " (* " << loss_weight
                 << " = " << stat.loss_weight * state.mean_loss(k) << " loss)";
      }
      LOG(INFO) << "    Train net output #"
          << i << ": " << stat.blob_name() << " = "
          << state.mean_loss(k) << loss_msg_stream.str();
    }
  }
}

void TrainStatsDisplay::ReceiveTestStats(const TestStats& stats)
{

}

}
