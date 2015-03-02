#include "TrainStatsDisplay.h"

namespace caffe {

void TrainStatsDisplay::ReceiveTrainStats(const TrainStats& stats) {
  LOG(INFO) << "Iteration " << stats.get_iter()
                << ", loss = " << stats.get_loss();

  ostringstream loss_msg_stream;
  for(int i = 0; i < stats.get )
  if ( loss_weight) {
    loss_msg_stream << " (* " << loss_weight
                    << " = " << loss_weight * result_vec[k] << " loss)";
  }
  LOG(INFO) << "    Train net output #"
      << score_index++ << ": " << output_name << " = "
      << result_vec[k] << loss_msg_stream.str();
}

void TrainStatsDisplay::ReceiveTestStats(const TestStats& stats)
{

}

}
