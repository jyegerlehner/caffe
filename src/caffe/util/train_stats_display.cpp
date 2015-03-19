#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/train_stats_display.hpp"

namespace caffe {

void TrainStatsDisplay::ReceiveTrainStats(const TrainStats& stats) {
  LOG(INFO) << "Iteration " << stats.iter()
                << ", loss = " << stats.loss();

  ostringstream loss_msg_stream;
  for(int i = 0; i < stats.stat_size(); ++i )
  {
    std::ostringstream loss_msg_stream;
    const TrainStat stat = stats.stat(i);
    if (stat.has_loss_weight()) {
      loss_msg_stream << " (* " << stat.loss_weight()
                      << " = " << stat.loss_weight() * stat.loss(i) << " loss)";
    }
    LOG(INFO) << "    Train net output #"
        << i << ": " << stat.blob_name() << " = "
        << stat.loss(i) << loss_msg_stream.str();
  }
}

void TrainStatsDisplay::ReceiveTestStats(const TestStats& stats)
{
   LOG(INFO) << "Test loss: " << stats.loss();
   for(int i=0; i < stats.stat_size(); ++i) {
     TestStat stat = stats.stat(i);
     ostringstream loss_msg_stream;

     if(stat.has_loss_weight())
     {
       loss_msg_stream << " (* " << stat.loss_weight()
            << " = " << stat.loss_weight() * stat.mean_score() << " loss)";
     }
     LOG(INFO) << "    Test net output #" << i << ": " << stat.blob_name()
               << " = " << stat.mean_score() << loss_msg_stream.str();
   }
}

}
