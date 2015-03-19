#include "caffe/util/train_stats_log.h"

namespace caffe {

TrainStatsLog::TrainStatsLog( bool create_log, const std::string& path_prefix ):
  stream_(),
  create_log_(create_log),
  path_prefix_(path_prefix)
{
}

}
