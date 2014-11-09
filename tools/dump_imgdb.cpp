#include <boost/filesystem.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include "caffe/util/io.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/dataset_factory.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;
using caffe::Dataset;
using caffe::DatasetFactory;
using caffe::Datum;
using caffe::shared_ptr;
using namespace boost::filesystem;

cv::Mat DatumToCvMat( const Datum& datum )
{
  int img_type;
  switch( datum.channels() )
  {
  case 1:
      img_type = CV_8UC1;
      std::cout << "1" << std::endl;
      break;
  case 2:
      img_type = CV_8UC2;
      std::cout << "2" << std::endl;
      break;
  case 3:
      img_type = CV_8UC3;
      std::cout << "3" << std::endl;
      break;
  default:
      CHECK(false) << "Invalid number of channels.";
      break;
  }

  cv::Mat mat( datum.height(), datum.width(), img_type );
//  cvCreateData( &mat );

//  CvMat* mat_p = cvCreateMat( datum.height(), datum.width(), img_type );
  int datum_channels = datum.channels();
  int datum_height = datum.height();
  int datum_width = datum.width();

  for (int h = 0; h < datum_height; ++h) {
    uchar* ptr = mat.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        ptr[img_index++] = datum.data()[datum_index];
      }
    }
  }
  return mat;
}

void dump_dataset( const std::string& input_dir,
                   const std::string& output_dir,
                   const std::string& num )
{
  path output_path( output_dir );

  // The lmdb database is at the input_dir.
  shared_ptr<Dataset<string, Datum> > dataset =
                        DatasetFactory<string, Datum>( "lmdb" );
  // Open db.
  CHECK(dataset->open( input_dir, Dataset<string, Datum>::ReadOnly ) );

  int num_to_write = atoi( num.c_str() );
  int ctr = 0;
  // For each datum in the db, write it to a jpg file.
  for (Dataset<std::string, Datum>::const_iterator iter = dataset->begin();
      iter != dataset->end() && ctr++ < num_to_write; ++iter)
  {
    const Datum& datum = iter->value;
    cv::Mat cvmat = DatumToCvMat( datum );
    std::string keystring = iter->key;
    path output_file_path = output_path;
    output_file_path /= keystring;
    output_file_path.replace_extension( ".jpg" );
    std::cout << "output file: " << output_file_path.string() << std::endl;
    bool result = cv::imwrite( output_file_path.string(), cvmat );
  }
}

int main(int argc, char** argv)
{
  srand(time(NULL));
  if (argc != 4 )
  {
    printf("This util dumps the contents of an lmdb database out as jpg files."
           "\nUsage:\n"
           "dump_imgdb input_dir output_dir number_to_dump\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    dump_dataset(string(argv[1]), string(argv[2]), string(argv[3]) );
  }
  return 0;
}
