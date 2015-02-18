#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/filesystem.hpp>
#include <fstream>  // NOLINT(readability/streams)
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"

using std::string;
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
      break;
  case 2:
      img_type = CV_8UC2;
      break;
  case 3:
      img_type = CV_8UC3;
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

namespace NoiseType
{
  enum Enum { SALT_AND_PEPPER, GAUSSIAN, NONE };
}

NoiseType::Enum GetNoiseType( int percent_sp, int percent_gauss )
{
  int val = rand() % 100;
  if ( val >= 0 && val < percent_sp )
  {
    return NoiseType::SALT_AND_PEPPER;
  }
  else if ( val >= percent_sp && val < percent_gauss + percent_sp )
  {
    return NoiseType::GAUSSIAN;
  }
  else
  {
    return NoiseType::NONE;
  }
}

struct PixelChoice
{
  int x;
  int y;

  static PixelChoice CreateRandom( int height, int width )
  {
    PixelChoice result;
    result.x = rand() % width;
    result.y = rand() % height;
    return result;
  }

  bool operator<( const PixelChoice& other ) const
  {
    if ( x < other.x)
      return true;
    else if ( x == other.x && y < other.y )
      return true;
    return false;
  }
};

cv::Mat SaltAndPepper( const cv::Mat& input_mat, int percent_pix_sp )
{
  typedef std::set<PixelChoice> PixSet;
  PixSet pixels_to_corrupt;
  int pixel_count = input_mat.rows * input_mat.cols;
  int pixel_count_to_corrupt = pixel_count * percent_pix_sp / 100;
  while ( pixels_to_corrupt.size() < pixel_count_to_corrupt )
  {
    pixels_to_corrupt.insert( PixelChoice::CreateRandom( input_mat.rows,
                                                         input_mat.cols));
  }

  cv::Mat result;
  input_mat.assignTo(result);
  for( PixSet::iterator pixit = pixels_to_corrupt.begin();
       pixit != pixels_to_corrupt.end();
       ++pixit )
  {
    cv::Vec3b& pel = result.at<cv::Vec3b>(pixit->x,pixit->y);
    int white_or_black = rand() % 2;
    if ( white_or_black == 0 )
    {
      pel[0] = 255;
      pel[1] = 255;
      pel[2] = 255;
    }
    else
    {
      pel[0] = 0;
      pel[1] = 0;
      pel[2] = 0;
    }
  }
  return result;
}

int round_int( float r )
{
    return (r > 0.0f) ? (r + 0.5f) : (r - 0.5f);
}

cv::Mat Gaussian( const cv::Mat& input, double std_dev )
{
  cv::Mat result;
  input.assignTo(result);
  boost::mt19937 rng;
  boost::normal_distribution<> nd(0.0, (float) std_dev);
  boost::variate_generator<boost::mt19937&,
                             boost::normal_distribution<> > noiser(rng, nd);
  int rows = input.rows;
  int cols = input.cols;
  int channels = input.channels();
  for( int row = 0; row < rows; ++row )
  {
    for( int col = 0; col < cols; ++col )
    {
      cv::Vec3b& pel = result.at<cv::Vec3b>(row,col);
      for( int chann = 0; chann < channels; ++chann )
      {
        int val = pel[chann];
        float noise = noiser();
        int noise_int = round_int(noise);
        int new_val = val + noise_int;
        if ( new_val > 255 )
        {
          new_val = 255;
        }
        else if ( new_val < 0 )
        {
          new_val = 0;
        }
        pel[chann] = static_cast<uchar>(new_val);
//        int after_assign =  result.at<uchar>(row,col,chann);
//        std::cout << "val==" << val << ", noise_int=" << noise_int
//                  << "new_val==" << new_val << ", after assign="
//                     << after_assign << std::endl;
      }
    }
  }
  return result;
}

cv::Mat BenoiseCvMat( const cv::Mat& cvmat,
                                    int percent_sp,
                                    int percent_pix_sp,
                                    int percent_gauss,
                                    double std_dev,
                                    int& sp_ctr,
                                    int& g_ctr,
                                    int& n_ctr )
{
    std::cout << "sp=" << percent_sp << ", percent_gauss=" << percent_gauss << std::endl;
    NoiseType::Enum noise_type = GetNoiseType( percent_sp, percent_gauss );
    switch(noise_type)
    {
      case NoiseType::SALT_AND_PEPPER:
        sp_ctr++;
        return SaltAndPepper( cvmat, percent_pix_sp );
      case NoiseType::GAUSSIAN:
        g_ctr++;
        return Gaussian( cvmat, std_dev );
      case NoiseType::NONE:
        n_ctr++;
        return cvmat;
      default:
        throw std::runtime_error("Unexpected noise type.");
    }
}

void benoise_dataset( const std::string& input_dir,
                   const std::string& output_dir,
                   const std::string& percent_sp_str,
                   const std::string& sp_pix_percent_str,
                   const std::string& percent_gauss_str,
                   const std::string& std_dev_str,
                   const std::string& max_images_str )
{
  std::srand(time(NULL));
  path output_path( output_dir );

  std::cout << "Opening input_dir:" << input_dir << std::endl;
  // The lmdb database is at the input_dir.
  shared_ptr<caffe::db::DB> input_db(caffe::db::GetDB("lmdb"));
  input_db->Open(input_dir, caffe::db::READ);

  std::cout << "Opening output_dir:" << output_dir << std::endl;
  shared_ptr<caffe::db::DB> output_db(caffe::db::GetDB("lmdb"));
  output_db->Open(output_dir, caffe::db::NEW);
  caffe::db::Transaction* tx = output_db->NewTransaction();

  int percent_sp = atoi(percent_sp_str.c_str());
  int percent_gauss = atoi(percent_gauss_str.c_str());
  double std_dev = atof(std_dev_str.c_str());
  int percent_pix_sp = atoi(sp_pix_percent_str.c_str());
  int num_to_write = atoi(max_images_str.c_str());

  int ctr = 0;

  int sp_ctr = 0;
  int g_ctr = 0;
  int n_ctr = 0;

  bool put_since_commit = false;
  caffe::db::Cursor* cursor = input_db->NewCursor();
  for (cursor->SeekToFirst();
       cursor->valid() && ( ctr < num_to_write || num_to_write <= 0 );
       cursor->Next() )
  {
    Datum datum;
    datum.ParseFromString(cursor->value());
    cv::Mat cvmat = DatumToCvMat( datum );
    std::string keystring = cursor->key();
    cv::Mat cvmat_noisy = BenoiseCvMat( cvmat,
                                        percent_sp,
                                        percent_pix_sp,
                                        percent_gauss,
                                        std_dev,
                                        sp_ctr,
                                        g_ctr,
                                        n_ctr );

    // Comment this out to write out image files instead.
    Datum noisy_datum;
    CVMatToDatum(cvmat_noisy, &noisy_datum);
    std::string noisy_datum_string;
    noisy_datum.SerializeToString(&noisy_datum_string);
    tx->Put(keystring, noisy_datum_string);
    if ( ctr % 1000  == 0)
    {
      tx->Commit();
      tx = output_db->NewTransaction();
      put_since_commit = false;
    }
    else
    {
      put_since_commit = true;
    }


// Use this instead to write out image files to the output directory.
//    path output_file_path = output_dir;
//    output_file_path /= keystring;
//    output_file_path.replace_extension( ".jpg" );
//    std::cout << "output file: " << output_file_path.string() << std::endl;
//    bool result = cv::imwrite( output_file_path.string(), cvmat_noisy );

    ctr++;
  }

  if ( put_since_commit )
  {
    tx->Commit();
  }
  output_db->Close();

  std::cout << "salt and pepper images: " << sp_ctr << std::endl;
  std::cout << "gaussian images: " << g_ctr << std::endl;
  std::cout << "no noise images: " << n_ctr << std::endl;

}

int main(int argc, char** argv)
{
  srand(time(NULL));
  if (argc != 8)
  {
    printf("This util takes a caffe lmdb database of images, adds noise, and"
           "saves it to a new database."
           "\nUsage:\n"
           "make_noisy input_dir output_dir sp_percent sp_pix_percent max "
           "gauss_percent std_dev max_images\n"
           "Where: \n input_dir is the directory of the db to be noised\n"
           "output_dir is where to put the noisy database\n"
           "sp_percent is the percent of images that are to have salt and"
           " pepper noise added. \n"
           "sp_pix_percent is the percentage of pixels that are corrupted"
           "with salt and pepper noise."
           "gauss_percent is the percent of images that are to have normally-"
           "distributed noise added to them, and \n"
           "std_dev is the standard deviation of the noise that is to be"
           " added.\n"
           "max_images is the maximum number of images that should be"
           " generated.\n"
           );
  } else {
    google::InitGoogleLogging(argv[0]);
    benoise_dataset(string(argv[1]), string(argv[2]), string(argv[3]),
                    string(argv[4]), string(argv[5]), string(argv[6]),
                    string(argv[7]));
  }
  return 0;
}
