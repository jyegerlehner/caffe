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
#include "caffe/util/db.hpp"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"
#include "stdint.h"

#include "caffe/proto/caffe.pb.h"

using std::string;
using caffe::Datum;
using caffe::shared_ptr;
using namespace boost::filesystem;

typedef std::vector<std::string> Directories;

Directories GetInputDirectories( const std::string& input_file )
{
  std::ifstream fs;
  fs.open( input_file.c_str() );
  if ( !fs.is_open())
  {
    throw std::runtime_error( "failed to open input file." );
  }
  std::string line;
  Directories directories;
  while ( getline( fs, line ) )
  {
    directories.push_back( line );
  }
  return directories;
}

double Area( const cv::Size& size )
{
    return size.width * size.height;
}

cv::Size operator*( uint left, const cv::Size& right )
{
    cv::Size res( left * right.width, left * right.height );
    return res;
}

struct Rect
{
  cv::Size Size;
  cv::Point Center;
};

void convert_dataset(const string& input_file,
                     const string& output_directory,
                     int num_pictures )
{
  bool is_lmdb = true;
  is_lmdb = true;

  path output_path( output_directory );

  // Open new db
  //  shared_ptr<Dataset<string, Datum> > dataset;
  shared_ptr<caffe::db::DB> output_db;
  caffe::db::Transaction* tx = NULL;
  if ( is_lmdb )
  {
    output_db =  shared_ptr<caffe::db::DB>( caffe::db::GetDB("lmdb") );
    output_db->Open(output_directory, caffe::db::NEW);
    tx = output_db->NewTransaction();
  }

  bool put_since_commit = false;
  int count = 0;
  Directories input_directories = GetInputDirectories(input_file);
  for( Directories::const_iterator dit = input_directories.begin();
       dit != input_directories.end(); ++dit )
  {
    path in_path( *dit );

    directory_iterator end_itr;
    for( directory_iterator itr(in_path);
         itr != end_itr && count < num_pictures; ++itr )
    {
      std::map<int,int> scale_counts;
      if ( is_regular_file( itr->path() ) )
      {
        string current_file = itr->path().string();
        LOG(INFO) << "Reading " << current_file;
        cv::Mat cv_img_original = cv::imread( current_file, CV_LOAD_IMAGE_COLOR );
        if (!cv_img_original.data)
        {
          LOG(ERROR) << "Could not open or find file " << current_file;
        }
        else
        {
          // Skip if it's not big enough.
          if ( cv_img_original.cols < 256 || cv_img_original.rows < 256)
          {
            // Make a name for the output file.
            path filename( current_file );
            std::cout << "source filename =" << filename.filename() << std::endl;
            std::string src_filename = filename.filename().string();
            std::cout << "parent path=" << filename.parent_path().string()
                      << std::endl;
            std::string directory = filename.parent_path().string();
            path directory_path( directory );
            path stem( filename.stem() );

            cv::Mat subimg;
            cv::Point2i center( cv_img_original.cols/2, cv_img_original.rows/2 );
            Rect rect;
            rect.Size = cv::Size2i(256, 256);
            rect.Center = center;
            getRectSubPix( cv_img_original, rect.Size, rect.Center, subimg );

            std::stringstream key;
            key << rand();
            key << "_" << rect.Size.width << "x" << rect.Size.height
                   << "_" << src_filename;

            // Saving to lmdb database.
            Datum datum;
            CVMatToDatum( subimg, &datum );
            // Set a random label, since we're just benchmarking speed, not
            // actually trying to train anything real.
            datum.set_label(count % 1000);
            std::string datum_string;
            datum.SerializeToString(&datum_string);
            tx->Put( key.str(), datum_string );
            count++;
            if (count % 1000 == 0) {
              tx->Commit();
              tx = output_db->NewTransaction();
              put_since_commit = false;
            }
            else
            {
              put_since_commit = true;
            }
          }
        }
      }
    }
  }
  if ( is_lmdb )
  {
    if ( put_since_commit )
    {
      tx->Commit();
    }
    output_db->Close();
  }
}

int main(int argc, char** argv)
{
  srand(time(NULL));
  if (argc != 4)
  {
    printf("This util converts jpg files to lmdb format. It selects 256x256 \n"
           "pixel rectangles from files in the source directory and writes \n"
           "them into the database \n"
           "Usage:\n"
           "Convert input_file output_directory num_pictures\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), atoi(argv[3]) );
  }
  return 0;
}
