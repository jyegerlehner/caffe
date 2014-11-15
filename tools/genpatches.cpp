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

namespace
{
  int PATCH_SIZE = 32;
}

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
  int line_ctr = 0;
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

struct PatchScale
{
    static PatchScale First( cv::Size screen )
    {
        const uint FIRST_SIZE = 32;
        cv::Size rect( FIRST_SIZE, FIRST_SIZE );
        PatchScale result( rect, screen );
//        result.m_lower_cutoff = 0;
//        result.m_upper_cutoff = Count();
        return result;
    }

    PatchScale Next() const
    {
        static const uint RATIO = 2;
        cv::Size rect = RATIO * m_rect;
        PatchScale result( rect, m_screen );
//        result.m_lower_cutoff = m_upper_cutoff;
//        result.m_upper_cutoff = m_upper_cutoff + Count();
        return result;
    }

    uint Count() const
    {
        return m_count;
    }

    // Returns true if the next larger rectangle size is still no larger
    // than the size of the original source image.
    bool HasNext() const
    {
        PatchScale next = Next();
        return next.m_rect.height <= m_screen.height &&
                next.m_rect.width <= m_screen.width &&
                next.Count() > 0;
    }

    Rect CreateRandomSource() const
    {
      uint min_x = m_rect.width / 2;
      uint min_y = m_rect.width / 2;
      uint max_x = m_screen.width - m_rect.width / 2;
      uint max_y = m_screen.height - m_rect.height / 2;

      uint NUM_SELECTS = 10000U;
      uint x_select = rand() % (NUM_SELECTS + 1);
      uint y_select = rand() % (NUM_SELECTS + 1);

      uint x_center = ( ( max_x - min_x ) * x_select ) / NUM_SELECTS + min_x;
      uint y_center = ( ( max_y - min_y ) * y_select ) / NUM_SELECTS + min_y;
      cv::Point2i center( x_center, y_center );
      Rect result;
      result.Size = m_rect;
      result.Center = center;
      return result;
    }

    cv::Size Size() const
    {
      return m_rect;
    }
private:
    PatchScale( const cv::Size& rect, const cv::Size& screen ):
        m_rect( rect ),
        m_screen( screen )
    {
        // m_count = static_cast<uint>( Area( screen ) / Area( m_rect ) );
        // If we choose count as merely the area ratio, then a high-res picture
        // has e.g. (2048/32)^2 small area patches for only a single patch that
        // covers the whole area of the photo. Thousands to one. Which
        // imbalances the representations of the scales.
        // So instead, raise to the 0.667 to make the ratio not so lopsided.

        double area_ratio = Area( screen ) / Area( m_rect );
        // Multiply a number greater than 1 to get more than a single patch
        // at the largest scale. Each will still be unique because each has a
        // randomly-chosen center.
        if ( area_ratio > 1000.0 )
        {
          area_ratio *= 0.5;
        }
        else if ( area_ratio > 100.0 )
        {
          area_ratio *= 0.75;
        }
        else if ( area_ratio > 10.0 )
        {
          area_ratio *= 1.0;
        }
        else
        {
          area_ratio *= 2.0;
        }

        m_count = std::pow( area_ratio, 0.8 );
    }

    // Number patches we should create at this scale.
    uint m_count;
    // Size of this patch scale.
    cv::Size m_rect;
    // Size of the full image from which we are extracting patches from.
    cv::Size m_screen;
};

typedef std::vector<PatchScale> PatchScales;

PatchScales CreateScales( const cv::Mat& cv_image )
{
  PatchScales scales;
  PatchScale scale =
      PatchScale::First( cv::Size( cv_image.cols, cv_image.rows ) );
  scales.push_back( scale );
  while ( scale.HasNext() )
  {
    scale = scale.Next();
    scales.push_back( scale );
  }
  return scales;
}

cv::Mat DownSize( const cv::Mat& rect )
{
  int width = PATCH_SIZE;
  int height = PATCH_SIZE;
  cv::Mat patch;
  cv::resize( rect, patch,
              cv::Size( width, height ),
              0.0, 0.0, cv::INTER_AREA );
  return patch;
}

void convert_dataset(const string& input_file,
                     const string& output_directory,
                     const string& output_type )
{
  bool is_lmdb;
  if ( output_type == "lmdb" )
  {
    is_lmdb = true;
  }
  else if ( output_type == "jpg" )
  {
    is_lmdb = false;
  }
  else
  {
    std::stringstream ss;
    ss << "Invalid output type:";
    ss << output_type;
    throw std::runtime_error( ss.str() );
  }

  path output_path( output_directory );

  // Open new db
  shared_ptr<Dataset<string, Datum> > dataset;
  if ( is_lmdb )
  {
    dataset = DatasetFactory<string, Datum>(output_type);
    // Open db.
    CHECK(dataset->open(output_path.string(),
                        Dataset<string, Datum>::New ) );
  }

  int count = 0;
  Directories input_directories = GetInputDirectories(input_file);
  for( Directories::const_iterator dit = input_directories.begin();
       dit != input_directories.end(); ++dit )
  {
    path in_path( *dit );

    directory_iterator end_itr;
    for( directory_iterator itr(in_path); itr != end_itr; ++itr )
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
          // Make a name for the output file.
          path filename( current_file );
          std::cout << "source filename =" << filename.filename() << std::endl;
          std::string src_filename = filename.filename().string();
          std::cout << "parent path=" << filename.parent_path().string()
                    << std::endl;
          std::string directory = filename.parent_path().string();
          path directory_path( directory );
          path stem( filename.stem() );

          PatchScales scales = CreateScales( cv_img_original );

          // Select and save 100 random 32x32 patches from the source.
          for( PatchScales::const_iterator scale_it = scales.begin();
               scale_it != scales.end(); ++scale_it )
          {
            std::cout << "scale: " << scale_it->Size().width << " x " <<
                         scale_it->Size().height << ", count=" <<
                         scale_it->Count() << std::endl;
            for( int i = 0; i < scale_it->Count(); ++i )
            {
              Rect rect = scale_it->CreateRandomSource();

              cv::Mat subimg;
              getRectSubPix( cv_img_original, rect.Size, rect.Center, subimg );

              // The patch is the subimg downsized to 32x32.
              cv::Mat patch;
              if ( subimg.rows != PATCH_SIZE || subimg.cols != PATCH_SIZE )
              {
                patch = DownSize( subimg );
              }
              else
              {
                patch = subimg;
              }

              if ( !is_lmdb )
              {
                // saving to jpeg images.
                std::stringstream ss;
                ss << stem.string() << "_" << rect.Size.width << "_" << i << ".jpg";
                path output_file( output_directory );
                output_file /= ss.str();
                std::cout << "Writing file " << output_file.string() << std::endl;
                cv::imwrite( output_file.string(), patch );
              }
              else
              {
                std::stringstream key;
                key << rand();
                key << "_" << rect.Size.width << "x" << rect.Size.height
                       << "_" << src_filename;

                // Saving to lmdb database.
                Datum datum;
                CVMatToDatum( patch, &datum );
                CHECK(dataset->put( key.str(), datum ));
                count++;
                if (count % 1000 == 0) {
                  CHECK(dataset->commit());
                }
              }
            }
          }
        }
      }
    }
  }
  if ( is_lmdb )
  {
    // write the last batch
    if (count % 1000 != 0) {
      CHECK(dataset->commit());
    }
    dataset->close();
  }
}

int main(int argc, char** argv)
{
  srand(time(NULL));
  if (argc != 4)
  {
    printf("This util converts jpg files to lmdb format. It selects 32x32 \n"
           "pixel rectangles from files in the source directory and writes \n"
           "them into the database \n"
           "Usage:\n"
           "Convert input_file output_directory [lmdb | jpg ]\n");
  } else {
    google::InitGoogleLogging(argv[0]);
    convert_dataset(string(argv[1]), string(argv[2]), string(argv[3]) );
  }
  return 0;
}
