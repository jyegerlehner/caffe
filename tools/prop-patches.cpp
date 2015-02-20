#include <boost/filesystem.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using boost::shared_ptr;
using namespace caffe;
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
        float datum_float_val = datum.float_data(datum_index);
        if ( datum_float_val >= 255.0 )
        {
          ptr[img_index++] = 255;
        }
        else if ( datum_float_val <= 0.0 )
        {
          ptr[img_index++] = 0;
        }
        else
        {
          ptr[img_index++] = static_cast<uchar>( lrint( datum_float_val) );
        }
      }
    }
  }
  return mat;
}

int main( int argc, char** argv )
{
  if ( argc != 5 )
  {
    LOG(ERROR) << "Usage: prop-patches model_proto model_weights"
                  " input_dir_name output_dir_name";
    return 1;
  }
  std::string net_proto( argv[1] );
  std::string model_weights( argv[2] );
  path input_dir( argv[3] );
  path output_dir( argv[4] );


  shared_ptr<Net<float> > net( new Net<float>( net_proto, TEST ) );
  {
    std::vector<std::string> model_names;
    boost::split(model_names, model_weights, boost::is_any_of(",") );
    for(int i = 0; i < model_names.size(); ++i) {
      net->CopyTrainedLayersFrom( model_names[i] );
    }
  }

  const float img_to_net_scale = 0.0039215684;
  TransformationParameter input_xform_param;
  input_xform_param.set_scale( img_to_net_scale );
  DataTransformer<float> input_xformer( input_xform_param, TEST );

  TransformationParameter output_xform_param;
  output_xform_param.set_scale( 1.0 / img_to_net_scale );
  DataTransformer<float> output_xformer( output_xform_param, TEST );

  path input_path( input_dir );
  directory_iterator end_iter;
  for( directory_iterator file_iter( input_path );
       file_iter != end_iter;
       ++ file_iter )
  {
    std::string input_file_str = file_iter->path().string();
    cv::Mat patch = cv::imread( input_file_str, CV_LOAD_IMAGE_COLOR );
    if ( !patch.data )
    {
      LOG(ERROR) << "Could not open or find file " << input_file_str;
    }
    else
    {
      Blob<float> input_blob;
      input_blob.Reshape(1, patch.channels(), patch.rows, patch.cols );

      input_xformer.Transform( patch, &input_blob );
      std::vector<Blob<float>*> input;
      input.push_back( &input_blob );
      std::vector<Blob<float>*> output = net->Forward( input );
      CHECK(output.size() == 1 ) << "Wrong size output of net.";
      Blob<float>* raw_blob_ptr = output[0];
      CHECK( raw_blob_ptr->num() == 1 ) << "Should have one output blob.";

      Blob<float> output_blob;
      if ( raw_blob_ptr->channels() != 3 )
      {
        int output_size = raw_blob_ptr->channels() *
                                raw_blob_ptr->height() *
                                raw_blob_ptr->width();
        int input_size = input_blob.channels() *
                         input_blob.height() * input_blob.width();
        CHECK( output_size == input_size ) << "Output size " << output_size <<
                                              " should equal input size " <<
                                              input_size;
      }
//      CHECK( raw_blob_ptr->channels() == 3 ) <<
//                                          "Should have three output channels.";

      // Scale the output blob back to the original image scaling
      output_blob.Reshape( raw_blob_ptr->num(), raw_blob_ptr->channels(),
                           raw_blob_ptr->height(),raw_blob_ptr->width() );

      output_xformer.Transform( raw_blob_ptr, &output_blob );

      // Shape the blob to match the
      output_blob.Reshape( raw_blob_ptr->num(), input_blob.channels(),
                           input_blob.height(),input_blob.width() );

      // Convert the output blob back to an image.
      Datum datum;
      datum.set_height( output_blob.height() );
      datum.set_width( output_blob.width() );
      datum.set_channels( output_blob.channels() );
      datum.clear_data();
      datum.clear_float_data();
      const float* blob_data = output_blob.cpu_data();
      for( int i = 0; i < output_blob.count(); ++i )
      {
        datum.add_float_data( blob_data[i] );
      }

      cv::Mat mat = DatumToCvMat( datum );

      path output_file_path = output_dir;
      output_file_path /= file_iter->path().filename().string();
      output_file_path.replace_extension( ".jpg" );
      std::cout << "output file: " << output_file_path.string() << std::endl;
      bool result = cv::imwrite( output_file_path.string(), mat );
    }
  }

}
