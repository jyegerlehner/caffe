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

namespace
{
  const float img_to_net_scale = 0.0039215684;
}

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

void GetHiddenActivations(shared_ptr<Net<float> > net,
                          const path& input_file,
                          const std::string& blob_name,
                          Blob<float>& activations)
{
  int channel_flag;
  // Get input blob and check its shape.
  {
    const std::vector<Blob<float>*>& blobs = net->ForwardPrefilled();
    CHECK(blobs.size() == 1) << "Net input should have one blob.";

    int channels = blobs[0]->channels();
    CHECK(channels == 1 || channels == 3) << "Bad number of channels.";
    if ( channels == 1)
      channel_flag = CV_LOAD_IMAGE_GRAYSCALE;
    else if ( channels == 3)
      channel_flag = CV_LOAD_IMAGE_COLOR;
  }

  std::string input_file_str = input_file.string();
  cv::Mat patch = cv::imread(input_file_str, channel_flag);
  if ( !patch.data )
  {
    LOG(ERROR) << "Could not open or find file " << input_file_str;
  }
  else
  {
    Blob<float> input_blob;
    input_blob.Reshape(1, patch.channels(), patch.rows, patch.cols );

    TransformationParameter input_xform_param;
    input_xform_param.set_scale( img_to_net_scale );
    DataTransformer<float> input_xformer( input_xform_param, TEST );
    input_xformer.Transform( patch, &input_blob );
    std::vector<Blob<float>*> input;
    input.push_back( &input_blob );
    std::vector<Blob<float>*> output = net->Forward( input );

    const shared_ptr<Blob<float> > feature_blob = net->blob_by_name(blob_name);

    activations.CopyFrom(*feature_blob, false, true);
  }
}


//    CHECK(output.size() == 1 ) << "Wrong size output of net.";
//    Blob<float>* raw_blob_ptr = output[0];
//    CHECK( raw_blob_ptr->num() == 1 ) << "Should have one output blob.";

//    Blob<float> output_blob;
//    if ( raw_blob_ptr->channels() != 3 )
//    {
//      int output_size = raw_blob_ptr->channels() *
//                              raw_blob_ptr->height() *
//                              raw_blob_ptr->width();
//      int input_size = input_blob.channels() *
//                       input_blob.height() * input_blob.width();
//      CHECK( output_size == input_size ) << "Output size " << output_size <<
//                                            " should equal input size " <<
//                                            input_size;
//    }

//    // Scale the output blob back to the original image scaling
//    output_blob.Reshape( raw_blob_ptr->num(), raw_blob_ptr->channels(),
//                         raw_blob_ptr->height(),raw_blob_ptr->width() );

//    output_xformer.Transform( raw_blob_ptr, &output_blob );

//    // Shape the blob to match the
//    output_blob.Reshape( raw_blob_ptr->num(), input_blob.channels(),
//                         input_blob.height(),input_blob.width() );

void Interpolate( Blob<float>& interpolated_blob,
                  const Blob<float>& activations1,
                  const Blob<float>& activations2,
                  float ratio)
{
  const float* from_data1 = activations1.cpu_data();
  const float* from_data2 = activations2.cpu_data();
  float* interp_data = interpolated_blob.mutable_cpu_data();

  for( int i = 0; i < activations1.count(); ++i)
  {
    float fromval1 = from_data1[i];
    float fromval2 = from_data2[i];
    float val = ratio* (fromval2 - fromval1) + fromval1;
    interp_data[i] = val;
  }
}

int FindEarliestLayerWithBottomBlob(Net<float>& net,
                                    const std::string& blob_name)
{
  const vector<shared_ptr<Layer<float> > >& layers = net.layers();
//  const Blob<float>& desired_blob = *net.blob_by_name(blob_name);

  for(int layer_index = 0; layer_index < layers.size(); ++layer_index)
  {
    Layer<float>& layer = *layers[layer_index];
//    std::vector<shared_ptr<Blob<float> > >& blobs = layer.blobs();
    int bottom_blobs_size = layer.layer_param().bottom_size();
    for(int blob_index = 0; blob_index < bottom_blobs_size; ++blob_index)
    {
      if ( layer.layer_param().bottom(blob_index) == blob_name )
      {
        return layer_index;
      }
    }
  }
  return -1;
}

void WriteBlobToOutput(int interp_index,
                       const path& output_dir,
                       Blob<float>& raw_blob)
{
  Blob<float> output_blob;
  // Scale the output blob back to the original image scaling
  output_blob.Reshape( raw_blob.num(), raw_blob.channels(),
                       raw_blob.height(), raw_blob.width() );

  TransformationParameter output_xform_param;
  output_xform_param.set_scale( 1.0 / img_to_net_scale );
  DataTransformer<float> output_xformer( output_xform_param, TEST );
  output_xformer.Transform( &raw_blob, &output_blob );

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
  // Create the output filename.

  std::string output_filename;
  {
    std::stringstream ss;
    ss << "interpolated";
    ss << interp_index;
    ss << ".png";
    output_filename = ss.str();
  }

  output_file_path /= output_filename;
  std::cout << "output file: " << output_file_path.string() << std::endl;
  bool result = cv::imwrite( output_file_path.string(), mat );
}

int main( int argc, char** argv )
{
  if ( argc < 7 || argc > 8)
  {
    LOG(ERROR) << "Usage: interp model_proto model_weights"
                  " input1_file input2_file blob_name output_dir [gpu=1]";
    return 1;
  }
  std::string net_proto( argv[1] );
  std::string model_weights( argv[2] );
  path input1_file(argv[3]);
  path input2_file(argv[4]);
  std::string blob_names_arg(argv[5]);
  path output_dir( argv[6]);
  if ( argc == 8) {
    int gpu = atoi(argv[7]);
    Caffe::SetDevice(gpu);
    Caffe::set_mode(Caffe::GPU);
    LOG(INFO) << "Set mode to GPU: " << gpu << std::endl;
  } else {
    Caffe::set_mode(Caffe::CPU);
    LOG(INFO) << "Set mode to CPU" << std::endl;
  }

  shared_ptr<Net<float> > net( new Net<float>( net_proto, TEST ) );

  {
    std::vector<std::string> model_names;
    boost::split(model_names, model_weights, boost::is_any_of(",") );
    for(int i = 0; i < model_names.size(); ++i) {
      net->CopyTrainedLayersFrom( model_names[i] );
    }
  }

  std::vector<std::string> blob_names;
  boost::split(blob_names, blob_names_arg, boost::is_any_of(","));

//  path input_path( input_dir );
//  directory_iterator end_iter;
//  for( directory_iterator file_iter( input_path );
//       file_iter != end_iter;
//       ++ file_iter )
//  {

  Blob<float> activations1;
  Blob<float> activations2;
  std::vector<Blob<float>* > activations1_vect;
  std::vector<Blob<float>* > activations2_vect;
  for(int i = 0; i < blob_names.size(); ++i)
  {
    Blob<float>* activations1 = new Blob<float>();
    GetHiddenActivations(net, input1_file, blob_names[i], *activations1);
    Blob<float>* activations2 = new Blob<float>();
    GetHiddenActivations(net, input2_file, blob_names[i], *activations2);
    activations1_vect.push_back(activations1);
    activations2_vect.push_back(activations2);
  }

  // Now interpolate between the two activations, and forward prop from each
  // interpolated hidden activation.
  const int NUM_INTERPOLATIONS = 10;
  for(int i = 0; i <= NUM_INTERPOLATIONS; ++i)
  {

    for( int blob_inx = 0; blob_inx < blob_names.size(); ++blob_inx)
    {
      float ratio = i / static_cast<float>(NUM_INTERPOLATIONS);
      Blob<float> interpolated_blob;
      interpolated_blob.ReshapeLike(*activations1_vect[blob_inx]);

      Interpolate( interpolated_blob,
                   *activations1_vect[blob_inx],
                   *activations2_vect[blob_inx],
                   ratio);

      //Assign interpolated blob.
      shared_ptr<Blob<float> > target_blob =
          net->blob_by_name(blob_names[blob_inx]);
      CHECK( target_blob->shape() == interpolated_blob.shape() );
      target_blob->CopyFrom(interpolated_blob);
    }

    // Forward prop from interpolated blob.
    {
      // Use the first blob name to find the layer where we start forward prop.
      int start_forward = FindEarliestLayerWithBottomBlob(*net, blob_names[0]);
      net->ForwardFrom(start_forward);
      const std::vector<Blob<float>*>& output = net->output_blobs();
      CHECK(output.size() == 1 ) << "Wrong size output of net.";
      Blob<float>* raw_blob_ptr = output[0];
      CHECK( raw_blob_ptr->num() == 1 ) << "Should have one output blob.";
      WriteBlobToOutput(i, output_dir, *raw_blob_ptr);
    }
  }
}
