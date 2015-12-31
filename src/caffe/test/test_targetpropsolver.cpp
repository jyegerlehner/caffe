#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/solver.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/TargetPropSolver.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TargetPropSolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  TargetPropSolverTest()
  {
  }

  void AddSolver(std::stringstream& ss)
  {
    ss << "base_lr: 0.1 \n";
    ss << "lr_policy: 'fixed' \n";
    ss << "display: 500 \n";
    ss << "momentum: 0.90 \n";
    ss << "solver_type: SGD \n";
    ss << "solver_mode: GPU \n";
    ss << "device_id: 0 \n";
    ss << "max_iter: 10000 \n";
    ss << "test_iter: 1 \n";
    ss << "test_interval: 500 \n ";
//    ss << "debug_info: true \n";
  }

//  void AddDataLayer(std::stringstream& ss)
//  {
//    ss << "  layer {\n";
//    ss << "    name: 'data_layer' \n";
//    ss << "    type: 'DummyData' \n";
//    ss << "    top: 'h0' \n";
//    ss << "    dummy_data_param {\n";
//    ss << "      shape { \n";
//    ss << "        dim: 4 \n"; // batch size 4
//    ss << "        dim: 6 \n"; // 6 scalars in the input vector.
//    ss << "       } \n";
//    ss << "    } \n";
//    ss << "  } \n";
//  }

  std::string IntToStr(int i)
  {
    std::stringstream ss;
    ss << i;
    return ss.str();
  }

  void AddDataLayer(std::stringstream& ss,
                          const std::string& mirror_blob,
                          int shape0, int shape1)
  {
    std::stringstream ln;
    ln << "data_layer_" << mirror_blob;
    ss << "  layer {\n";
    ss << "    name: '" << ln.str() << "' \n";
    ss << "    type: 'DummyData' \n";
    // Top blob is named the same as the blob it mirrors (which was created
    // in a different net).
    ss << "    top: '" << mirror_blob << "' \n";
    ss << "    dummy_data_param {\n";

    // Data layer mirrors a blob that the blob_finder already
    // knows about.
    ss << "      shape { \n";
    ss << "        dim: " << IntToStr(shape0) << "\n"; // batch size
    ss << "        dim: " << IntToStr(shape1) << "\n";
    ss << "       } \n";
    ss << "    } \n";
    ss << "  } \n";
  }


  void AddFullyConnectedLayer(std::stringstream& ss,
                              bool encoder_not_decoder,
                              int level,
                              std::string& bottom_blob_name = string())
  {
    std::string prefix = encoder_not_decoder ? "encoder" : "decoder";
    std::string layer_name;
    {
      std::stringstream ns;
      ns << prefix << level << "_fc";
      layer_name = ns.str();
    }

    string num_output;
    if (encoder_not_decoder)
    {
      num_output = level == 1 ? "4" : "2";
    }
    else
    {
      num_output = level == 1 ? "6" : "4";
    }

    std::string bottom_name = bottom_blob_name;
    if(bottom_name.size() == 0) {
      std::stringstream bn;
      if (encoder_not_decoder)
      {
        bn << "h" << (level-1);
      }
      else
      {
        if (level == 1)
          bn << "h" << level << "_hat";
        else
          bn << "h" << level;
      }
      bottom_name = bn.str();
    }

    std::string top_name;
    {
      std::stringstream tn;
      tn << prefix << level <<"_fc";
      top_name = tn.str();
    }

    ss << "layer { \n";
    ss << "  name: '" << layer_name << "' \n";
    ss << "  type: 'InnerProduct' \n";
    ss << "  inner_product_param { \n";
    ss << "    num_output: "<< num_output << " \n";
    ss << "    weight_filler { \n";
    ss << "      type: 'gaussian' \n";
    ss << "      std: 0.1 \n";
    ss << "    } \n";
    ss << "    bias_filler { \n";
    ss << "      type: 'constant' \n";
    ss << "    } \n";
    ss << "  } \n";
    ss << "  bottom: '" << bottom_name << "' \n";
    ss << "  top: '" << top_name << "' \n";
    ss << "} \n";
    ss << std::endl;
  }

  void AddSigmoidLayer(std::stringstream& ss,
                       bool encoder_not_decoder,
                       int level)
  {
    std::string prefix = encoder_not_decoder ? "encoder" : "decoder";
    std::string layer_name;
    {
      std::stringstream ln;
      ln << prefix << level << "_sigmoid";
      layer_name = ln.str();
    }

    std::string bottom_name;
    {
      std::stringstream bn;
      bn << prefix << level << "_fc";
      bottom_name = bn.str();
    }

    std::string top_name;
    {
      std::stringstream tn;
      int index = encoder_not_decoder ? level : (level-1);
      tn << "h" << index;
      tn << (encoder_not_decoder ? "" : "_hat");
      top_name = tn.str();
    }

    ss << "layer { \n";
    ss << "  name: '"<< layer_name << "' \n";
    ss << "  type: 'Sigmoid' \n";
    ss << "  bottom: '" << bottom_name << "' \n";
    ss << "  top: '" << top_name << "' \n";
    ss << "} \n";
  }

  void AddProcessingLayers(std::stringstream& ss,
                           bool encoder_not_decoder,
                           int level,
                           std::string decoder_bottom_blob_name = string())
  {
    AddFullyConnectedLayer(ss, encoder_not_decoder, level,
                           decoder_bottom_blob_name);
    AddSigmoidLayer(ss, encoder_not_decoder, level);
  }

  void AddEuclideanLoss(std::stringstream& ss,
                        const std::string& bottom1,
                        const std::string& bottom2,
                        const std::string& name)
  {
    ss << "layer { \n";
    ss << "  name: '" << name << "' \n";
    ss << "  type: 'EuclideanLoss' \n";
    ss << "  bottom: '" << bottom1 << "' \n";
    ss << "  bottom: '" << bottom2 <<"' \n";
    ss << "  top: '" << name << "' \n";
    ss << "} \n";
  }

  void AddLongLoop(std::stringstream& ss)
  {
    AddDataLayer(ss, "h0", 4,6);
    AddProcessingLayers(ss, true, 1);
    AddProcessingLayers(ss, true, 2);
    AddProcessingLayers(ss, false, 2);
    AddProcessingLayers(ss, false, 1);
    AddEuclideanLoss(ss, "h0", "h0_hat", "LongLoopLoss");
  }

  void AddEncoderIdentityLoop(std::stringstream& ss, int level)
  {
    ss << "dependent_net { \n";
    ss << "  name: 'encoder_loop" << level << "' \n";
    if ( level == 1)
    {
      AddDataLayer(ss, "h0", 4, 6);
    }
    else
    {
      AddDataLayer(ss, "h1", 4, 4);
    }

    std::string decoder_bottom_blob_name = (level == 1) ? "h1" : "h2";

    AddProcessingLayers(ss, true, level);
    AddProcessingLayers(ss, false, level, decoder_bottom_blob_name);

    std::string loss_blob1 = (level == 1) ? "h0" : "h1";
    std::string loss_blob2 = (level == 1) ? "h0_hat" : "h1_hat";
    std::string loss_name;
    {
      std::stringstream ln;
      ln << "encoder_loop" << level << "_loss";
      loss_name = ln.str();
    }
    AddEuclideanLoss(ss, loss_blob1, loss_blob2, loss_name );
    ss << "} \n";
  }

  void AddDecoderIdentityLoop(std::stringstream& ss, int level)
  {
    ss << "dependent_net { \n";
    ss << "  name: 'encoder_loop" << level << "' \n";
    if ( level == 1)
    {
      AddDataLayer(ss, "h0", 4,6);
      AddDataLayer(ss, "h1_hat", 4, 4);
    }
    else
    {
      throw std::runtime_error("Only level 1 supported.");
    }

    AddProcessingLayers(ss, false, level);
    AddProcessingLayers(ss, true, level, "h0_hat");

    std::string loss_blob1 = (level == 1) ? "h1" : "xxxxx";
    std::string loss_blob2 = (level == 1) ? "h1_hat" : "zzzzzz";
    std::string loss_name;
    {
      std::stringstream ln;
      ln << "decoder_loop" << level << "_loss_h1";
      loss_name = ln.str();
    }
    AddEuclideanLoss(ss, loss_blob1, loss_blob2, loss_name );
    AddEuclideanLoss(ss, "h0", "h0_hat", "decoder_loop_loss_h0" );

    ss << "} \n";
  }


  void StartNet(std::stringstream& ss)
  {
    ss << "net_param { \n";
    ss << "  name: 'XorLongLoop'\n";
  }

  void EndNet(std::stringstream& ss)
  {
    ss << "} \n";
  }

  std::string CreateXorProtoText()
  {
    std::stringstream ss;
    AddSolver(ss);
    StartNet(ss);
    AddLongLoop(ss);
    AddEncoderIdentityLoop(ss,1);
    AddEncoderIdentityLoop(ss,2);
    AddDecoderIdentityLoop(ss,1);
    EndNet(ss);
    return ss.str();
  }

  template<typename Dtype>
  struct InputOutputVal
  {
    InputOutputVal(Dtype lx0, Dtype lx1):
      x0(lx0),
      x1(lx1)
    {
    }

    Dtype x0;
    Dtype x1;
  };

  void AssignTrainingData(TargetPropSolver<Dtype>& solver)
  {
    typedef InputOutputVal<Dtype> IOVal;
    shared_ptr<Blob<Dtype> > input = solver.BlobByName("h0");
    std::vector<IOVal> xor_outputs;
    std::vector<IOVal> or_outputs;
    std::vector<IOVal> and_outputs;
    xor_outputs.push_back(IOVal(0.0, 1.0));
    xor_outputs.push_back(IOVal(1.0, 0.0));
    xor_outputs.push_back(IOVal(1.0, 0.0));
    xor_outputs.push_back(IOVal(0.0, 1.0));

    or_outputs.push_back(IOVal(0.0, 1.0));
    or_outputs.push_back(IOVal(1.0, 0.0));
    or_outputs.push_back(IOVal(1.0, 0.0));
    or_outputs.push_back(IOVal(1.0, 0.0));

    and_outputs.push_back(IOVal(0.0, 1.0));
    and_outputs.push_back(IOVal(0.0, 1.0));
    and_outputs.push_back(IOVal(0.0, 1.0));
    and_outputs.push_back(IOVal(1.0, 0.0));

    for(int i = 0; i < xor_outputs.size(); ++i)
    {
      // XOR
      int offset = input->offset(i,0,0,0);
      input->mutable_cpu_data()[offset] = xor_outputs[i].x0;
      offset = input->offset(i,1,0,0);
      input->mutable_cpu_data()[offset] = xor_outputs[i].x1;

      //OR
      offset = input->offset(i,2,0,0);
      input->mutable_cpu_data()[offset] = or_outputs[i].x0;
      offset = input->offset(i,3,0,0);
      input->mutable_cpu_data()[offset] = or_outputs[i].x1;

      //AND
      offset = input->offset(i,4,0,0);
      input->mutable_cpu_data()[offset] = and_outputs[i].x0;
      offset = input->offset(i,5,0,0);
      input->mutable_cpu_data()[offset] = and_outputs[i].x1;
    }
  }

};

template <typename Dtype>
void AssertBlobsEqual(const Blob<Dtype>& b1, const Blob<Dtype>& b2) {
  ASSERT_EQ( b1.count(), b2.count());
  int count = b1.count();
  for(int index = 0; index < count; ++index) {
    Dtype val1 = b1.cpu_data()[index];
    Dtype val2 = b2.cpu_data()[index];
    if (val1 != 0.0 || val2 != 0.0 )
    {
      ASSERT_NEAR(val1, val2, 0.00001);
    } else {
      ASSERT_NEAR(val1, val2, 0.00001);
    }
  }
}


TYPED_TEST_CASE(TargetPropSolverTest, TestDtypesAndDevices);

// Forward propagating through the Softmaxima layer should produce the same
// result as forward propagating through two individual Softmax layers whose
// outputs are concatenated appropriately.
TYPED_TEST(TargetPropSolverTest, PrintXorProto) {
  typedef typename TypeParam::Dtype Dtype;
  std::string proto = this->CreateXorProtoText();

  std::cout << proto;

  SolverParameter param;
  CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
  switch (Caffe::mode()) {
    case Caffe::CPU:
      param.set_solver_mode(SolverParameter_SolverMode_CPU);
      break;
    case Caffe::GPU:
      param.set_solver_mode(SolverParameter_SolverMode_GPU);
      break;
    default:
      LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
  }

  TargetPropSolver<Dtype> solver(param);
  std::vector<int> gpus;
  gpus.push_back(0);
  this->AssignTrainingData(solver);
  Dtype loss = 0.0;
  solver.Run(gpus, loss);

  ASSERT_NE(loss, 0.0);
  const Dtype TOL = 0.001f;
  ASSERT_NEAR(loss, 0.0, TOL);

  {
    shared_ptr<Blob<Dtype> > long_loop_output = solver.ForwardLongLoop();
    shared_ptr<Blob<Dtype> > loops_output = solver.ForwardLoops();
    AssertBlobsEqual(*long_loop_output, *loops_output);
  }
}

} // namespace caffe
