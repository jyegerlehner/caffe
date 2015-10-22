#if defined(USE_EIGEN)
#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/orthogonal.hpp"
#include "caffe/proto/caffe.pb.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

namespace caffe {

template <typename Dtype>
class OrthogonalizerTest : public ::testing::Test {
protected:
  typedef typename Orthogonalizer<Dtype>::Matrix Matrix;
  OrthogonalizerTest()
    : filler_param_() {
    filler_param_.set_mean(0.2);
    filler_param_.set_std(2.0);
    filler_param_.set_sparse(3.0);
    filler_.reset( new GaussianFiller<Dtype>(filler_param_));
  }

  FillerParameter filler_param_;
  shared_ptr<GaussianFiller<Dtype> > filler_;
  // Tolerance for ASSERT_NEAR.
  static const Dtype TOL = static_cast<Dtype>(0.0001);

  void AssertNear(const Blob<Dtype>& blob1,
                 const Blob<Dtype>& blob2) {

    ASSERT_EQ(blob1.count(), blob2.count());
    for(int i = 0; i < blob1.count(); ++i) {
      ASSERT_NEAR(blob1.cpu_data()[i], blob2.cpu_data()[i], TOL);
    }
  }

  std::string MakeConvolutionParam(int kernel_size, int num_output) {
    std::stringstream ss;
    ss << "convolution_param {" << std::endl;
    ss << "  kernel_size: " <<kernel_size << std::endl;
    ss << "  bias_term: false" << std::endl;
    ss << "  num_output: " << num_output << std::endl;
    ss << "  weight_filler { " << std::endl;
    ss << "    type: 'uniform'" << std::endl;
    ss << "    min: -1.0" << std::endl;
    ss << "    max: 1.0" << std::endl;
    ss << "  }" << std::endl;
    ss << "}" << std::endl;
    return ss.str();
  }

  Dtype TestConvDeconv( int kernel_size, int in_out_chans, int bottom_size, int middle_channels ) {
    using namespace caffe;
    typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;

    const int NUM_INSTANCES = 4;
//    const int MIDDLE_CHANS = 100;
//    const int IN_OUT_CHANS = 10;
//    const int IN_OUT_HEIGHT_WIDTH = 12;
//    const int KERNEL_SIZE = 3;

    Blob<Dtype> bottom_blob(NUM_INSTANCES, in_out_chans, bottom_size, bottom_size);
    this->filler_->Fill(&bottom_blob);
    Blob<Dtype> middle_blob;
    Blob<Dtype> top_blob;

    std::vector<Blob<Dtype>* > first_bott_vec;
    std::vector<Blob<Dtype>* > first_top_vec;
    std::vector<Blob<Dtype>* > second_bott_vec;
    std::vector<Blob<Dtype>* > second_top_vec;

    first_bott_vec.push_back(&bottom_blob);
    first_top_vec.push_back(&middle_blob);

    second_bott_vec.push_back(&middle_blob);
    second_top_vec.push_back(&top_blob);

    // Create a deconvolution layer.
    std::string param_str = MakeConvolutionParam(kernel_size, middle_channels);
    LayerParameter layer_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(param_str, &layer_param));
    shared_ptr<DeconvolutionLayer<Dtype> > layer1(
        new DeconvolutionLayer<Dtype>(layer_param));
    layer1->SetUp(first_bott_vec, first_top_vec);

    // Create a convolution layer meant to be the inverse.
    LayerParameter layer_param2;
    param_str = MakeConvolutionParam(kernel_size, in_out_chans);
    CHECK(google::protobuf::TextFormat::ParseFromString(param_str, &layer_param2));
    shared_ptr<ConvolutionLayer<Dtype> > layer2(
          new ConvolutionLayer<Dtype>(layer_param2));
    layer2->SetUp(second_bott_vec, second_top_vec);

    Orthogonalizer<Dtype>::InvertDeconv(*layer1->blobs()[0],
        *layer2->blobs()[0]);

    Matrix layer1_wts = Orthogonalizer<Dtype>::BlobToMat(*layer1->blobs()[0]);
    Matrix layer2_wts = Orthogonalizer<Dtype>::BlobToMat(*layer2->blobs()[0]);

    Matrix wt_prod = layer2_wts*layer1_wts.transpose();
    std::cout << "wt_prod:" << std::endl << wt_prod << std::endl;
    this->AssertNear(wt_prod, Matrix::Identity(wt_prod.rows(), wt_prod.rows()));

    layer1->Forward(first_bott_vec, first_top_vec);
    layer2->Forward(second_bott_vec, second_top_vec);

    Matrix m_bot = Orthogonalizer<Dtype>::BlobToMat(bottom_blob);
    Matrix m_middle = Orthogonalizer<Dtype>::BlobToMat(middle_blob);
    Matrix m_top = Orthogonalizer<Dtype>::BlobToMat(top_blob);

    std::cout << "bot mat rows,cols = " << m_bot.rows() << "," << m_bot.cols() << std::endl;
    std::cout << "mid mat rows,cols = " << m_middle.rows() << "," << m_middle.cols() << std::endl;
    std::cout << "top mat rows,cols = " << m_top.rows() << "," << m_top.cols() << std::endl;

  //  std::cout << "m_bot':" << std::endl << m_bot.transpose() << std::endl;
  //  std::cout << "m_top':" << std::endl << m_top.transpose() << std::endl;

    Matrix diff = m_bot - m_top;
    Dtype diff_norm = diff.norm();
    Dtype bot_norm = m_bot.norm();
    Dtype diff_ratio = diff_norm / bot_norm;
    std::cout << "Diff-to-Bottom norm ratio:" << diff_ratio << std::endl;
    return diff_ratio;
  }

  void AssertNear(const Matrix& mat1,
                  const Matrix& mat2)
  {
    ASSERT_GT(mat1.rows(), 0);
    ASSERT_GT(mat1.cols(), 0);
    ASSERT_EQ(mat1.rows(), mat2.rows() );
    ASSERT_EQ(mat1.rows(), mat2.rows() );

    for(int row = 0; row < mat1.rows(); ++row) {
      for(int col = 0; col < mat1.cols(); ++col) {
        ASSERT_NEAR(mat1(row,col), mat2(row,col), TOL);
      }
    }
  }

  void TestOrthogonalization(Blob<Dtype>& blob) {
    typedef typename caffe::Orthogonalizer<Dtype>::Matrix Matrix;

    // Fill blob.
    this->filler_->Fill(&blob);
    // Orthogonalize.
    Orthogonalizer<Dtype>::Execute(blob,
      FillerParameter_Orthogonalization_SIMPLE);

    // Get blob as a matrix, so it is easy to multiply it by
    // its transpose.
    Matrix mat = Orthogonalizer<Dtype>::BlobToMat(blob);

    // Test orthogonality by multiplying blob matrix by its
    // transpose, which should yield identity matrix.
    if (mat.rows() < mat.cols() )
    {
      this->AssertNear(mat*mat.transpose(),
                       Matrix::Identity(mat.rows(),mat.rows()));
    }
    else
    {
      this->AssertNear(mat.transpose()*mat,
                       Matrix::Identity(mat.cols(), mat.cols()));
    }
  }

  void TestOrthogonalization_Preserve(Blob<Dtype>& blob,
      FillerParameter_Orthogonalization orthog) {
    typedef typename caffe::Orthogonalizer<Dtype>::Matrix Matrix;

    // Fill blob.
    this->filler_->Fill(&blob);
    // Orthogonalize.
    Orthogonalizer<Dtype>::Execute(blob, orthog);

    // Get blob as a matrix, so it is easy to multiply it by
    // its transpose.
    Matrix mat = Orthogonalizer<Dtype>::BlobToMat(blob);

    // Test orthogonality by multiplying blob matrix by its
    // transpose, which should yield identity matrix.
    if (mat.rows() < mat.cols() )
    {
      Matrix prod = mat*mat.transpose();
      this->AssertNear(prod,
                       prod(0,0)*Matrix::Identity(mat.rows(),mat.rows()));
    }
    else
    {
      Matrix prod = mat.transpose()*mat;
      this->AssertNear(prod,
                       prod(0,0)*Matrix::Identity(mat.cols(), mat.cols()));
    }
  }

  void AssertBlobAndMatNear( const Blob<Dtype>& blob, const Matrix& mat)
  {
    std::vector<int> blob_shape = blob.shape();
    int blob_cols = 1;
    for(int i = 1; i < blob_shape.size(); ++i)
    {
      blob_cols *= blob_shape[i];
    }
    int blob_rows = blob_shape[0];
    ASSERT_EQ(mat.rows(), blob_rows);
    ASSERT_EQ(mat.cols(), blob_cols);

    const Dtype* blob_data = blob.cpu_data();
    for(int row = 0; row < mat.rows(); ++row)
    {
      for(int col = 0; col < mat.cols(); ++col)
      {
        Dtype val = *blob_data++;
        ASSERT_NEAR( mat(row,col), val, TOL);
      }
    }
  }


  void AssertFlatBlobAndMatNear( const Blob<Dtype>& blob, const Matrix& mat)
  {
    std::vector<int> blob_shape = blob.shape();

    int blob_rows = blob_shape[0];
    int blob_cols = blob_shape[1];

    ASSERT_EQ(mat.rows(), blob_rows);
    ASSERT_EQ(mat.cols(), blob_cols);

    const Dtype TOL = (Dtype) 0.00001;
    for(int row = 0; row < mat.rows(); ++row)
    {
      for(int col = 0; col < mat.cols(); ++col)
      {
        int offset = blob.offset(row,col,0,0);
        Dtype val = blob.cpu_data()[offset];
        ASSERT_NEAR( mat(row,col), val, TOL);
      }
    }
  }

};

TYPED_TEST_CASE(OrthogonalizerTest, TestDtypes);

TYPED_TEST(OrthogonalizerTest, MatAndFlatBlobSame_5x10) {
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;
  Blob<Dtype> blob(5,10,1,1);
  this->filler_->Fill(&blob);
  Matrix mat = Orthogonalizer<Dtype>::BlobToMat(blob);
  this->AssertFlatBlobAndMatNear(blob,mat);
}

TYPED_TEST(OrthogonalizerTest, MatAndFlatBlobSame_10x5) {
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;
  Blob<Dtype> blob(10,5,1,1);
  this->filler_->Fill(&blob);
  Matrix mat = Orthogonalizer<Dtype>::BlobToMat(blob);
  this->AssertFlatBlobAndMatNear(blob,mat);
}

TYPED_TEST(OrthogonalizerTest, BlobToMatPreservesShapeAndIdentity) {
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;
  Blob<Dtype> blob(4,2,2,3);
  this->filler_->Fill(&blob);

  Matrix mat = Orthogonalizer<Dtype>::BlobToMat(blob);
  this->AssertBlobAndMatNear(blob, mat);
}

TYPED_TEST(OrthogonalizerTest, MatToBlobPreservesShapeAndIdentity) {
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;
  Blob<Dtype> blob(4,2,2,3);
  this->filler_->Fill(&blob);

  Matrix mat = Orthogonalizer<Dtype>::BlobToMat(blob);

  Blob<TypeParam> new_blob( blob.shape());
  Orthogonalizer<Dtype>::MatToBlob(mat, new_blob);
  this->AssertBlobAndMatNear(new_blob,mat);
}

TYPED_TEST(OrthogonalizerTest, MatBlobRoundTrip)
{
  typedef TypeParam Dtype;
  typedef typename caffe::Orthogonalizer<Dtype>::Matrix Matrix;
  Blob<Dtype> orig_blob(100, 50, 2, 3);
  this->filler_->Fill(&orig_blob);
  Matrix mat = Orthogonalizer<Dtype>::BlobToMat(orig_blob);
  Blob<Dtype> new_blob(100, 50, 2, 3);
  Orthogonalizer<Dtype>::MatToBlob(mat, new_blob);
  this->AssertNear(orig_blob, new_blob);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_50x20) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(50, 20, 1, 1);
  this->TestOrthogonalization(blob);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_20x50) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(20, 50, 1, 1);
  this->TestOrthogonalization(blob);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_50x20x2x3) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(50, 20, 2, 3);
  this->TestOrthogonalization(blob);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_100x10x2x3) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(100, 10, 2, 3);
  this->TestOrthogonalization(blob);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_PreserveTraceNorm_5x4) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(5, 4, 1, 1);
  this->TestOrthogonalization_Preserve(blob,
      FillerParameter_Orthogonalization_PRESERVE_TRACE_NORM);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_PreserveTraceNorm_4x5) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(4, 5, 1, 1);
  this->TestOrthogonalization_Preserve(blob,
      FillerParameter_Orthogonalization_PRESERVE_TRACE_NORM);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_PreserveFrobNorm_5x4) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(5, 4, 1, 1);
  this->TestOrthogonalization_Preserve(blob,
      FillerParameter_Orthogonalization_PRESERVE_FROB_NORM);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_PreserveFrobNorm_4x5) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(4, 5, 1, 1);
  this->TestOrthogonalization_Preserve(blob,
      FillerParameter_Orthogonalization_PRESERVE_FROB_NORM);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_Preserve1Norm_5x4) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(5, 4, 1, 1);
  this->TestOrthogonalization_Preserve(blob,
      FillerParameter_Orthogonalization_PRESERVE_1_NORM);
}

TYPED_TEST(OrthogonalizerTest, TestOrthogonal_Preserve1Norm_4x5) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(4, 5, 1, 1);
  this->TestOrthogonalization_Preserve(blob,
      FillerParameter_Orthogonalization_PRESERVE_1_NORM);
}

TYPED_TEST(OrthogonalizerTest, TestInvertMat) {
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;

  Matrix m1(4,10);
  Matrix m2(10,4);

  int ctr = 0;
  for(int row=0; row < m1.rows(); ++row)
  {
    for(int col=0; col < m1.cols(); ++col)
    {
      m1(row,col) = (rand() %100 - 50) / 10.0f;
      ctr++;
    }
  }

  Orthogonalizer<Dtype>::Invert(m1,m2);

  Matrix prod = m1*m2;
  this->AssertNear(prod, Matrix::Identity(m1.rows(), m2.cols()));
}

TYPED_TEST(OrthogonalizerTest, TestInvertBlob_4x10) {
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;

  Blob<Dtype> blob(4,10,1,1);
  Blob<Dtype> inverted_blob(10,4,1,1);

  this->filler_->Fill(&blob);

  Orthogonalizer<Dtype>::Invert(blob, inverted_blob);

  Matrix m = Orthogonalizer<Dtype>::BlobToMat(blob);
  Matrix m_inverted = Orthogonalizer<Dtype>::BlobToMat(inverted_blob);

  Matrix prod = m*m_inverted;
  this->AssertNear(prod, Matrix::Identity(m.rows(), m_inverted.cols()));
}

TYPED_TEST(OrthogonalizerTest, TestInvertBlob_20x40) {
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;

  Blob<Dtype> blob(5,40,1,1);
  Blob<Dtype> inverted_blob(40,5,1,1);

  this->filler_->Fill(&blob);

  Orthogonalizer<Dtype>::Invert(blob, inverted_blob);

  Matrix m = Orthogonalizer<Dtype>::BlobToMat(blob);
  Matrix m_inverted = Orthogonalizer<Dtype>::BlobToMat(inverted_blob);

  Matrix prod = m*m_inverted;
  this->AssertNear(prod, Matrix::Identity(m.rows(), m_inverted.cols()));
}

TYPED_TEST(OrthogonalizerTest, TestInvertInnerProductLayer) {
  using namespace caffe;
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;

  const int MIDDLE_CHANS = 10;
  const int IN_OUT_CHANS = 4;
  Blob<Dtype> bottom_blob(3,IN_OUT_CHANS,1,1);
  this->filler_->Fill(&bottom_blob);
  Blob<Dtype> middle_blob;
  Blob<Dtype> top_blob;

  std::vector<Blob<Dtype>* > first_bott_vec;
  std::vector<Blob<Dtype>* > first_top_vec;
  std::vector<Blob<Dtype>* > second_bott_vec;
  std::vector<Blob<Dtype>* > second_top_vec;

  first_bott_vec.push_back(&bottom_blob);
  first_top_vec.push_back(&middle_blob);

  second_bott_vec.push_back(&middle_blob);
  second_top_vec.push_back(&top_blob);

  // Create an inner product layer.
  LayerParameter layer_param;
  InnerProductParameter* innerprod_param =
      layer_param.mutable_inner_product_param();
  innerprod_param->set_num_output(MIDDLE_CHANS);
  innerprod_param->mutable_weight_filler()->set_type("gaussian");
  innerprod_param->set_bias_term(false);
  shared_ptr<InnerProductLayer<Dtype> > layer1(
      new InnerProductLayer<Dtype>(layer_param));
  layer1->SetUp(first_bott_vec, first_top_vec);

  // Create an inner product layer meant to be the inverse.
  LayerParameter layer_param2;
  InnerProductParameter* innerprod_param2 =
      layer_param2.mutable_inner_product_param();
  innerprod_param2->set_num_output(IN_OUT_CHANS);
  innerprod_param2->mutable_weight_filler()->set_type("gaussian");
  innerprod_param2->set_bias_term(false);
  shared_ptr<InnerProductLayer<Dtype> > layer2(
        new InnerProductLayer<Dtype>(layer_param2));
  layer2->SetUp(second_bott_vec, second_top_vec);
   //layer2->blobs()[0]

  Orthogonalizer<Dtype>::Invert(*layer1->blobs()[0],
      *layer2->blobs()[0]);

  Matrix layer1_wts = Orthogonalizer<Dtype>::BlobToMat(*layer1->blobs()[0]);
  Matrix layer2_wts = Orthogonalizer<Dtype>::BlobToMat(*layer2->blobs()[0]);

  this->AssertNear(layer2_wts*layer1_wts, Matrix::Identity(layer2_wts.rows(),
                                                     layer1_wts.cols()));

  layer1->Forward(first_bott_vec, first_top_vec);
  layer2->Forward(second_bott_vec, second_top_vec);

  Matrix m_bot = Orthogonalizer<Dtype>::BlobToMat(bottom_blob);
  Matrix m_top = Orthogonalizer<Dtype>::BlobToMat(top_blob);
  this->AssertNear(m_bot, m_top);
}

TYPED_TEST(OrthogonalizerTest, TestConvolutionDeconv) {
  typedef TypeParam Dtype;
  const int KERNEL_SIZE = 3;
  const int IN_OUT_CHANNELS = 10;
  const int BOTTOM_SIZE = 12;
  const int MIDDLE_CHANNELS = 100;
  Dtype diff_norm_ratio = this->TestConvDeconv( KERNEL_SIZE, IN_OUT_CHANNELS, BOTTOM_SIZE, MIDDLE_CHANNELS );
  ASSERT_LT(diff_norm_ratio, static_cast<Dtype>(0.3));
  //this->AssertNear(m_bot, m_top);
}

TYPED_TEST(OrthogonalizerTest, TestConvolutionDeconv_1x1) {
  typedef TypeParam Dtype;
  const int KERNEL_SIZE = 1;
  const int IN_OUT_CHANNELS = 10;
  const int BOTTOM_SIZE = 12;
  const int MIDDLE_CHANNELS = 10;
  Dtype diff_norm_ratio = this->TestConvDeconv( KERNEL_SIZE, IN_OUT_CHANNELS, BOTTOM_SIZE, MIDDLE_CHANNELS );
  ASSERT_LT(diff_norm_ratio, static_cast<Dtype>(0.001));
  //this->AssertNear(m_bot, m_top);
}

TYPED_TEST(OrthogonalizerTest, TestConvolutionDeconv_9x9) {
  typedef TypeParam Dtype;
  const int KERNEL_SIZE = 5;
  const int IN_OUT_CHANNELS = 5;
  const int BOTTOM_SIZE = 12;
  const int MIDDLE_CHANNELS = 50;
  Dtype diff_norm_ratio = this->TestConvDeconv( KERNEL_SIZE, IN_OUT_CHANNELS, BOTTOM_SIZE, MIDDLE_CHANNELS );
  ASSERT_LT(diff_norm_ratio, static_cast<Dtype>(0.3));
  //this->AssertNear(m_bot, m_top);
}


//TYPED_TEST(OrthogonalizerTest, TestOrthogonal_Nearest_50x20) {
//  typedef TypeParam Dtype;
//  Blob<Dtype> blob(50, 20, 1, 1);
//  this->TestOrthogonalization_Nearest(blob);
//}

//TYPED_TEST(OrthogonalizerTest, TestOrthogonal_Nearest_20x50) {
//  typedef TypeParam Dtype;
//  Blob<Dtype> blob(20, 50, 1, 1);
//  this->TestOrthogonalization_Nearest(blob);
//}

//TYPED_TEST(OrthogonalizerTest, TestOrthogonal_Nearest_50x20x2x3) {
//  typedef TypeParam Dtype;
//  Blob<Dtype> blob(50, 20, 2, 3);
//  this->TestOrthogonalization_Nearest(blob);
//}

//TYPED_TEST(OrthogonalizerTest, TestOrthogonal_Nearest_100x10x2x3) {
//  typedef TypeParam Dtype;
//  Blob<Dtype> blob(100, 10, 2, 3);
//  this->TestOrthogonalization_Nearest(blob);
//}

}
#endif  // USE_EIGEN
