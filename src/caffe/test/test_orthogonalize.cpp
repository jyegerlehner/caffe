#if defined(USE_EIGEN)
#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/orthogonal.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OrthogonalizerTest : public ::testing::Test {
protected:
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  OrthogonalizerTest()
    : filler_param_() {
    filler_param_.set_mean(0.2);
    filler_param_.set_std(0.25);
    filler_param_.set_sparse(3);
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
      std::cout << std::endl << prod << std::endl;
      this->AssertNear(prod,
                       prod(0,0)*Matrix::Identity(mat.rows(),mat.rows()));
    }
    else
    {
      Matrix prod = mat.transpose()*mat;
      std::cout << std::endl << prod << std::endl;
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
    int cols = 1;
    for(int i = 1; i < blob.shape().size(); ++i)
    {
      cols *= blob.shape()[i];
    }

    ASSERT_EQ(mat.rows(), blob_rows);
    ASSERT_EQ(mat.cols(), cols);

    const Dtype TOL = (Dtype) 0.0001;
    const Dtype* blob_data = blob.cpu_data();
    for(int col = 0; col < mat.cols(); ++col)
    {
      for(int row = 0; row < mat.rows(); ++row)
      {
        ASSERT_NEAR( (Dtype) mat(row,col), *blob_data++, TOL);
      }
    }
  }
};

TYPED_TEST_CASE(OrthogonalizerTest, TestDtypes);

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

//TYPED_TEST(OrthogonalizerTest, TestOrthogonalConvolutionDeconv) {
//  // Create a deconvolution layer.
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  ConvolutionParameter* convolution_param =
//      layer_param.mutable_convolution_param();
//  convolution_param->add_kernel_size(3);
//  convolution_param->add_stride(1);
//  convolution_param->set_num_output(128);
//  convolution_param->mutable_weight_filler()->set_type("gaussian");
//  convolution_param->set_bias_term(false);
//  shared_ptr<Layer<Dtype> > layer(
//      new DeconvolutionLayer<Dtype>(layer_param));
//  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

//  // Create a convolution layer, which we intend to be the inverse of
//  // the deconv layer.


//  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);


//}

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


//TYPED_TEST(OrthogonalizerTest, TestInvertBlob) {
//  using namespace caffe;

//  Blob<Dtype> bottom_blob(2,4,1,1);
//  Blob<Dtype> middle_blob();
//  Blob<Dtype> top_blob();


//  std::vector<Blob<Dtype> > first_bott_vec;
//  std::vector<Blob<Dtype> > first_top_vec;
//  std::vector<Blob<Dtype> > second_bott_vec;
//  std::vector<Blob<Dtype> > second_top_vec;

//  first_bott_vec.push_back(&bottom_blob);
//  first_top_vec.push_back(&middle_blob);

//  second_bott_vec.push_back(&middle_blob);
//  second_top_vec.push_back(&top_blob);

//  // Create an inner product layer.
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  InnerProductParameter* innerprod_param =
//      layer_param.mutable_inner_product_param();
//  innerprod_param->set_num_output(16);
//  innerprod_param->mutable_weight_filler()->set_type("gaussian");
//  innerprod_param->set_bias_term(false);
//  shared_ptr<InnerProductLayer<Dtype> > layer(
//      new InnerProductLayer<Dtype>(layer_param));
//  layer->SetUp(first_bott_vec, first_top_vec);

//  // Create an inner product layer meant to be the inverse.
//  LayerParameter layer_param2;
//  InnerProductParameter* innerprod_param2 =
//      layer_param2.mutable_inner_product_param();
//  innerprod_param2->set_num_output(BOTTOM_CHANS);
//  innerprod_param2->mutable_weight_filler()->set_type("gaussian");
//  innerprod_param2->set_bias_term(false);
//  shared_ptr<InnerProductLayer<Dtype> > layer2(
//        new InnerProductLayer(layer_param2));
//  layer2->SetUp(second_bott_vec, second_top_vec);
//   //layer2->blobs()[0]

//  Orthogonalizer<Dtype>::Invert(*layer->blobs()[0],
//      *layer2->blobs()[0]);

//  layer1->Forward(first_bott_vec, first_top_vec);
//  layer2->Forward(second_bott_vec, second_top_vec);

//  AssertNear(bottom_blob, top_blob);
//}


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
