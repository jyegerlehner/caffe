#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "gtest/gtest.h"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/util/orthogonal.hpp"

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

  std::pair<int,int> FlatShape(const Blob<Dtype>& blob)
  {
    std::vector<int> shape = blob.shape();
    int rows = shape[0];
    int cols = 1;
    for(int i = 1; i < shape.size(); ++i)
    {
      cols *= shape[i];
    }
    return std::pair<int,int>( rows, cols );
  }

  Eigen::Map<const Matrix> BlobToMat( const Blob<Dtype>& blob )
  {
    std::pair<int,int> shape = FlatShape(blob);

    int rows = shape.first;
    int cols = shape.second;

    Eigen::Map<const Matrix> m( blob.cpu_data(), rows, cols );
    return m;
  }

//  Matrix BlobToMat(const Blob<Dtype>& blob )
//  {
//    std::pair<int,int> shape = FlatShape(blob);

//    int rows = shape.first;
//    int cols = shape.second;

//    Eigen::Map<const Matrix> m( blob.cpu_data(), rows, cols );

//    Matrix copy(m.rows(), m.cols());
//    for(int col = 0; col < cols; ++col)
//    {
//      for(int row = 0; row < rows; ++row)
//      {
//        copy(row,col) = m(row,col);
//      }
//    }

//    return copy;
//  }

  void MatToBlob( const Matrix& mat, Blob<Dtype>& blob )
  {
    std::vector<int> blob_shape = blob.shape();
    int blob_cols = 1;
    for(int i = 1; i < blob_shape.size(); ++i)
    {
      blob_cols *= blob_shape[i];
    }

    int blob_rows = blob_shape[0];

    std::cout << "mat rows, cols==" << mat.rows() << "," << mat.cols() <<
                 std::endl;
    std::cout << "blob rows, cols==" << blob_rows << "," << blob_cols <<
                 std::endl;

    ASSERT_EQ( blob_cols, mat.cols());
    ASSERT_EQ( blob_rows, mat.rows());

    Dtype* blob_data = blob.mutable_cpu_data();
    for( int col = 0; col < mat.cols(); ++col)
    {
      for( int row = 0; row < mat.rows(); ++row)
      {
        *blob_data++ = mat(row,col);
      }
    }
  }

  void AssertNear(const Blob<Dtype>& blob1,
                 const Blob<Dtype>& blob2) {

    const Dtype TOL = static_cast<Dtype>(0.00001);
    ASSERT_EQ(blob1.count(), blob2.count());
    for(int i = 0; i < blob1.count(); ++i) {
      ASSERT_NEAR(blob1.cpu_data()[i], blob2.cpu_data()[i], TOL);
    }
  }

  void AssertNear(const Matrix& mat1,
                  const Matrix& mat2)
  {
    const Dtype TOL = static_cast<Dtype>(0.00001);
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

  void AssertNearIdentity( const Matrix& mat)
  {
    const float TOL = 0.0001;
    ASSERT_EQ(mat.rows(), mat.cols() );
    ASSERT_GT(mat.rows(), 1);
    for(int row = 0; row < mat.rows(); ++row) {
      for(int col = 0; col < mat.cols(); ++col) {
        ASSERT_NEAR(mat(row,col), (row == col) ? 1.0f : 0.0f, TOL );
      }
    }
  }

//  void Invert( const Blob<Dtype>& blob )
//  {
//    Eigen::Map<const Matrix> m( blob.cpu_data(), rows, cols );

//  }

  void Orthogonalize( const Blob<Dtype>& blob )
  {
//    using namespace Eigen;
    Eigen::Map<const Matrix> m = BlobToMat(blob);
    std::cout << std::endl << "original matrix:" << std::endl;
    std::cout << m;

    unsigned int options = Eigen::ComputeThinU | Eigen::ComputeThinV;

    Eigen::JacobiSVD<Matrix> svd(m, options);
    std::cout << std::endl << std::endl << "u matrix:"
              << svd.matrixU() << std::endl;
    const Matrix& u = svd.matrixU();
    std::cout << std::endl << "v matrix:" << svd.matrixV() << std::endl;
    const Matrix& v = svd.matrixV();

    bool use_u = (u.rows() == m.rows() && u.cols() == m.cols());
    bool use_v = (v.rows() == m.cols() && v.cols() == m.rows());
    Blob<Dtype> orthog_blob(blob.shape());
    if(use_u)
    {
      this->MatToBlob(u, orthog_blob);
    }
    else if ( use_v )
    {
      this->MatToBlob(v.transpose(), orthog_blob);
    }
    else
    {
      ASSERT_TRUE(false);
    }

    Eigen::Map<const Matrix> orthog_mat = this->BlobToMat(orthog_blob);
    std::cout << std::endl << "Orthogonalized blob:" << std::endl;
    std::cout << orthog_mat;

    if ( orthog_mat.rows() < orthog_mat.cols() )
    {
      Matrix should_be_identity = orthog_mat * orthog_mat.transpose();
      std::cout << std::endl << "identity? : " << std::endl;
      std::cout << should_be_identity << std::endl;
      AssertNearIdentity(should_be_identity);
    }
    else
    {
      Matrix should_be_identity = orthog_mat.transpose() * orthog_mat;
      std::cout << std::endl << "identity? : " << std::endl;
      std::cout << should_be_identity << std::endl;
      AssertNearIdentity(should_be_identity);
    }

  }

  Matrix OrthogonalizeMinFrobNorm( const Blob<Dtype>& blob )
  {
    using namespace Eigen;
    Eigen::Map<const Matrix> m = BlobToMat(blob);
    std::cout << std::endl << "original matrix:" << std::endl;
    std::cout << m;

//    unsigned int options = m.cols() > m.rows() ?
//          Eigen::ComputeThinU | Eigen::ComputeFullV :
//          Eigen::ComputeFullU | Eigen::ComputeThinV;

//    unsigned int options = Eigen::ComputeThinU | Eigen::ComputeThinV;

    unsigned int options = m.cols() > m.rows() ?
          Eigen::ComputeThinU | Eigen::ComputeFullV :
          Eigen::ComputeThinU | Eigen::ComputeThinV;

    Eigen::JacobiSVD<Matrix> svd(m, options);
    std::cout << std::endl << "u matrix:" << std::endl << svd.matrixU() << std::endl;
    const Matrix& u = svd.matrixU();
    std::cout << std::endl << "v matrix:" << std::endl << svd.matrixV() << std::endl;
    const Matrix& v = svd.matrixV();

    Matrix s = Matrix::Zero(u.cols(), v.rows());
    int min_dim = s.rows() < s.cols() ?  s.rows() : s.cols();
    for(int i = 0; i < min_dim; ++i)
    {
      s(i,i) = 1.0;
    }
    std::cout << "s matrix:" << std::endl << s  << std::endl;

    Matrix orthog_mat = u * s * v;
    Blob<Dtype> orthog_blob(blob.shape());

//    this->MatToBlob(orthog_mat, orthog_blob);

    std::cout << std::endl << "Orthogonalized blob:" << std::endl;
    std::cout << orthog_mat;

    if ( orthog_mat.rows() < orthog_mat.cols() )
    {
      Matrix should_be_identity = orthog_mat * orthog_mat.transpose();
      std::cout << std::endl << "identity? : " << std::endl;
      std::cout << should_be_identity << std::endl;
      AssertNearIdentity(should_be_identity);
    }
    else
    {
      Matrix should_be_identity = orthog_mat.transpose() * orthog_mat;
      std::cout << std::endl << "identity? : " << std::endl;
      std::cout << should_be_identity << std::endl;
      AssertNearIdentity(should_be_identity);
    }
    return orthog_mat;
  }

  Matrix OrthogonalizeMinFrobNormFullUV( const Blob<Dtype>& blob )
  {
    using namespace Eigen;
    Eigen::Map<const Matrix> m = BlobToMat(blob);
    std::cout << std::endl << "original matrix:" << std::endl;
    std::cout << m;

    unsigned int options = Eigen::ComputeFullU | Eigen::ComputeFullV;

    Eigen::JacobiSVD<Matrix> svd(m, options);
    std::cout << std::endl << "u matrix:" << std::endl << svd.matrixU() << std::endl;
    const Matrix& u = svd.matrixU();
    std::cout << std::endl << "v matrix:" << std::endl << svd.matrixV() << std::endl;
    const Matrix& v = svd.matrixV();

    Matrix s = Matrix::Zero(u.cols(), v.rows());
    int min_dim = s.rows() < s.cols() ?  s.rows() : s.cols();
    for(int i = 0; i < min_dim; ++i)
    {
      s(i,i) = 1.0;
    }
    std::cout << "s matrix:" << std::endl << s  << std::endl;

    Matrix orthog_mat = u * s * v;
    Blob<Dtype> orthog_blob(blob.shape());

//    this->MatToBlob(orthog_mat, orthog_blob);

    std::cout << std::endl << "Orthogonalized blob:" << std::endl;
    std::cout << orthog_mat;

    if ( orthog_mat.rows() < orthog_mat.cols() )
    {
      Matrix should_be_identity = orthog_mat * orthog_mat.transpose();
      std::cout << std::endl << "identity? : " << std::endl;
      std::cout << should_be_identity << std::endl;
      AssertNearIdentity(should_be_identity);
    }
    else
    {
      Matrix should_be_identity = orthog_mat.transpose() * orthog_mat;
      std::cout << std::endl << "identity? : " << std::endl;
      std::cout << should_be_identity << std::endl;
      AssertNearIdentity(should_be_identity);
    }

    return orthog_mat;
  }


  void CompareBlobAndMat( const Blob<Dtype>& blob, const Matrix& mat)
  {
    std::vector<int> blob_shape = blob.shape();
    int blob_cols = 1;
    for(int i = 1; i < blob_shape.size(); ++i)
    {
      blob_cols *= blob_shape[i];
    }
    int blob_rows = blob_shape[0];

    std::cout << "mat rows, cols==" << mat.rows() << "," << mat.cols() <<
                 std::endl;
    std::cout << "blob rows, cols==" << blob_rows << "," << blob_cols <<
                 std::endl;

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

  Matrix mat = this->BlobToMat(blob);
  this->CompareBlobAndMat(blob, mat);
}

TYPED_TEST(OrthogonalizerTest, MatToBlobPreservesShapeAndIdentity) {
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;
  Blob<Dtype> blob(4,2,2,3);
  this->filler_->Fill(&blob);

  Matrix mat = this->BlobToMat(blob);

  Blob<TypeParam> new_blob( blob.shape());
  this->MatToBlob(mat, new_blob);
  this->CompareBlobAndMat(new_blob,mat);
}

TYPED_TEST(OrthogonalizerTest, TestSimple) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(4,2,2,3);
  this->filler_->Fill(&blob);

  this->Orthogonalize(blob);
}

TYPED_TEST(OrthogonalizerTest, TestSimple2) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(6,2,2,1);
  this->filler_->Fill(&blob);

  this->Orthogonalize(blob);
}

TYPED_TEST(OrthogonalizerTest, TestSimple3) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(5,10,1,1);
  this->filler_->Fill(&blob);

  this->Orthogonalize(blob);
}

TYPED_TEST(OrthogonalizerTest, TestSimple4) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(10,5,1,1);
  this->filler_->Fill(&blob);

  this->Orthogonalize(blob);
}

TYPED_TEST(OrthogonalizerTest, TestSimple5) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(10,5,1,1);
  this->filler_->Fill(&blob);

  this->OrthogonalizeMinFrobNorm(blob);
}

TYPED_TEST(OrthogonalizerTest, TestSimple6) {
  typedef TypeParam Dtype;
  Blob<Dtype> blob(5,10,1,1);
  this->filler_->Fill(&blob);

  this->OrthogonalizeMinFrobNorm(blob);
}

TYPED_TEST(OrthogonalizerTest, TestFullUVFromNormMin)
{
  typedef TypeParam Dtype;
  Blob<Dtype> blob(5,10,1,1);
  this->filler_->Fill(&blob);

  (void) this->OrthogonalizeMinFrobNormFullUV(blob);
}

TYPED_TEST(OrthogonalizerTest, TestFullAndThinAreSame)
{
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;
  Blob<Dtype> blob(5,10,1,1);
  this->filler_->Fill(&blob);

  Matrix mat_thin = this->OrthogonalizeMinFrobNorm(blob);
  Matrix mat_full = this->OrthogonalizeMinFrobNormFullUV(blob);

  this->AssertNear(mat_thin, mat_full);
}

TYPED_TEST(OrthogonalizerTest, TestFullAndThinAreSame2)
{
  typedef TypeParam Dtype;
  typedef typename OrthogonalizerTest<Dtype>::Matrix Matrix;
  Blob<Dtype> blob(10,5,1,1);
  this->filler_->Fill(&blob);

  Matrix mat_thin = this->OrthogonalizeMinFrobNorm(blob);
  Matrix mat_full = this->OrthogonalizeMinFrobNormFullUV(blob);

  this->AssertNear(mat_thin, mat_full);
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

TYPED_TEST(OrthogonalizerTest, TestOrthogonal) {
  typedef TypeParam Dtype;
  typedef typename caffe::Orthogonalizer<Dtype>::Matrix Matrix;
  Blob<Dtype> blob(100, 50, 2, 3);
  this->filler_->Fill(&blob);

  Orthogonalizer<Dtype>::Fast(blob);

  Matrix mat = Orthogonalizer<Dtype>::BlobToMat(blob);

  if (mat.rows() < mat.cols() )
  {
    this->AssertNear(mat * mat.transpose(),
                     Matrix::Identity(mat.rows(),mat.rows()));
  }
  else
  {
    this->AssertNear(mat.transpose()*mat,
                     Matrix::Identity(mat.cols(), mat.cols()));
  }
}

}
