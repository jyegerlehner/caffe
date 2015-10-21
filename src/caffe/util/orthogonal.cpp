#ifdef USE_EIGEN
#include <cmath>
#include "caffe/util/orthogonal.hpp"

namespace caffe {

// Create the shape of the matrix corresponding to the blob.
template<typename Dtype>
std::pair<int,int> Orthogonalizer<Dtype>::FlatShape(const Blob<Dtype>& blob)
{
  std::vector<int> shape = blob.shape();
  int rows = shape[0];
  int cols = 1;
  for(int i = 1; i < shape.size(); ++i)
  {
    cols *= shape[i];
  }
  return std::pair<int,int>(rows, cols);
}

// Returns a "matrix map" which is an Eigen matrix, except its data storage
// is provided at time of construction via pointer to data allocated
// externally to Eigen.
template<typename Dtype>
typename Orthogonalizer<Dtype>::MatrixMap Orthogonalizer<Dtype>::BlobToMat(
    Blob<Dtype>& blob)
{
  std::pair<int,int> shape = FlatShape(blob);
  int rows = shape.first;
  int cols = shape.second;
  return MatrixMap( blob.mutable_cpu_data(), rows, cols );
}

// Version of BlobToMat taking a const blob.
template<typename Dtype>
typename Orthogonalizer<Dtype>::ConstMatrixMap Orthogonalizer<Dtype>::BlobToMat(
    const Blob<Dtype>& blob )
{
  std::pair<int,int> shape = FlatShape(blob);
  int rows = shape.first;
  int cols = shape.second;
  return ConstMatrixMap( blob.cpu_data(), rows, cols );
}

template<typename Dtype>
void Orthogonalizer<Dtype>::MatToBlob(const Matrix& m, Blob<Dtype>& blob) {
  MatrixMap blob_as_mat = BlobToMat(blob);
  CHECK_EQ(blob_as_mat.cols(), m.cols());
  CHECK_EQ(blob_as_mat.rows(), m.rows());
  blob_as_mat = m;
}

// Assign the blob an orthogonal matrix created from the blob.
template<typename Dtype>
void Orthogonalizer<Dtype>::Simple(MatrixMap& blob_mat) {
  unsigned int options = Eigen::ComputeThinU | Eigen::ComputeThinV;
  Eigen::JacobiSVD<Matrix> svd(blob_mat, options);
  const Matrix& u = svd.matrixU();
  const Matrix& v = svd.matrixV();

  bool use_u = (u.rows() == blob_mat.rows() && u.cols() == blob_mat.cols());
  bool use_v = (v.cols() == blob_mat.rows() && v.rows() == blob_mat.cols());
  CHECK(use_u || use_v) << "Orthogonalize error: bad matrix dimensions.";
  if (use_u) {
    // This assignment modifies the underlying blob through its mutable_cpu_data
    // pointer, since blob_mat is a MatrixMap.
    blob_mat = u;
  }
  else
  {
    // This assignment modifies the underlying blob through its mutable_cpu_data
    // pointer, since blob_mat is a MatrixMap.
    blob_mat = v.transpose();
  }
}

// Compute the mean of the singular values of the original matrix so we can
// preserve it in the orthogonalized matrix (PRESERVE_TRACE_NORM).
template<typename Dtype>
Dtype Orthogonalizer<Dtype>::SVAverage(const Matrix& S) {
  Dtype avg = 0.0;
  for(int i=0; i < S.rows(); ++i) {
    avg += S(i,0);
  }
  avg /= S.rows();
  return avg;
}

// Compute the 1/N * square root of the sum of the squares of the original
// matrix's singular values, so we can preserve it (PRESERVE_FROB_NORM).
template<typename Dtype>
Dtype Orthogonalizer<Dtype>::FrobNormSV(const Matrix& S) {
  Dtype accum = 0.0;
  for(int i=0; i < S.rows(); ++i) {
    Dtype entry = S(i,0);
    accum += entry*entry;
  }
  accum = std::sqrt(accum);
  accum /= S.rows();
  return accum;
}

// Find the largest singular value so we can preserve it (PRESERVE_1_NORM).
template<typename Dtype>
Dtype Orthogonalizer<Dtype>::LargestSV(const Matrix& S) {
  Dtype largest = 0.0;
  for(int i=0; i < S.rows(); ++i) {
    Dtype entry = S(i,0);
    if (entry > largest) {
      largest = entry;
    }
  }
  return largest;
}

// Set the s matrix diagonal values used to create the new orthogonalized
// matrix.
template<typename Dtype>
void Orthogonalizer<Dtype>::SetSingularVals(
    const Eigen::JacobiSVD<Matrix>& svd,
    Matrix& s,
    FillerParameter_Orthogonalization orthog) {

  const Matrix& singular_vals = svd.singularValues();
  CHECK(singular_vals.cols() == 1)
      << "Singular values matrix should be Nx1 col vect";

  Dtype s_val;
  switch(orthog) {
    case(FillerParameter_Orthogonalization_PRESERVE_TRACE_NORM):
    {
      s_val = SVAverage(singular_vals);
      break;
    }
    case(FillerParameter_Orthogonalization_PRESERVE_FROB_NORM):
    {
      s_val = FrobNormSV(singular_vals);
      break;
    }
    case(FillerParameter_Orthogonalization_PRESERVE_1_NORM):
    {
      s_val = LargestSV(singular_vals);
      break;
    }
    default:
    {
      LOG(FATAL) << "Bad option in orthogonalization.";
    }
  }

  int min_dim = s.rows() < s.cols() ? s.rows() : s.cols();
  for(int i=0; i < min_dim; ++i) {
    s(i,i) = s_val;
  }
}

// Assign the blob its nearest orthogonal matrix.
template<typename Dtype>
void Orthogonalizer<Dtype>::PreserveNorm(MatrixMap& blob_mat,
    FillerParameter_Orthogonalization orthog) {
  unsigned int options = blob_mat.cols() > blob_mat.rows() ?
        Eigen::ComputeThinU | Eigen::ComputeFullV :
        Eigen::ComputeThinU | Eigen::ComputeThinV;

  Eigen::JacobiSVD<Matrix> svd(blob_mat, options);
  const Matrix& u = svd.matrixU();
  const Matrix& v = svd.matrixV();

  Matrix s = Matrix::Zero(u.cols(), v.rows());
  SetSingularVals(svd, s, orthog);

  // This assignment modifies the underlying blob through its mutable_cpu_data
  // pointer, since m is a MatrixMap.
  blob_mat = u*s*v;
}

template<typename Dtype>
void Orthogonalizer<Dtype>::Execute(Blob<Dtype>& blob,
                                    FillerParameter_Orthogonalization orthog) {
  CHECK(orthog != FillerParameter_Orthogonalization_NONE) <<
      "Bad orthogonalization option.";
  MatrixMap m = BlobToMat(blob);
  if (FillerParameter_Orthogonalization_SIMPLE == orthog) {
    Simple(m);
  } else {
    PreserveNorm(m, orthog);
  }
}

template<typename Dtype>
void Orthogonalizer<Dtype>::Invert(const Matrix& source, Matrix& target)
{
  unsigned int options = Eigen::ComputeFullU | Eigen::ComputeFullV;
  Eigen::JacobiSVD<Matrix> svd(source, options);
  const Matrix& u = svd.matrixU();
  const Matrix& v = svd.matrixV();
  const Matrix& sing_vals = svd.singularValues();

  Matrix sprime = Matrix::Zero(v.cols(), u.rows());

  const Dtype EPSILON = 0.000001;
  for(int i=0; i < sing_vals.rows(); ++i)
  {
    Dtype val = sing_vals(i,0);
    if (val > EPSILON)
    {
      sprime(i,i) = static_cast<Dtype>(1)/val;
    }
    else
    {
      sprime(i,i) = 0.0;
    }
  }

  target = v * sprime * u.transpose();
}

template<typename Dtype>
void Orthogonalizer<Dtype>::Invert(const Blob<Dtype>& source,
                                          Blob<Dtype>& target) {
//  std::cout << "Blob source shape:";
//  for(int i=0; i < source.shape().size(); ++i)
//    std::cout << source.shape()[i] << ",";
//  std::cout << std::endl;

//  std::cout << "Blob target shape:";
//  for(int i=0; i < target.shape().size(); ++i)
//    std::cout << target.shape()[i] << ",";
//  std::cout << std::endl;

  ConstMatrixMap m_source = BlobToMat(source);
//  bool transpose = m_source.rows() > m_source.cols();
//  if ( transpose)
//  {
//    Matrix m_target = BlobToMat(target).transpose();
//    Invert(m_source.transpose(), m_target);
//    MatToBlob(m_target.transpose(), target);
//  }
//  else
//  {
    Matrix m_target = BlobToMat(target);
    Invert(m_source, m_target);
    MatToBlob(m_target, target);
//  }
}


// explicit instantiation
template class Orthogonalizer<float>;
template class Orthogonalizer<double>;

}  // namespace caffe
#endif  // USE_EIGEN
