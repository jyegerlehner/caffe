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
  int min_dim = S.rows() < S.cols() ? S.rows() : S.cols();

  Dtype avg = 0.0;
  for(int i=0; i < min_dim; ++i) {
    avg += S(i,i);
  }
  avg /= min_dim;
  return avg;
}

// Compute the 1/N * square root of the sum of the squares of the original
// matrix's singular values, so we can preserve it (PRESERVE_FROB_NORM).
template<typename Dtype>
Dtype Orthogonalizer<Dtype>::FrobNormSV(const Matrix& S) {
  int min_dim = S.rows() < S.cols() ? S.rows() : S.cols();

  Dtype accum = 0.0;
  for(int i=0; i < min_dim; ++i) {
    Dtype entry = S(i,i);
    accum += entry*entry;
  }
  accum = std::sqrt(accum);
  accum /= min_dim;
  return accum;
}

// Find the largest singular value so we can preserve it (PRESERVE_1_NORM).
template<typename Dtype>
Dtype Orthogonalizer<Dtype>::LargestSV(const Matrix& S) {
  int min_dim = S.rows() < S.cols() ? S.rows() : S.cols();

  Dtype largest = 0.0;
  for(int i=0; i < min_dim; ++i) {
    Dtype entry = S(i,i);
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

  Dtype s_val;
  switch(orthog) {
    case(FillerParameter_Orthogonalization_PRESERVE_TRACE_NORM):
    {
      s_val = SVAverage(svd.singularValues());
      break;
    }
    case(FillerParameter_Orthogonalization_PRESERVE_FROB_NORM):
    {
      s_val =FrobNormSV(svd.singularValues());
      break;
    }
    case(FillerParameter_Orthogonalization_PRESERVE_1_NORM):
    {
      s_val = LargestSV(svd.singularValues());
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

// explicit instantiation
template class Orthogonalizer<float>;
template class Orthogonalizer<double>;

}  // namespace caffe
#endif  // USE_EIGEN
