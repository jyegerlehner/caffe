#ifdef USE_EIGEN
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
void Orthogonalizer<Dtype>::Fast(Blob<Dtype>& blob) {
  MatrixMap m = BlobToMat(blob);

  unsigned int options = Eigen::ComputeThinU | Eigen::ComputeThinV;
  Eigen::JacobiSVD<Matrix> svd(m, options);
  const Matrix& u = svd.matrixU();
  const Matrix& v = svd.matrixV();

  bool use_u = (u.rows() == m.rows() && u.cols() == m.cols());
  bool use_v = (v.cols() == m.rows() && v.rows() == m.cols());
  CHECK(use_u || use_v) << "Orthogonalize error: bad matrix dimensions.";
  if (use_u) {
    // This assignment modifies the underlying blob through its mutable_cpu_data
    // pointer, since m is a MatrixMap.
    m = u;
  }
  else
  {
    // This assignment modifies the underlying blob through its mutable_cpu_data
    // pointer, since m is a MatrixMap.
    m = v.transpose();
  }
}

// Assign the blob its nearest orthogonal matrix.
template<typename Dtype>
void Orthogonalizer<Dtype>::Nearest(Blob<Dtype>& blob) {
  MatrixMap m = BlobToMat(blob);

  unsigned int options = m.cols() > m.rows() ?
        Eigen::ComputeThinU | Eigen::ComputeFullV :
        Eigen::ComputeThinU | Eigen::ComputeThinV;

  Eigen::JacobiSVD<Matrix> svd(m, options);
  const Matrix& u = svd.matrixU();
  const Matrix& v = svd.matrixV();

  Matrix s = Matrix::Zero(u.cols(), v.rows());
  int min_dim = s.rows() < s.cols() ?  s.rows() : s.cols();
  for(int i = 0; i < min_dim; ++i)
  {
    s(i,i) = 1.0;
  }

  // This assignment modifies the underlying blob through its mutable_cpu_data
  // pointer, since m is a MatrixMap.
  m = u*s*v;
}

// explicit instantiation
template class Orthogonalizer<float>;
template class Orthogonalizer<double>;

}  // namespace caffe
#endif  // USE_EIGEN
