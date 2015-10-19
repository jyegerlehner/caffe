#ifdef USE_EIGEN
#ifndef ORTHOGONAL_HPP
#define ORTHOGONAL_HPP

#include "caffe/blob.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

namespace caffe {

template<typename Dtype>
class Orthogonalizer {
public:
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  typedef Eigen::Map<const Matrix> ConstMatrixMap;
  typedef Eigen::Map<Matrix> MatrixMap;

  // Returns the blob in the form of a "matrix map" which is an Eigen matrix,
  // except it does not own its data storage; rather the data is provided at the
  // time of its construction via an externally-allocated pointer.
  static ConstMatrixMap BlobToMat( const Blob<Dtype>& blob );
  static MatrixMap BlobToMat( Blob<Dtype>& blob );

  // Assigns the contents of the matrix to the blob. Shapes of the arguments
  // must be consistent.
  static void MatToBlob(const Matrix& m, Blob<Dtype>& blob);

  // Compute an orthogonal matrix from the blob.
  static void Fast(Blob<Dtype>& blob);

  // Compute the nearest orthogonal matrix to that of the blob.
  static void Nearest(Blob<Dtype>& blob);
protected:
  // Create the shape of the matrix corresponding to the blob. Returned
  // rows and columns.
  static std::pair<int,int> FlatShape(const Blob<Dtype>& blob);
};

} // namespace caffe

#endif  // ORTHOGONAL_HPP
#endif  // USE_EIGEN
