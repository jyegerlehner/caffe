#ifdef USE_EIGEN
#ifndef ORTHOGONAL_HPP
#define ORTHOGONAL_HPP

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

namespace caffe {

template<typename Dtype>
class Orthogonalizer {
public:
  typedef Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matrix;
  typedef Eigen::Map<const Matrix> ConstMatrixMap;
  typedef Eigen::Map<Matrix> MatrixMap;

  // Returns the blob in the form of a "matrix map" which is an Eigen matrix,
  // except it does not own its data storage; rather the data is provided at
  // the time of its construction via an externally-allocated pointer.
  static ConstMatrixMap BlobToMat( const Blob<Dtype>& blob );
  static MatrixMap BlobToMat( Blob<Dtype>& blob );

  // Assigns the contents of the matrix to the blob. Shapes of the arguments
  // must be consistent.
  static void MatToBlob(const Matrix& m, Blob<Dtype>& blob);

  // Compute an orthogonal blob, optionally preserving a norm of the original
  // matrix.
  static void Execute(Blob<Dtype>& blob,
                       FillerParameter_Orthogonalization orthog);

  static void Invert(const Matrix& source, Matrix& target);
  static void Invert(const Blob<Dtype>& source, Blob<Dtype>& target);
  static void InvertDeconv(const Blob<Dtype>& source, Blob<Dtype>& target);
protected:
  // Create the shape of the matrix corresponding to the blob. Returned
  // rows and columns.
  static std::pair<int,int> FlatShape(const Blob<Dtype>& blob);

  // Make the blob's matrix orthogonal.
  static void Simple(MatrixMap& blob_mat);

  static void PreserveNorm(MatrixMap& blob_mat,
                           FillerParameter_Orthogonalization orthog);

  static void SetSingularVals(const Eigen::JacobiSVD<Matrix>& svd,
                              Matrix& s,
                              FillerParameter_Orthogonalization orthog);

  static Dtype LargestSV(const Matrix& S);
  static Dtype FrobNormSV(const Matrix& S);
  static Dtype SVAverage(const Matrix& S);
};

} // namespace caffe

#endif  // ORTHOGONAL_HPP
#endif  // USE_EIGEN
