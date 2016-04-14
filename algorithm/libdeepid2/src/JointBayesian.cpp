#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "deepid2/JointBayesian.hpp"

using cv::Mat;
using cv::FileStorage;
using std::string;

JointBayesian::JointBayesian(const Mat& A, const Mat& G) {
  A_ = A.clone();
  G_ = G.clone();
}

/**
* parameter yaml_file_A, yaml file for Mat A_
* parameter matrix_name_A, opencv-matrix name for Mat A_,
* ...
*/
JointBayesian::JointBayesian(const string& yaml_file_A, const string& matrix_name_A,
    const string& yaml_file_G, const string& matrix_name_G) {
  FileStorage fs_A(yaml_file_A.c_str(), FileStorage::READ);
  FileStorage fs_G(yaml_file_G.c_str(), FileStorage::READ);
   
  fs_A[matrix_name_A.c_str()] >> A_;
  fs_G[matrix_name_G.c_str()] >> G_;
  // matrix in matlab is col-major, while matrix in opencv is row-major
  // so when we serialize matrix in matlab to yaml file
  // and deserialize the yaml file into matrix in opencv
  // we need to transpose it
  A_ = A_.t();
  G_ = G_.t();
    
  fs_A.release();
  fs_G.release();
}

/** 
* we need to make sure x1 and x2 are both col vector
* that's if A_ and G_ are d by d matrixes, then x1 and x2 must be d by 1
*/
float JointBayesian::distance(const Mat& x1, const Mat& x2) {
  // actually, result is a scalar
  Mat result = x1.t() * A_ * x1 + x2.t() * A_ * x2 - 2 * x1.t() * G_ * x2;
  return result.at<float>(0,0);
}
