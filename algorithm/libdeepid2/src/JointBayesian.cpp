#include <opencv2/core/core.hpp>

#include "deepid2/JointBayesian.hpp"

using cv::Mat;

JointBayesian::JointBayesian(const Mat& A, const Mat& G, const Mat& Sw, const Mat& Su) {
  A_ = A.clone();
  G_ = G.clone();
  Sw_ = Sw.clone();
  Su_ = Su.clone();
}

// we need to make sure x1 and x2 are both col vector
double JointBayesian::distance(const Mat& x1, const Mat& x2) {
  return ((cv::Mat)(x1.t() * A_ * x1 + x2.t() * A_ * x2 - 2 * x1.t() * G_ * x2)).at<double>(0);
}
