#ifndef JOINT_BAYESIAN_HPP
#define JOINT_BAYESIAN_HPP

#include <opencv2/core/core.hpp>

using cv::Mat;
class JointBayesian {
 public:
  JointBayesian(const Mat& A, const Mat& G, const Mat& Sw, const Mat& Su);
  double distance(const Mat& source1, const Mat& source2);
 private:
  Mat A_;
  Mat G_;
  Mat Sw_;
  Mat Su_;
};
#endif
