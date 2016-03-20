#ifndef JOINT_BAYESIAN_HPP
#define JOINT_BAYESIAN_HPP

#include <opencv2/core/core.hpp>
#include <string>

using std::string;
using cv::Mat;

class JointBayesian {
 public:
  JointBayesian(const string& yaml_file_A, const string& matrix_name_A,
      const string& yaml_file_G, const string& matrix_name_G);
  JointBayesian(const Mat& A, const Mat& G);
  float distance(const Mat& source1, const Mat& source2);
 private:
  Mat A_;
  Mat G_;
};
#endif
