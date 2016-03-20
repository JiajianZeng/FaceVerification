#ifndef SVM_CLASSIFIER_HPP
#define SVM_CLASSIFIER_HPP

#include <opencv2/core/core.hpp>
#include <string>

using cv::Mat;
using std::string;

class SvmClassifier {
 public:
  SvmClassifier(const string& svm_model_file);
  bool classify(float distance);
 private:
  Mat Parameters_;
  Mat nr_class_;          /* number of class, = 2 in regression/one class svm */
  Mat totalSV_;           /* total #SV */
  Mat rho_;               /* constants in decision functions(rho[k*(k-1)/2]) */
  Mat Label_;             /* label of each class(Label[k]) */
  Mat sv_indices_;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_training_data] to indicate SVs in the training set */
  
  /* pairwise probability information */
  Mat ProbA_;
  Mat ProbB_;
  /* number of SVs for each class(nSV[K]) */
  /* nSV[0] + nSV[1] + ... + nSV[k-1] = 1 */
  Mat nSV_;

  Mat sv_coef_;
  Mat SVs_;
  float weighted_sum_;
 private:
  static const string Parameters_name_;
  static const string nr_class_name_;
  static const string totalSV_name_;
  static const string rho_name_;
  static const string Label_name_;
  static const string sv_indices_name_;
  static const string ProbA_name_;
  static const string ProbB_name_;
  static const string nSV_name_;
  static const string sv_coef_name_;
  static const string SVs_name_;
  
};

#endif
