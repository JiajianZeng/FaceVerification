#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "deepid2/SvmClassifier.hpp"

using std::string;
using cv::Mat;
using cv::FileStorage;

// static member initialization
const string SvmClassifier::Parameters_name_("Parameters");
const string SvmClassifier::nr_class_name_("nr_class");
const string SvmClassifier::totalSV_name_("totalSV");
const string SvmClassifier::rho_name_("rho");
const string SvmClassifier::Label_name_("Label");
const string SvmClassifier::sv_indices_name_("sv_indices");
const string SvmClassifier::ProbA_name_("ProbA");
const string SvmClassifier::ProbB_name_("ProbB");
const string SvmClassifier::nSV_name_("nSV");
const string SvmClassifier::sv_coef_name_("sv_coef");
const string SvmClassifier::SVs_name_("SVs");

SvmClassifier::SvmClassifier(const string& svm_model_file) {
  FileStorage fs_svm_model(svm_model_file.c_str(), FileStorage::READ);
  
  fs_svm_model[Parameters_name_.c_str()] >> Parameters_;
  fs_svm_model[nr_class_name_.c_str()] >> nr_class_;
  fs_svm_model[totalSV_name_.c_str()] >> totalSV_;
  fs_svm_model[rho_name_.c_str()] >> rho_;
  fs_svm_model[Label_name_.c_str()] >> Label_;
  fs_svm_model[sv_indices_name_.c_str()] >> sv_indices_;
  fs_svm_model[ProbA_name_.c_str()] >> ProbA_;
  fs_svm_model[ProbB_name_.c_str()] >> ProbB_;
  fs_svm_model[nSV_name_.c_str()] >> nSV_;
  fs_svm_model[sv_coef_name_.c_str()] >> sv_coef_;
  fs_svm_model[SVs_name_.c_str()] >> SVs_;

  Parameters_ = Parameters_.t();
  nr_class_ = nr_class_.t();
  totalSV_ = totalSV_.t();
  rho_ = rho_.t();
  Label_ = Label_.t();
  sv_indices_ = sv_indices_.t();
  // ProbA_ = ProbA_.t();
  // ProbB_ = ProbB_.t();
  nSV_ = nSV_.t();
  sv_coef_ = sv_coef_.t();
  SVs_ = SVs_.t();

  weighted_sum_ = sv_coef_.dot(SVs_);
  std::cout << "weighted_sum = " << weighted_sum_ << std::endl;
  std::cout << "rho = " << rho_.at<float>(0) << std::endl;

  fs_svm_model.release();
}

bool SvmClassifier::classify(float distance) {
  // using linear kernel here
  return distance * weighted_sum_ - rho_.at<float>(0) > 0;
}
