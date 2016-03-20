#ifndef VERIFICATOR_HPP_
#define VERIFICATOR_HPP_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "deepid2/FeatureExtractor.hpp"
#include "deepid2/JointBayesian.hpp"
#include "deepid2/SvmClassifier.hpp"
#include "caffe/caffe.hpp"

using cv::Mat;
using caffe::Blob;
using std::vector;
using std::string;
using boost::shared_ptr;

class Verificator {
 public:
  Verificator(const string& yaml_config_file);
  ~Verificator();
  bool verificate(const Mat& image1, const Mat& image2, vector<string> feature_blob_names, 
      Mat& feature1, Mat& feature2);

  FeatureExtractor* get_feature_extractor();
  JointBayesian* get_joint_bayesian();
  SvmClassifier* get_svm_classifier();
 private:
  FeatureExtractor* fe_;
  JointBayesian* jb_;
  SvmClassifier* svm_;
  Mat mean_;
  Mat feature_mean_;
private:
  static const string param_file_name_;
  static const string trained_model_file_name_;
  static const string use_gpu_name_;
  static const string device_id_name_;
  static const string yaml_file_A_name_;
  static const string yaml_file_G_name_;
  static const string matrix_A_name_;
  static const string matrix_G_name_;
  static const string svm_model_file_name_;
  static const string mean_file_name_;  
  static const string feature_mean_file_name_;
  static const string matrix_feature_mean_name_;

 protected:
  void set_mean(const string& mean_file);
  void wrap_input_layer(vector<shared_ptr<vector<shared_ptr<Mat> > > >& input_channels_vec);
  void preprocess(const Mat& img, shared_ptr<vector<Mat> > input_channels, Blob<float>* input_layer);
  void extract_feature(const Mat& image1, const Mat& image2, vector<string> feature_blob_names, 
      Mat& feature1, Mat& feature2);
};
#endif
