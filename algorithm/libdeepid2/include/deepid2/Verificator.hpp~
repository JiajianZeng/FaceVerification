#ifndef VERIFICATOR_HPP_
#define VERIFICATOR_HPP_

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

#include "deepid2/FeatureExtractor.hpp"
#include "caffe/caffe.hpp"

using cv::Mat;
using caffe::Blob;
using std::vector;
using std::string;

class Verificator {
 public:
  Verificator(const string& param_file,
              const string& trained_model_file,
              const string& mean_file,
              const bool use_gpu = false,
              const int device_id = -1);
  ~Verificator();
  bool verificate(const Mat& image1, const Mat& image2, vector<string> feature_blob_names, 
      Mat* feature1, Mat* feature2);
  FeatureExtractor* get_feature_extractor();
 private:
  FeatureExtractor* fe_;
  Mat mean_;  

 protected:
  void set_mean(const string& mean_file);
  void wrap_input_layer(vector<vector<Mat*>* >& input_channels_vec);
  void preprocess(const Mat& img, vector<Mat>* input_channels, Blob<float>* input_layer);
  void extract_feature(const Mat& image1, const Mat& image2, vector<string> feature_blob_names, 
      Mat* feature1, Mat* feature2);
};
#endif
