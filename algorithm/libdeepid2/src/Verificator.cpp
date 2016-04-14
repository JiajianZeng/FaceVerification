#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#include "deepid2/Verificator.hpp"
#include "caffe/caffe.hpp"

using std::vector;
using std::string;
using cv::Mat;
using cv::FileStorage;
using caffe::Net;
using caffe::Blob;
using caffe::BlobProto;
using std::ofstream;
using boost::shared_ptr;

// static member initialization
const string Verificator::param_file_name_("param_file");
const string Verificator::trained_model_file_name_("trained_model_file");
const string Verificator::use_gpu_name_("use_gpu");
const string Verificator::device_id_name_("device_id");
const string Verificator::yaml_file_A_name_("yaml_file_A");
const string Verificator::yaml_file_G_name_("yaml_file_G");
const string Verificator::matrix_A_name_("matrix_A_name");
const string Verificator::matrix_G_name_("matrix_G_name");
const string Verificator::svm_model_file_name_("svm_model_file");
const string Verificator::mean_file_name_("mean_file");

Verificator::Verificator(const string& yaml_config_file) {
  FileStorage fs(yaml_config_file.c_str(), FileStorage::READ);
  string param_file;  
  string trained_model_file;
  bool use_gpu;
  int device_id;
  string yaml_file_A;
  string yaml_file_G;
  string matrix_A_name;
  string matrix_G_name;
  string svm_model_file;
  string mean_file;
  
  // deserialize
  fs[param_file_name_.c_str()] >> param_file;
  fs[trained_model_file_name_.c_str()] >> trained_model_file;
  fs[use_gpu_name_.c_str()] >> use_gpu;
  fs[device_id_name_.c_str()] >> device_id;
  fs[yaml_file_A_name_.c_str()] >> yaml_file_A;
  fs[yaml_file_G_name_.c_str()] >> yaml_file_G;
  fs[matrix_A_name_.c_str()] >> matrix_A_name;
  fs[matrix_G_name_.c_str()] >> matrix_G_name;
  fs[svm_model_file_name_.c_str()] >> svm_model_file;
  fs[mean_file_name_.c_str()] >> mean_file;

  // initialize
  fe_ = new FeatureExtractor(param_file, trained_model_file, use_gpu, device_id);
  jb_ = new JointBayesian(yaml_file_A, matrix_A_name, yaml_file_G, matrix_G_name);
  svm_ = new SvmClassifier(svm_model_file);
  set_mean(mean_file);

  fs.release();
}

Verificator::~Verificator() {
  delete fe_;
  delete jb_;
  delete svm_;
}

void Verificator::set_mean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
  
  /* convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);

  vector<Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for(int i = 0;i < mean_blob.channels();i++){
    Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* merge the seprate channels into a single image */
  cv::merge(channels, mean_);
}

/**
* point single Mat channel to input layer, and when preprocess is done, the data will be filled
*/
void Verificator::wrap_input_layer(vector<shared_ptr<vector<shared_ptr<Mat> > > >& input_channels_vec) {
  Net<float>* net = fe_->get_net();
  for(int i = 0;i < net->input_blobs().size();i++){
    shared_ptr<vector<shared_ptr<Mat> > > input_channels(new vector<shared_ptr<Mat> >() );
    Blob<float>* input_layer = net->input_blobs()[i];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for(int j = 0;j < input_layer->channels();j++){
      shared_ptr<Mat> channel(new Mat(height, width, CV_32FC1, input_data));
      input_channels->push_back(channel);
      input_data += width * height;
    }
    input_channels_vec.push_back(input_channels);
  }  
}

void Verificator::preprocess(const Mat& img, shared_ptr<vector<Mat> > input_channels, Blob<float>* input_layer){
  Mat sample;
  int num_channels = input_channels->size();
  /* convert the input image to the input image format of the network */
  if(img.channels() == 3 && num_channels == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if(img.channels() == 4 && num_channels == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if(img.channels() == 4 && num_channels == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if(img.channels() == 1 && num_channels == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  /* resize */
  cv::Size input_geometry(input_layer->width(), input_layer->height());
  Mat sample_resized;
  if(sample.size() != input_geometry)
    cv::resize(sample, sample_resized, input_geometry);
  else
    sample_resized = sample;

  /* float */
  Mat sample_float;
  if(sample_resized.channels() == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  /* normalized */
  Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);
  
  /* write the separate BGR planes directly to the input layer of the network */
  //cv::split(sample_normalized, *input_channels);
  cv::split(sample_float, *input_channels);
}

void Verificator::extract_feature(const Mat& image1, const Mat& image2, const vector<string>& feature_blob_names, 
      Mat& feature1, Mat& feature2){
  /* prepare the source image for extracting feature*/
  Mat img1, img2;
  vector<Mat> img_vec;
  img1 = image1.clone();
  img2 = image2.clone();
  img_vec.push_back(img1);
  img_vec.push_back(img2);

  /* wrap input layer */
  vector<shared_ptr<vector<shared_ptr<Mat> > > >  input_channels_vec;
  wrap_input_layer(input_channels_vec);

  /* preprocess the source image and copy data to input layer of the net */
  /* we need to make sure input_channels_vec.size() == 2 */
  Net<float>* net = fe_->get_net();
  for(int i = 0;i < input_channels_vec.size();i++){
    shared_ptr<vector<shared_ptr<Mat> > > input_channels = input_channels_vec[i];
    shared_ptr<vector<Mat> > v(new vector<Mat>());
    for(int j = 0;j < input_channels->size();j++)
      v->push_back(*input_channels->at(j));
    preprocess(img_vec[i], v, net->input_blobs()[i]);
  }

  /* actually, we don't need this operation */
  /* we need make sure net->input_blobs().size() == 2 */
  vector<Blob<float>* > net_input_blobs;
  for(int i = 0;i < net->input_blobs().size();i++)
    net_input_blobs.push_back(net->input_blobs()[i]);

  /* there are two blobs we want to extract features from */
  vector<vector<float*> > feature_blob_data;
  vector<float*> blob_data1, blob_data2;
  feature_blob_data.push_back(blob_data1);
  feature_blob_data.push_back(blob_data2);

  /* extract features */
  vector<int> feature_dim_vecs;
  fe_->extract(feature_blob_names, net_input_blobs, feature_dim_vecs, feature_blob_data);

  // col vector
  feature1 = Mat(feature_dim_vecs[0], 1, CV_32FC1, feature_blob_data[0][0]);
  /* debug
  std::ofstream of("feature1.txt");
  for(int i = 0;i < feature_dim_vecs[0];i++)
    of << feature1.at<float>(i) << ",";
  of.close();
  end debug */
  
  // col vector
  feature2 = Mat(feature_dim_vecs[1], 1, CV_32FC1, feature_blob_data[1][0]);
  /* debug
  std::ofstream of1("feature2.txt");
  for(int i = 0;i < feature_dim_vecs[1];i++)
    of1 << feature2.at<float>(i) << ",";
  of1.close();
  end debug */
}

FeatureExtractor* Verificator::get_feature_extractor() {
  return fe_;
}

JointBayesian* Verificator::get_joint_bayesian() {
  return jb_;
}

SvmClassifier* Verificator::get_svm_classifier() {
  return svm_;
}

bool Verificator::verificate(const Mat& image1, const Mat& image2, const vector<string>& feature_blob_names, 
      Mat& feature1, Mat& feature2) {
  extract_feature(image1, image2, feature_blob_names, feature1, feature2);
  // norm normalize 
  cv::normalize(feature1, feature1, 1, 0, cv::NORM_L1);
  cv::normalize(feature2, feature2, 1, 0, cv::NORM_L1);
  // compute Euclidean distance
  double distance = cv::norm(feature1, feature2);
  return svm_->classify(distance);
}
