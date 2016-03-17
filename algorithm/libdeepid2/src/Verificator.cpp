#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>

#include "deepid2/Verificator.hpp"
#include "caffe/caffe.hpp"

using std::vector;
using std::string;
using cv::Mat;
using caffe::Net;
using caffe::Blob;
using caffe::BlobProto;
using std::ofstream;

Verificator::Verificator(const string& param_file,
                         const string& trained_model_file,
                         const string& mean_file,
                         const bool use_gpu,
                         const int device_id) {
  fe_ = new FeatureExtractor(param_file, trained_model_file, use_gpu, device_id);
  set_mean(mean_file);
}

Verificator::~Verificator() {
  delete fe_;
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
  Mat mean;
  cv::merge(channels, mean);
  
  /* compute the global mean pixel value and create a mean image filled with this value */
  cv::Scalar channel_mean = cv::mean(mean);
  cv::Size sz(mean_blob.width(), mean_blob.height());
  mean_ = Mat(sz, mean.type(), channel_mean);
}

// 
void Verificator::wrap_input_layer(vector<vector<Mat*>* >& input_channels_vec) {
  Net<float>* net = fe_->get_net();
  for(int i = 0;i < net->input_blobs().size();i++){
    vector<Mat*>* input_channels = new vector<Mat*>();

    Blob<float>* input_layer = net->input_blobs()[i];
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();
    for(int j = 0;j < input_layer->channels();j++){
      Mat* channel = new Mat(height, width, CV_32FC1, input_data);
      input_channels->push_back(channel);
      input_data += width * height;
    }
    input_channels_vec.push_back(input_channels);
  }  
}

void Verificator::preprocess(const Mat& img, vector<Mat>* input_channels, Blob<float>* input_layer){
  Mat sample;
  int num_channels = input_channels->size();
  /* convert the input image to the input image format of the network */
  if(img.channels() == 3 && num_channels == 1)
    cv::cvtColor(img, sample, cv::COLOR_RGB2GRAY);
  else if(img.channels() == 4 && num_channels == 1)
    cv::cvtColor(img, sample, cv::COLOR_RGBA2GRAY);
  else if(img.channels() == 4 && num_channels == 3)
    cv::cvtColor(img, sample, cv::COLOR_RGBA2RGB);
  else if(img.channels() == 1 && num_channels == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2RGB);
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
  
  /* write the separate RGB planes directly to the input layer of the network */
  cv::split(sample_normalized, *input_channels);

  bool equal = (float*)(input_channels->at(0).data) == input_layer->cpu_data();
  
}

void Verificator::extract_feature(const Mat& image1, const Mat& image2, vector<string> feature_blob_names, 
      Mat* feature1, Mat* feature2){
  /* prepare the source image for extracting feature*/
  Mat img1, img2;
  cv::cvtColor(image1, img1, cv::COLOR_BGR2RGB);
  cv::cvtColor(image2, img2, cv::COLOR_BGR2RGB);

  vector<Mat> img_vec;
  img_vec.push_back(img1);
  img_vec.push_back(img2);

  /* wrap input layer */
  vector<vector<Mat*>* >  input_channels_vec;
  wrap_input_layer(input_channels_vec);

  /* preprocess the source image and copy data to input layer of the layer */
  Net<float>* net = fe_->get_net();
  for(int i = 0;i < input_channels_vec.size();i++){
    vector<Mat*>* input_channels = input_channels_vec[i];
    vector<Mat>* v = new vector<Mat>();
    for(int j = 0;j < input_channels->size();j++)
      v->push_back(*input_channels->at(j));
    preprocess(img_vec[i], v, net->input_blobs()[i]);
  }

  /* */
  vector<Blob<float>* > net_input_blobs;
  for(int i = 0;i < net->input_blobs().size();i++)
    net_input_blobs.push_back(net->input_blobs()[i]);

  /* */
  vector<vector<const float*> > feature_blob_data;
  vector<const float*> blob_data1;
  vector<const float*> blob_data2;
  feature_blob_data.push_back(blob_data1);
  feature_blob_data.push_back(blob_data2);

  /* */
  vector<int> feature_dim_vecs;
  fe_->extract(feature_blob_names, net_input_blobs, feature_dim_vecs, feature_blob_data);

  ofstream of("feature2.txt");
  for(int i = 0;i < feature_dim_vecs.size();i++) {
    std::cout << feature_blob_names[i] << "," << feature_dim_vecs[i] << std::endl;
    of << "feature of " << feature_blob_names[i] << std::endl;
    for(int j = 0;j < feature_blob_data[i].size();j++){
      for(int n = 0;n < feature_dim_vecs[i];n++){
        of << *(feature_blob_data[i][j]++) << ",";
      }
      of << std::endl;
    }
    of << std::endl;
  }
  of.close();
  
}

FeatureExtractor* Verificator::get_feature_extractor() {
  return fe_;
}

bool Verificator::verificate(const Mat& image1, const Mat& image2, vector<string> feature_blob_names, 
      Mat* feature1, Mat* feature2) {
  extract_feature(image1, image2, feature_blob_names, feature1, feature2);
  return true;
}









