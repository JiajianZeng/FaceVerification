#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "deepid2/FeatureExtractor.hpp"

using std::string;
using caffe::Net;
using caffe::Blob;
using caffe::Caffe;
using std::vector;
using boost::shared_ptr;

FeatureExtractor::FeatureExtractor(const string& param_file, const string& trained_model_file,
      bool use_gpu, int device_id) {
  if(use_gpu && device_id >= 0){
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  }else{
    Caffe::set_mode(Caffe::CPU);
  }

  net_ = new Net<float>(param_file, caffe::TEST);
  net_->CopyTrainedLayersFrom(trained_model_file); 
}

FeatureExtractor::~FeatureExtractor() {
  delete net_;
}

void FeatureExtractor::extract(vector<string> feature_blob_names, vector<Blob<float>* > net_input_blobs, vector<int>& feature_dim_vecs,
      vector<vector<const float*> >& feature_blob_data) {
  vector<shared_ptr<Blob<float> > > feature_blobs;
  int num_features = feature_blob_names.size();
  net_->Forward(net_input_blobs);

  for(int i = 0;i < num_features;i++){
    feature_blobs.push_back(net_->blob_by_name(feature_blob_names[i]));
  }

  parse_blob_data(feature_blobs, feature_dim_vecs, feature_blob_data);
}

void FeatureExtractor::parse_blob_data(vector<shared_ptr<Blob<float> > > feature_blobs, vector<int>& feature_dim_vecs,
      vector<vector<const float*> >& feature_blob_data) {
  int num_blobs = feature_blobs.size();

  for(int i = 0;i < num_blobs;i++){
    const shared_ptr<Blob<float> > feature_blob = feature_blobs[i];
    int batch_size = feature_blob->num();
    int dim_features = feature_blob->count() / batch_size;
    
    feature_dim_vecs.push_back(dim_features);
    
    for(int n = 0;n < batch_size;n++){
      feature_blob_data[i].push_back(feature_blob->cpu_data() + feature_blob->offset(n));
    }
  }
}

Net<float>* FeatureExtractor::get_net() {
  return net_;
}
