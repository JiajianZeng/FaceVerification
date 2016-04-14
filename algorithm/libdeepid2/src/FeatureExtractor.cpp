#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "deepid2/FeatureExtractor.hpp"

using std::string;
using caffe::Net;
using caffe::Blob;
using caffe::Caffe;
using std::vector;
using boost::shared_ptr;

/**
* parameter param_file, the definition file of the net structure
* parameter trained_model_file, the model file of the net 
*/
FeatureExtractor::FeatureExtractor(const string& param_file, const string& trained_model_file,
      bool use_gpu, int device_id) {
  if(use_gpu && device_id >= 0){
    Caffe::set_mode(Caffe::GPU);    
    Caffe::SetDevice(device_id);
  }else{
    Caffe::set_mode(Caffe::CPU);
  }

  net_ = new Net<float>(param_file, caffe::TEST);
  net_->CopyTrainedLayersFrom(trained_model_file); 
}

FeatureExtractor::~FeatureExtractor() {
  delete net_;
}

/**
* parameter feature_blob_names, holds the blob names which we want to extract feature from
* parameter net_input_blobs, holds the data of the input layer of the net
* parameter feature_dim_vecs, when this function return, will store dimension of feature extracted from each layer
* parameter feature_blob_data, holds the data extracted from each layer
*/
void FeatureExtractor::extract(const vector<string>& feature_blob_names, const vector<Blob<float>* >& net_input_blobs, 
    vector<int>& feature_dim_vecs, vector<vector<float*> >& feature_blob_data) {
  net_->Forward(net_input_blobs);
  
  vector<shared_ptr<Blob<float> > > feature_blobs;
  for(int i = 0;i < feature_blob_names.size();i++){
    feature_blobs.push_back(net_->blob_by_name(feature_blob_names[i]));
  }
  // parse data 
  parse_blob_data(feature_blobs, feature_dim_vecs, feature_blob_data);
}

void FeatureExtractor::parse_blob_data(const vector<shared_ptr<Blob<float> > >& feature_blobs, vector<int>& feature_dim_vecs,
      vector<vector<float*> >& feature_blob_data) {
  for(int i = 0;i < feature_blobs.size();i++){
    const shared_ptr<Blob<float> > feature_blob = feature_blobs[i];
    int batch_size = feature_blob->num();
    int dim_features = feature_blob->count() / batch_size;
    
    feature_dim_vecs.push_back(dim_features);
    // store data
    for(int n = 0;n < batch_size;n++){
      feature_blob_data[i].push_back(feature_blob->mutable_cpu_data() + feature_blob->offset(n));
    }
  }
}

Net<float>* FeatureExtractor::get_net() {
  return net_;
}
