#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"
#include "deepid2/FeatureExtractor.hpp"

using std::string;
using caffe::Net;
using caffe::Blob;
using caffe::Caffe;
using std::vector;
using boost::shared_ptr;

template <typename Dtype>
FeatureExtractor<Dtype>::FeatureExtractor(const string& param_file, const string& trained_model_file,
      const Net<Dtype>* root_net, bool use_gpu, int device_id) {
  if(use_gpu && device_id >= 0){
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  }else{
    Caffe::set_mode(Caffe::CPU);
  }

  net_ = new Net<Dtype>(param_file, caffe::TEST, root_net);
  net_->CopyTrainedLayersFrom(trained_model_file); 
}

template <typename Dtype>
FeatureExtractor<Dtype>::~FeatureExtractor() {
  delete net_;
}

template <typename Dtype>
void FeatureExtractor<Dtype>::extract(vector<string> feature_blob_names, vector<Blob<Dtype>* > net_input_blobs, 
      vector<shared_ptr<Blob<Dtype> > > feature_blobs) {
  int num_features = feature_blob_names.size();
  net_->Forward(net_input_blobs);

  for(int i = 0;i < num_features;i++){
    feature_blobs.push_back(net_->blob_by_name(feature_blob_names[i]));
  }
}

template <typename Dtype>
vector<vector<Dtype*> > FeatureExtractor<Dtype>::parse_blob_data(vector<shared_ptr<Blob<Dtype> > > feature_blobs,
      int* dim_features) {
  int num_blobs = feature_blobs.size();
  vector<vector<Dtype*> > feature_data_vecs;

  for(int i = 0;i < num_blobs;i++){
    const shared_ptr<Blob<Dtype> > feature_blob = feature_blobs[i];
    vector<Dtype*> feature_data;
    int batch_size = feature_blob->num();
    *dim_features = feature_blob->count() / batch_size;
    
    for(int n = 0;n < batch_size;n++){
      Dtype* feature_blob_data = feature_blob->cpu_data() + feature_blob->offset(n);
      feature_data.push_back(feature_blob_data);
    }

    feature_data_vecs.push_back(feature_data);
  }
  return feature_data_vecs;
}
