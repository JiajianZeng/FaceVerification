#ifndef FEATURE_EXTRACTOR_HPP_
#define FEATURE_EXTRACTOR_HPP_

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"

using std::string;
using std::vector;
using caffe::Net;
using caffe::Blob;
using boost::shared_ptr;

template <typename Dtype>
class FeatureExtractor {
 public:
  FeatureExtractor(const string& param_file, const string& trained_model_file,
      const Net<Dtype>* root_net = NULL, bool use_gpu = false, int device_id = -1);
  ~FeatureExtractor();

  void extract(vector<string> feature_blob_names, vector<Blob<Dtype>* > net_input_blobs, 
      vector<shared_ptr<Blob<Dtype> > > feature_blobs);

 private:
  Net<Dtype>* net_;   

 protected:
  vector<vector<Dtype*> > parse_blob_data(vector<shared_ptr<Blob<Dtype> > > feature_blobs, int* dim_features);
};
#endif
