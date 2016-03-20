#ifndef FEATURE_EXTRACTOR_HPP_
#define FEATURE_EXTRACTOR_HPP_

#include <string>
#include <vector>

#include "caffe/caffe.hpp"

using std::string;
using std::vector;
using caffe::Net;
using caffe::Blob;
using boost::shared_ptr;

class FeatureExtractor {
 public:
  FeatureExtractor(const string& param_file, const string& trained_model_file,
      bool use_gpu = false, int device_id = -1);
  ~FeatureExtractor();

  void extract(vector<string> feature_blob_names, vector<Blob<float>* > net_input_blobs, vector<int>& feature_dim_vecs,
      vector<vector<float*> >& feature_blob_data);

  Net<float>* get_net();

 private:
  Net<float>* net_;   

 protected:
  void parse_blob_data(vector<shared_ptr<Blob<float> > > feature_blobs, vector<int>& feature_dim_vecs, 
      vector<vector<float*> >& feature_blob_data);
};
#endif
