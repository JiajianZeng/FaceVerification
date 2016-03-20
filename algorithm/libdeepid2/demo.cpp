#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "deepid2/Verificator.hpp"

using std::string;
using std::vector;
using cv::Mat;

int main(int argc, char** argv) {
  string config_file("/share/jiajian/FaceRecognitionSystem/algorithm/libdeepid2/data/libdeepid2.yaml");
  Verificator* ver = new Verificator(config_file);
  
  vector<string> feature_blob_names;
  string fc1("fc1");
  string fc1_p("fc1_p");
  feature_blob_names.push_back(fc1);
  feature_blob_names.push_back(fc1_p);

  Mat image1, image2;
  image1 = cv::imread("/share/jiajian/dataset/CASIA/100/3614913/056-r.jpg");
  image2 = cv::imread("/share/jiajian/dataset/CASIA/100/3614913/056-l.jpg");

  Mat feature1, feature2;
  bool identical = ver->verificate(image1, image2, feature_blob_names, feature1, feature2);
  std::cout << identical << std::endl;

  return 0;
}
