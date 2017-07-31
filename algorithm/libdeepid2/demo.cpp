#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "deepid2/Verificator.hpp"
#include "deepid2/FaceDetector.hpp"

using std::string;
using std::vector;
using std::cout;
using cv::Mat;
using cv::Rect;

int main(int argc, char** argv) {
  if(argc != 3) {
    cout << "Usage: demo path_to_image1 path_to_image2" << "\n";
    return -1;
   }
  char* image1_file = argv[1];
  char* image2_file = argv[2];

  string config_file("./data/libdeepid2.yaml");
  Verificator* ver = new Verificator(config_file);
  
  // feature blobs to extract
  vector<string> feature_blob_names;
  string fc1("fc1");
  string fc1_p("fc1_p");
  feature_blob_names.push_back(fc1);
  feature_blob_names.push_back(fc1_p);
  
  // source images
  Mat image1, image2;
  image1 = cv::imread(image1_file);
  image2 = cv::imread(image2_file);
   
  if(image1.data == NULL) {
     cout << "can not read image1: " << image1_file << "\n";
     return -1; 
  }else if(image2.data == NULL){
     cout << "can not read image2: " << image2_file << "\n";
     return -1;
   }

  // detect faces
  vector<Rect> faces1, faces2;
  string frontal_model("./data/frontal_face.xml");
  string profile_model("./data/profile_face.xml");
  FaceDetector* fd = new FaceDetector(frontal_model, profile_model);

  fd->detect_face(image1, faces1);
  fd->detect_face(image2, faces2);
  // some output
  cout << "faces1 size : " << faces1.size() << "\n";
  cout << "faces2 size : " << faces2.size() << "\n";
  
  // region of interest
  Mat image_roi1, image_roi2;
  if(faces1.size() > 0) 
    image_roi1 = Mat(image1, faces1[0]);
  else
    image_roi1 = image1;
  
  if(faces2.size() > 0)
    image_roi2 = Mat(image2, faces2[0]);
  else
    image_roi2 = image2;
  
  // extract features
  Mat feature1, feature2;
  bool identical = ver->verificate(image_roi1, image_roi2, feature_blob_names, feature1, feature2);
  double distance = cv::norm(feature1, feature2);
  std::cout << "Euclidean distance of the two face image is " << distance << std::endl;
  std::cout << "The two face image belong to " << (identical ? "same" : "defferent") << " identity." << std::endl;
  
  if(faces1.size() > 0)
    cv::rectangle(image1, faces1[0], (255));
  if(faces2.size() > 0)
    cv::rectangle(image2, faces2[0], (255));

  cv::imshow("image1", image1);
  cv::imshow("image2", image2);
  cv::waitKey(0);
  
  return 0;
}
