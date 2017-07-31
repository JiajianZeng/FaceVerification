#include "deepid2/FaceDetector.hpp"

FaceDetector::FaceDetector(const std::string frontal_model, const std::string profile_model) 
{
  haar_frontal_detector_.load(frontal_model);
  haar_profile_detector_.load(profile_model);
}

void FaceDetector::detect_face(const cv::Mat image, std::vector<cv::Rect>& faces)
{
  cv::Mat sample = image.clone();
  cv::Mat sample_resized;
  // resize
  if (sample.cols > 2000)
    cv::resize(sample, sample_resized, cv::Size(sample.cols / 3, sample.rows / 3), 0, 0, cv::INTER_LINEAR); //ground_truth_shape /= 3.0;
  else if (sample.cols > 1400)
    cv::resize(sample, sample_resized, cv::Size(sample.cols / 2, sample.rows / 2), 0, 0, cv::INTER_LINEAR); //ground_truth_shape /= 2.0;
  else
    sample_resized = sample;

  // detect face
  std::vector<cv::Rect> frontal_faces;
  std::vector<cv::Rect> profile_faces;
  haar_frontal_detector_.detectMultiScale(sample_resized, frontal_faces, 1.1, 2, 0, cv::Size(30, 30));
  haar_profile_detector_.detectMultiScale(sample_resized, profile_faces, 1.1, 2, 0, cv::Size(30, 30));

  // merge frontal and profile faces detected
  merge_frontal_profile_faces(profile_faces, frontal_faces);
  
  for(int i = 0;i < frontal_faces.size();i++)
    faces.push_back(frontal_faces[i]);
}

void FaceDetector::merge_frontal_profile_faces(const std::vector<cv::Rect>& faces_profile, std::vector<cv::Rect>& faces_frontal)
{ 
  for(int i = 0;i <faces_profile.size() ;i++) {
    int count = 0;
    for(int j = 0;j < faces_frontal.size();j++) {
      cv::Rect intersect = faces_profile[i] & faces_frontal[j];
      if(intersect.area() < 0.5 * faces_frontal[j].area() && 
          intersect.area() < 0.5 * faces_profile[i].area() )
        count++;
      else
        break;      
    }
    if(count == faces_frontal.size())
      faces_frontal.push_back(faces_profile[i]);
  }
}
