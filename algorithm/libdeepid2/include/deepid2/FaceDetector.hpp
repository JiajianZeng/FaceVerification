#ifndef FACE_DETECTOR_HPP
#define FACE_DETECTOR_HPP
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class FaceDetector {
 public:
  FaceDetector(const std::string frontal_model, const std::string profile_model);
  void detect_face(const cv::Mat image, std::vector<cv::Rect>& faces);
 private:
  cv::CascadeClassifier haar_frontal_detector_;
  cv::CascadeClassifier haar_profile_detector_;
 protected:
  void merge_frontal_profile_faces(const std::vector<cv::Rect>& faces_profile, std::vector<cv::Rect>& faces_frontal);
};

#endif
