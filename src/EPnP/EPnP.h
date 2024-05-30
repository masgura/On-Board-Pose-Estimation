#ifndef EPNP_H
#define EPNP_H

#include <opencv2/opencv.hpp>    //opencv header file
#include <vector>
#include <string>
#include <numeric>

class EPnP {
    public:
        float f = 0.0176;
        float rho = 5.86e-6;
        float cx = 1920 / 2;
        float cy = 1200 / 2;
        float fx = f / rho;
        float fy = f / rho;
        std::vector<cv::Point3f> bodyPoints = {cv::Point3f(-0.37, 0.304,  0), 
                                                cv::Point3f(-0.37, -0.264, 0),   
                                                cv::Point3f(0.37,  -0.264, 0),
                                                cv::Point3f(0.37,  0.304,  0),
                                                cv::Point3f(-0.37, 0.385,  0.3215),
                                                cv::Point3f(-0.37, -0.385, 0.3215),
                                                cv::Point3f(0.37,  -0.385, 0.3215),
                                                cv::Point3f(0.37,  0.385,  0.3215),
                                                cv::Point3f(-0.5427, 0.4877, 0.2535),
                                                cv::Point3f(0.305, -0.579, 0.2515),
                                                cv::Point3f(0.5427, 0.4877, 0.2591)};

        cv::Point3f bodyCenter = cv::Point3_<float>(0, 0, 0.3215/2);
        float charactLenght = 1.05*cv::norm(bodyPoints[4] - bodyPoints[6]);   
        std::vector<cv::Point3f> objectPoints;
        cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << 
                                fx, 0, cx,
                                0, fy, cy,
                                0,  0,  1);	
        float confThr = 0.75;
        int minLM = 5;
        cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F); // no distortion

        float computeReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                const std::vector<cv::Point2f>& imagePoints,
                                const cv::Mat& rvec, const cv::Mat& tvec);

        std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>> filterLandmarks(const std::vector<cv::Point3f>& bodyPoints,
                                                                                const std::vector<std::vector<float>>& keypoints,
                                                                                const std::vector<float>& confScores,
                                                                                const int& minLandmarks, const float& confThr);
        
        void poseOutlierOrRefinement(const std::vector<cv::Point3f>& objectPoints,
                            const std::vector<cv::Point2f>& imagePoints,
                            cv::Mat& rvec, cv::Mat& tvec, bool& isOutlier,
                            const float& reprojError,
                            const std::vector<int>& bbox,
                            const std::vector<float>& confScores);

        std::tuple<cv::Mat, cv::Vec4f, bool> solveEPnP(const std::vector<std::vector<float>>& keypoints,
                                                       const std::vector<float>& confScores,
                                                       const std::vector<int>& bbox);

        void dcm2quat(const cv::Mat& dcm, cv::Vec4f& q);
        float boxScore;
};

#endif
