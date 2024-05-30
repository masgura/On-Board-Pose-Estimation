#ifndef YOLOV8_H
#define YOLOV8_H

#include <opencv2/opencv.hpp>    //opencv header file
#include <openvino/openvino.hpp> //openvino header file
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <time.h>

struct Prediction {

    std::vector<int> bbox; 
    float classScore;
    std::vector<std::vector<float>> keypoints;
    std::vector<float> kptsScores;
    float time;

};


class YOLOv8 {

    public:

        float confThreshold = 0.01;
        float nmsThreshold = 0.5;    
        std::string device = "CPU";
        int numThreads = std::thread::hardware_concurrency();
        void LoadModel(const std::string &model_path);
        void run(cv::Mat &image, Prediction &outPred);

    private:

        ov::Core core;
        ov::InferRequest infer_request;
        ov::CompiledModel model;
        ov::Shape in_shape;
        ov::element::Type in_type;
        int img_width;
        int img_height;
        float scale;

        float generate_scale(cv::Mat& image, const int& target_size);
        void letterbox(cv::Mat &input_image, cv::Mat &output_image, const int &target_size);
        void nms(const cv::Mat &ouput_buffer, Prediction &outPred);
        void scale_boxes(const std::vector<int> &img1_shape, std::vector<int> &boxes, const std::vector<int> &img0_shape);
        void scale_coords(const std::vector<int> &img1_shape, std::vector<std::vector<float>> &kpts, const std::vector<int> &img0_shape);
};        

#endif