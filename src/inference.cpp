#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/inotify.h>
#include <csignal>
#include <chrono>
#include <thread>
#include <time.h>
#include <filesystem>
#include <numeric>

#include <nlohmann/json.hpp>
#include "yolov8.h"

namespace fs = std::filesystem;
using json = nlohmann::json;   

void printProgressBar(int num_images, int curr_image) {

    const int barWidth = 70;
    float progress = float(curr_image) / float(num_images);
    int pos = barWidth * progress;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << curr_image << "/" << num_images << "; " << int(progress * 100.0) << " %\r";
    if(int(progress * 100.0) < 100){
        std::cout.flush();
    }
}

int main() {

    YOLOv8 yolov8n;
    yolov8n.LoadModel("../yolov8n_352.xml");

    YOLOv8 yolov8n_pose;
    yolov8n_pose.LoadModel("../yolov8n_pose_320_int8.xml");

    std::string test_img_path = "../test/images/";
    int num_images = std::distance(fs::directory_iterator(test_img_path), fs::directory_iterator{});

    std::ifstream testfile("../test/test.json");

    json testData;
    try {
        testfile >> testData;
    } catch (json::parse_error& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return 1;
    }
    
    for (int numThreads = std::thread::hardware_concurrency(); numThreads > 0; numThreads--){

        yolov8n.numThreads = numThreads;
        yolov8n_pose.numThreads = numThreads;
        std::cout << "\nSTARTING INFERENCE ON TEST IMAGES WITH # THREADS = " << numThreads  << std::endl;
        int processed_images = 0;
        std::vector<std::map<std::string, json>> timeResults;
        std::vector<std::map<std::string, json>> infResults;

            for (const auto& obj : testData) {
                
                ++processed_images;
                printProgressBar(num_images, processed_images);
                std::map<std::string, json> results;
                std::map<std::string, json> time;
                std::string img_name = obj["filename"];
                std::string img_path = test_img_path + img_name;

                
                cv::Mat img = cv::imread(img_path);

                // Object detection:
                Prediction detPred;
                yolov8n.run(img, detPred);
                std::vector<int> bbox = detPred.bbox;
                cv::Mat roi = img(cv::Rect(bbox[0], bbox[1], bbox[2], bbox[3])).clone();

                // LM regression:
                Prediction keyPred;
                yolov8n_pose.run(roi, keyPred);
                std::vector<std::vector<float>> keypoints = keyPred.keypoints;
                std::vector<float> confScores = keyPred.kptsScores;
                
                for (int i = 0; i < 11; i++) {
                    keypoints[i][0] += float(bbox[0]);
                    keypoints[i][1] += float(bbox[1]);
                }

                results["filename"] = img_name;
                results["bbox"] = bbox;
                results["classScore"] = detPred.classScore;
                results["keypoints"] = keypoints;
                results["kptsScores"] = keyPred.kptsScores;
                if (numThreads == std::thread::hardware_concurrency()) {
                    infResults.push_back(results);
                }

                time["det_time"] = detPred.time;
                time["key_time"] = keyPred.time;
                timeResults.push_back(time);

            }

        if (numThreads == std::thread::hardware_concurrency()){

            std::string json_path = "../output/inf_results.json";
            json json_data(infResults);
            std::ofstream file(json_path);
            file << json_data.dump(4);
            file.close();

        }

        std::string time_json_path = "../output/time_results_" + std::to_string(numThreads) + ".json";
        json time_json_data(timeResults);
        std::ofstream file(time_json_path);
        file << time_json_data.dump(4);
        file.close(); 
    }

    std::cout <<"\nENDED, RESULTS SAVED TO: ../output/inf_results.json" << std::endl;
    
    return 0;
}