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
#include "EPnP.h"

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

std::vector<float> computeTranslationError(const std::vector<float>& t_GT, const std::vector<float>& t_pred) {

    float squareErr = 0.0f;
    float t_GT_norm = 0.0f;

    for (size_t i = 0; i < t_GT.size(); i++) {
        squareErr += std::pow(t_pred[i] - t_GT[i], 2);
        t_GT_norm += t_GT[i] * t_GT[i];
    }

    t_GT_norm = std::sqrt(t_GT_norm);

    std::vector<float> Err;
    Err.push_back(std::sqrt(squareErr));
    Err.push_back(Err[0] / t_GT_norm);

    return Err;
}

float computeQuaternionError(const std::vector<float> &q_GT, const std::vector<float> &q_pred) {

    float q_dot = 0.0f;
    float q_GT_norm = 0.0f; 
    float q_pred_norm = 0.0f;

    for (size_t i = 0; i < q_GT.size(); i++) {

        q_dot += q_GT[i] * q_pred[i];
        q_GT_norm += q_GT[i] * q_GT[i];
        q_pred_norm += q_pred[i] * q_pred[i];

    }

    q_GT_norm = std::sqrt(q_GT_norm);
    q_pred_norm = std::sqrt(q_pred_norm);

    float Err = 2*std::acos(std::max(-1.0f, std::min(1.0f, std::abs(q_dot/ (q_GT_norm * q_pred_norm)))));

    return Err;
}

int main() {

    EPnP epnp;

    std::ifstream inffile("../output/inf_results.json");
    json infResults;
    try {
        inffile >> infResults;
    } catch (json::parse_error& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return 1;
    }

    std::ifstream testfile("../test/test.json");
    json testData;
    try {
        testfile >> testData;
    } catch (json::parse_error& e) {
        std::cerr << "Parse error: " << e.what() << std::endl;
        return 1;
    }

    std::string test_img_path = "../test/images/";
    int num_images = std::distance(fs::directory_iterator(test_img_path), fs::directory_iterator{});

    // Evaluation of SLAB Score on min # landmarks and min kpts score
    std::vector<int> MIN_LANDMARK_VEC(6);
    std::iota(MIN_LANDMARK_VEC.begin(), MIN_LANDMARK_VEC.end(), 5);
    std::vector<float> MIN_CONF_VEC;
    for(float i = 0.0; i <= 0.9; i += 0.05){
        MIN_CONF_VEC.push_back(i);
    }
    std::vector<std::vector<float>> eslab_mat;
    int tot_iterations = MIN_CONF_VEC.size() * MIN_LANDMARK_VEC.size();
    int iteration = 1;
    
    std::cout << "\nEVALUATING SLAB SCORE:" << std::endl;
    float opt_eslab = 1e3;
    int opt_min_LM;
    float opt_min_conf;
    float opt_mean_eq;
    float opt_mean_et;

    std::vector<std::vector<float>> opt_quat;
    std::vector<std::vector<float>> opt_tvec;
    std::vector<float> opt_time;
    std::vector<float> opt_et;
    std::vector<float> opt_eq;
    std::vector<float> opt_Et;

    for(int MIN_LANDMARK : MIN_LANDMARK_VEC){

        std::vector<float> eslab_vec;

        for(float MIN_CONF : MIN_CONF_VEC){

            epnp.minLM = MIN_LANDMARK;
            epnp.confThr = MIN_CONF;

            std::vector<float> Et;
            std::vector<float> et;
            std::vector<float> eq;
            std::vector<float> eslab;    
            printProgressBar(tot_iterations, iteration);
            ++iteration;

            std::vector<std::vector<float>> quat_vec;
            std::vector<std::vector<float>> tvec_vec;
            std::vector<float> time_vec;

            for (int i = 0; i < num_images; i++) {
                
                std::vector<int> bbox = infResults[i]["bbox"];
                float classScore = infResults[i]["classScore"];
                epnp.boxScore = classScore;
                std::vector<std::vector<float>> keypoints = infResults[i]["keypoints"];
                std::vector<float> kptsScores = infResults[i]["kptsScores"];

                auto start_epnp = std::chrono::high_resolution_clock::now(); // start time
                auto [tvec, quat, isOutlier] = epnp.solveEPnP(keypoints, kptsScores, bbox);
                auto end_epnp = std::chrono::high_resolution_clock::now(); // end time
                std::chrono::duration<float> epnp_time = end_epnp - start_epnp;

                std::vector<float> quat_json;
                for(int c = 0; c < quat.rows; c++){
                    quat_json.push_back(quat[c]);
                }

                std::vector<float> tvec_json(tvec.begin<float>(), tvec.end<float>());
                
                quat_vec.push_back(quat_json);
                tvec_vec.push_back(tvec_json);
                time_vec.push_back(epnp_time.count());

                std::vector<float> Et_vec = computeTranslationError(testData[i]["r_Vo2To_vbs_true"], tvec_json);
                Et.push_back(Et_vec[0]);
                et.push_back(Et_vec[1]);
                eq.push_back(computeQuaternionError(testData[i]["q_vbs2tango"], quat_json));
                eslab.push_back(et.back() + eq.back());

            }

            float mean_eslab = std::accumulate(eslab.begin(), eslab.end(), 0.0)/eslab.size();
            eslab_vec.push_back(mean_eslab);

            if (mean_eslab < opt_eslab) {

                opt_eslab = mean_eslab;
                opt_min_conf = MIN_CONF;
                opt_min_LM = MIN_LANDMARK;
                opt_quat = quat_vec;
                opt_tvec = tvec_vec;
                opt_time = time_vec;
                opt_eq = eq;
                opt_et = et;
                opt_Et = Et;

            }

        }

        eslab_mat.push_back(eslab_vec);

    }

    std::cout << "\nEVALUATION ENDED, BEST RESULT:" << std::endl;
    std::cout << "- Min. # LM: " << opt_min_LM << std::endl;
    std::cout << "- Min. LM conf.: " << opt_min_conf << std::endl;
    std::cout << "- Mean quat. error [deg]: " << std::accumulate(opt_eq.begin(), opt_eq.end(), 0.0)*(180.0/3.141592653589793238463)/opt_eq.size() << std::endl;
    std::cout << "- Mean trasl. error [m]: " << std::accumulate(opt_Et.begin(), opt_Et.end(), 0.0)/opt_Et.size() << std::endl;
    std::cout << "- SLAB score: " << opt_eslab << std::endl;

    json param_optimization;
    param_optimization["MIN_LANDMARK"] = MIN_LANDMARK_VEC;
    param_optimization["MIN_CONFIDENCE"] = MIN_CONF_VEC;
    param_optimization["SLAB_score"] = eslab_mat;
    std::ofstream outFile("../output/param_optimization.json");
    outFile << param_optimization.dump(4); 
    outFile.close();

    std::cout << "UPDATING ../output/inf_results.json WITH EPNP RESULTS" << std::endl;
    for (int i = 0; i < num_images; i++) {
        infResults[i]["quat"] = -opt_quat[i];
        infResults[i]["tvec"] = opt_tvec[i];
        infResults[i]["epnp_time"] = opt_time[i];
        infResults[i]["eq"] = opt_eq[i];
        infResults[i]["et"] = opt_et[i];
        infResults[i]["Et"] = opt_Et[i];
    }

    std::ofstream infResultsFile("../output/inf_results.json");
    infResultsFile << infResults.dump(4);
    infResultsFile.close();

    return 0;
}