#include "EPnP.h"

float EPnP::computeReprojectionError(const std::vector<cv::Point3f>& objectPoints,
                                     const std::vector<cv::Point2f>& imagePoints,
                                     const cv::Mat& rvec, const cv::Mat& tvec) {
    std::vector<cv::Point2f> reprojectedPoints;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, reprojectedPoints);

    float meanError = cv::norm(imagePoints, reprojectedPoints, cv::NORM_L2) / objectPoints.size();

    return meanError;
}

std::tuple<std::vector<cv::Point2f>, std::vector<cv::Point3f>, std::vector<float>>
	EPnP::filterLandmarks(const std::vector<cv::Point3f>& bodyPoints,
                          const std::vector<std::vector<float>>& keypoints,
                          const std::vector<float>& confScores,
                          const int& minLM, const float& confThr) {
					   
	std::vector<bool> filter_low_conf_idx;
    for (float c : confScores) {
        filter_low_conf_idx.push_back(c >= confThr);
    }

    // If insufficient landmarks detected, relax requirements and consider min_landmark best landmarks
    if (std::count(filter_low_conf_idx.begin(), filter_low_conf_idx.end(), true) < minLM) {
        // Get indices of sorted confidence vector (descending)
        std::vector<size_t> indices(confScores.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&confScores](size_t i1, size_t i2) { return confScores[i1] > confScores[i2]; });

        // Set true for min_landmark best landmarks
        for (int i = 0; i < minLM; ++i) {
            filter_low_conf_idx[indices[i]] = true;
        }
    }
    
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Point3f> objectPoints;
    std::vector<float> filtConf;
    for (size_t i = 0; i < keypoints.size(); ++i) {

        if (filter_low_conf_idx[i]) {
            imagePoints.push_back(cv::Point2f(keypoints[i][0], keypoints[i][1]));
            objectPoints.push_back(bodyPoints[i]);
            filtConf.push_back(confScores[i]);
        }
    }

    return std::make_tuple(imagePoints, objectPoints, filtConf);
}

void EPnP::poseOutlierOrRefinement(const std::vector<cv::Point3f>& objectPoints,
                                   const std::vector<cv::Point2f>& imagePoints,
                                   cv::Mat& rvec, cv::Mat& tvec, bool& isOutlier,
                                   const float& reprojError,
                                   const std::vector<int>& bbox,
                                   const std::vector<float>& confScores) {
								
	float left = static_cast<float>(bbox[0]);
    float top = static_cast<float>(bbox[1]);
    float w = static_cast<float>(bbox[2]);
    float h = static_cast<float>(bbox[3]);
    float xc = left + w/2;
    float yc = top + h/2;
    
    std::vector<cv::Point2f> projectedBodyCenter;
    std::vector<cv::Point3f> bodyCenterVec;
    bodyCenterVec.push_back(bodyCenter);
    
    cv::projectPoints(bodyCenterVec, rvec, tvec, cameraMatrix, distCoeffs, projectedBodyCenter);
    float xc_pose = projectedBodyCenter[0].x;
    float yc_pose = projectedBodyCenter[0].y;
    
    // 1. Calculate relative center mismatch
    float dx_rel = abs(xc_pose - xc) / w;
    float dy_rel = abs(yc_pose - yc) / h;

    // 2. Calculate distance mismatch
    float diag_BB = std::sqrt(std::pow(w / (1.1), 2) + std::pow(h / (1.1), 2));
    float dist_norm_BB_approx = (fx + fy) / 2 * charactLenght / diag_BB;

    // 3. Calculate relative reprojection error
    float err_reproj_relative = reprojError / diag_BB;   
	float mean_conf = std::accumulate(confScores.begin(), confScores.end(), 0.0)/confScores.size();

    if ((dx_rel > 0.5 || dy_rel > 0.5 ||
       (abs(cv::norm(tvec) - dist_norm_BB_approx) / dist_norm_BB_approx) > 0.75 ||
       ((abs(cv::norm(tvec) - dist_norm_BB_approx) / dist_norm_BB_approx) > 0.15 &&
       ((mean_conf < 0.5) || err_reproj_relative > 0.1))) && boxScore>0.25) {
            isOutlier = true;
            
			float Cx = cx;
			float Cy = cy;			
			float alpha = std::atan((xc - Cx) / fx);
			float beta = std::atan((yc - Cy) / fy);
			cv::Mat R_alpha = (cv::Mat_<float>(3, 3) << std::cos(alpha), 0, std::sin(alpha),
														0, 1, 0,
														-std::sin(alpha), 0, std::cos(alpha));
			cv::Mat R_beta = (cv::Mat_<float>(3, 3) << 1, 0, 0,
													   0, std::cos(beta), std::sin(beta),
													   0, -std::sin(beta), std::cos(beta));
			cv::Mat1f dcm;
			cv::Rodrigues(-rvec, dcm);

			cv::Mat v = cv::Mat_<float>({0, 0, dist_norm_BB_approx});
			cv::Mat t = cv::Mat_<float>({0, 0, -bodyCenter.z});

			tvec = R_alpha * (R_beta * v) + dcm.t() * t;

		} else {
		cv::solvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, 
							 distCoeffs, rvec, tvec);
	}						
}

void EPnP::dcm2quat(const cv::Mat& dcm, cv::Vec4f& q) {
    
    float t, qx, qy, qz, qw;
    float r11 = dcm.at<float>(0, 0);
    float r12 = dcm.at<float>(0, 1);
    float r13 = dcm.at<float>(0, 2);
    float r21 = dcm.at<float>(1, 0);
    float r22 = dcm.at<float>(1, 1);
    float r23 = dcm.at<float>(1, 2);
    float r31 = dcm.at<float>(2, 0);
    float r32 = dcm.at<float>(2, 1);
    float r33 = dcm.at<float>(2, 2);

    // Calculate the trace of the matrix
    float tr = r11 + r22 + r33;

    if (tr > 0) {
        float S = sqrt(tr + 1.0) * 2.0; // S=4*qw
        qw = 0.25 * S;
        qx = (r32 - r23) / S;
        qy = (r13 - r31) / S;
        qz = (r21 - r12) / S;
    } else if ((r11 > r22) && (r11 > r33)) {
        float S = sqrt(1.0 + r11 - r22 - r33) * 2.0; // S=4*qx
        qw = (r32 - r23) / S;
        qx = 0.25 * S;
        qy = (r12 + r21) / S;
        qz = (r13 + r31) / S;
    } else if (r22 > r33) {
        float S = sqrt(1.0 + r22 - r11 - r33) * 2.0; // S=4*qy
        qw = (r13 - r31) / S;
        qx = (r12 + r21) / S;
        qy = 0.25 * S;
        qz = (r23 + r32) / S;
    } else {
        float S = sqrt(1.0 + r33 - r11 - r22) * 2.0; // S=4*qz
        qw = (r21 - r12) / S;
        qx = (r13 + r31) / S;
        qy = (r23 + r32) / S;
        qz = 0.25 * S;
    }

    q[0] = -qw; q[1] = qx; q[2] = qy; q[3] = qz;

}

std::tuple<cv::Mat, cv::Vec4f, bool> EPnP::solveEPnP(const std::vector<std::vector<float>>& keypoints,
                                                     const std::vector<float>& confScores,
                                                     const std::vector<int>& bbox) {
										   
	// Rotation and translation vectors
    cv::Mat rvec(4,1,CV_32F), tvec(3,1,CV_32F);
    
    // Filter detected keypoints
    auto [imagePoints, objectPoints, filtConf] = filterLandmarks(bodyPoints, keypoints, confScores, 
															     minLM, confThr);
    // EPnP solver
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, 
                  rvec, tvec, false, cv::SOLVEPNP_EPNP);

    bool isOutlier = false;
    auto reprojErr = computeReprojectionError(objectPoints, imagePoints, rvec, tvec);
    poseOutlierOrRefinement(objectPoints, imagePoints, rvec, tvec, isOutlier, reprojErr, bbox, filtConf);
    
	cv::Mat1f dcm;
	cv::Rodrigues(rvec, dcm);
    cv::Vec4f quat;
	dcm2quat(dcm.t(), quat);
    
	return {tvec, quat, isOutlier};
}