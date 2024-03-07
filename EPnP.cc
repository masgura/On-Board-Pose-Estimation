#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>


// Function to convert Rodrigues rotation vector to quaternion
cv::Vec4d rodriguesToQuaternion(const cv::Vec3d& rvec) {
    // Normalize the rotation axis
    double angle = cv::norm(rvec);
    cv::Vec3d axis = rvec / angle;

    // Calculate quaternion components
    double s = sin(angle / 2);
    double w = cos(angle / 2);
    double x = axis[0] * s;
    double y = axis[1] * s;
    double z = axis[2] * s;

    // Return the quaternion
    return cv::Vec4d(w, x, y, z);
}

int main(int argc, char* argv[]) {
    // Define 3D points
    std::vector<cv::Point3f> objectPoints;
    objectPoints.push_back(cv::Point3f(-0.37, 0.30,  0));   
    objectPoints.push_back(cv::Point3f(-0.37, -0.26, 0));   
    objectPoints.push_back(cv::Point3f(0.37,  -0.26, 0));
    objectPoints.push_back(cv::Point3f(0.37,  0.30,  0));
    objectPoints.push_back(cv::Point3f(-0.37, 0.38,  0.32));
    objectPoints.push_back(cv::Point3f(-0.37, -0.38, 0.32));
    objectPoints.push_back(cv::Point3f(0.37,  -0.38, 0.32));
    objectPoints.push_back(cv::Point3f(0.37,  0.38,  0.32));
    objectPoints.push_back(cv::Point3f(-0.54, 0.49, 0.255));
    objectPoints.push_back(cv::Point3f(0.31, -0.56, 0.255));
    objectPoints.push_back(cv::Point3f(0.54, 0.49, 0.255));

    // Read image points from file
    std::ifstream infile(argv[1]);
    std::vector<cv::Point2f> imagePoints;
    float x, y;
    while (infile >> x >> y) {
        imagePoints.push_back(cv::Point2f(x, y));
    }
    infile.close();
    
    // Camera parameters
    double f = 0.0176;
    double rho = 5.86e-6;
    int cx = 1920/2;
    int cy = 1200/2;
    double fx = f/rho;
    double fy = f/rho;

    // Camera matrix
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
                            fx, 0, cx,
                            0, fy, cy,
                            0,  0,  1);

    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F); // no distortion

    // Rotation and translation vectors
    cv::Mat rvec, tvec;

    // EPnP solver
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, cv::SOLVEPNP_EPNP);

    // Convert Rodrigues rotation vector to quaternion
    cv::Vec4d quaternion = rodriguesToQuaternion(rvec);

    // Print the quaternion components
    std::cout << "Rodrigues: " << rvec << std::endl;
    std::cout << "Quaternion: " << quaternion << std::endl;
    std::cout << "Translation vector:\n" << tvec << std::endl;

    return 0;
}
