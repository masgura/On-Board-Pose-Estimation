#include "yolov8.h"

void YOLOv8::LoadModel(const std::string &model_path) {

    /* 
    Compile the model, start the inference request and get the input shape [B,C,H,W].
    INPUT:
        - model_path    |   path to the .xml model
    */
    core.set_property(device, ov::hint::enable_cpu_pinning(false), ov::inference_num_threads(numThreads));
    std::shared_ptr<ov::Model> read_model = core.read_model(model_path);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(read_model);

    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({ 255, 255, 255 });
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);

    read_model = ppp.build();
    model = core.compile_model(read_model);

    infer_request = model.create_infer_request();
    auto input_port = infer_request.get_input_tensor();

    in_shape = input_port.get_shape();
    in_type = input_port.get_element_type();
}

float YOLOv8::generate_scale(cv::Mat &image, const int  &target_size) {

    float ratio_h = static_cast<float>(target_size) / static_cast<float>(image.cols);
    float ratio_w = static_cast<float>(target_size) / static_cast<float>(image.rows);
    float resize_scale = std::min(ratio_h, ratio_w);
    
    return resize_scale;
}

void YOLOv8::letterbox(cv::Mat &input_image, cv::Mat &output_image, const int &target_size) {

    /*
    Add symmetric padding to the smaller size of an image, making it square. This way, when resizing the image,
    the original aspect ratio is manteined, avoiding distortion when resizing the image to match the network input shape.
    Input:
        - input_image   |   the original image
        - output_imatge |   an empty cv::Mat in which the padded and resized image will be saved
        - target_size   |   the desired size of the output image 
    */

    scale = generate_scale(input_image, target_size);
        
    int new_shape_w = std::round(input_image.cols * scale);
    int new_shape_h = std::round(input_image.rows * scale);
    float padw = (target_size - new_shape_w) / 2.;
    float padh = (target_size - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image, cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

}

void YOLOv8::scale_boxes(const std::vector<int> &img1_shape, std::vector<int> &boxes, const std::vector<int> &img0_shape) {

    /*
    Scale boxes to original image size
    */

    auto gain = std::min((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);
    
    boxes[0] -= pad0; boxes[0] /= gain;
    boxes[1] -= pad1; boxes[1] /=gain;
    boxes[2] /= gain; boxes[3] /= gain;
}

void YOLOv8::scale_coords(const std::vector<int> &img1_shape, std::vector<std::vector<float>> &kpts, const std::vector<int> &img0_shape) {

    /*
    Scale coordinates to original image size
    */
   
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    for (int i = 0; i < 11; i++){
        kpts[i][0] -= pad0; kpts[i][0] /= gain; 
        kpts[i][1] -= pad1; kpts[i][1] /= gain; 

    }

}

void YOLOv8::nms(const cv::Mat &output_buffer, Prediction &outPred) {

    cv::Rect bbox;
    std::vector<float> objKeypoints;
    float maxScore = 0.0f;
    for (int i = 0; i < output_buffer.rows; i++) {

        float classScore = output_buffer.at<float>(i, 4); 
        
        if (classScore > maxScore) {
            
            maxScore = classScore;

            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);
            int left, top,  width, height;

            left = int((cx - 0.5 * w));
            top = int((cy - 0.5 * h));
            width = int(w);
            height = int(h);

            if (output_buffer.cols > 5) {
                std::vector<std::vector<float>> keypoints;
                std::vector<float> confScores;
                cv::Mat kpts = output_buffer.row(i).colRange(5, 38);
                for (int i = 0; i < 11; i++) {                
                    float x = kpts.at<float>(0, i * 3 + 0);
                    float y = kpts.at<float>(0, i * 3 + 1);
                    float s = kpts.at<float>(0, i * 3 + 2);
                    keypoints.push_back({x, y});
                    confScores.push_back(s);
                   
                }
                outPred.keypoints = keypoints;
                outPred.kptsScores = confScores;
            } 
            bbox =  cv::Rect(left, top, width, height);
            
        }
    }

    outPred.bbox = {bbox.x, bbox.y, bbox.width, bbox.height};
    outPred.classScore = maxScore;
}

void YOLOv8::run(cv::Mat &image, Prediction &outPred) {

    img_width = image.cols;
    img_height = image.rows;

    auto start_time = std::chrono::high_resolution_clock::now();
    
    cv::Mat in_image;
    letterbox(image, in_image, int(in_shape[2]));

    //auto blob = cv::dnn::blobFromImage(in_image, 1.0 / 255.0, cv::Size(in_shape[2], in_shape[2]), cv::Scalar(), true, false);
    float *input_data = (float *)in_image.data;
    ov::Tensor input_tensor(in_type, in_shape, input_data);
    infer_request.set_input_tensor(input_tensor);

    // Start inference
    infer_request.infer();

    // Get the inference result
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    
    // Postprocess the result
    float* data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    cv::transpose(output_buffer, output_buffer); 
    nms(output_buffer, outPred);

    scale_boxes({in_image.rows, in_image.cols}, outPred.bbox, {img_height, img_width});
    if(outPred.bbox[0]<= 0){
            outPred.bbox[0] = 0;
        }
        if(outPred.bbox[1] <= 0){
            outPred.bbox[1] = 0;
        }
        if(outPred.bbox[0] + outPred.bbox[2] >= img_width) {
            outPred.bbox[2] = img_width - outPred.bbox[0];
        }

        if(outPred.bbox[1] + outPred.bbox[3] >= img_height) {
            outPred.bbox[3] = img_height - outPred.bbox[1];
    }
    

    if (output_buffer.cols > 5) {
        scale_coords({int(in_shape[2]), int(in_shape[2])}, outPred.keypoints, {img_height, img_width});
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> time = end_time - start_time;
    outPred.time = time.count();
}
