
#include <fstream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp> 
using namespace std;
using namespace cv;
using namespace ov;
vector<string> readClassNames(const string& filename) {
    vector<string> classNames;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return classNames;
    }
    string line;
    while (getline(file, line)) {
        if (!line.empty()) {
            classNames.push_back(line);
        }
    }
    file.close();
    return classNames;
}
int main(int argc, char** argv)
{
    string filename = "coco.names";
    vector<string> labels = readClassNames(filename);
    Mat image = imread("bus.jpg");
    int ih = image.rows;
    int iw = image.cols;
    string model_path = "yolov8n.xml";
    float conf_threshold = 0.25;
    float nms_threshold = 0.4;
    float score_threshold = 0.25;
    Core core;
    auto compiled_model = core.compile_model(model_path, "CPU");
    auto input_port = compiled_model.input();
    auto input_shape = input_port.get_shape();
    InferRequest infer_request = compiled_model.create_infer_request();
    int64 start = getTickCount();
    int w = image.cols;
    int h = image.rows;
    int _max = max(h, w);
    Mat image_ = Mat::zeros(Size(_max, _max), CV_8UC3);
    Rect roi(0, 0, w, h);
    image.copyTo(image_(roi));
    int input_w = input_shape[2];
    int input_h = input_shape[3];
    float x_factor = image_.cols / static_cast<float>(input_w);
    float y_factor = image_.rows / static_cast<float>(input_h);
    Mat blob = dnn::blobFromImage(image, 1.0 / 255.0, Size(input_w, input_h), Scalar(), true);
    Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    float* data = output.data<float>();
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); 
    vector<Rect> boxes;
    vector<int> classIds;
    vector<float> confidences;
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, 84);
        Point classIdPoint;
        double score;
        minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);
        if (score > score_threshold)
        {
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float ow = output_buffer.at<float>(i, 2);
            float oh = output_buffer.at<float>(i, 3);
            int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
            int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
            int width = static_cast<int>(ow * x_factor);
            int height = static_cast<int>(oh * y_factor);
            Rect box;
            box.x = x;
            box.y = y;
            box.width = width;
            box.height = height;
            boxes.push_back(box);
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
        }
    }
    vector<int> indexes;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indexes);
    for (size_t i = 0; i < indexes.size(); i++) {
        int index = indexes[i];
        int idx = classIds[index];
        rectangle(image, boxes[index], Scalar(0, 0, 255), 2, 8);
        rectangle(image, Point(boxes[index].tl().x, boxes[index].tl().y - 20),
            Point(boxes[index].br().x, boxes[index].tl().y), Scalar(0, 255, 255), -1);
        putText(image, labels[idx], Point(boxes[index].tl().x, boxes[index].tl().y), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 0), 2, 8);
    }
    float t = (getTickCount() - start) / static_cast<float>(getTickFrequency());
    putText(image, format("FPS: %.2f", 1 / t), Point(20, 40), FONT_HERSHEY_PLAIN, 2.0, Scalar(255, 0, 0), 2, 8);
    imshow("YOLOV8-OPENVINO", image);
    waitKey(0);
    return 0;
}
