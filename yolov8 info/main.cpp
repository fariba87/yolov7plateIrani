#include <opencv2/opencv.hpp>
#include <fstream>
// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;
// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);
void draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}
vector<Mat> pre_process(Mat& input_image, Net& net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1. / 255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}
Mat post_process(Mat input_image, vector<Mat>& outputs, const vector<string>& class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;
    const int dimensions = 85;
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float* classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((cx - 0.5 * w) * x_factor);
                int top = int((cy - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
            }
        }
        // Jump to the next row.
        data += 85;
    }
    // Perform Non-Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3 * THICKNESS);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}

int main()
{
    // Load class list.
    vector<string> class_list{"knife","gun"};
    //ifstream ifs("coco.names");
    //string line;
    //while (getline(ifs, line))
    //{
    //    class_list.push_back(line);
    //}
    // Load image.
    Mat frame;
    frame = imread("a2.png");
    // Load model.
    Net net;
    //net = cv::dnn::readNetFromONNX("dr_kiae.onnx");
    net = cv::dnn::readNetFromTorch("B:/yolov5/best.pt", true, true);
    //net = readNet("YOLOv5s.onnx");
    vector<Mat> detections;     // Process the image.
    detections = pre_process(frame, net);
    Mat img = post_process(frame.clone(), detections, class_list);
    // Put efficiency information.
    // The function getPerfProfile returns the overall time for     inference(t) and the timings for each of the layers(in layersTimes).
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time : %.2f ms", t);
    putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
    imshow("Output", img);
    waitKey(0);
    return 0;
}
/*
#include <opencv2/opencv.hpp>
#include <iostream>

#include <opencv2/opencv.hpp>

#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <assert.h>
#include <chrono>
#include <onnxruntime_cxx_api.h>
using namespace std;
using namespace cv;
using namespace Ort;
using namespace cv;

struct Detection
{
    cv::Rect box;
    float conf{};
    int classId{};
};

template <typename T>
T clip(const T& n, const T& lower, const T& upper)
{
    return std::max(lower, std::min(n, upper));
}

cv::Rect2f scaleCoords(const cv::Size& imageShape, cv::Rect2f coords, const cv::Size& imageOriginalShape, bool p_Clip = false)
{
    cv::Rect2f l_Result;
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
        (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = { (int)std::round((((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f) - 0.1f),
                 (int)std::round((((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f) - 0.1f) };

    l_Result.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
    l_Result.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

    l_Result.width = (int)std::round(((float)coords.width / gain));
    l_Result.height = (int)std::round(((float)coords.height / gain));

    // // clip coords, should be modified for width and height
    if (p_Clip)
    {
        l_Result.x = clip(l_Result.x, (float)0, (float)imageOriginalShape.width);
        l_Result.y = clip(l_Result.y, (float)0, (float)imageOriginalShape.height);
        l_Result.width = clip(l_Result.width, (float)0, (float)(imageOriginalShape.width - l_Result.x));
        l_Result.height = clip(l_Result.height, (float)0, (float)(imageOriginalShape.height - l_Result.y));
    }
    return l_Result;
}

void getBestClassInfo(const cv::Mat& p_Mat, const int& numClasses,
    float& bestConf, int& bestClassId)
{
    bestClassId = 0;
    bestConf = 0;

    if (p_Mat.rows && p_Mat.cols)
    {
        for (int i = 0; i < numClasses; i++)
        {
            if (p_Mat.at<float>(0, i + 4) > bestConf)
            {
                bestConf = p_Mat.at<float>(0, i + 4);
                bestClassId = i;
            }
        }
    }
}

std::vector<Detection> postprocessing(const cv::Size& resizedImageShape,
    const cv::Size& originalImageShape,
    std::vector<Ort::Value>& outputTensors,
    const float& confThreshold, const float& iouThreshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    auto* rawOutput = outputTensors[0].GetTensorData<float>();
    std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    cv::Mat l_Mat = cv::Mat(outputShape[1], outputShape[2], CV_32FC1, (void*)rawOutput);
    cv::Mat l_Mat_t = l_Mat.t();
    //std::vector<float> output(rawOutput, rawOutput + count);

    // for (const int64_t& shape : outputShape)
    //     std::cout << "Output Shape: " << shape << std::endl;

    // first 5 elements are box[4] and obj confidence
    int numClasses = l_Mat_t.cols - 4;

    // only for batch size = 1

    for (int l_Row = 0; l_Row < l_Mat_t.rows; l_Row++)
    {
        cv::Mat l_MatRow = l_Mat_t.row(l_Row);
        float objConf;
        int classId;

        getBestClassInfo(l_MatRow, numClasses, objConf, classId);

        if (objConf > confThreshold)
        {
            float centerX = (l_MatRow.at<float>(0, 0));
            float centerY = (l_MatRow.at<float>(0, 1));
            float width = (l_MatRow.at<float>(0, 2));
            float height = (l_MatRow.at<float>(0, 3));
            float left = centerX - width / 2;
            float top = centerY - height / 2;

            float confidence = objConf;
            cv::Rect2f l_Scaled = scaleCoords(resizedImageShape, cv::Rect2f(left, top, width, height), originalImageShape, true);

            // Prepare NMS filtered per class id's
            nms_boxes.emplace_back((int)std::round(l_Scaled.x) + classId * 7680, (int)std::round(l_Scaled.y) + classId * 7680,
                (int)std::round(l_Scaled.width), (int)std::round(l_Scaled.height));
            boxes.emplace_back((int)std::round(l_Scaled.x), (int)std::round(l_Scaled.y),
                (int)std::round(l_Scaled.width), (int)std::round(l_Scaled.height));
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(nms_boxes, confs, confThreshold, iouThreshold, indices);
    // std::cout << "amount of NMS indices: " << indices.size() << std::endl;

    std::vector<Detection> detections;

    for (int idx : indices)
    {
        Detection det;
        det.box = boxes[idx];
        det.conf = confs[idx];
        det.classId = classIds[idx];
        detections.emplace_back(det);
    }

    return detections;
}


void letterbox(const cv::Mat& image, cv::Mat& outImage,
    const cv::Size& newShape = cv::Size(640, 640),
    const cv::Scalar& color = cv::Scalar(114, 114, 114),
    bool auto_ = true,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32)
{
    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
        (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{ r, r };
    int newUnpad[2]{ (int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r) };

    auto dw = (float)(newShape.width - newUnpad[0]);
    auto dh = (float)(newShape.height - newUnpad[1]);

    if (auto_)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        newUnpad[0] = newShape.width;
        newUnpad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != newUnpad[0] || shape.height != newUnpad[1])
    {
        cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}


int calculate_product(const std::vector<int64_t>& v)
{
    int total = 1;
    for (auto& i : v)
        total *= i;
    return total;
}

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v)
{
    std::stringstream ss("");
    for (size_t i = 0; i < v.size() - 1; i++)
        ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        printf("usage: DisplayImage.out <Image_Path> <onnxmodel> CPU|GPU\n");
        return -1;
    }

    bool useGPU = false;
    std::string l_GpuOption(argv[3]);
    std::transform(l_GpuOption.begin(), l_GpuOption.end(), l_GpuOption.begin(), [](unsigned char c)
        { return std::tolower(c); });
    if (l_GpuOption == "gpu")
    {
        useGPU = true;
        std::cout << "Using GPU" << std::endl;
    }

#ifdef _WIN32
    std::string str = argv[2];
    std::wstring wide_string = std::wstring(str.begin(), str.end());
    std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
    std::string model_file = argv[2];
#endif

    // onnxruntime setup
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");

    auto providers = Ort::GetAvailableProviders();
    std::cout << "Available providers" << std::endl;
    for (auto provider : providers)
    {
        std::cout << provider << std::endl;
    }
    Ort::SessionOptions session_options;

    if (useGPU)
    {
        OrtCUDAProviderOptions l_CudaOptions;
        l_CudaOptions.device_id = 0;
        std::cout << "Before setting session options" << std::endl;
        session_options.AppendExecutionProvider_CUDA(l_CudaOptions);
        std::cout << "set session options" << std::endl;
    }
    else
    {
        // session_options.SetIntraOpNumThreads(12);
    }

    Ort::Session::Session(std::nullptr_t)
    Ort::Session::Session(const Env & env,
        const char* model_path,
        const SessionOptions & options
    )
    Ort::Session::Session session
    session = Session(env, model_file,std::allocator<>, session_options ); // access experimental components via the Experimental namespace


    // print name/shape of inputs
    std::vector<std::string> input_names = session.GetInputNames();
    std::vector<std::vector<int64_t>> input_shapes = session.GetInputShapes();
    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (size_t i = 0; i < input_names.size(); i++)
    {
        std::cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << std::endl;
    }

    // print name/shape of outputs
    std::vector<std::string> output_names = session.GetOutputNames();
    std::vector<std::vector<int64_t>> output_shapes = session.GetOutputShapes();
    std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (size_t i = 0; i < output_names.size(); i++)
    {
        std::cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << std::endl;
    }

    if (useGPU)
    {
        std::cout << "Perform wamup on CUDA inference" << std::endl;
        // Create a single Ort tensor of random numbers
        auto input_shape = input_shapes[0];
        int total_number_elements = calculate_product(input_shape);
        std::vector<float> input_tensor_values(total_number_elements);
        std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&]
            { return 0.0f; }); // generate random numbers in the range [0, 255]
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));
        for (int count = 0; count < 2; count++)
        {
            auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
            std::cout << "Warmup done" << std::endl;
            std::cout << "output_tensor_shape: " << print_shape(output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;
        }
    }

    Mat image;
    image = imread(argv[1], 1);
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }

    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    letterbox(resizedImage, resizedImage, cv::Size(640, 640),
        cv::Scalar(114, 114, 114), false,
        false, true, 32);
    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    float* blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{ floatImage.cols, floatImage.rows };
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
    std::vector<float> inputTensorValues(blob, blob + 3 * floatImageSize.width * floatImageSize.height);
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(inputTensorValues.data(), inputTensorValues.size(), input_shapes[0]));
    auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
    std::vector<Detection> result = postprocessing(cv::Size(640, 640), image.size(), output_tensors, 0.5, 0.45);
    for (auto detection : result)
    {
        std::cout << "Detection : (" << detection.box.x << "," << detection.box.y << "," << detection.box.x + detection.box.width
            << "," << detection.box.y + detection.box.height << ")" << " Score : " << std::round(detection.conf * 100) / 100.0
            << " Object ID : " << detection.classId << std::endl;
        cv::Point l_P1(detection.box.x, detection.box.y);
        cv::Point l_P2(detection.box.x + detection.box.width, detection.box.y + detection.box.height);
        cv::rectangle(image, l_P1, l_P2, cv::Scalar(0, 255, 0), 1);
        std::stringstream l_ss;
        l_ss << "Object ID:" << int(detection.classId) << " Score:";
        l_ss.setf(std::ios::fixed);
        l_ss.precision(2);
        l_ss << detection.conf;
        std::string l_Label = l_ss.str();
        std::cout << l_Label << std::endl;
        int l_BaseLine = 0;
        cv::Size l_TextSize = cv::getTextSize(l_Label, 0, 0.5, 1, &l_BaseLine);
        bool l_outSide = false;
        if (l_P1.y - l_TextSize.height >= 3)
        {
            l_outSide = true;
        }
        cv::Point l_P3;
        l_P3.x = l_P1.x + l_TextSize.width;
        if (l_outSide)
        {
            l_P3.y = l_P1.y - l_TextSize.height - 3;
        }
        else
        {
            l_P3.y = l_P1.y + l_TextSize.height + 3;
        }
        cv::rectangle(image, l_P1, l_P3, cv::Scalar(0, 255, 0), -1);
        cv::Point l_TextPos;
        l_TextPos.x = l_P1.x;
        if (l_outSide)
        {
            l_TextPos.y = l_P1.y - 2;
        }
        else
        {
            l_TextPos.y = l_P1.y + l_TextSize.height + 2;
        }
        cv::putText(image, l_Label, l_TextPos, 0, 0.5, cv::Scalar(0, 0, 0));
    }
//display
    namedWindow("Display Image", WINDOW_AUTOSIZE);
    imshow("Display Image", image);
    cv::imwrite("output.png", image);
    waitKey(0);
    return 0;
}
*/