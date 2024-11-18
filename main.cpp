#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <filesystem>

using namespace cv;
using namespace std;

// Function to extract frames from a video
void extractFrames(const std::string &video_path, const std::string &output_path) {
    namespace fs = std::filesystem;

    // Ensure the output directory exists
    if (!fs::exists(output_path)) {
        fs::create_directories(output_path);
    }

    VideoCapture video(video_path);
    if (!video.isOpened()) {
        cerr << "Error: video can't be read" << std::endl;
        return;
    }

    int i = 0;

    while (true) {
        Mat frame;
        video >> frame; 
        if (frame.empty()) break;

        // HSV Masking
        Mat hsvframe;
        cvtColor(frame, hsvframe, COLOR_BGR2HSV);

        Scalar lower_bound1(0, 50, 50); 
        Scalar upper_bound1(10, 255, 255); 

        Scalar lower_bound2(170, 50, 50); 
        Scalar upper_bound2(180, 255, 255); 

        Mat mask1, mask2, mask;
        inRange(hsvframe, lower_bound1, upper_bound1, mask1);
        inRange(hsvframe, lower_bound2, upper_bound2, mask2);

        mask = mask1 | mask2;

        // Generate contours
        vector<vector<Point>> contour_vec;
        findContours(mask, contour_vec, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


        double max_area = 0;
        vector<Point> largest_contour;
        for (const auto &contour : contour_vec) {
            double area = contourArea(contour);
            if (area > max_area) {
                max_area = area;
                largest_contour = contour;
            }
        }

        if (!largest_contour.empty()) {
            Rect bounding_rect = boundingRect(largest_contour);
            rectangle(frame, bounding_rect, Scalar(255, 0, 0), 3); // Blue rectangle
        }

        // Save processed frame
        ostringstream afterframe;
        afterframe << output_path << "/afterframe_" << i++ << ".jpg";

        if (!imwrite(afterframe.str(), frame)) {
            cerr << "Error: Could not save processed frame to " << afterframe.str() << endl;
            return;
        }

        imshow("frame", frame);
    }

cout << "Frames successfully parsed to " << output_path << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Error" << endl;
        return -1;
    }

    const string videoPath = argv[1];
    const string outputDir = argv[2];

    extractFrames(videoPath, outputDir);

    return 0;
}
