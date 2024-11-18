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

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Error" << endl;
        return -1;
    }

    const string video_path = argv[1];
    const string output_path = argv[2];

    if (!filesystem::exists(output_path)) {
        filesystem::create_directories(output_path);
    }

    // Open video
    VideoCapture video(video_path);
    if (!video.isOpened()) {
        cerr << "Error" << endl;
        return -1;
    }

    int i = 0;

    // Parse every frame until video ends
    while (true) {
        Mat frame;
        video >> frame; 
        if (frame.empty()) break;

        // Convert to HSV
        Mat hsvframe;
        cvtColor(frame, hsvframe, COLOR_BGR2HSV);

        // HSV boundaries for red
        Scalar lower_bound1(0, 50, 50);
        Scalar upper_bound1(10, 255, 255);
        Scalar lower_bound2(170, 50, 50);
        Scalar upper_bound2(180, 255, 255);

        // Masking
        Mat mask1, mask2, mask;
        inRange(hsvframe, lower_bound1, upper_bound1, mask1);
        inRange(hsvframe, lower_bound2, upper_bound2, mask2);
        mask = mask1 | mask2; // Using bitwise OR to combine both mask

        // Generate contours
        vector<vector<Point>> contour_vec;
        findContours(mask, contour_vec, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // Find largest contour
        double max = 0;
        vector<Point> largest_contour;
        for (const auto &contour : contour_vec) {
            double area = contourArea(contour);
            if (area > max) {
                max = area;
                largest_contour = contour;
            }
        }

        // Make bounding rectangle based on largest contour
        if (!largest_contour.empty()) {
            Rect bounding_rect = boundingRect(largest_contour);
            rectangle(frame, bounding_rect, Scalar(255, 0, 0), 3); // Blue bounding rectangle
        }

        // Save processed frame to folder
        ostringstream afterframe;
        afterframe << output_path << "/afterframe_" << i++ << ".jpg";
        if (!imwrite(afterframe.str(), frame)) {
            cerr << "Error: Could not save" << endl;
            return -1;
        }
    }

    cout << "Frames successfully parsed to: " << output_path << endl;
    return 0;
}
