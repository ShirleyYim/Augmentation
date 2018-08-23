#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

char file_xml[] = "D:\\opencv310\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";

int main(int argc, char** argv)
{
	cv::VideoCapture video_capture;
	cv::Mat captured_image;

	cv::CascadeClassifier classifier;
	classifier.load(file_xml);

	//captured_image = cv::imread(curr_img_file, -1);
	video_capture = cv::VideoCapture(0);
	while (1)
	{
		video_capture >> captured_image;

		vector<cv::Rect> faces;
		cv::Rect rect;
		classifier.detectMultiScale(captured_image, faces, 1.3, 3, 0, cv::Size(60, 60));

		if (faces.size() > 0)
		{
			rect = faces[0];
			cv::rectangle(captured_image, rect, CV_RGB(255, 0, 0), 3, 8, 0);
		}

		cv::imshow("img", captured_image);

		// detect key presses
		char character_press = waitKey(5);
		if (character_press == 'q')
		{
			break;
		}
	}

	return 0;
}

