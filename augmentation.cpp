#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "filesystem.h"

using namespace cv;
using namespace std;

char file_xml[] = "D:\\opencv310\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml";

enum EmotionType { Neutral = 0, Anger = 1, Contempt = 2, Disgust = 3, Fear = 4, Happy = 5, Sad = 6, Surprise = 7 };

void DetectFacesInCk(string &emotion_file, string &out_root, string &out_file, cv::CascadeClassifier &classifier);
void FlipImages(string &src_root, string &src_file, string &out_file);
void ContrastImages(string &src_root, string &src_file, string &out_file);
void UnsharpMask(const IplImage* src, IplImage* dst, float amount, float radius, uchar threshold, int contrast);
void ImageEnhance(string &src_root, string &src_file, string &out_root, string &out_file);
void AdjustContrast(const IplImage* src, IplImage* dst, int contrast);
void GetMaxFace(vector<cv::Rect> &faces, cv::Rect &rect);


int main(int argc, char** argv)
{
	cv::VideoCapture video_capture;
	cv::Mat captured_image;

	cv::CascadeClassifier classifier;
	classifier.load(file_xml);

	string ck = "E:\\dataset\\CK\\CK+\\EmotionList.txt";
	string out_root = "E:\\dataset\\CK\\CKAugment\\";
	string out_file = "E:\\dataset\\CK\\CKAugment\\list.txt";
	//DetectFacesInCk(ck, out_root, out_file, classifier);

	string out_file2 = "E:\\dataset\\CK\\CKAugment\\flip.txt";
	FlipImages(out_root, out_file, out_file2);

	string out_file3 = "E:\\dataset\\CK\\CKAugment\\contrast.txt";
	ContrastImages(out_root, out_file2, out_file3);

	////image enchance
	////string out_file4 = "E:\\dataset\\CK\\test\\enhance.txt";
	////string out_dir = "E:\\dataset\\CK\\test\\";
	////ImageEnhance(out_root, out_file2, out_dir, out_file4);


	return 0;
}


void ImageEnhance(string &src_root, string &src_file, string &out_root, string &out_file)
{
	std::ifstream ifs(src_file, ios_base::in);
	if (!ifs.is_open())
	{
		cout << "Couldn't open the src_file, aborting" << endl;
		return;
	}
	std::ofstream ofs(out_file, ios_base::out);
	if (!ofs.is_open())
	{
		cout << "Couldn't open the list file, aborting" << endl;
		return;
	}

	string line, imagename, imagename2;
	int label;
	cv::Mat image, image2, gray;

	while (!ifs.eof())
	{
		getline(ifs, line);
		stringstream lineStream(line);
		lineStream >> imagename;
		lineStream >> label;
		image = cv::imread(src_root + imagename, -1);
		ofs << imagename << " " << label << endl;

		//contrast
		filesystem::path dir(out_root);
		if (image.channels() > 1)
		{
			cv::cvtColor(image, gray, CV_BGR2GRAY);
		}
		else
		{
			image.copyTo(gray);
		}

		cv::imwrite((dir / imagename).str(), gray);

		size_t found = imagename.find(".");
		imagename2 = imagename.replace(found, 0, "_e");
		image2.create(gray.rows, gray.cols, CV_8UC1);

		IplImage src_image = gray;
		IplImage dst_image = image2;
		UnsharpMask(&src_image, &dst_image, 30, 5, 10, 200);

		cv::imwrite((dir / imagename2).str(), image2);
		ofs << imagename2 << " " << label << endl;

		cv::namedWindow("EnhanceImage");
		cv::imshow("EnhanceImage", image2);
		cv::waitKey(5);

	}
}

void AdjustContrast(const IplImage* src, IplImage* dst, int contrast)
{
	if (!src) return;

	int imagewidth = src->width;
	int imageheight = src->height;
	int channel = src->nChannels;

	//求原图均值
	CvScalar mean = { 0,0,0,0 };
	for (int y = 0; y < imageheight; y++)
	{
		for (int x = 0; x < imagewidth; x++)
		{
			CvScalar ori = cvGet2D(src, y, x);
			for (int k = 0; k < channel; k++)
			{
				mean.val[k] += ori.val[k];
			}
		}
	}
	for (int k = 0; k < channel; k++)
	{
		mean.val[k] /= imagewidth * imageheight;
	}

	//调整对比度
	if (contrast <= -255)
	{
		//当增量等于-255时，是图像对比度的下端极限，此时，图像RGB各分量都等于阀值，图像呈全灰色，灰度图上只有1条线，即阀值灰度；
		for (int y = 0; y < imageheight; y++)
		{
			for (int x = 0; x < imagewidth; x++)
			{
				cvSet2D(dst, y, x, mean);
			}
		}
	}
	else if (contrast > -255 && contrast <= 0)
	{
		//(1)nRGB = RGB + (RGB - Threshold) * Contrast / 255
		// 当增量大于-255且小于0时，直接用上面的公式计算图像像素各分量
		//公式中，nRGB表示调整后的R、G、B分量，RGB表示原图R、G、B分量，Threshold为给定的阀值，Contrast为处理过的对比度增量。
		for (int y = 0; y < imageheight; y++)
		{
			for (int x = 0; x < imagewidth; x++)
			{
				CvScalar nRGB;
				CvScalar ori = cvGet2D(src, y, x);
				for (int k = 0; k < channel; k++)
				{
					nRGB.val[k] = ori.val[k] + (ori.val[k] - mean.val[k]) *contrast / 255;
				}
				cvSet2D(dst, y, x, nRGB);
			}
		}
	}
	else if (contrast > 0 && contrast < 255)
	{
		//当增量大于0且小于255时，则先按下面公式(2)处理增量，然后再按上面公式(1)计算对比度：
		//(2)、nContrast = 255 * 255 / (255 - Contrast) - 255
		//公式中的nContrast为处理后的对比度增量，Contrast为给定的对比度增量。                

		CvScalar nRGB;
		int nContrast = 255 * 255 / (255 - contrast) - 255;

		for (int y = 0; y < imageheight; y++)
		{
			for (int x = 0; x < imagewidth; x++)
			{
				CvScalar ori = cvGet2D(src, y, x);
				for (int k = 0; k < channel; k++)
				{
					nRGB.val[k] = ori.val[k] + (ori.val[k] - mean.val[k]) *nContrast / 255;
				}
				cvSet2D(dst, y, x, nRGB);
			}
		}
	}
	else
	{
		//当增量等于 255时，是图像对比度的上端极限，实际等于设置图像阀值，图像由最多八种颜色组成，灰度图上最多8条线，
		//即红、黄、绿、青、蓝、紫及黑与白；        
		for (int y = 0; y < imageheight; y++)
		{
			for (int x = 0; x < imagewidth; x++)
			{
				CvScalar rgb;
				CvScalar ori = cvGet2D(src, y, x);
				for (int k = 0; k < channel; k++)
				{
					if (ori.val[k] > mean.val[k])
					{
						rgb.val[k] = 255;
					}
					else
					{
						rgb.val[k] = 0;
					}
				}
				cvSet2D(dst, y, x, rgb);
			}
		}
	}
}


void UnsharpMask(const IplImage* src, IplImage* dst, float amount, float radius, uchar threshold, int contrast)
{
	if (!src)return;

	int imagewidth = src->width;
	int imageheight = src->height;
	int channel = src->nChannels;

	IplImage* blurimage = cvCreateImage(cvSize(imagewidth, imageheight), src->depth, channel);
	IplImage* DiffImage = cvCreateImage(cvSize(imagewidth, imageheight), 8, channel);

	//原图的高对比度图像
	IplImage* highcontrast = cvCreateImage(cvSize(imagewidth, imageheight), 8, channel);
	AdjustContrast(src, highcontrast, contrast);

	//原图的模糊图像
	cvSmooth(src, blurimage, CV_GAUSSIAN, radius);

	//原图与模糊图作差
	for (int y = 0; y < imageheight; y++)
	{
		for (int x = 0; x < imagewidth; x++)
		{
			CvScalar ori = cvGet2D(src, y, x);
			CvScalar blur = cvGet2D(blurimage, y, x);
			CvScalar val;
			val.val[0] = abs(ori.val[0] - blur.val[0]);
			val.val[1] = abs(ori.val[1] - blur.val[1]);
			val.val[2] = abs(ori.val[2] - blur.val[2]);

			cvSet2D(DiffImage, y, x, val);
		}
	}

	//锐化
	for (int y = 0; y < imageheight; y++)
	{
		for (int x = 0; x < imagewidth; x++)
		{
			CvScalar hc = cvGet2D(highcontrast, y, x);
			CvScalar diff = cvGet2D(DiffImage, y, x);
			CvScalar ori = cvGet2D(src, y, x);
			CvScalar val;

			for (int k = 0; k < channel; k++)
			{
				if (diff.val[k] > threshold)
				{
					//最终图像 = 原始*(1-r) + 高对比*r
					val.val[k] = ori.val[k] * (100 - amount) + hc.val[k] * amount;
					val.val[k] /= 100;
				}
				else
				{
					val.val[k] = ori.val[k];
				}
			}
			cvSet2D(dst, y, x, val);
		}
	}

	cvReleaseImage(&blurimage);
	cvReleaseImage(&DiffImage);
}

void ContrastImages(string &src_root, string &src_file, string &out_file)
{
	std::ifstream ifs(src_file, ios_base::in);
	if (!ifs.is_open())
	{
		cout << "Couldn't open the src_file, aborting" << endl;
		return;
	}
	std::ofstream ofs(out_file, ios_base::out);
	if (!ofs.is_open())
	{
		cout << "Couldn't open the list file, aborting" << endl;
		return;
	}

	string line, imagename, imagename2;
	int label;
	cv::Mat image, image2, gray;

	while (!ifs.eof())
	{
		getline(ifs, line);
		stringstream lineStream(line);
		lineStream >> imagename;
		lineStream >> label;
		image = cv::imread(src_root+ imagename, -1);
		ofs << imagename << " " << label << endl;

		//contrast
		filesystem::path dir(src_root);
		size_t found = imagename.find(".");
		imagename2 = imagename.replace(found, 0, "_c");

		if (image.channels() > 1)
		{
			cv::cvtColor(image, gray, CV_BGR2GRAY);
		}
		else
		{
			image.copyTo(gray);
		}

		cv::equalizeHist(gray, image2);
		cv::imwrite((dir / imagename2).str(), image2);
		ofs << imagename2 << " " << label << endl;

		cv::namedWindow("ContrastImage");
		cv::imshow("ContrastImage", image2);
		cv::waitKey(5);

	}
}

void FlipImages(string &src_root, string &src_file, string &out_file)
{
	std::ifstream ifs(src_file, ios_base::in);
	if (!ifs.is_open())
	{
		cout << "Couldn't open the src_file, aborting" << endl;
		return;
	}
	std::ofstream ofs(out_file, ios_base::out);
	if (!ofs.is_open())
	{
		cout << "Couldn't open the list file, aborting" << endl;
		return;
	}

	string line, imagename, imagename2;
	int label;
	cv::Mat image, image2, gray;

	while (!ifs.eof())
	{
		getline(ifs, line);
		stringstream lineStream(line);
		lineStream >> imagename;
		lineStream >> label;
		image = cv::imread(src_root + imagename, -1);
		ofs << imagename << " " << label << endl;

		//flip
		filesystem::path dir(src_root);
		size_t found = imagename.find(".");
		
		cv::flip(image, image2, 1);
		imagename2 = imagename.replace(found, 0, "_f");

		cv::imwrite((dir / imagename2).str(), image2);
		ofs << imagename2 << " " << label << endl;

		cv::namedWindow("FlipImage");
		cv::imshow("FlipImage", image2);
		cv::waitKey(5);
	}
}

void DetectFacesInCk(string &emotion_file, string &out_root, string &out_file, cv::CascadeClassifier &classifier)
{
	std::ifstream locations(emotion_file, ios_base::in);
	if (!locations.is_open())
	{
		cout << "Couldn't open the emotion_file, aborting" << endl;
		return;
	}
	std::ofstream ofs(out_file, ios_base::out);
	if (!ofs.is_open())
	{
		cout << "Couldn't open the list file, aborting" << endl;
		return;
	}

	string line, labelline;
	while (!locations.eof())
	{
		getline(locations, line);

		//string image, image_directory;
		float label;
		cv::Mat image, cropimage;
		string  imagename;

		if (line.size())
		{
			//emotion label
			std::ifstream emotion_file(line.c_str(), ios_base::in);
			getline(emotion_file, labelline);

			stringstream lineStream(labelline);
			lineStream >> label;
			if (label != Contempt) //except contempt
			{
				//get image location
				string str("Emotion");
				string str2("cohn-kanade-images");
				int size = str.size();
				std::size_t found = line.find(str);
				line = line.replace(found, size, str2);

				//get corresponding image face
				str = "_emotion.txt";
				str2 = ".png";
				size = str.size();
				found = line.find(str);
				line = line.replace(found, size, str2);

				vector<cv::Rect> faces;
				cv::Rect rect;
				image = cv::imread(line, -1);
				classifier.detectMultiScale(image, faces, 1.1, 3, 0);
				if (faces.size() > 0)
				{
					GetMaxFace(faces, rect);
					cropimage = image(rect);
					filesystem::path loc = filesystem::path(line).parent_path();
					imagename = line.substr(loc.str().size() + 1);
					filesystem::path dir(out_root);
					dir = dir / imagename;
					cv::imwrite(dir.str(), cropimage);
					ofs << imagename << " " << label << endl;
					cv::imshow("CropFace", cropimage);
					cv::waitKey(5);
				}
				faces.clear();

				//the fourth image
				if (label == Sad || label == Fear)
				{
					found = line.find_last_of('_');
					str = line.substr(found + 1, 8);

					int index;
					sscanf(str.c_str(), "%08d", &index);
					index = index - 4;
					sprintf((char *)(str.c_str()), "%08d", index);
					line = line.replace(found + 1, 8, str);
					image = cv::imread(line, -1);
					classifier.detectMultiScale(image, faces, 1.1, 3, 0);
					if (faces.size() > 0)
					{
						GetMaxFace(faces, rect);
						cropimage = image(rect);
						filesystem::path loc = filesystem::path(line).parent_path();
						imagename = line.substr(loc.str().size() + 1);
						filesystem::path dir(out_root);
						dir = dir / imagename;
						cv::imwrite(dir.str(), cropimage);
						ofs << imagename << " " << label << endl;
						cv::imshow("CropFace", cropimage);
						cv::waitKey(5);
					}
					faces.clear();
				}

				//Neutral image
				found = line.find_last_of("_");
				line = line.replace(line.begin() + found, line.end(), "_00000001.png");
				image = cv::imread(line, -1);
				classifier.detectMultiScale(image, faces, 1.1, 3, 0);
				if (faces.size() > 0)
				{
					GetMaxFace(faces, rect);
					cropimage = image(rect);
					filesystem::path loc = filesystem::path(line).parent_path();
					imagename = line.substr(loc.str().size() + 1);
					filesystem::path dir(out_root);
					dir = dir / imagename;
					cv::imwrite(dir.str(), cropimage);
					ofs << imagename << " " << 0 << endl;
					cv::imshow("CropFace", cropimage);
					cv::waitKey(5);
				}
				faces.clear();
			}
		}
		else
		{
			cout << "Couldn't find the emotion file, aborting" << endl;
			return;
		}
	}
	
	locations.close();
	ofs.close();
}

void GetMaxFace(vector<cv::Rect> &faces, cv::Rect &rect)
{
	int max_height = 0;
	int idx = 0;

	if (faces.size() == 0)
	{
		return;
	}
	else if (faces.size() == 1)
	{
		rect = faces[0];
	}
	else
	{
		max_height = faces[0].height;
		for (int i = 0; i < faces.size(); ++i)
		{
			if (faces[i].height > max_height)
			{
				max_height = faces[i].height;
				idx = i;
			}
		}

		rect = faces[idx];
	}
}