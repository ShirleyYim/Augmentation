#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void GetInputMat(std::ifstream &ifs, cv::Mat &src, vector<int> &labels, int &sample_num, int &feature_num);

#define PCA_MEAN    "mean"  
#define PCA_EIGEN_VECTOR    "eigen_vector"


//Training  
int main(int argc, char** argv)
{
	if (argc < 3)
	{
		std::cout << "Configuration error!\n";
	}

	std::ifstream ifs;
	ifs.open(argv[1], ios_base::in);

	cv::Mat SampleSet;
	vector<int> labels;
	int feature_num, sample_num;
	if (ifs.is_open())
	{
		GetInputMat(ifs, SampleSet, labels, sample_num, feature_num);
	}
	else
	{
		std::cout << "open input file failed!\n";
		return -1;
	}
	ifs.close();

	PCA *pca = new PCA(SampleSet, Mat(), CV_PCA_DATA_AS_ROW);
	//cout << "eigenvalues:" << endl << pca->eigenvalues << endl << endl;

	//calculate the decreased dimensions  
	int index;
	float sum = 0, sum0 = 0, ratio;
	for (int d = 0; d < pca->eigenvalues.rows; ++d)
	{
		sum += pca->eigenvalues.at<float>(d, 0);
	}
	for (int d = 0; d < pca->eigenvalues.rows; ++d)
	{
		sum0 += pca->eigenvalues.at<float>(d, 0);
		ratio = sum0 / sum;
		if (ratio > 0.95) {
			index = d;
			break;
		}
	}
	cout << "index" << endl << index << endl;
	Mat eigenvetors_d;
	eigenvetors_d.create((index + 1), feature_num, CV_32FC1);//eigen values of decreased dimension  
	for (int i = 0; i < (index + 1); ++i)
	{
		pca->eigenvectors.row(i).copyTo(eigenvetors_d.row(i));
	}
	//cout << "eigenvectors" << endl << eigenvetors_d << endl;
	FileStorage fs_w(argv[2], FileStorage::WRITE);//write mean and eigenvalues into xml file  
	fs_w << PCA_MEAN << pca->mean;
	fs_w << PCA_EIGEN_VECTOR << eigenvetors_d;
	fs_w.release();

	//encoding
	PCA *pca_encoding = new PCA();
	FileStorage fs_r(argv[2], FileStorage::READ);
	fs_r[PCA_MEAN] >> pca_encoding->mean;
	fs_r[PCA_EIGEN_VECTOR] >> pca_encoding->eigenvectors;
	fs_r.release();

	//Mat output_encode(SampleSet.rows, pca_encoding->eigenvectors.rows, CV_32FC1);
	//pca_encoding->project(SampleSet, output_encode);
	//cout << endl << "pca_encode:" << endl << output_encode;

	Mat newset(sample_num, pca_encoding->eigenvectors.rows, CV_32FC1);

	newset = SampleSet * pca_encoding->eigenvectors.t();

	//decoding
	//Mat output_decode(SampleSet.rows, feature_num, CV_32FC1);
	//pca_encoding->backProject(output_encode, output_decode);
	////cout << endl << "pca_Decode:" << endl << output_decode;

	//output
	std::ofstream hog_file;
	hog_file.open(argv[3], ios_base::out);
	if (!hog_file.is_open())
	{
		std::cout << "open output file failed!\n";
		return -1;
	}

	for (int sample_id = 0; sample_id < sample_num; ++sample_id)
	{
		int index = 1;
		hog_file << labels[sample_id] << " ";
		for (int feature_id = 0; feature_id < pca_encoding->eigenvectors.rows; ++feature_id)
		{
			float hog_data = newset.at<float>(sample_id, feature_id);
			if (hog_data)
			{
				hog_file << index << ":";
				hog_file << hog_data << " ";
			}
			++index;
		}
		hog_file << endl;
	}

	hog_file.close();

	return 0;
}

void GetInputMat(std::ifstream &ifs, cv::Mat &src, vector<int> &labels, int &sample_num, int &feature_num)
{
	int row_id = 0, label;
	string line;

	getline(ifs, line);
	stringstream linestream(line);
	linestream >> sample_num;
	linestream >> feature_num;

	src.create(sample_num, feature_num, CV_32FC1);
	while (!ifs.eof())
	{
		getline(ifs, line);
		stringstream linestream(line);

		linestream >> label;
		labels.push_back(label);

		int col_id = 0;
		while (col_id < feature_num)
		{
			linestream >> src.at<float>(row_id, col_id);
			++col_id;
		}
		++row_id;
	}

}