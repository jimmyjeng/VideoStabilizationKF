#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <time.h>
#include "videostab.h"

using namespace std;
using namespace cv;
const int CUBEMAP_LENGTH = 256;
const int HORIZONTAL_BORDER_CROP = 30;


// This class redirects cv::Exception to our process so that we can catch it and handle it accordingly.
class cvErrorRedirector {
public:
    int cvCustomErrorCallback( )
    {
        std::cout << "A cv::Exception has been caught. Skipping this frame..." << std::endl;
        return 0;
    }

    cvErrorRedirector() {
        cvRedirectError((cv::ErrorCallback)cvCustomErrorCallback(), this);
    }
};


float faceTransform[6][2] =
{
	{0, 0},
	{M_PI / 2, 0},
	{M_PI, 0},
	{-M_PI / 2, 0},
	{0, -M_PI / 2},
	{0, M_PI / 2}
};


void createCubeMapFace(const Mat in, Mat &face,
	int faceId = 0, const int width = -1,
	const int height = -1) {

	float inWidth = in.cols;
	float inHeight = in.rows;

	// Allocate map
	Mat mapx(height, width, CV_32F);
	Mat mapy(height, width, CV_32F);

	// Calculate adjacent (ak) and opposite (an) of the
	// triangle that is spanned from the sphere center 
	//to our cube face.
	const float an = sin(M_PI / 4);
	const float ak = cos(M_PI / 4);

	const float ftu = faceTransform[faceId][0];
	const float ftv = faceTransform[faceId][1];

	// For each point in the target image, 
	// calculate the corresponding source coordinates. 
	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			//cout << "y: " << y << " x:" << x << endl;

			// Map face pixel coordinates to [-1, 1] on plane
			float nx = (float)y / (float)height - 0.5f;
			float ny = (float)x / (float)width - 0.5f;

			nx *= 2;
			ny *= 2;

			// Map [-1, 1] plane coords to [-an, an]
			// thats the coordinates in respect to a unit sphere 
			// that contains our box. 
			nx *= an;
			ny *= an;

			float u, v;

			// Project from plane to sphere surface.
			if (ftv == 0) {
				// Center faces
				u = atan2(nx, ak);
				v = atan2(ny * cos(u), ak);
				u += ftu;
			}
			else if (ftv > 0) {
				// Bottom face 
				float d = sqrt(nx * nx + ny * ny);
				v = M_PI / 2 - atan2(d, ak);
				u = atan2(ny, nx);
			}
			else {
				// Top face
				float d = sqrt(nx * nx + ny * ny);
				v = -M_PI / 2 + atan2(d, ak);
				u = atan2(-ny, nx);
			}

			// Map from angular coordinates to [-1, 1], respectively.
			u = u / (M_PI);
			v = v / (M_PI / 2);

			// Warp around, if our coordinates are out of bounds. 
			while (v < -1) {
				v += 2;
				u += 1;
			}
			while (v > 1) {
				v -= 2;
				u += 1;
			}

			while (u < -1) {
				u += 2;
			}
			while (u > 1) {
				u -= 2;
			}

			// Map from [-1, 1] to in texture space
			u = u / 2.0f + 0.5f;
			v = v / 2.0f + 0.5f;

			u = u * (inWidth - 1);
			v = v * (inHeight - 1);

			// Save the result for this pixel in map
			mapx.at<float>(x, y) = u;
			mapy.at<float>(x, y) = v;
		}
	}

	// Recreate output image if it has wrong size or type. 
	if (face.cols != width || face.rows != height ||
		face.type() != in.type()) {
		face = Mat(width, height, in.type());
	}

	// Do actual resampling using OpenCV's remap
	remap(in, face, mapx, mapy,
		CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
}

int main(int argc, char **argv)
{
	//Create a object of stabilization class
	VideoStab stab0;
	VideoStab stab1;
	VideoStab stab2;
	VideoStab stab3;

	// Read input video
	String videoName = "test";
	String inputVideo = videoName + ".mp4";
	String inputPathVideo0 = videoName + "_ori_0" + ".mp4";
	String inputPathVideo1 = videoName + "_ori_1" + ".mp4";
	String inputPathVideo2 = videoName + "_ori_2" + ".mp4";
	String inputPathVideo3 = videoName + "_ori_3" + ".mp4";

	String outputPathVideo0 = videoName + "_smooth_0" + ".mp4";
	String outputPathVideo1 = videoName + "_smooth_1" + ".mp4";
	String outputPathVideo2 = videoName + "_smooth_2" + ".mp4";
	String outputPathVideo3 = videoName + "_smooth_3" + ".mp4";

	//Initialize the VideoCapture object
	VideoCapture cap(inputVideo);
	Mat tmp;
	Mat currFrame0, currFrameGray0;
	Mat preFrame0, preFrameGray0;

	Mat currFrame1, currFrameGray1;
	Mat preFrame1, preFrameGray1;

	Mat currFrame2, currFrameGray2;
	Mat preFrame2, preFrameGray2;

	Mat currFrame3, currFrameGray3;
	Mat preFrame3, preFrameGray3;

	cap >> tmp;

	// Get frame count
	int n_frames = int(cap.get(CAP_PROP_FRAME_COUNT));

	// Get width and height of video stream
	int w = int(cap.get(CAP_PROP_FRAME_WIDTH));
	int h = int(cap.get(CAP_PROP_FRAME_HEIGHT));
	// Get frames per second (fps)
	double fps = cap.get(CV_CAP_PROP_FPS);

	cout << "total frame " << n_frames << " FPS: " << fps << endl;

    cvErrorRedirector redir;


	VideoWriter srcVideo0, srcVideo1, srcVideo2, srcVideo3;
	VideoWriter outputVideo, outputVideo0, outputVideo1, outputVideo2, outputVideo3;


	srcVideo0 = VideoWriter(inputPathVideo0, CV_FOURCC('H', '2', '6', '4'), fps, Size(CUBEMAP_LENGTH, CUBEMAP_LENGTH));
	srcVideo1 = VideoWriter(inputPathVideo1, CV_FOURCC('H', '2', '6', '4'), fps, Size(CUBEMAP_LENGTH, CUBEMAP_LENGTH));
	srcVideo2 = VideoWriter(inputPathVideo2, CV_FOURCC('H', '2', '6', '4'), fps, Size(CUBEMAP_LENGTH, CUBEMAP_LENGTH));
	srcVideo3 = VideoWriter(inputPathVideo3, CV_FOURCC('H', '2', '6', '4'), fps, Size(CUBEMAP_LENGTH, CUBEMAP_LENGTH));


	outputVideo0 = VideoWriter(outputPathVideo0 , CV_FOURCC('H', '2', '6', '4'), fps, Size(CUBEMAP_LENGTH, CUBEMAP_LENGTH));
	outputVideo1 = VideoWriter(outputPathVideo1, CV_FOURCC('H', '2', '6', '4'), fps, Size(CUBEMAP_LENGTH, CUBEMAP_LENGTH));
	outputVideo2 = VideoWriter(outputPathVideo2, CV_FOURCC('H', '2', '6', '4'), fps, Size(CUBEMAP_LENGTH, CUBEMAP_LENGTH));
	outputVideo3 = VideoWriter(outputPathVideo3, CV_FOURCC('H', '2', '6', '4'), fps, Size(CUBEMAP_LENGTH, CUBEMAP_LENGTH));

	createCubeMapFace(tmp, preFrame0, 0, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
	createCubeMapFace(tmp, preFrame1, 1, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
	createCubeMapFace(tmp, preFrame2, 2, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
	createCubeMapFace(tmp, preFrame3, 3, CUBEMAP_LENGTH, CUBEMAP_LENGTH);


	srcVideo0.write(preFrame0);
	srcVideo1.write(preFrame1);
	srcVideo2.write(preFrame2);
	srcVideo3.write(preFrame3);

    cvtColor(preFrame0, preFrameGray0, COLOR_BGR2GRAY);
	cvtColor(preFrame1, preFrameGray1, COLOR_BGR2GRAY);
	cvtColor(preFrame2, preFrameGray2, COLOR_BGR2GRAY);
	cvtColor(preFrame3, preFrameGray3, COLOR_BGR2GRAY);

    Mat smoothedMat(2,3,CV_64F);

    while(true)
    {
        try {
            cap >> tmp;

            if(tmp.data == NULL)
            {
                break;
            }
			createCubeMapFace(tmp, currFrame0, 0, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
			createCubeMapFace(tmp, currFrame1, 1, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
			createCubeMapFace(tmp, currFrame2, 2, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
			createCubeMapFace(tmp, currFrame3, 3, CUBEMAP_LENGTH, CUBEMAP_LENGTH);

			srcVideo0.write(currFrame0);
			srcVideo1.write(currFrame1);
			srcVideo2.write(currFrame2);
			srcVideo3.write(currFrame3);

            cvtColor(currFrame0, currFrameGray0, COLOR_BGR2GRAY);
			cvtColor(currFrame1, currFrameGray1, COLOR_BGR2GRAY);
			cvtColor(currFrame2, currFrameGray2, COLOR_BGR2GRAY);
			cvtColor(currFrame3, currFrameGray3, COLOR_BGR2GRAY);

            Mat smoothedFrame0, smoothedFrame1, smoothedFrame2, smoothedFrame3;

            smoothedFrame0 = stab0.stabilize(preFrame0, currFrame0, 0);
			smoothedFrame1 = stab1.stabilize(preFrame1, currFrame1, 1);
			smoothedFrame2 = stab2.stabilize(preFrame2, currFrame2, 2);
			smoothedFrame3 = stab3.stabilize(preFrame3, currFrame3, 3);

			outputVideo.write(smoothedFrame0);

            outputVideo0.write(smoothedFrame0);
			outputVideo1.write(smoothedFrame1);
			outputVideo2.write(smoothedFrame2);
			outputVideo3.write(smoothedFrame3);

            //imshow("Stabilized Video" , smoothedFrame);

            preFrame0 = currFrame0.clone();
			preFrameGray0 = currFrameGray0.clone();

			preFrame1 = currFrame1.clone();
			preFrameGray1 = currFrameGray1.clone();

			preFrame2 = currFrame2.clone();
			preFrameGray2 = currFrameGray2.clone();

			preFrame3 = currFrame3.clone();
			preFrameGray3 = currFrameGray3.clone();
        } catch (cv::Exception& e) {
			cout << "Exception" << endl;

            cap >> tmp;
			if (tmp.data == NULL)
			{
				break;
			}
			createCubeMapFace(tmp, preFrame0, 0, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
			createCubeMapFace(tmp, preFrame1, 1, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
			createCubeMapFace(tmp, preFrame2, 2, CUBEMAP_LENGTH, CUBEMAP_LENGTH);
			createCubeMapFace(tmp, preFrame3, 3, CUBEMAP_LENGTH, CUBEMAP_LENGTH);

            cvtColor(preFrame0, preFrameGray0, COLOR_BGR2GRAY);
			cvtColor(preFrame1, preFrameGray1, COLOR_BGR2GRAY);
			cvtColor(preFrame2, preFrameGray2, COLOR_BGR2GRAY);
			cvtColor(preFrame3, preFrameGray3, COLOR_BGR2GRAY);
        }
    }
    return 0;
}


