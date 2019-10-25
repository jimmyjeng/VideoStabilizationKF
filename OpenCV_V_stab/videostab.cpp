#define _USE_MATH_DEFINES

#include "dbscan.h"
#include "videostab.h"
#include <cmath>

//Parameters for Kalman Filter
#define Q1 0.004
#define R1 0.5

//To see the results of before and after stabilization simultaneously
#define test 1

// 0
std::ofstream warp_X0("Warp_X0.txt");
std::ofstream warp_Y0("Warp_Y0.txt");
std::ofstream warp_A0("Warp_A0.txt");

// 1
std::ofstream warp_X1("Warp_X1.txt");
std::ofstream warp_Y1("Warp_Y1.txt");
std::ofstream warp_A1("Warp_A1.txt");

// 2
std::ofstream warp_X2("Warp_X2.txt");
std::ofstream warp_Y2("Warp_Y2.txt");
std::ofstream warp_A2("Warp_A2.txt");

// 3
std::ofstream warp_X3("Warp_X3.txt");
std::ofstream warp_Y3("Warp_Y3.txt");
std::ofstream warp_A3("Warp_A3.txt");

struct Polar
{
	Polar() {}
	Polar(double _length, double _angle)
	{
		length = _length;
		angle = _angle;
	}

	double length;
	double angle;
};

VideoStab::VideoStab()
{

    smoothedMat.create(2 , 3 , CV_64F);

    k = 1;

    errscaleX = 1;
    errscaleY = 1;
    errthetha = 1;
    errtransX = 1;
    errtransY = 1;

    Q_scaleX = Q1;
    Q_scaleY = Q1;
    Q_thetha = Q1;
    Q_transX = Q1;
    Q_transY = Q1;

    R_scaleX = R1;
    R_scaleY = R1;
    R_thetha = R1;
    R_transX = R1;
    R_transY = R1;

    sum_scaleX = 0;
    sum_scaleY = 0;
    sum_thetha = 0;
    sum_transX = 0;
    sum_transY = 0;

    scaleX = 0;
    scaleY = 0;
    thetha = 0;
    transX = 0;
    transY = 0;

}

//The main stabilization function
Mat VideoStab::stabilize(Mat preFrame, Mat currFrame, int direction)
{
    cvtColor(preFrame, preFrameGray, COLOR_BGR2GRAY);
    cvtColor(currFrame, currFrameGray, COLOR_BGR2GRAY);

    int vert_border = HORIZONTAL_BORDER_CROP * preFrame.rows / preFrame.cols;

    vector <Point2f> features1, features2;
    vector <Point2f> goodFeatures1, goodFeatures2;
	vector <Point2f> RS_Features1, RS_Features2;
    vector <uchar> status;
    vector <float> err;

    //Estimating the features in frame1 and frame2
    goodFeaturesToTrack(preFrameGray, features1, 2000, 0.01 , 10 );
    calcOpticalFlowPyrLK(preFrameGray, currFrameGray, features1, features2, status, err );
	cout <<"Index :"<< k << " features1 :" << features1.size() << " features2 : " << features2.size() <<endl;
	vector<ClusterPoint> points;

    for(size_t i=0; i < status.size(); i++)
    {
        if(status[i])
        {
            goodFeatures1.push_back(features1[i]);
            goodFeatures2.push_back(features2[i]);

			/*Polar p;
			p.angle = atan(features2[i].y - features1[i].y / features2[i].x - features1[i].x);
			p.length = sqrt(pow(features2[i].x - features1[i].x, 2) + pow(features2[i].y - features1[i].y, 2) );
			cout << "angle : " << p.angle<< " length : " << p.length<<endl;*/

			//ClusterPoint p;
			//p.x = features2[i].x - features1[i].x;
			//p.y = features2[i].y - features1[i].y;
			//points.push_back(p);
        }
    }
	//cout << "goodFeatures1 :" << goodFeatures1.size() << " goodFeatures2 : " << goodFeatures2.size() << endl;

	//vector<int> labels;

	//int num = dbscan(points, labels, 0.5, 10);

	//cout << "cluster size is " << num << endl;
	//if (num > 1) {
	//	for (int i = 0; i < (int)points.size(); i++) {
	//		std::cout << "Point(" << points[i].x << ", " << points[i].y << "): " << labels[i] << std::endl;
	//	}
	//}

	// //RANSAC
	//vector<uchar> RansacStatus;
	//Mat Fundamental = findFundamentalMat(goodFeatures1, goodFeatures2, RansacStatus, FM_RANSAC);

	//for (size_t i = 0; i < status.size(); i++)
	//{
	//	if (RansacStatus[i] != 0)
	//	{
	//		RS_Features1.push_back(goodFeatures1[i]);
	//		RS_Features2.push_back(goodFeatures2[i]);
	//	}
	//}
	//cout << "RS_Features1 :" << RS_Features1.size() << " RS_Features2 : " << RS_Features2.size() << endl;
	//if (goodFeatures1.size() > RS_Features1.size()) {
	//	cout << "features1 :" << features1.size() << " features2 : " << features2.size() <<endl;
	//	cout << "goodFeatures1 :" << goodFeatures1.size() << " goodFeatures2 : " << goodFeatures2.size() << endl;
	//	cout << "RS_Features1 :" << RS_Features1.size() << " RS_Features2 : " << RS_Features2.size() << endl;
	//}

	//All the parameters scale, angle, and translation are stored in affine
    affine = estimateRigidTransform(goodFeatures1, goodFeatures2, false);

    // If an affine transformation is not found, return the most recent frame received.
    if(affine.size().height == 0 || affine.size().width == 0)
        return currFrame;
    
    //cout<<affine;
    //flush(cout);

    //affine = affineTransform(goodFeatures1 , goodFeatures2);

    dx = affine.at<double>(0,2);
    dy = affine.at<double>(1,2);
    da = atan2(affine.at<double>(1,0), affine.at<double>(0,0));
    ds_x = affine.at<double>(0,0)/cos(da);
    ds_y = affine.at<double>(1,1)/cos(da);

    sx = ds_x;
    sy = ds_y;

    sum_transX += dx;
    sum_transY += dy;
    sum_thetha += da;
    sum_scaleX += ds_x;
    sum_scaleY += ds_y;


    //Don't calculate the predicted state of Kalman Filter on 1st iteration
    if(k==1)
    {
        k++;
    }
    else
    {
		k++;
        Kalman_Filter(&scaleX , &scaleY , &thetha , &transX , &transY);

    }

    diff_scaleX = scaleX - sum_scaleX;
    diff_scaleY = scaleY - sum_scaleY;
    diff_transX = transX - sum_transX;
    diff_transY = transY - sum_transY;
    diff_thetha = thetha - sum_thetha;

    ds_x = ds_x + diff_scaleX;
    ds_y = ds_y + diff_scaleY;
    dx = dx + diff_transX;
    dy = dy + diff_transY;
    da = da + diff_thetha;

    //Creating the smoothed parameters matrix
    smoothedMat.at<double>(0,0) = sx * cos(da);
    smoothedMat.at<double>(0,1) = sx * -sin(da);
    smoothedMat.at<double>(1,0) = sy * sin(da);
    smoothedMat.at<double>(1,1) = sy * cos(da);

	if (dx > 50) {
		dx = 50;
	}
	if (dx < -50) {
		dx = -50;
	}
	if (dy > 50) {
		dy = 50;
	}
	if (dy < -50) {
		dy = -50;
	}
    smoothedMat.at<double>(0,2) = dx;
    smoothedMat.at<double>(1,2) = dy;

    //Uncomment if you want to see smoothed values
    //cout<<smoothedMat;
    //flush(cout);

	// Log
	if (direction == 0) {
		warp_X0 << dx << ", ";
		warp_Y0 << dy << ", ";
		warp_A0 << sx * da << ", ";
	} 
	else if (direction == 1) {
		warp_X1 << dx << ", ";
		warp_Y1 << dy << ", ";
		warp_A1 << sx * da << ", ";
	}
	else if (direction == 2) {
		warp_X2 << dx << ", ";
		warp_Y2 << dy << ", ";
		warp_A2 << sx * da << ", ";
	}
	else if (direction == 3) {
		warp_X3 << dx << ", ";
		warp_Y3 << dy << ", ";
		warp_A3 << sx * da << ", ";
	}

    //Warp the new frame using the smoothed parameters
    warpAffine(preFrame, smoothedFrame, smoothedMat, currFrame.size());

    //Crop the smoothed frame a little to eliminate black region due to Kalman Filter
    smoothedFrame = smoothedFrame(Range(vert_border, smoothedFrame.rows-vert_border), Range(HORIZONTAL_BORDER_CROP, smoothedFrame.cols-HORIZONTAL_BORDER_CROP));
    resize(smoothedFrame, smoothedFrame, currFrame.size());

    //Change the value of test if you want to see both unstabilized and stabilized video
    if(test)
    {
        Mat canvas = Mat::zeros(currFrame.rows, currFrame.cols*2+10, currFrame.type());

        preFrame.copyTo(canvas(Range::all(), Range(0, smoothedFrame.cols)));

        smoothedFrame.copyTo(canvas(Range::all(), Range(smoothedFrame.cols+10, smoothedFrame.cols*2+10)));

        if(canvas.cols > 1920)
        {
            resize(canvas, canvas, Size(canvas.cols/2, canvas.rows/2));
        }
		//if (direction == 0) {
		//	imshow("before and after 0", canvas);
		//}
		//else if (direction == 1) {
		//	imshow("before and after 1", canvas);
		//}
		//else if (direction == 2) {
		//	imshow("before and after 2", canvas);
		//}
		//else if (direction == 3) {
		//	imshow("before and after 3", canvas);
		//}
    }

    return smoothedFrame;

}

TransformParam VideoStab::stabilize(Mat preFrame, Mat currFrame)
{
	cvtColor(preFrame, preFrameGray, COLOR_BGR2GRAY);
	cvtColor(currFrame, currFrameGray, COLOR_BGR2GRAY);

	int vert_border = HORIZONTAL_BORDER_CROP * preFrame.rows / preFrame.cols;

	vector <Point2f> features1, features2;
	vector <Point2f> goodFeatures1, goodFeatures2;
	vector <uchar> status;
	vector <float> err;

	//Estimating the features in frame1 and frame2
	goodFeaturesToTrack(preFrameGray, features1, 2000, 0.01, 10);
	calcOpticalFlowPyrLK(preFrameGray, currFrameGray, features1, features2, status, err);
	for (size_t i = 0; i < status.size(); i++)
	{
		if (status[i])
		{
			goodFeatures1.push_back(features1[i]);
			goodFeatures2.push_back(features2[i]);
		}
	}

	//All the parameters scale, angle, and translation are stored in affine
	affine = estimateRigidTransform(goodFeatures1, goodFeatures2, false);

	// If an affine transformation is not found, return the most recent frame received.
	//if (affine.size().height == 0 || affine.size().width == 0)
	//	return currFrame;

	//cout<<affine;
	//flush(cout);

	//affine = affineTransform(goodFeatures1 , goodFeatures2);

	dx = affine.at<double>(0, 2);
	dy = affine.at<double>(1, 2);
	da = atan2(affine.at<double>(1, 0), affine.at<double>(0, 0));
	ds_x = affine.at<double>(0, 0) / cos(da);
	ds_y = affine.at<double>(1, 1) / cos(da);

	sx = ds_x;
	sy = ds_y;

	sum_transX += dx;
	sum_transY += dy;
	sum_thetha += da;
	sum_scaleX += ds_x;
	sum_scaleY += ds_y;


	//Don't calculate the predicted state of Kalman Filter on 1st iteration
	if (k == 1)
	{
		k++;
	}
	else
	{
		Kalman_Filter(&scaleX, &scaleY, &thetha, &transX, &transY);

	}

	diff_scaleX = scaleX - sum_scaleX;
	diff_scaleY = scaleY - sum_scaleY;
	diff_transX = transX - sum_transX;
	diff_transY = transY - sum_transY;
	diff_thetha = thetha - sum_thetha;

	ds_x = ds_x + diff_scaleX;
	ds_y = ds_y + diff_scaleY;
	dx = dx + diff_transX;
	dy = dy + diff_transY;
	da = da + diff_thetha;

	//Creating the smoothed parameters matrix
	smoothedMat.at<double>(0, 0) = sx * cos(da);
	smoothedMat.at<double>(0, 1) = sx * -sin(da);
	smoothedMat.at<double>(1, 0) = sy * sin(da);
	smoothedMat.at<double>(1, 1) = sy * cos(da);

	smoothedMat.at<double>(0, 2) = dx;
	smoothedMat.at<double>(1, 2) = dy;

	//Uncomment if you want to see smoothed values
	//cout<<smoothedMat;
	//flush(cout);

	return TransformParam(dx, dy, sx * da);

}

//Kalman Filter implementation
void VideoStab::Kalman_Filter(double *scaleX , double *scaleY , double *thetha , double *transX , double *transY)
{
    double frame_1_scaleX = *scaleX;
    double frame_1_scaleY = *scaleY;
    double frame_1_thetha = *thetha;
    double frame_1_transX = *transX;
    double frame_1_transY = *transY;

    double frame_1_errscaleX = errscaleX + Q_scaleX;
    double frame_1_errscaleY = errscaleY + Q_scaleY;
    double frame_1_errthetha = errthetha + Q_thetha;
    double frame_1_errtransX = errtransX + Q_transX;
    double frame_1_errtransY = errtransY + Q_transY;

    double gain_scaleX = frame_1_errscaleX / (frame_1_errscaleX + R_scaleX);
    double gain_scaleY = frame_1_errscaleY / (frame_1_errscaleY + R_scaleY);
    double gain_thetha = frame_1_errthetha / (frame_1_errthetha + R_thetha);
    double gain_transX = frame_1_errtransX / (frame_1_errtransX + R_transX);
    double gain_transY = frame_1_errtransY / (frame_1_errtransY + R_transY);

    *scaleX = frame_1_scaleX + gain_scaleX * (sum_scaleX - frame_1_scaleX);
    *scaleY = frame_1_scaleY + gain_scaleY * (sum_scaleY - frame_1_scaleY);
    *thetha = frame_1_thetha + gain_thetha * (sum_thetha - frame_1_thetha);
    *transX = frame_1_transX + gain_transX * (sum_transX - frame_1_transX);
    *transY = frame_1_transY + gain_transY * (sum_transY - frame_1_transY);

    errscaleX = ( 1 - gain_scaleX ) * frame_1_errscaleX;
    errscaleY = ( 1 - gain_scaleY ) * frame_1_errscaleX;
    errthetha = ( 1 - gain_thetha ) * frame_1_errthetha;
    errtransX = ( 1 - gain_transX ) * frame_1_errtransX;
    errtransY = ( 1 - gain_transY ) * frame_1_errtransY;
}
