/************************************************************************
* File:	RunTracker.cpp
* Brief: C++ demo for paper: Kaihua Zhang, Lei Zhang, Ming-Hsuan Yang,"Real-Time Compressive Tracking," ECCV 2012.
* Version: 1.0
* Author: Yang Xian
* Email: yang_xian521@163.com
* Date:	2012/08/03
* History:
* Revised by Kaihua Zhang on 14/8/2012, 23/8/2012
* Email: zhkhua@gmail.com
* Homepage: http://www4.comp.polyu.edu.hk/~cskhzhang/
* Project Website: http://www4.comp.polyu.edu.hk/~cslzhang/CT/CT.htm
************************************************************************/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include "CompressiveTracker.h"

#include <stdlib.h>
// #include <Windef.h>
// #include <WinBase.h>
#define MAX_PATH 260

#define FANG_FIX 1

#define ONLY_SHOW 0

#define SAVE_OUTPUT 1
#define INIT_BY_VIDEO 0
using namespace cv;
using namespace std;


void readConfig(char* configFileName, char* imgFilePath, Rect &box);
/*  Description: read the tracking information from file "config.txt"
    Arguments:	
	-configFileName: config file name
	-ImgFilePath:    Path of the storing image sequences
	-box:            [x y width height] intial tracking position
	History: Created by Kaihua Zhang on 15/8/2012
*/
void readImageSequenceFiles(char* ImgFilePath,vector <string> &imgNames);
/*  Description: search the image names in the image sequences 
    Arguments:
	-ImgFilePath: path of the image sequence
	-imgNames:  vector that stores image name
	History: Created by Kaihua Zhang on 15/8/2012
*/

#if FANG_FIX > 0
#if ONLY_SHOW > 0

int main(void)
{
	bool bFrame = false;
	cv::Mat frame;
	// cv::VideoCapture cap("D:/Wildlife.wmv");
	cv::VideoCapture cap("D:/Work/TMP_DATA/CT_C++_sq_v1/FCT_Test/CompressiveTracking/data/IMG_2301_s.avi");

	if (!cap.isOpened()) {
		std::cout << "Cannot open the video file on C++ API" << std::endl;
		return -1;
	}

	while(true)
	{
		bFrame = cap.read(frame);

		if (!bFrame)
		{
			std::cout << "Null frame or End of video" << std::endl;
			return -1;
		}
	
		cv::imshow("Video Frame", frame);
		cv::waitKey(27);
	}
	return 0;
}
#else
int main(void)
{
	char imgFilePath[100];
	char  conf[100];
	// strcpy(conf, "./config.txt");
	strcpy(conf, "./config_1.txt");
	char tmpDirPath[MAX_PATH + 1];
	int lFrameCnt = 0;
	Rect box; // [x y width height] tracking position
	vector <string> imgNames;

	readConfig(conf, imgFilePath, box);
	readImageSequenceFiles(imgFilePath, imgNames);

	// CT framework
	CompressiveTracker ct;

	Mat frame;
	Mat grayImg;
	double eTime = 0;
	char acText[MAX_PATH] = "";
#if SAVE_OUTPUT > 0
	// for video write
	Size tInputSize;
	int lFourcc = 0;
	int lFps = 0;
	VideoWriter outputVideo; // Open the output
	// for video write
#endif

	// cv::VideoCapture cap("D:/Work/TMP_DATA/CT_C++_sq_v1/FCT_Test/CompressiveTracking/data/IMG_2301_s.avi");
	// cv::VideoCapture cap("./data/IMG_2301_s.avi");
	cv::VideoCapture cap("./data/test.avi");
	// cv::VideoCapture cap("./data/test_1.avi");
	bool bFrame = false;
	bool bDetected = false;
	float fRadioMax = 0;

	if (!cap.isOpened()) {
		std::cout << "Cannot open the video file on C++ API" << std::endl;
		return -1;
	}

#if SAVE_OUTPUT > 0
	// Acquire input size
	tInputSize = Size((int)cap.get(CAP_PROP_FRAME_WIDTH), (int)cap.get(CAP_PROP_FRAME_HEIGHT));
	lFourcc = static_cast<int>(cap.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
	lFps = cap.get(CAP_PROP_FPS);
	outputVideo.open("./Out.avi", -1, lFps, tInputSize, true);

	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write: " << endl;
		return -1;
	}
#endif

#if INIT_BY_VIDEO > 0
	// get frame
	bFrame = cap.read(frame);
	if (!bFrame)
	{
		std::cout << "Null frame or End of video" << std::endl;
		return -1;
	}
#else
	// frame = imread("./data/Sample_1.jpg", CV_LOAD_IMAGE_COLOR);
	frame = imread("./data/Sample_2.jpg", CV_LOAD_IMAGE_COLOR);
#endif
	cvtColor(frame, grayImg, CV_RGB2GRAY);
	ct.init(grayImg, box);	// init FCT with first frame
	char strFrame[MAX_PATH];

	while(true)
	{
		// get frame
		bFrame = cap.read(frame);
		if (!bFrame)
		{
			std::cout << "Null frame or End of video" << std::endl;
			break;
		}
		// frame = imread("./data/Sample_2.jpg", CV_LOAD_IMAGE_COLOR);
		eTime = (double)getTickCount();
		cvtColor(frame, grayImg, CV_RGB2GRAY);

		// ct.processFrame(grayImg, box);// Process frame
		fRadioMax = ct.processFrameTest(grayImg, box);// Process frame
		// fRadioMax = ct.processFrameTest1(grayImg, box);// Process frame
		eTime = ((double)getTickCount() - eTime) / getTickFrequency();

		sprintf(acText, "Time Cost: %f", eTime);

		if (fRadioMax > -40)
		{
			rectangle(frame, box, Scalar(0, 200, 0), 2);// Draw rectangle
			putText(frame, "MATCH", cvPoint(0, 90), FONT_HERSHEY_COMPLEX, 1, CV_RGB(0, 200, 0), 2, LINE_AA);
		}
		else
		{
			rectangle(frame, box, Scalar(200, 0, 0), 2);// Draw rectangle
			putText(frame, "MISS", cvPoint(0, 90), FONT_HERSHEY_COMPLEX, 1, CV_RGB(200, 0, 0), 2, LINE_AA);
		}

		sprintf(strFrame, "#%d RatioMax: %f", lFrameCnt, fRadioMax);

		putText(frame, strFrame, cvPoint(0, 20), 1, 1, CV_RGB(25, 200, 25), 2, LINE_AA);

		putText(frame, acText,
			Point(20, 50),
			FONT_HERSHEY_COMPLEX, 1, // font face and scale
			Scalar(0, 0, 255),
			2, LINE_AA); // line thickness and type

#if SAVE_OUTPUT > 0
		outputVideo.write(frame); //save
#endif
		imshow("CT", frame);// Display
		
		int c = waitKey(10);
		if ((char)c == 27) { break; } // escape

		lFrameCnt++;
	}
	std::cout << "Finish ???" << std::endl;
	return 0;
}
#endif
#else
int main(int argc, char * argv[])
{
	char imgFilePath[100];
    char  conf[100];
	strcpy(conf,"./config.txt");

	char tmpDirPath[MAX_PATH+1];
	
	Rect box; // [x y width height] tracking position

	vector <string> imgNames;
    
	readConfig(conf,imgFilePath,box);
	readImageSequenceFiles(imgFilePath,imgNames);

	// CT framework
	CompressiveTracker ct;

	Mat frame;
	Mat grayImg;

	sprintf(tmpDirPath, "%s/", imgFilePath);
	imgNames[0].insert(0,tmpDirPath);
	frame = imread(imgNames[0]);
    cvtColor(frame, grayImg, CV_RGB2GRAY);    
	ct.init(grayImg, box);    

	char strFrame[20];

    FILE* resultStream;
	resultStream = fopen("TrackingResults.txt", "w");
	fprintf (resultStream,"%i %i %i %i\n",(int)box.x,(int)box.y,(int)box.width,(int)box.height);

	for(int i = 1; i < imgNames.size()-1; i ++)
	{
		
		sprintf(tmpDirPath, "%s/", imgFilePath);
        imgNames[i].insert(0,tmpDirPath);
        		
		frame = imread(imgNames[i]);// get frame
		cvtColor(frame, grayImg, CV_RGB2GRAY);
		
		ct.processFrame(grayImg, box);// Process frame
		
		rectangle(frame, box, Scalar(200,0,0),2);// Draw rectangle

		fprintf (resultStream,"%i %i %i %i\n",(int)box.x,(int)box.y,(int)box.width,(int)box.height);

		sprintf(strFrame, "#%d ",i) ;

		putText(frame,strFrame,cvPoint(0,20),2,1,CV_RGB(25,200,25));
		
		imshow("CT", frame);// Display
		waitKey(1);		
	}
	fclose(resultStream);

	return 0;
}
#endif

void readConfig(char* configFileName, char* imgFilePath, Rect &box)	
{
	int x;
	int y;
	int w;
	int h;

	fstream f;
	char cstring[1000];
	int readS=0;

	f.open(configFileName, fstream::in);

	char param1[200]; strcpy(param1,"");
	char param2[200]; strcpy(param2,"");
	char param3[200]; strcpy(param3,"");

	f.getline(cstring, sizeof(cstring));
	readS=sscanf (cstring, "%s %s %s", param1,param2, param3);

	strcpy(imgFilePath,param3);

	f.getline(cstring, sizeof(cstring)); 
	f.getline(cstring, sizeof(cstring)); 
	f.getline(cstring, sizeof(cstring));

	readS=sscanf (cstring, "%s %s %i %i %i %i", param1,param2, &x, &y, &w, &h);

	box = Rect(x, y, w, h);
	
}

void readImageSequenceFiles(char* imgFilePath,vector <string> &imgNames)
{	

	//imgNames.clear();

	//char tmpDirSpec[MAX_PATH+1];
	//sprintf (tmpDirSpec, "%s/*", imgFilePath);

	//WIN32_FIND_DATA f;
	//HANDLE h = FindFirstFile(tmpDirSpec , &f);
	//if(h != INVALID_HANDLE_VALUE)
	//{
	//	FindNextFile(h, &f);	//read ..
	//	FindNextFile(h, &f);	//read .
	//	do
	//	{
	//		imgNames.push_back(f.cFileName);
	//	} while(FindNextFile(h, &f));

	//}
	//FindClose(h);	
}