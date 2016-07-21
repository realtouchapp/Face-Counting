
#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>
#include <vector>
#include <stdlib.h> 
#include "CompressiveTracker.h"
#include <omp.h>

#ifndef LINE_AA
/** \brief opencv putText word thickness  */
#define LINE_AA 16
#endif

#ifndef LINE_8
/** \brief opencv putText word thickness  */
#define LINE_8 8
#endif

/** \brief 0: release mode else: debug mode   Note: useless if you build opencv by yourself !!!! */
#define RUN_IN_DEBUG_MODE 0

/** \brief 0: read from file else: read from webcam */
#define READ_FROM_CAM 0

/** \brief 0: run sample code else: do detect & track */
#define USE_FANG_TRACK 1

/** \brief max string size */
#define MAX_PATH 260

/** \brief 0: do not show else: show detected eyes */
#define SHOW_EYE 0

/** \brief 0: do not show else: show detected faces */
#define SHOW_FACE 1

/** \brief 0: do not show else: show all tracking bounding box */
#define SHOW_TRACK 1

/** \brief 0: do not detect eyes else: detect eyes in detected faces */
#define DETECT_EYE 1

/** \brief 0: do not save else: save result frame as video */
#define SAVE_OUTPUT 0

/** \brief counter threshold to decide if face tracking should be delete */
#define MISS_CNT_TH 10

/** \brief counter threshold to decide if face focus on camera */
#define LOOK_CNT_TH 2

/** \brief 0: do not track else: using tracking */
#define DO_TRACK 1

/** \brief 0: do nothing else: detect memory leak in visual studio in windows */
#define WIN_VC_MEM_LEAK_DETECT 1

/** \brief Down size scale */
#define DOWN_SIZE_SCALE 0

/** \brief 0: do not modify 1:modify CT parameter */
#define SET_CT_PARAMETER 1

/** \brief 0: Full image 1: use ROI */
#define SET_FACE_DETECT_ROI 0

#if WIN_VC_MEM_LEAK_DETECT > 0
#define _CRTDBG_MAP_ALLOC 
#include <crtdbg.h>
#endif

/*
double t = (double)getTickCount();
// do something ...
t = ((double)getTickCount() - t)/getTickFrequency();
*/
using namespace std;
using namespace cv;

/** Function Headers */
/** \brief only detect with classifier */
void detectAndDisplay( Mat frame );

#if USE_FANG_TRACK > 0
/** \brief detect with classifier & tracking with FCT*/
void Detect_TrackAndDisplay(Mat frame);

/** \brief matching detected objects to tracking objects*/
int VerifyNewObj(Rect Detect, vector<Rect>& Track);

/** \brief alloc the global vector of detect ROI for OpenCV BUG in Debug mode*/
void InitVector();

/** \brief free the global vector of detect ROI for OpenCV BUG in Debug mode*/
void FreeVector();
#endif


/** Global variables */
String face_cascade_name = "haarcascade_profileface.xml";	// 這模型包含側臉
// String face_cascade_name = "haarcascade_frontalface_alt.xml";	// 這模型只有正臉
String eyes_cascade_name = "haarcascade_eye.xml";
String window_name = "Face Detection & Tracking";

/** \brief face detect classifier */
CascadeClassifier CFace_Cascade;

/** \brief eye detect classifier */
CascadeClassifier CEyes_Cascade;

/** \brief frame counter */
int g_lFrameCnt = 0;
#if USE_FANG_TRACK > 0
/** \brief vector of CT tracking framework class */
vector<CompressiveTracker> vCFCT;

/** \brief vector of CT tracking face result */
vector<Rect> FacesList;

/** \brief miss detect counter of tracking objects */
vector<int> g_VlMissCnt;

/** \brief face fcous counter of tracking objects */
vector<int> g_VlLookCntForEnable;

/** \brief face fcous flag of tracking objects */
vector<bool> g_VbLookEnable;

/** \brief face detect flag of tracking objects */
vector<bool> g_VbDetect;

/** \brief local face detect result ROI */
vector<Rect> g_faces;

/** \brief local eye detect result ROI */
vector<Rect> g_eyes;

/** \brief global face detect result ROI */
vector<Rect>* g_pvEyesDetect = NULL;

/** \brief global eye detect result ROI */
vector<Rect>* g_pvFaceDetect = NULL;

/** \brief face counter */
int g_lFaceCnt = 0;

/** \brief focus counter */
int g_lLookCnt = 0;

#if SAVE_OUTPUT > 0
Size tInputSize;
int lFourcc = 0;
double eFps = 0;
VideoWriter outputVideo; // Open the output
#endif

#endif

#define MP_TEST_NUM 10000

void TestMP()
{
	float afPos[MP_TEST_NUM] = { 0 }, afNeg[MP_TEST_NUM] = { 0 };
	float afValueTmp[MP_TEST_NUM] = { 0 };
	float afSigmaPos[MP_TEST_NUM] = { 0 };
	float afSigmaNeg[MP_TEST_NUM] = { 0 };
	float sumRadio = 0, sumRadioNoMP = 0;
	int j = 0;
	double t0 = 0;
	double t1 = 0;

	for (int i = 0; i < MP_TEST_NUM; i++)
	{
		afValueTmp[i] = 1;
	}

	j = 2;

	t0 = (double)getTickCount();
	for (int i = 0; i < MP_TEST_NUM; i++)
	{
		float pPosTmp = 0, pNegTmp = 0, *pfValueTmpMP = NULL, fTmpValue = 0;
		double eTmpValuePos = 0, eTmpValueNeg = 0;

		fTmpValue = afValueTmp[j];
		eTmpValuePos = (fTmpValue - afPos[i])*(fTmpValue - afPos[i]);
		eTmpValueNeg = (fTmpValue - afNeg[i])*(fTmpValue - afNeg[i]);
		pPosTmp = (float)(exp(eTmpValuePos / -(2.0f*afSigmaPos[i] * afSigmaPos[i] + 1e-30)) / (afSigmaPos[i] + 1e-30));
		pNegTmp = (float)(exp(eTmpValueNeg / -(2.0f*afSigmaNeg[i] * afSigmaNeg[i] + 1e-30)) / (afSigmaNeg[i] + 1e-30));
		sumRadioNoMP += (float)(log(pPosTmp + 1e-30) - log(pNegTmp + 1e-30));	// equation 4
	}

	t0 = ((double)getTickCount() - t0) / getTickFrequency();
	t1 = (double)getTickCount();

	#pragma omp parallel for reduction( +:sumRadio)
	for (int i = 0; i < MP_TEST_NUM; i++)
	{
		float pPosTmp = 0, pNegTmp = 0, *pfValueTmpMP = NULL, fTmpValue = 0;
		double eTmpValuePos = 0, eTmpValueNeg = 0;

		fTmpValue = afValueTmp[j];
		eTmpValuePos = (fTmpValue - afPos[i])*(fTmpValue - afPos[i]);
		eTmpValueNeg = (fTmpValue - afNeg[i])*(fTmpValue - afNeg[i]);
		pPosTmp = (float)(exp(eTmpValuePos / -(2.0f*afSigmaPos[i] * afSigmaPos[i] + 1e-30)) / (afSigmaPos[i] + 1e-30));
		pNegTmp = (float)(exp(eTmpValueNeg / -(2.0f*afSigmaNeg[i] * afSigmaNeg[i] + 1e-30)) / (afSigmaNeg[i] + 1e-30));
		sumRadio += (float)(log(pPosTmp + 1e-30) - log(pNegTmp + 1e-30));	// equation 4
	}

	t1 = ((double)getTickCount() - t1) / getTickFrequency();
	printf("%.3f\n", t0 - t1);
}

void TestMP_1()
{
	float afValueTmp[MP_TEST_NUM] = { 0 };
	float sum = 0, sumNoMP = 0;
	float fDiff = 0;
	int j = 0;
	double eTDiff = 0;
	double t0 = 0;
	double t1 = 0;
	double tt = 0;

	sum = 49995000;

	for (int i = 0; i < MP_TEST_NUM; i++)
	{
		tt += i;
	}

	for (int i = 0; i < MP_TEST_NUM; i++)
	{
		afValueTmp[i] = (float)i;
	}

	t0 = (double)getTickCount();
	for (int i = 0; i < MP_TEST_NUM; i++)
	{
		// for (int k = 0; k < MP_TEST_NUM; k++);	// just for delay
		
		if (sumNoMP-49992896 == 0)
		{
			sumNoMP = sumNoMP;
		}
		sumNoMP += afValueTmp[i];
	}

	t0 = ((double)getTickCount() - t0) / getTickFrequency();
	t1 = (double)getTickCount();

	#pragma omp parallel for reduction( +:sum)
	for (int i = 0; i < MP_TEST_NUM; i++)
	{
		for (int k = 0; k < MP_TEST_NUM; k++);	// just for delay

		sum += afValueTmp[i];
	}

	t1 = ((double)getTickCount() - t1) / getTickFrequency();
	eTDiff = t0 - t1;	// time improve
	fDiff = sum - sumNoMP;	// check result
	printf("%.3f\n", eTDiff);
}

#if USE_FANG_TRACK > 0
void InitVector()
{
	g_pvFaceDetect = new std::vector<cv::Rect>; //never call delete whatever you do
	g_pvEyesDetect = new std::vector<cv::Rect>; //never call delete whatever you do
}

void FreeVector()
{
#if RUN_IN_DEBUG_MODE > 0	// 這段只能在debug mode 跑 >> 不然編譯不會過
	int size = 0;
	Rect *ptFreePtr = NULL;
	double *peFreePtr = NULL;
	unsigned char *pucFreePtr = NULL;
	unsigned char *pucFreePtr_1 = NULL;
	std::_Container_proxy *pPtr = NULL;
	void *pPtr1 = NULL;

	size = sizeof(Rect);
	g_pvFaceDetect->clear();
	g_pvEyesDetect->clear();
	
	// 這裡很奇怪 他是function pointer 會mem leak >> 用leak大小的指標指到那邊去free
	pPtr = g_pvEyesDetect->_Myproxy();
	pPtr1 = pPtr;
	peFreePtr = (double*)pPtr1;
	free(peFreePtr);
	pPtr = g_pvFaceDetect->_Myproxy();
	pPtr1 = pPtr;
	peFreePtr = (double*)pPtr1;
	free(peFreePtr);

	// 不知為何不能直接delete vector 所以用同長度的型態指標去幫忙free
	ptFreePtr = (Rect*)(g_pvFaceDetect);
	free(ptFreePtr);
	ptFreePtr = (Rect*)(g_pvEyesDetect);
	free(ptFreePtr);

	ptFreePtr = NULL;
	peFreePtr = NULL;
#else
	delete g_pvEyesDetect;
	delete g_pvFaceDetect;
#endif
}
#endif

/** @function main */
int main( void )
{
#if WIN_VC_MEM_LEAK_DETECT > 0
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif	

#if USE_FANG_TRACK > 0
	InitVector();
#endif	

    VideoCapture capture;
    Mat CFrame;

	face_cascade_name = "./AdaboostModel/" + face_cascade_name;
	eyes_cascade_name = "./AdaboostModel/" + eyes_cascade_name;

    //-- 1. Load the cascades
    if( !CFace_Cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !CEyes_Cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    //-- 2. Read the video stream form cam or file
#if READ_FROM_CAM > 0
    capture.open( -1 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
#else
	capture.open("./FaceTest.avi");
	if (!capture.isOpened()) {
		std::cout << "Cannot open the video file on C++ API" << std::endl;
		return -1;
	}
#endif

#if SAVE_OUTPUT > 0
	// Acquire input size
	tInputSize = Size((int)capture.get(CAP_PROP_FRAME_WIDTH), (int)capture.get(CAP_PROP_FRAME_HEIGHT));
#if DOWN_SIZE_SCALE > 0
	tInputSize.width = tInputSize.width / DOWN_SIZE_SCALE;
	tInputSize.height = tInputSize.height / DOWN_SIZE_SCALE;
#endif
	lFourcc = static_cast<int>(capture.get(CAP_PROP_FOURCC));     // Get Codec Type- Int form
	eFps = capture.get(CAP_PROP_FPS);
	outputVideo.open("./Out.avi", -1, eFps, tInputSize, true);

	if (!outputVideo.isOpened())
	{
		cout << "Could not open the output video for write: " << endl;
		return -1;
	}
#endif
    while ( capture.read(CFrame) )
    {
        if(CFrame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
		g_lFrameCnt++;

        //-- 3. Apply the classifier to the frame (you could choose tracking or not)
#if USE_FANG_TRACK > 0
		Detect_TrackAndDisplay(CFrame);
#else
		detectAndDisplay(CFrame);
#endif
        int c = waitKey(1);
        if( (char)c == 27 ) { break; } // escape
    }
#if USE_FANG_TRACK > 0
	FreeVector();
#endif
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    Mat frame_gray;
	static vector<Rect> facesResult;
	static vector<Rect> eyes;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    CFace_Cascade.detectMultiScale( frame_gray, facesResult, 1.1, 3, 0|CASCADE_SCALE_IMAGE, Size(80, 80) );

    for ( size_t i = 0; i < facesResult.size(); i++ )
    {
        Point center( facesResult[i].x + facesResult[i].width/2, facesResult[i].y + facesResult[i].height/2 );
        ellipse( frame, center, Size( facesResult[i].width/2, facesResult[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( facesResult[i] );
        

        //-- In each face, detect eyes
        CEyes_Cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        for ( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( facesResult[i].x + eyes[j].x + eyes[j].width/2, facesResult[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
		eyes.clear();
    }

	facesResult.clear();

    //-- Show what you got
    imshow( window_name, frame );
}

#if USE_FANG_TRACK > 0
void Detect_TrackAndDisplay(Mat frame)
{	
	vector<bool> FacesEye;	// 記錄有沒有偵測到眼睛
	Rect NewFace;
	CompressiveTracker CFCT_Current;
	Mat frame_gray;
	char strFrame[MAX_PATH];
	float fRadioMax = 0;
	bool bNewObj = false;
	int lMinIndex = 0;
	size_t tSizeTmp;
	double eTime = 0, eTimeDetect = 0, eTimeTrack = 0;

	Size tFaceSize(90, 90);
	Size tEyeSize(30, 30);

	
#if DOWN_SIZE_SCALE > 0
	cv::resize(frame, frame, cv::Size(frame.cols / DOWN_SIZE_SCALE, frame.rows / DOWN_SIZE_SCALE));
	tFaceSize.width = tFaceSize.width / DOWN_SIZE_SCALE;
	tFaceSize.height = tFaceSize.height / DOWN_SIZE_SCALE;
	tEyeSize.width = tEyeSize.width / DOWN_SIZE_SCALE;
	tEyeSize.height = tEyeSize.height / DOWN_SIZE_SCALE;
#endif

#if SET_FACE_DETECT_ROI > 0
	Rect tFaceROI;

	tFaceROI.x = 0;
	tFaceROI.y = 0;
	tFaceROI.width = 640;
	tFaceROI.height = 340;
#endif
	eTime = (double)getTickCount();
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
#if SET_FACE_DETECT_ROI > 0
	Mat frame_grayHist;	// 取ROI一定要有BUFFER
	equalizeHist(frame_gray, frame_grayHist);
	frame_gray = frame_grayHist(tFaceROI);
	// imwrite("./Gray_Image_ROI.jpg", frame_gray);
#else
	equalizeHist(frame_gray, frame_gray);
#endif

	eTimeDetect = (double)getTickCount();
	//-- Detect faces
	CFace_Cascade.detectMultiScale(frame_gray, *g_pvFaceDetect, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, tFaceSize);

	tSizeTmp = g_pvFaceDetect->size();
	FacesEye.assign(g_pvFaceDetect->size(), false);

#if DETECT_EYE > 0
	//-- Detect eyes in each faces
	for (size_t i = 0; i < g_pvFaceDetect->size(); i++)
	{
		Point center((*g_pvFaceDetect)[i].x + (*g_pvFaceDetect)[i].width / 2, (*g_pvFaceDetect)[i].y + (*g_pvFaceDetect)[i].height / 2);
		Mat faceROI = frame_gray((*g_pvFaceDetect)[i]);
		
		//-- In each face, detect eyes
		CEyes_Cascade.detectMultiScale(faceROI, *g_pvEyesDetect, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, tEyeSize);
		
#if SHOW_FACE > 0
		sprintf(strFrame, "(%d %d)", (*g_pvFaceDetect)[i].width, (*g_pvFaceDetect)[i].height);
		putText(frame, strFrame, cvPoint((*g_pvFaceDetect)[i].x, (*g_pvFaceDetect)[i].y), 1, 1, CV_RGB(0, 0, 200), 1, LINE_AA);
#endif
		if (g_pvEyesDetect->size() > 0)
		{
			FacesEye[i] = true;
#if SHOW_FACE > 0
			ellipse(frame, center, Size((*g_pvFaceDetect)[i].width / 2, (*g_pvFaceDetect)[i].height / 2), 0, 0, 360, Scalar(0, 200, 0), 4, 8, 0);
#endif
		}
		else
		{
			FacesEye[i] = false;
#if SHOW_FACE > 0
			ellipse(frame, center, Size((*g_pvFaceDetect)[i].width / 2, (*g_pvFaceDetect)[i].height / 2), 0, 0, 360, Scalar(0, 200, 200), 4, 8, 0);
#endif
		}

		
#if SHOW_EYE > 0
		for (size_t j = 0; j < g_pvEyesDetect->size(); j++)
		{
			Point eye_center((*g_pvFaceDetect)[i].x + (*g_pvEyesDetect)[j].x + (*g_pvEyesDetect)[j].width / 2, (*g_pvFaceDetect)[i].y + (*g_pvEyesDetect)[j].y + (*g_pvEyesDetect)[j].height / 2);
			int radius = cvRound(((*g_pvEyesDetect)[j].width + (*g_pvEyesDetect)[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}
#endif
	}
#endif
	eTimeDetect = ((double)getTickCount() - eTimeDetect) / getTickFrequency();

	if (g_lFrameCnt == 34)
	{
		g_lFrameCnt = g_lFrameCnt;
	}

#if DO_TRACK > 0
	printf("Frame: %d FaceNum: %d TrackNum: %d\n", g_lFrameCnt, (*g_pvFaceDetect).size(), FacesList.size());
	eTimeTrack = (double)getTickCount();
	// tracking current faces list
	for (size_t i = 0; i < FacesList.size(); i++)
	{
		fRadioMax = vCFCT[i].ProcessFrameWithScore(frame_gray, FacesList[i]);// Process frame
	}
	eTimeTrack = ((double)getTickCount() - eTimeTrack) / getTickFrequency();
	

	// update FacesList
	g_VbDetect.clear();
	tSizeTmp = FacesList.size();
	g_VbDetect.assign(tSizeTmp, false);
	for (size_t i = 0; i < (*g_pvFaceDetect).size(); i++)
	{
		NewFace = (*g_pvFaceDetect)[i];
		lMinIndex = VerifyNewObj(NewFace, FacesList);
		if (lMinIndex < 0)
		{	// 新物件就init追蹤

#if SET_CT_PARAMETER > 0
			CFCT_Current.CT_SetParameter(-1, -1, 20, -1, 15, -1);
#endif
			CFCT_Current.init(frame_gray, NewFace);	// init FCT with new object


			vCFCT.push_back(CFCT_Current);
			FacesList.push_back(NewFace);
			g_VlMissCnt.push_back(0);
			g_VbDetect.push_back(true);
			if (FacesEye[i])
			{
				g_VlLookCntForEnable.push_back(1);
			}
			else
			{
				g_VlLookCntForEnable.push_back(0);
			}
			g_VbLookEnable.push_back(false);
			g_lFaceCnt++;
		}
		else
		{	// 有match
			g_VbDetect[lMinIndex] = true;
			if (FacesEye[i] && !g_VbLookEnable[lMinIndex])
			{
				g_VlLookCntForEnable[lMinIndex]++;
				if (g_VlLookCntForEnable[lMinIndex] > LOOK_CNT_TH)
				{
					g_VbLookEnable[lMinIndex] = true;
					g_lLookCnt++;
				}
				
			}
		}
	}

	// 計算miss cnt
	for (size_t i = 0; i < g_VbDetect.size(); i++)
	{
		if (!g_VbDetect[i])
		{	//有追蹤 但沒有偵測到臉
			g_VlMissCnt[i]++;
		}
		else
		{	// 有偵測到 >> miss cnt 歸零
			g_VlMissCnt[i]= 0;
		}
	}

	// 將miss cnt 過高的移除追蹤
	for (size_t i = 0; i < g_VbDetect.size(); i++)
	{
		if (g_VlMissCnt[i] > MISS_CNT_TH)
		{	// miss cnt 過高 >> 移除
			vCFCT.erase(vCFCT.begin() + i);
			FacesList.erase(FacesList.begin() + i);
			g_VbDetect.erase(g_VbDetect.begin() + i);
			g_VlMissCnt.erase(g_VlMissCnt.begin() + i);
			g_VlLookCntForEnable.erase(g_VlLookCntForEnable.begin() + i);
			g_VbLookEnable.erase(g_VbLookEnable.begin() + i);
			i--;
		}
	}
#endif

#if SHOW_TRACK > 0
	// show bounding box of tracking current face
	for (size_t i = 0; i < FacesList.size(); i++)
	{
		rectangle(frame,
			Point(FacesList[i].x, FacesList[i].y),
			Point(FacesList[i].x+ FacesList[i].width, FacesList[i].y+ FacesList[i].height),
			Scalar(0, 0, 255), 1, 8);
	}
#endif

	// log 的黑底
	rectangle(frame, Point(0, 0), Point(300, 50), Scalar(0, 0, 0), -1, 8);

	sprintf(strFrame, "Face Cnt: %d Focus Cnt:%d", g_lFaceCnt, g_lLookCnt);
	putText(frame, strFrame, cvPoint(0, 10), 1, 1, CV_RGB(200, 120, 0), 1, LINE_8);

	sprintf(strFrame, "Frame: %d ", g_lFrameCnt);
	putText(frame, strFrame, cvPoint(0, 25), 1, 1, CV_RGB(200, 200, 0), 1, LINE_8);

	eTime = ((double)getTickCount() - eTime) / getTickFrequency();

	sprintf(strFrame, "FPS: %.3f ", 1/eTime);
	putText(frame, strFrame, cvPoint(100, 25), 1, 1, CV_RGB(200, 200, 0), 1, LINE_8);

	sprintf(strFrame, "detect: %.1f%% track: %.1f%%", eTimeDetect / eTime*100, eTimeTrack / eTime*100);
	putText(frame, strFrame, cvPoint(0, 40), 1, 1, CV_RGB(200, 200, 0), 1, LINE_8);

	printf("Face Cnt: %d Focus Cnt:%d FPS: %.3f\n", g_lFaceCnt, g_lLookCnt, 1 / eTime);
	printf("detect: %.1f%% track: %.1f%% \n", eTimeDetect / eTime * 100, eTimeTrack / eTime * 100);
	
#if SAVE_OUTPUT > 0
	outputVideo.write(frame); //save
#endif
	//-- Show what you got
	imshow(window_name, frame);

	printf("this frame end\n");
}

double GetDistance(Point a_P1, Point a_P2)
{
	double eDistance = 0;

	eDistance = sqrt((a_P1.x - a_P2.x)*(a_P1.x - a_P2.x) + (a_P1.y - a_P2.y)*(a_P1.y - a_P2.y));

	return eDistance;
}

int VerifyNewObj(Rect Detect, vector<Rect>& Track)
{
	Point DetectCenter;
	Point TrackCenter;
	double eDistance = 0;
	double eDistanceMin = 99999;
	int lMinIndex = -1;

	DetectCenter.x = Detect.x + Detect.width / 2;
	DetectCenter.y = Detect.y + Detect.height / 2;

	// find min diatance in  FacesList
	for (size_t i = 0; i < Track.size(); i++)
	{
		TrackCenter.x = Track[i].x + Track[i].width / 2;
		TrackCenter.y = Track[i].y + Track[i].height / 2;

		eDistance =  GetDistance(DetectCenter, TrackCenter);
		if (eDistance < eDistanceMin)
		{
			eDistanceMin = eDistance;
			lMinIndex = i;
		}
	}

	if (eDistanceMin < Detect.width && lMinIndex >= 0)
	{	// match >> update tracking box
		Track[lMinIndex].x = Detect.x;
		Track[lMinIndex].width = Detect.width;
		Track[lMinIndex].y = Detect.y;
		Track[lMinIndex].height = Detect.height;
		return lMinIndex;
	}

	return -1;
}
#endif