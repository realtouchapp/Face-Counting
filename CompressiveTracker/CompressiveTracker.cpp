
#include "CompressiveTracker.h"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;

/** \brief 0: none else: log time */
#define LOG_INIT_PERFORMANCE 0

/** \brief 0: none else: log time */
#define LOG_TRACK_PERFORMANCE 0

/** \brief 0: none else: use */
#define USE_OPENMP 2

/** \brief 0: none else: use */
#define USE_OPENMP_debug 1

/** \brief 0: none else: use */
#define USE_POINTER_FEATURE_CAL 1

#if USE_OPENMP > 0
#include <omp.h>
#endif

//------------------------------------------------
CompressiveTracker::CompressiveTracker(void)
{
	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
	featureNum = 50;	// number of all weaker classifiers, i.e,feature pool
	rOuterPositive = 4;	// radical scope of positive samples
	rSearchWindow = 25; // size of search window
	muPositive = vector<float>(featureNum, 0.0f);
	muNegative = vector<float>(featureNum, 0.0f);
	sigmaPositive = vector<float>(featureNum, 1.0f);
	sigmaNegative = vector<float>(featureNum, 1.0f);
	learnRate = 0.85f;	// Learning rate parameter
}

CompressiveTracker::~CompressiveTracker(void)
{
}


void CompressiveTracker::HaarFeature(Rect& _objectBox, int _numFeature)
/*Description: compute Haar features
  Arguments:
  -_objectBox: [x y width height] object rectangle
  -_numFeature: total number of features.The default is 50.
*/
{
	features = vector<vector<Rect> >(_numFeature, vector<Rect>());
	featuresWeight = vector<vector<float> >(_numFeature, vector<float>());
	
	int numRect;
	Rect rectTemp;
	float weightTemp;
      
	for (int i=0; i<_numFeature; i++)
	{
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
	
		for (int j=0; j<numRect; j++)
		{
			rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
			rectTemp.width = cvCeil(rng.uniform(0.0, (double)(_objectBox.width - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(0.0, (double)(_objectBox.height - rectTemp.y - 2)));
			features[i].push_back(rectTemp);

			weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
			featuresWeight[i].push_back(weightTemp);
           
		}
	}
}


void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox)
/* Description: compute the coordinate of positive and negative sample image templates
   Arguments:
   -_image:        processing frame
   -_objectBox:    recent object position 
   -_rInner:       inner sampling radius
   -_rOuter:       Outer sampling radius
   -_maxSampleNum: maximal number of sampled images
   -_sampleBox:    Storing the rectangle coordinates of the sampled images.
*/
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _rInner*_rInner;
	float outradsq = _rOuter*_rOuter;

  	
	int dist;

	int minrow = max(0,(int)_objectBox.y-(int)_rInner);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_rInner);
	int mincol = max(0,(int)_objectBox.x-(int)_rInner);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_rInner);
    
	
	
	int i = 0;

	float prob = ((float)(_maxSampleNum))/(maxrow-minrow+1)/(maxcol-mincol+1);

	int r;
	int c;
    
    _sampleBox.clear();//important
    Rect rec(0,0,0,0);

	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ ){
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( rng.uniform(0.,1.)<prob && dist < inradsq && dist >= outradsq ){

                rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;
				
                _sampleBox.push_back(rec);				
				
				i++;
			}
		}
	
		_sampleBox.resize(i);
		
}

void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox)
/* Description: Compute the coordinate of samples when detecting the object.*/
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _srw*_srw;	
	int dist;
	int minrow = max(0,(int)_objectBox.y-(int)_srw);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_srw);
	int mincol = max(0,(int)_objectBox.x-(int)_srw);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_srw);

	int i = 0;

	int r;
	int c;

	Rect rec(0,0,0,0);
    _sampleBox.clear();//important

	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ ){
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( dist < inradsq ){

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;

				_sampleBox.push_back(rec);				

				i++;
			}
		}
	
		_sampleBox.resize(i);

}
// Compute the features of samples
void CompressiveTracker::getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue)
{
	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	

#if USE_OPENMP > 1

	#pragma omp parallel for
	for (int i = 0; i < featureNum; i++)
	{
		// #pragma omp parallel for
		for (int j = 0; j < sampleBoxSize; j++)
		{
			float tempValueMP = 0.0f;
			int lSize = (int)features[i].size();
			int lBox_X = 0, lBox_Y = 0;
			lBox_X = _sampleBox[j].x;
			lBox_Y = _sampleBox[j].y;

			for (int k = 0; k < lSize; k++)
			{
				int xMin = 0, xMax = 0, yMin = 0, yMax = 0;

				xMin = lBox_X + features[i][k].x;
				xMax = xMin + features[i][k].width;
				yMin = lBox_Y + features[i][k].y;
				yMax = yMin + features[i][k].height;
#if USE_POINTER_FEATURE_CAL > 0
				unsigned char *pucPtr = NULL;
				float *pfPtrMin = NULL;
				float *pfPtrMax = NULL;

				pucPtr = _imageIntegral.data;
				pfPtrMin = (float *)(pucPtr + _imageIntegral.step.p[0] * yMin);
				pfPtrMax = (float *)(pucPtr + _imageIntegral.step.p[0] * yMax);

				tempValueMP += featuresWeight[i][k] * (pfPtrMin[xMin] + pfPtrMax[xMax] - pfPtrMin[xMax] - pfPtrMax[xMin]);
#else
				tempValueMP += featuresWeight[i][k] *
					(_imageIntegral.at<float>(yMin, xMin) +
						_imageIntegral.at<float>(yMax, xMax) -
						_imageIntegral.at<float>(yMin, xMax) -
						_imageIntegral.at<float>(yMax, xMin));
#endif
			}

			_sampleFeatureValue.at<float>(i, j) = tempValueMP;
		}
	}
#else
	float tempValue = 0;
	for (int i=0; i<featureNum; i++)
	{
		for (int j=0; j<sampleBoxSize; j++)
		{
			tempValue = 0.0f;

			for (int k=0; k<features[i].size(); k++)
			{
				int xMin = 0, xMax = 0, yMin = 0, yMax = 0;

				/*xMin = _sampleBox[j].x + features[i][k].x;
				xMax = _sampleBox[j].x + features[i][k].x + features[i][k].width;
				yMin = _sampleBox[j].y + features[i][k].y;
				yMax = _sampleBox[j].y + features[i][k].y + features[i][k].height;*/

				xMin = _sampleBox[j].x + features[i][k].x;
				xMax = xMin + features[i][k].width;
				yMin = _sampleBox[j].y + features[i][k].y;
				yMax = yMin + features[i][k].height;
				tempValue += featuresWeight[i][k] * 
					(_imageIntegral.at<float>(yMin, xMin) +
					_imageIntegral.at<float>(yMax, xMax) -
					_imageIntegral.at<float>(yMin, xMax) -
					_imageIntegral.at<float>(yMax, xMin));
			}

			_sampleFeatureValue.at<float>(i,j) = tempValue;
		}
	}
#endif
}

// Update the mean and variance of the gaussian classifier
void CompressiveTracker::classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate)
{
	Scalar muTemp;
	Scalar sigmaTemp;
    
	for (int i=0; i<featureNum; i++)
	{
		meanStdDev(_sampleFeatureValue.row(i), muTemp, sigmaTemp);
	   
		_sigma[i] = (float)sqrt( _learnRate*_sigma[i]*_sigma[i]	+ (1.0f-_learnRate)*sigmaTemp.val[0]*sigmaTemp.val[0] 
		+ _learnRate*(1.0f-_learnRate)*(_mu[i]-muTemp.val[0])*(_mu[i]-muTemp.val[0]));	// equation 6 in paper

		_mu[i] = (float)(_mu[i]*_learnRate + (1.0f-_learnRate)*muTemp.val[0]);	// equation 6 in paper
	}
}

// Compute the ratio classifier 
void CompressiveTracker::radioClassifier(
	vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg,
	vector<float>& _sigmaNeg, Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex)
{
	double sumRadio = 0, pPos = 0, pNeg = 0;
	double aeValueTmp[10] = {};
	_radioMax = -FLT_MAX;
	_radioMaxIndex = 0;
	int sampleBoxNum = _sampleFeatureValue.cols;

#if USE_OPENMP > 0
	int lRankCnt = 0;;
	vector<double> vSumRadio(sampleBoxNum, 0);
	
	#pragma omp parallel for
	for (int j=0; j<sampleBoxNum; j++)
	{
		double eSumRadioTmp = 0;
		eSumRadioTmp = 0.0f;

		for (int i = 0; i<featureNum; i++)
		{
			double ePosTmp = 0, eNegTmp = 0;
			ePosTmp = (float)(exp((_sampleFeatureValue.at<float>(i, j) - _muPos[i])*(_sampleFeatureValue.at<float>(i, j) - _muPos[i]) / -(2.0f*_sigmaPos[i] * _sigmaPos[i] + 1e-30)) / (_sigmaPos[i] + 1e-30));
			eNegTmp = (float)(exp((_sampleFeatureValue.at<float>(i, j) - _muNeg[i])*(_sampleFeatureValue.at<float>(i, j) - _muNeg[i]) / -(2.0f*_sigmaNeg[i] * _sigmaNeg[i] + 1e-30)) / (_sigmaNeg[i] + 1e-30));
			eSumRadioTmp += (float)(log(ePosTmp + 1e-30) - log(eNegTmp + 1e-30));	// equation 4
		}
		vSumRadio[j] = eSumRadioTmp;
	}

	for (int lRankCnt = 0; lRankCnt < sampleBoxNum; lRankCnt++)
	{
		if (_radioMax < vSumRadio[lRankCnt])
		{
			_radioMax = (float)vSumRadio[lRankCnt];
			_radioMaxIndex = lRankCnt;
		}
	}
	
#else
	for (int j = 0; j<sampleBoxNum; j++)
	{
		sumRadio = 0.0f;

		for (int i = 0; i<featureNum; i++)
		{
			pPos = (float)(exp((_sampleFeatureValue.at<float>(i, j) - _muPos[i])*(_sampleFeatureValue.at<float>(i, j) - _muPos[i]) / -(2.0f*_sigmaPos[i] * _sigmaPos[i] + 1e-30)) / (_sigmaPos[i] + 1e-30));
			pNeg = (float)(exp((_sampleFeatureValue.at<float>(i, j) - _muNeg[i])*(_sampleFeatureValue.at<float>(i, j) - _muNeg[i]) / -(2.0f*_sigmaNeg[i] * _sigmaNeg[i] + 1e-30)) / (_sigmaNeg[i] + 1e-30));
			sumRadio += (float)(log(pPos + 1e-30) - log(pNeg + 1e-30));	// equation 4
		}

		if (_radioMax < sumRadio)
		{
			_radioMax = (float)sumRadio;
			_radioMaxIndex = j;
		}
	}
#endif
}

// Compute the ratio classifier 
void CompressiveTracker::radioClassifier_1(
	vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg,
	vector<float>& _sigmaNeg, Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex)
{
	double sumRadio;
	_radioMax = -FLT_MAX;
	_radioMaxIndex = 0;
	double pPos;
	double pNeg;
	int sampleBoxNum = _sampleFeatureValue.cols;

	for (int j = 0; j<sampleBoxNum; j++)
	{
		sumRadio = 0.0f;
		for (int i = 0; i<featureNum; i++)
		{
			pPos = exp((_sampleFeatureValue.at<float>(i, j) - _muPos[i])*(_sampleFeatureValue.at<float>(i, j) - _muPos[i]) / -(2.0f*_sigmaPos[i] * _sigmaPos[i] + 1e-30)) / (_sigmaPos[i] + 1e-30);
			pNeg = exp((_sampleFeatureValue.at<float>(i, j) - _muNeg[i])*(_sampleFeatureValue.at<float>(i, j) - _muNeg[i]) / -(2.0f*_sigmaNeg[i] * _sigmaNeg[i] + 1e-30)) / (_sigmaNeg[i] + 1e-30);
			sumRadio += log(pPos + 1e-30) - log(pNeg + 1e-30);	// equation 4
		}
		if (_radioMax < sumRadio)
		{
			_radioMax = (float)sumRadio;
			_radioMaxIndex = j;
		}
	}
}

void CompressiveTracker::init_Test(Mat& _frame, Rect& _objectBox)
{
	// compute feature template
	HaarFeature(_objectBox, featureNum);

	// compute sample templates
	sampleRect(_frame, _objectBox, (float)rOuterPositive, 0, 1000000, samplePositiveBox);

	integral(_frame, imageIntegral, CV_32F);
	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
}

void CompressiveTracker::init(Mat& _frame, Rect& _objectBox)
{
#if LOG_INIT_PERFORMANCE > 0
	double aeTime[10];
	double aeTimeGap[10];
#endif
	
#if LOG_INIT_PERFORMANCE > 0
	aeTime[0] = (double)getTickCount();
#endif
	// compute feature template
	HaarFeature(_objectBox, featureNum);

#if LOG_INIT_PERFORMANCE > 0
	aeTime[1] = (double)getTickCount();
#endif

	// compute sample templates
	sampleRect(_frame, _objectBox, (float)rOuterPositive, 0, 1000000, samplePositiveBox);
	sampleRect(_frame, _objectBox, (float)(rSearchWindow*1.5), (float)(rOuterPositive+4.0), 100, sampleNegativeBox);

#if LOG_INIT_PERFORMANCE > 0
	aeTime[2] = (double)getTickCount();
#endif

	// get integral image
	integral(_frame, imageIntegral, CV_32F);

#if LOG_INIT_PERFORMANCE > 0
	aeTime[3] = (double)getTickCount();
#endif

	// calculate feature by integral image & feature template
	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);

	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);


#if LOG_INIT_PERFORMANCE > 0
	aeTime[4] = (double)getTickCount();
#endif

	// update classifier
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);

#if LOG_INIT_PERFORMANCE > 0
	aeTime[5] = (double)getTickCount();
#endif

#if LOG_INIT_PERFORMANCE > 0
	aeTimeGap[0] = ((double)aeTime[1] - aeTime[0]) / getTickFrequency() * 1000;
	aeTimeGap[1] = ((double)aeTime[2] - aeTime[1]) / getTickFrequency() * 1000;
	aeTimeGap[2] = ((double)aeTime[3] - aeTime[2]) / getTickFrequency() * 1000;
	aeTimeGap[3] = ((double)aeTime[4] - aeTime[3]) / getTickFrequency() * 1000;
	aeTimeGap[4] = ((double)aeTime[5] - aeTime[4]) / getTickFrequency() * 1000;
	aeTimeGap[5] = ((double)aeTime[5] - aeTime[0]) / getTickFrequency() * 1000;

	FILE *fFptr = NULL;
	fFptr = fopen("TrackInitTimeLog.txt", "a+");
	fprintf(fFptr, "--------------------\n");
	fprintf(fFptr, "HaarFeature: %.2f ms %.2f %%\n", aeTimeGap[0], aeTimeGap[0] / aeTimeGap[5] * 100);
	fprintf(fFptr, "sampleRect All: %.2f ms %.2f %%\n", aeTimeGap[1], aeTimeGap[1] / aeTimeGap[5] * 100);
	fprintf(fFptr, "integral: %.2f ms %.2f %%\n", aeTimeGap[2], aeTimeGap[2] / aeTimeGap[5] * 100);
	fprintf(fFptr, "getFeatureValue All: %.2f ms %.2f %%\n", aeTimeGap[3], aeTimeGap[3] / aeTimeGap[5] * 100);
	fprintf(fFptr, "classifierUpdate All: %.2f ms %.2f %%\n", aeTimeGap[4], aeTimeGap[4] / aeTimeGap[5] * 100);
	fprintf(fFptr, "--------------------\n");
#endif
}
void CompressiveTracker::processFrame(Mat& _frame, Rect& _objectBox)
{
	// predict
	sampleRect(_frame, _objectBox, (float)rSearchWindow,detectBox);
	integral(_frame, imageIntegral, CV_32F);
	getFeatureValue(imageIntegral, detectBox, detectFeatureValue);
	int radioMaxIndex;
	float radioMax;
	radioClassifier(muPositive, sigmaPositive, muNegative, sigmaNegative, detectFeatureValue, radioMax, radioMaxIndex);
	_objectBox = detectBox[radioMaxIndex];

	printf("radioMax=%f\n", radioMax);

	// update
	sampleRect(_frame, _objectBox, (float)rOuterPositive, 0.0, 1000000, samplePositiveBox);
	sampleRect(_frame, _objectBox, (float)(rSearchWindow*1.5), (float)(rOuterPositive+4.0), 100, sampleNegativeBox);
	
	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
}

float CompressiveTracker::ProcessFrameWithScore(Mat& _frame, Rect& _objectBox)
{
	int radioMaxIndex = 0;
	float radioMax = 0;
#if LOG_TRACK_PERFORMANCE > 0
	double aeTime[10];
	double aeTimeGap[10];
#endif
#if LOG_TRACK_PERFORMANCE > 0
	aeTime[0] = (double)getTickCount();
#endif
	// compute feature template
	// HaarFeature(_objectBox, featureNum);

	/*------------------------predict----------------------*/
 	sampleRect(_frame, _objectBox, (float)rSearchWindow, detectBox);
#if LOG_TRACK_PERFORMANCE > 0
	aeTime[1] = (double)getTickCount();
#endif
	integral(_frame, imageIntegral, CV_32F);
#if LOG_TRACK_PERFORMANCE > 0
	aeTime[2] = (double)getTickCount();
#endif

	getFeatureValue(imageIntegral, detectBox, detectFeatureValue);

#if LOG_TRACK_PERFORMANCE > 0
	aeTime[3] = (double)getTickCount();
#endif
	radioClassifier(muPositive, sigmaPositive, muNegative, sigmaNegative, detectFeatureValue, radioMax, radioMaxIndex);
	_objectBox = detectBox[radioMaxIndex];
#if LOG_TRACK_PERFORMANCE > 0
	aeTime[4] = (double)getTickCount();
#endif
	/*------------------------predict----------------------*/

	/*------------------------update----------------------*/
	sampleRect(_frame, _objectBox, (float)rOuterPositive, 0.0, 1000000, samplePositiveBox);
	sampleRect(_frame, _objectBox, (float)(rSearchWindow*1.5), (float)(rOuterPositive + 4.0), 100, sampleNegativeBox);
#if LOG_TRACK_PERFORMANCE > 0
	aeTime[5] = (double)getTickCount();
#endif

	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);

	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);

#if LOG_TRACK_PERFORMANCE > 0
	aeTime[6] = (double)getTickCount();
#endif
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
#if LOG_TRACK_PERFORMANCE > 0
	aeTime[7] = (double)getTickCount();
#endif
	/*------------------------update----------------------*/
#if LOG_TRACK_PERFORMANCE > 0
	aeTimeGap[0] = ((double)aeTime[1] - aeTime[0]) / getTickFrequency() * 1000;
	aeTimeGap[1] = ((double)aeTime[2] - aeTime[1]) / getTickFrequency() * 1000;
	aeTimeGap[2] = ((double)aeTime[3] - aeTime[2]) / getTickFrequency() * 1000;
	aeTimeGap[3] = ((double)aeTime[4] - aeTime[3]) / getTickFrequency() * 1000;
	aeTimeGap[4] = ((double)aeTime[5] - aeTime[4]) / getTickFrequency() * 1000;
	aeTimeGap[5] = ((double)aeTime[6] - aeTime[5]) / getTickFrequency() * 1000;
	aeTimeGap[6] = ((double)aeTime[7] - aeTime[6]) / getTickFrequency() * 1000;
	aeTimeGap[7] = ((double)aeTime[7] - aeTime[0]) / getTickFrequency() * 1000;

	FILE *fFptr = NULL;
	fFptr = fopen("TrackProcessTimeLog.txt", "a+");
	fprintf(fFptr, "--------------------\n");
	fprintf(fFptr, "predict >> sampleRect & HaarFeature: %.2f ms %.2f %%\n", aeTimeGap[0], aeTimeGap[0] / aeTimeGap[7] * 100);
	fprintf(fFptr, "integral: %.2f ms %.2f %%\n", aeTimeGap[1], aeTimeGap[1] / aeTimeGap[7] * 100);
	fprintf(fFptr, "predict >> getFeatureValue: %.2f ms %.2f %%\n", aeTimeGap[2], aeTimeGap[2] / aeTimeGap[7] * 100);
	fprintf(fFptr, "predict >>  radioClassifier: %.2f ms %.2f %%\n", aeTimeGap[3], aeTimeGap[3] / aeTimeGap[7] * 100);
	fprintf(fFptr, "update >> sampleRect All: %.2f ms %.2f %%\n", aeTimeGap[4], aeTimeGap[4] / aeTimeGap[7] * 100);
	fprintf(fFptr, "update >> getFeatureValue All: %.2f ms %.2f %%\n", aeTimeGap[5], aeTimeGap[5] / aeTimeGap[7] * 100);
	fprintf(fFptr, "update >> classifierUpdate All: %.2f ms %.2f %%\n", aeTimeGap[6], aeTimeGap[6] / aeTimeGap[7] * 100);
	fprintf(fFptr, "--------------------\n");
#endif
	return radioMax;
}

float CompressiveTracker::processFrameTest(Mat& _frame, Rect& _objectBox)
{
	// predict

	// Compute the coordinate of samples when detecting the object.
	sampleRect(_frame, _objectBox, (float)rSearchWindow, detectBox);

	// cal integral image
	integral(_frame, imageIntegral, CV_32F);

	// get Feature Value of each detected Box from integral image
	getFeatureValue(imageIntegral, detectBox, detectFeatureValue);
	
	
	int radioMaxIndex;
	float radioMax;
	radioClassifier_1(muPositive, sigmaPositive, muNegative, sigmaNegative, detectFeatureValue, radioMax, radioMaxIndex);
	
	// _objectBox = detectBox[radioMaxIndex];	// 更新位置 >> 似乎不需要 因為都在原地

	// update
	/*getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);*/
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);

	return radioMax;
}

void CompressiveTracker::CT_SetParameter(
	int a_lMinNumRect, int a_lMaxNumRect, int a_lFeatureNum,
	int a_lRadicalScopePos, int a_lSearchWindowSize,float a_flearnRate)
{
	featureMinNumRect = (a_lMinNumRect > 0) ? a_lMinNumRect : featureMinNumRect;
	featureMaxNumRect = (a_lMaxNumRect > 0) ? a_lMaxNumRect : featureMaxNumRect;
	featureNum = (a_lFeatureNum > 0) ? a_lFeatureNum : featureNum;
	rOuterPositive = (a_lRadicalScopePos > 0) ? a_lRadicalScopePos : rOuterPositive;
	rSearchWindow = (a_lSearchWindowSize > 0) ? a_lSearchWindowSize : rSearchWindow;
	learnRate = (a_flearnRate > 0) ? a_flearnRate : learnRate;
}