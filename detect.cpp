#include <ros/ros.h>
#include <std_msgs/Int32.h>
#include <sstream>

#include <cv.h>
#include <highgui.h>
#include <ml.h>

#include <iostream>
#include <math.h>
#include <string.h>
#include <time.h>

#include "LabelOCR.h"
#include "DetectLabel.h"


using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
	FILE *fp ;
	fp = fopen("test.txt","w");

	int count1 =0;
	int count2 =0;
	int count = 0;
	int count_tess1 = 0;
	int count_tess2 = 0;
	int count_tess1_total = 0;
	int count_tess2_total = 0;
	ros::init(argc, argv, "deTect"); //ros init
	ros::NodeHandle pnh;
	ros::Publisher value = pnh.advertise<std_msgs::Int32>("/keyop/value", 1000); // publish name = /keyop/value 
	ros::Rate loop_rate(100);
	std_msgs::Int32 msg;

    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return (-1);

    Mat normalImage, modImage, cropImage1, labelImage1;
    Mat cropImage2, labelImage2, binImage;
    vector<Point> contour;
    vector<vector<Point> > contours;
    Rect label1ROI;

    string text1, text2;

    DetectLabel detectLabels;
    LabelOCR labelOcr;
//    CvSVM svmClassifier;
    Ptr<ml::SVM> svmClassifier = ml::SVM::create();

    vector<Mat> possible_labels, label_1, label_2;
    vector<string> labelText1, labelText2;
    detectLabels.showBasicImages = true;
    detectLabels.showAllImages = true;

	namedWindow("normal",WINDOW_NORMAL);

	
	// SVM learning algorithm

	clock_t begin_time = clock();

	// Read file storage.
	FileStorage fs;
	fs.open("/home/gyeom/Detect/openCV_Tesseract_test/ml/SVM.xml", FileStorage::READ);
	Mat SVM_TrainingData;
	Mat SVM_Classes;
	fs["TrainingData"] >> SVM_TrainingData;
	fs["classes"] >> SVM_Classes;
	//Set SVM params
	svmClassifier->setType(ml::SVM::C_SVC);
	svmClassifier->setKernel(ml::SVM::LINEAR);
	svmClassifier->setGamma(1);
	svmClassifier->setDegree(0);
	svmClassifier->setCoef0(0);
	svmClassifier->setC(1);
	svmClassifier->setNu(0);
	svmClassifier->setP(0);
	svmClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 1000,  0.01 ) );
	svmClassifier->train( SVM_TrainingData,ml::ROW_SAMPLE, SVM_Classes);
#ifdef LEARNING	
	//Train SVM 
	svmClassifier.train( SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params);
	//svmClassifier.train_auto( SVM_TrainingData, SVM_Classes, Mat(), Mat(), SVM_params, 10);
#endif

	float timer = ( clock () - begin_time ) /  CLOCKS_PER_SEC;
	cout << "Time: " << timer << endl;

	while(ros::ok()){
		msg.data = 0;

		//std::stringstream ss;
		cap >> normalImage; // get a new frame from camera
		imshow("normal", normalImage);

		possible_labels.clear();
		label_1.clear();
		label_2.clear();


		// segmentation
		detectLabels.segment(normalImage,possible_labels);

		int posLabels = possible_labels.size();
		if (posLabels > 0){
			//For each possible label, classify with svm if it's a label or no
			for(int i=0; i< posLabels; i++)
				{
				if (!possible_labels[i].empty() ){
					Mat gray;
					cvtColor(possible_labels[i], gray, COLOR_RGB2GRAY);
					Mat p = gray.reshape(1, 1);
					p.convertTo(p, CV_32FC1); // CV_32FC1
			//	#ifdef LEARNING
					int response = (int)svmClassifier->predict(p);
					cout << "Class: " << response << endl;
					if(response==1)
						label_1.push_back(possible_labels[i]);
					if(response==2)
						label_2.push_back(possible_labels[i]);
					if(response==0)
						count= count+1;
			//	#endif
					}
			}
		}
		if ( label_1.size() > 0) {
			labelText1 = labelOcr.runRecognition(label_1,1);
			//publish 
			//ss << labelText1;
			if(labelText1[0] == "GYEOM"){
			msg.data = 1;
			count_tess1_total = count_tess1_total+1;
			count_tess1 = count_tess1+1;
			}
			else count_tess1_total = count_tess1_total+1;
			
			count = count +1;
			count1 = count1 +1;
		}
		if ( label_2.size() > 0) {
			labelText2 = labelOcr.runRecognition(label_2,2);
			if(labelText2[0]=="SEOULTECH")
			{
			msg.data = 2;
			count_tess2 = count_tess2 +1;
			count_tess2_total = count_tess2_total +1;
			}
			else count_tess2_total = count_tess2_total +1;

			count2 = count2 +1;
			count = count +1;
		}
		if(waitKey(30) >= 0) break;

		value.publish(msg);
	    ros::spinOnce();
    	loop_rate.sleep();
	}

	fprintf(fp, "count =%d\n", count);
	fprintf(fp, "count1=%d\n", count1);
	fprintf(fp, "count2=%d\n", count2);
	fprintf(fp, "-------------tesseract_ocr\n");
	fprintf(fp, "count_tess1_total=%d\n", count_tess1_total);
	fprintf(fp, "count_tess1=%d\n", count_tess1);
	fprintf(fp, "count_tess2_total=%d\n", count_tess2_total);
	fprintf(fp, "count_tess2=%d\n", count_tess2);
	fclose(fp);

	return (0);
}
