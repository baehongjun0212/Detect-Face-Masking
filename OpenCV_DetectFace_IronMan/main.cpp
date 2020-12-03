#include <opencv2/imgproc.hpp>		// Gaussian Blur
#include <opencv2/core.hpp>			// �⺻ OpenCV ����ü�� cv::Mat Scalar�� ����ϱ� ���� ����
#include <opencv2/videoio.hpp>		// ���� �Է� ��� ���� ��� ���� ����
#include <opencv2/highgui.hpp>		// OpenCV ������ â���� ��� ���� ����
#include <opencv2/features2d.hpp>	// Ư¡�� ���⿡ �ʿ��� ��� ����
#include <opencv2/objdetect.hpp>	// ������Ʈ�� �ν��ϱ� ���� ��� ���� ����
#include <stdio.h>					// ǥ�� �����

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;// ǥ�� ���ӽ����̽� ���
using namespace cv; // OpenCV�� ��� �Լ��� Ŭ������ cv namespace�� �ִ�.

const string WindowName = "20155127_��ȫ��";	// ī�޶� ������ â Ÿ��Ʋ
class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector	// ��ķ�� �ѱ� ���� �ʿ��� Ŭ����
{
public:
	CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector) : IDetector(), Detector(detector)
	{
		// �ǽð����� detector ���¸� üũ�ϱ� ����, ���� Exception�� �����ϱ� ����
		CV_Assert(detector);
	}

	void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects) CV_OVERRIDE
	{
		// �Էµ� �̹������� ����� �ٸ� ������Ʈ�� ����. ����� ������Ʈ�� ���簢������ ���ϵ�.
		// Params1 : �����ϰ��� �ϴ� ���� �̹����� �ǹ��մϴ�.
		// Params2 : ����� �̹����� ä������.
		// Params3 : ���� ũ�⸦ ���̴� ������ �����ϴ� �Ű�����.
		// Params4 : ���� �������� ũ�⿡�� minNeighbors Ƚ�� �̻� ����� object�� valid�ϰ� ������ ��.
		// Params5 : cvHaarDetectObjects �Լ������� ���� cascade �Ű�����(old cascade ���ÿ��� �ǹ̸� ������ �Ķ����)
		// Params6 : �ּ� ������ ��ü ũ��.
		// Params7 : �ִ� ������ ��ü ũ��.
		Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
	}

	virtual ~CascadeDetectorAdapter() CV_OVERRIDE	// ���� �Լ� ����
	{}

private:
	CascadeDetectorAdapter();					// private �Լ� ����
	cv::Ptr<cv::CascadeClassifier> Detector;	// Ptr<cv::CascadeClassifier>Ÿ���� Detector�� privateƯ������ ���� ����
};

// ���� �ڵ� ����
int main(int, char**)
{
	namedWindow(WindowName);		// ������ â Ÿ��Ʋ����

	VideoCapture VideoStream(0);	// ������ ��µ� ��Ʈ�� ����

	// �� �νĿ� �ʿ��� �⺻ �ʱ�ȭ �� �غ� ���� �ҽ��ڵ� 
	if (!VideoStream.isOpened())	// VideoStream�� ��� �������� üũ�Ѵ�.
	{
		printf("Error: Cannot open video stream from camera\n");	// VideoStream�� ��� �Ұ����ϸ� ���� �޽��� ���
		return 1;													// ������ ����
	}

	std::string cascadeFrontalfilename = samples::findFile("../data/lbpcascade_frontalface.xml");		
	// �� ���⿡ �ʿ��� �� ������ �ҷ��ͼ� cascadeFrontalfilename�� ����

	cv::Ptr<cv::CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);	
	// cascadeFrontalfilename�� ��� �ִ� ������ �̿��Ͽ� <cv::CascadeClassifier>Ÿ���� cascade���� ����

	cv::Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);	
	// Ptr<DetectionBasedTracker::IDetector> Ÿ���� MainDetector ����. �ᱹ MainDetector�� �����ϱ� ���� .xml������ �ҷ��Դ�.

	if (cascade->empty())	// cascade empty�Լ� load. ���������� cascade�� load�Ǿ����� �Ǵ��ϱ� ����.
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());	// cascade�� �������� ������ ���� �޽���
		return 2;											// �ν�				// ������ ����
	}

	cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);	// cascade�� .xml������ �ҷ��ͼ� makePtr ����ü �������� �����Ѵ�.
	cv::Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);	// Ptr<DetectionBasedTracker::IDetector> Ÿ���� TrackingDetector����
	if (cascade->empty())	// cascade �������� �Ǵ�
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());	// ���� �޽���
		return 2;															// ������ ����
	}

	DetectionBasedTracker::Parameters params;
	DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

	if (!Detector.run())	// Detector ���� ������ ��� ���ǹ� ������ ����.
	{
		printf("Error: Detector initialization failed\n");	// ���� �޽���
		return 2;											// ������ ����
	}

	// �� ���� ���� �� ����� �󱼿� �̹��� ���� �ҽ� �ڵ� 
	Mat ReferenceFrame;	// ��ķ �̹��� ���� ����
	Mat GrayFrame;		// gray�� ��ȯ�� ��ķ �̹��� ���� ����
	vector<Rect> Faces;	// �� ������ ��� vector

	// do while�� ����
	do
	{
		VideoStream >> ReferenceFrame;							// ķ �̹����� ReferenceFrame�� �ִ´�.
		cvtColor(ReferenceFrame, GrayFrame, COLOR_BGR2GRAY);	// ReferenceFrame �̹����� gray������ ����
		Detector.process(GrayFrame);							// gray������ ����� �̹����� process, �� �ν�
		Detector.getObjects(Faces);								// getObjects�� ���� Faces vector�� ����� �̹��� ����

		for (size_t i = 0; i < Faces.size(); i++)				// �νĵ� �� ������ ���� ����
		{
			// =========================================================
			// �νĵ� ���� �߾��� �˾Ƴ��� ����
			Point center;		// �߾��� ��ġ�� �����ϱ� ���� center ���� ����
			Rect r = Faces[i];	// Faces ������ ����ִ� �νĵ� ���� ���������� r�� ����

			//int radius;
			center.x = cvRound((r.x + r.width*0.5));	// �νĵ� ���� x�� ����
			center.y = cvRound((r.y + r.height*0.5));	// �νĵ� ���� y�� ����

			// �νĵ� �󱼿� ������ ���� ����
			float gain = 1.3f;										// ȭ�鿡 �ߴ� �̹����� ũ�� ������ ��ȭ ��Ű�� �����̴�.
			Mat floating_img_temp = imread("../data/iron_man.png");	// ���̾�� ���� �ҷ�����
			Mat floating_img = imread("");							// ���̾�� ���� ũ�� ���濡 �ʿ�(���� ī�޶� ������ Ŀ�����ϰ� �־����� �۾������ϱ� �����̴�)
			Mat mask_temp = imread("../data/iron_man.png", 0);		// ���̾�� ������ mask�� ����� ����
			Mat mask = imread("");									// ���̾�� ���� mask�� ũ�� ���濡 �ʿ�(���� ī�޶� ������ Ŀ�����ϰ� �־����� �۾������ϱ� �����̴�)
			resize(floating_img_temp, floating_img, Size((int)(Faces[i].width*gain), (int)(Faces[i].height*gain)), 0, 0, INTER_CUBIC);	// ���� ī�޶� ��������� ���� ũ�� ����
			resize(mask_temp, mask, Size((int)(Faces[i].width*gain), (int)(Faces[i].height*gain)), 0, 0, INTER_CUBIC);					// mask�� ũ�⸦ ���������� �����ϱ� ����
			try
			{
				int col = (int)(Faces[i].width*gain);	// ���̾�� ����� ����ؾ� �ϹǷ� �νĵ� �� ������ col ���
				int row = (int)(Faces[i].height*gain);	// ���̾�� ����� ����ؾ� �ϹǷ� �νĵ� �� ������ row ���
				Mat imageROI = ReferenceFrame(Rect(center.x - (col / 2), center.y - (row / 2), col, row));	// �̹����� matrix ������ �ִ´�.

				floating_img.copyTo(imageROI, mask);	// �̹����� mask�� ����
			}
			catch (Exception e)
			{
				printf("%s", e.what());	// ���� ���� try/catch��
			}
			// =========================================================
		}
		//waitKey(200);
		imshow(WindowName, ReferenceFrame);	// �̹����� ȭ�鿡 ����.
											//imshow(WindowName, test_mozaik(Faces[0]));
	} while (waitKey(30) < 0);	// ������ 30ms

	Detector.stop();	// ���� ����

	return 0;	// ���� ����
	
	Mat img;                // ������ ���� ����� ���� ��� ����
	VideoCapture cap(0);    // ķ���κ��� ������ �޾ƿ´� (Ȥ�� 1��)

	int count = 0;
	char savefile[200];        // �̹��� ���� �̸��� 200�� �̳��� �����ϱ� ���� char ���� ����

	while (1) {
		cap >> img;
		imshow("image", img);      // ���� ���    

		resize(img, img, Size(100, 100), 0, 0, INTER_CUBIC);  // �������� ������ ���� ũ�⸦ downsizing�ؼ� �����Ѵ�

		sprintf(savefile, "image%d.jpg", count++);
		imwrite(savefile, img);        // img�� ���Ϸ� �����Ѵ�.

		if (waitKey(100) == 27)  break; // esc�� ������ �����Ѵ�
	}
	return 0;
}