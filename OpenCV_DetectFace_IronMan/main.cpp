#include <opencv2/imgproc.hpp>		// Gaussian Blur
#include <opencv2/core.hpp>			// 기본 OpenCV 구조체들 cv::Mat Scalar를 사용하기 위해 선언
#include <opencv2/videoio.hpp>		// 비디오 입력 출력 관련 헤더 파일 선언
#include <opencv2/highgui.hpp>		// OpenCV 윈도우 창관련 헤더 파일 선언
#include <opencv2/features2d.hpp>	// 특징점 추출에 필요한 헤더 파일
#include <opencv2/objdetect.hpp>	// 오브젝트를 인식하기 위한 헤더 파일 선언
#include <stdio.h>					// 표준 입출력

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;// 표준 네임스페이스 사용
using namespace cv; // OpenCV의 모든 함수와 클래스는 cv namespace에 있다.

const string WindowName = "20155127_배홍준";	// 카메라 켜지는 창 타이틀
class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector	// 웹캠을 켜기 위해 필요한 클래스
{
public:
	CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector) : IDetector(), Detector(detector)
	{
		// 실시간으로 detector 상태를 체크하기 위해, 또한 Exception을 검출하기 위해
		CV_Assert(detector);
	}

	void detect(const cv::Mat &Image, std::vector<cv::Rect> &objects) CV_OVERRIDE
	{
		// 입력된 이미지에서 사이즈가 다른 오브젝트를 검출. 검출된 오브젝트는 직사각형으로 리턴됨.
		// Params1 : 검출하고자 하는 원본 이미지를 의미합니다.
		// Params2 : 검출된 이미지가 채워진다.
		// Params3 : 영상 크기를 줄이는 정도를 지정하는 매개변수.
		// Params4 : 여러 스케일의 크기에서 minNeighbors 횟수 이상 검출된 object는 valid하게 검출할 때.
		// Params5 : cvHaarDetectObjects 함수에서와 같은 cascade 매개변수(old cascade 사용시에만 의미를 가지는 파라미터)
		// Params6 : 최소 가능한 개체 크기.
		// Params7 : 최대 가능한 개체 크기.
		Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
	}

	virtual ~CascadeDetectorAdapter() CV_OVERRIDE	// 가상 함수 선언
	{}

private:
	CascadeDetectorAdapter();					// private 함수 선언
	cv::Ptr<cv::CascadeClassifier> Detector;	// Ptr<cv::CascadeClassifier>타입의 Detector를 private특성으로 변수 선언
};

// 메인 코드 시작
int main(int, char**)
{
	namedWindow(WindowName);		// 윈도우 창 타이틀설정

	VideoCapture VideoStream(0);	// 영상이 출력될 스트림 선언

	// 얼굴 인식에 필요한 기본 초기화 및 준비를 위한 소스코드 
	if (!VideoStream.isOpened())	// VideoStream이 사용 가능한지 체크한다.
	{
		printf("Error: Cannot open video stream from camera\n");	// VideoStream이 사용 불가능하면 에러 메시지 출력
		return 1;													// 비정상 종료
	}

	std::string cascadeFrontalfilename = samples::findFile("../data/lbpcascade_frontalface.xml");		
	// 얼굴 검출에 필요한 모델 정보를 불러와서 cascadeFrontalfilename에 저장

	cv::Ptr<cv::CascadeClassifier> cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);	
	// cascadeFrontalfilename에 담겨 있는 정보를 이용하여 <cv::CascadeClassifier>타입의 cascade변수 선언

	cv::Ptr<DetectionBasedTracker::IDetector> MainDetector = makePtr<CascadeDetectorAdapter>(cascade);	
	// Ptr<DetectionBasedTracker::IDetector> 타입의 MainDetector 선언. 결국 MainDetector를 선언하기 위해 .xml파일을 불러왔다.

	if (cascade->empty())	// cascade empty함수 load. 정상적으로 cascade가 load되었는지 판단하기 위해.
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());	// cascade가 정상이지 않으면 에러 메시지
		return 2;											// 인식				// 비정상 종료
	}

	cascade = makePtr<cv::CascadeClassifier>(cascadeFrontalfilename);	// cascade에 .xml정보를 불러와서 makePtr 구조체 형식으로 생성한다.
	cv::Ptr<DetectionBasedTracker::IDetector> TrackingDetector = makePtr<CascadeDetectorAdapter>(cascade);	// Ptr<DetectionBasedTracker::IDetector> 타입의 TrackingDetector선언
	if (cascade->empty())	// cascade 정상인지 판단
	{
		printf("Error: Cannot load %s\n", cascadeFrontalfilename.c_str());	// 에러 메시지
		return 2;															// 비정상 종료
	}

	DetectionBasedTracker::Parameters params;
	DetectionBasedTracker Detector(MainDetector, TrackingDetector, params);

	if (!Detector.run())	// Detector 동작 에러일 경우 조건문 안으로 들어간다.
	{
		printf("Error: Detector initialization failed\n");	// 에러 메시지
		return 2;											// 비정상 종료
	}

	// 얼굴 검출 시작 및 검출된 얼굴에 이미지 띄우기 소스 코드 
	Mat ReferenceFrame;	// 웹캠 이미지 저장 변수
	Mat GrayFrame;		// gray로 변환된 웹캠 이미지 저장 변수
	vector<Rect> Faces;	// 얼굴 정보가 담긴 vector

	// do while문 시작
	do
	{
		VideoStream >> ReferenceFrame;							// 캠 이미지를 ReferenceFrame에 넣는다.
		cvtColor(ReferenceFrame, GrayFrame, COLOR_BGR2GRAY);	// ReferenceFrame 이미지를 gray색으로 변경
		Detector.process(GrayFrame);							// gray색으로 변경된 이미지를 process, 얼굴 인식
		Detector.getObjects(Faces);								// getObjects를 통해 Faces vector에 검출된 이미지 저장

		for (size_t i = 0; i < Faces.size(); i++)				// 인식된 얼굴 갯수에 따라 동작
		{
			// =========================================================
			// 인식된 얼굴의 중앙을 알아내기 위해
			Point center;		// 중앙의 위치를 저장하기 위한 center 변수 선언
			Rect r = Faces[i];	// Faces 변수에 들어있는 인식된 얼굴을 순차적으로 r에 저장

			//int radius;
			center.x = cvRound((r.x + r.width*0.5));	// 인식된 얼굴의 x축 센터
			center.y = cvRound((r.y + r.height*0.5));	// 인식된 얼굴의 y축 센터

			// 인식된 얼굴에 사진을 띄우기 위해
			float gain = 1.3f;										// 화면에 뜨는 이미지의 크기 비율을 변화 시키기 위함이다.
			Mat floating_img_temp = imread("../data/iron_man.png");	// 아이언맨 사진 불러오기
			Mat floating_img = imread("");							// 아이어맨 사진 크기 변경에 필요(얼굴이 카메라에 가까우면 커져야하고 멀어지면 작아져야하기 때문이다)
			Mat mask_temp = imread("../data/iron_man.png", 0);		// 아이언맨 사진의 mask를 씌우기 위해
			Mat mask = imread("");									// 아이언맨 사진 mask의 크기 변경에 필요(얼굴이 카메라에 가까우면 커져야하고 멀어지면 작아져야하기 때문이다)
			resize(floating_img_temp, floating_img, Size((int)(Faces[i].width*gain), (int)(Faces[i].height*gain)), 0, 0, INTER_CUBIC);	// 얼굴이 카메라에 가까워짐에 따라 크기 변경
			resize(mask_temp, mask, Size((int)(Faces[i].width*gain), (int)(Faces[i].height*gain)), 0, 0, INTER_CUBIC);					// mask의 크기를 유동적으로 변경하기 위해
			try
			{
				int col = (int)(Faces[i].width*gain);	// 아이언맨 사이즈를 계산해야 하므로 인식된 얼굴 사이즈 col 계산
				int row = (int)(Faces[i].height*gain);	// 아이언맨 사이즈를 계산해야 하므로 인식된 얼굴 사이즈 row 계산
				Mat imageROI = ReferenceFrame(Rect(center.x - (col / 2), center.y - (row / 2), col, row));	// 이미지를 matrix 변수에 넣는다.

				floating_img.copyTo(imageROI, mask);	// 이미지를 mask에 삽입
			}
			catch (Exception e)
			{
				printf("%s", e.what());	// 에러 뜰경우 try/catch문
			}
			// =========================================================
		}
		//waitKey(200);
		imshow(WindowName, ReferenceFrame);	// 이미지를 화면에 띄운다.
											//imshow(WindowName, test_mozaik(Faces[0]));
	} while (waitKey(30) < 0);	// 딜레이 30ms

	Detector.stop();	// 검출 종료

	return 0;	// 정상 종료
	
	Mat img;                // 동영상 파일 재생을 위한 행렬 선언
	VideoCapture cap(0);    // 캠으로부터 영상을 받아온다 (혹은 1번)

	int count = 0;
	char savefile[200];        // 이미지 파일 이름을 200자 이내로 제한하기 위한 char 변수 선언

	while (1) {
		cap >> img;
		imshow("image", img);      // 영상 출력    

		resize(img, img, Size(100, 100), 0, 0, INTER_CUBIC);  // 사진으로 저장할 때는 크기를 downsizing해서 저장한다

		sprintf(savefile, "image%d.jpg", count++);
		imwrite(savefile, img);        // img를 파일로 저장한다.

		if (waitKey(100) == 27)  break; // esc를 누르면 종료한다
	}
	return 0;
}