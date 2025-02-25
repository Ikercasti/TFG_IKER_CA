
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream> // for writing CSV
#include <unistd.h>
#include <string>
#include "RaspberryCam/Include/OCV_Funcs.hpp"

using namespace cv;
using namespace std;

/* GUI */
bool show_borders = true;
bool make_ROI = true;
int lowThreshold = 0, rati=2, kernel_size=3;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		show_borders = show_borders ? false : true;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		make_ROI = make_ROI ? false : true;
	}
}

static void CannyThreshold(int lowT, void* v)
{
	lowThreshold = lowT;
}

static void CannyKSize(int kS, void* v)
{
	kernel_size = kS;
}
/* GUI */

int main(int argc, char* argv[])
{
	bool vid_cam = true;
	bool img_vid = false;

	Mat frame;
	// No sé si estos números valen para todos los videos
	int FRAME_SIZE_X = 1880;
	int FRAME_SIZE_Y = 1075;
	VideoCapture cap;

	// video original completo 1h
	//vid_cam ? cap.open("/home/ROBOGait/Videos/AthTrack.webm"):cap.open(0);
	// Video Vallehermoso
	//vid_cam ? cap.open("/home/ROBOGait/Videos/LGV_TFG/Vallehermoso/Vuelta_mixta.mp4"):cap.open(0); ///home/ROBOGait/Pictures/AthTrack.webm

	// video 1 vueta empezando CON sol de cara
	//vid_cam ? cap.open("/home/ROBOGait/Videos/vuelta_con_sol.avi"):cap.open(0);
	// video 1 vueta empezando SIN sol de cara
	vid_cam ? cap.open("/home/ROBOGait/Videos/vuelta_sin_sol.avi"):cap.open(0);
	
	double fps = cap.get(CAP_PROP_FPS); //NOT FOR CAMERA
	cout<<"FPS: "<<fps<<endl;
	
	int delay = (fps>=0) ? (1000 / fps) : 10; //ms between frames based on framerate
	delay = vid_cam ? delay : 0;
	cout<<"delay (ms): "<<delay<<endl;
	
	// Se queda esperando un Enter
	cout<<"Presiona ENTER para seguir"<<endl;
	cin.get();
	
	string window_name;
	if (img_vid) {
		frame = imread("/home/ROBOGait/Documents/TFG Lucas GV/TrackApprox/Track2.webp");
		if (frame.empty())
		{
			cout << "Could not open or find the image" << endl;
			cin.get(); //wait for any key press
			return -1;
		}
		window_name = "My image";
	}
	else {
		// if not success, exit program
		if (cap.isOpened() == false)
		{
			cout << "Cannot open the video camera" << endl;
			cin.get(); //wait for any key press
			return -1;
		}
		window_name = "My Video";
	}
	   
	namedWindow(window_name, WindowFlags::WINDOW_NORMAL);
	setMouseCallback(window_name, CallBackFunc, NULL);

	createTrackbar("Min Threshold:", window_name, &lowThreshold, 100, CannyThreshold, (void*)&window_name);
	createTrackbar("SobelKernelSize:", window_name, &kernel_size, 7, CannyKSize);

	// create a window called "Control"
	namedWindow("Control", WindowFlags::WINDOW_AUTOSIZE); 
	//int iLowH = 12;    
	//int iHighH = 102;  
	int iLowH = 30;    
	int iHighH = 75;  
	createTrackbar("LowH", "Control", &iLowH, 179);   //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);
	int iLowS = 0;
	//int iHighS = 72;
	int iHighS = 255;
	createTrackbar("LowS", "Control", &iLowS, 255);   //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);
	//int iLowV = 191;
	int iLowV = 190;
	int iHighV = 255;
	createTrackbar("LowV", "Control", &iLowV, 255);   //Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);

	if(false) {
		// SALVAR EL PRIMER FRAME
		cout<<"Salvar frames"<<endl;
		cap.read(frame); // read a new frame from video 
		cv::Mat cropped_image = frame(cv::Range(FRAME_SIZE_Y-300, FRAME_SIZE_Y),  cv::Range((FRAME_SIZE_X/2)-300,(FRAME_SIZE_X/2)+300));
		cv::imwrite("MyImageBGR.jpg", cropped_image);
		cv::cvtColor(cropped_image, cropped_image, cv::COLOR_BGR2HSV); 
		ParallelHueShift(cropped_image, cropped_image, 45);
		cv::imwrite("MyImageHSV.jpg", cropped_image);

		cout<<"Imagenes guardadas"<<endl;
		// Se queda esperando un Enter
	    cout<<"Presiona ENTER para seguir"<<endl;		
		cin.get();
		destroyAllWindows();
		exit(0);
	}

	Mat openKern, closeKern;
	openKern = getStructuringElement(MORPH_CROSS, Size(3, 3));
	closeKern = getStructuringElement(MORPH_CROSS, Size(5, 5));
	
	Mat perspFrame, frameCpy, morphOut, imgHSV, imgThresh, imgCanny;
	//ROI and laneLogic INIT
	LineProcess::laneLogic laneL;
	//reserve 4 spots to avoid costly memory reallocation operations
	std::vector<cv::Point> maskVerts;
	maskVerts.reserve(4); 
	// Numeros validos para Vallehermoso
/*	if(img_vid){
		laneL.setInitPoint(590);
		maskVerts.emplace_back(0, 720); 
		maskVerts.emplace_back(1280/9, 720/3); //emplace contructs the object directly inside the vector instead of copying
		maskVerts.emplace_back(1280/9*8, 720/3); 
		maskVerts.emplace_back(1280, 720);
	}else{
		laneL.setInitPoint(640);
		maskVerts.emplace_back(0, 720); 
		maskVerts.emplace_back(1280/9, 720/3*2); //emplace contructs the object directly inside the vector instead of copying
		maskVerts.emplace_back(1280/9*8, 720/3*2); 
		maskVerts.emplace_back(1280, 720);
	}*/
	// Numeros validos para Video pista Francia
	laneL.setInitPoint((FRAME_SIZE_X/2)-400);
	maskVerts.emplace_back((FRAME_SIZE_X/2)-400, FRAME_SIZE_Y); 
	maskVerts.emplace_back((FRAME_SIZE_X/2)-250, FRAME_SIZE_Y*2/3); 
	maskVerts.emplace_back((FRAME_SIZE_X/2)+250, FRAME_SIZE_Y*2/3); 
	maskVerts.emplace_back((FRAME_SIZE_X/2)+400, FRAME_SIZE_Y);
	cout << "LOOP" << endl; 

	int cuenta = 1;
	bool first = true;

	std::vector<cv::Point> left_line, left_points;
	std::vector<cv::Point> right_line, right_points; 
	std::vector<cv::Point> middle_line;
	std::vector<cv::Point> points, erased;
	std::vector<int> quitar;
	double distancia;
		
	while (true)
	{
		auto startTime = getTickCount();
		if (!img_vid){
			bool bSuccess = cap.read(frame); // read a new frame from video 

			//Breaking the while loop if the frames cannot be captured
			if (bSuccess == false)
			{
				cout << "Video camera is disconnected" << endl;
			    cout<<"Presiona ENTER para seguir"<<endl;		
				cin.get(); //Wait for any key press
				break; 
			}
		}
		
		//namedWindow("Original", WindowFlags::WINDOW_NORMAL);
		//imshow("Original", frame);

		// saves the original image
		// PARA QUE ??
		frame.copyTo(frameCpy);

		if (show_borders) {

			// To HSV + Shifts hue circle by (shift*2)º
			cv::cvtColor(frame, imgHSV, cv::COLOR_BGR2HSV); 
			ParallelHueShift(imgHSV, imgHSV, 45);           

			//Threshold the image
			//iLowH = 12;  iHighH = 102;  
            //iLowS = 0;   iHighS = 72;
			//iLowV = 191; iHighV = 255;
			cv::inRange(imgHSV, cv::Scalar(iLowH, iLowS, iLowV), cv::Scalar(iHighH, iHighS, iHighV), imgThresh); 
			
			// Realmente no recorta la imagen, solo le aplica una máscara
			if(make_ROI) {
				Apply_Draw_ROI(frameCpy, imgThresh, maskVerts);
			}
			namedWindow("Thresholded Image", WindowFlags::WINDOW_NORMAL);
			imshow("Thresholded Image", imgThresh); 
	
			// CLOSE AND OPEN
			// Por qué OPEN comentado? XQ Las 2 a la vez no tienen sentido	
			// Opening: erosion followed by dilation
			// Closing: dilation followed by Erosion
			//cv::morphologyEx(imgThresh, morphOut, cv::MORPH_CLOSE, closeKern);
			//cv::morphologyEx(morphOut, morphOut, cv::MORPH_OPEN, openKern);
			cv::morphologyEx(imgThresh, morphOut, cv::MORPH_OPEN, openKern);
			namedWindow("Morphed Image", WindowFlags::WINDOW_NORMAL);
			imshow("Morphed Image", morphOut);
				
			// Window = 100, 3 alturas || Window = 75, 4 alturas || Window = 70, 5 alturas
			points = DavidLanePoints(morphOut, FRAME_SIZE_Y, FRAME_SIZE_Y*2/3, 75, 8);
			//Draw_Contour_Points(frameCpy, points);

			//if(first)
			// Buscar DCHA e IZQ en base a (FRAME_SIZE_X/2) 
			// TODO:DEBERIA SER EN BASE A LA MITAD DE LA CALLE
			//recorremos los puntos y asignamos dcha e izq
			for( int i = 0; i < points.size(); i++ ) {
				if(points[i].x < (FRAME_SIZE_X/2) ) {
					left_points.push_back(points[i]);
				}	
				else {
					right_points.push_back(points[i]);
				}			
			}
			if(first) {
				first = false;
				left_line = left_points;
				right_line = right_points;

			} else {
				// Deteccion de pocos puntos en frame
				if(left_points.size() < 3) {
					cout << "Frame " << cuenta << " mala linea izq." << endl;
				}	
				else {

               // Se necesita tener linea completa al inicio 					
					for( int i = 0; i < left_points.size(); i++ ) {
						if((i+1) == left_line.size()) {
							distancia = distancePointToLine(left_line[i-1]
															,left_line[i]
															,left_points[i]);
							if( distancia > 5 ) {
								cout << "Frame " << cuenta << " Dist " << distancia << 
									" quitar altura " <<  left_points[i].y << endl;
					 			quitar.push_back(i);
							}
							else
								cout << "Frame " << cuenta << " Dist " << distancia << 
									" altura BIEN " <<  left_points[i].y << endl;	
						} else {
							distancia = distancePointToLine(left_line[i]
															,left_line[i+1]
															,left_points[i]);
							if(distancia > 5 ) {
								cout << "Frame " << cuenta << " Dist " << distancia << 
									" quitar altura " <<  left_line[i].y << endl;
								quitar.push_back(i);	
							}	
							else
								cout << "Frame " << cuenta << " Dist " << distancia << 
									" altura BIEN " <<  left_line[i].y << endl;	
						}	
						
						// Esta idea no está clara
					   /* for( int j = 0; j < left_line.size(); j++ ) {
							if (left_points[i].y == left_line[j].y) {
							    if((j+1) == left_line.size()) {
								    distancia = distancePointToLine(left_line[j-1]
									        			           ,left_line[j]
																   ,left_points[i]);
									if( distancia > 5 ) {
  								        cout << "Frame " << cuenta << " Dist " << distancia << 
										    " quitar altura " <<  left_line[j].y << endl;
										quitar.push_back(i);
									}
									else
										cout << "Frame " << cuenta << " Dist " << distancia << 
										    " altura BIEN " <<  left_line[j].y << endl;	
								} else {
									distancia = distancePointToLine(left_line[j]
																   ,left_line[j+1]
																   ,left_points[i]);
									if(distancia > 5 ) {
  								        cout << "Frame " << cuenta << " Dist " << distancia << 
										    " quitar altura " <<  left_line[j].y << endl;
										quitar.push_back(i);	
									}	
									else
										cout << "Frame " << cuenta << " Dist " << distancia << 
										    " altura BIEN " <<  left_line[j].y << endl;	
								}	
							}			
						} */
					} 		
				}		

				if(right_points.size() < 3) {
					cout << "Frame " << cuenta << "mala linea dch." << endl;
				}

				/* COMO SABEMOS CON QUE PUNTO COMPARA?
						for( int i = 0; i < left_points.size(); i++ ) {
							distance.push_back(distancePointToLine);
							///
						}	
				*/	

			}
			for( int i = 0; i < quitar.size(); i++ ) {
				erased.push_back(left_points[quitar[i]]);
				left_points.erase(left_points.begin()+quitar[i]);	
			}

			Draw_Contour_Points(frameCpy, left_points);
			Draw_Contour_Points(frameCpy, right_points);
			Draw_Contour_Points(frameCpy, erased, cv::Scalar(150, 50, 0));

		}
		else { 
			try {
				destroyWindow("Thresholded Image");
				destroyWindow("Morphed Image");
				//destroyWindow("Canny Image");
			}
			catch (Exception) { ; }
		}
		//show the frame in the created window
		//namedWindow("Line Image", WindowFlags::WINDOW_NORMAL);
		imshow(window_name, frameCpy);
		
		//processing time it took since the beggining of the frame loop
		int processTime = (getTickCount() - startTime)/ getTickFrequency()*1000; 

        if (waitKey(1) == 27) //obligatory for displaying purposes
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		} 
		cout<<"\u001b[31mProcess Time: "<<processTime<<"(ms) "<<endl;
		cout<<"Time between frames: "<<(getTickCount()-startTime)/ getTickFrequency()*1000<<"(ms) "<<endl;
		cout<<"Actual fps: "<<(float)1/((getTickCount()-startTime)/ getTickFrequency())
		<<"\u001b[0m\x1b[2;A \r";	
		
		left_points.clear();
		right_points.clear();
		quitar.clear();

		// Se queda esperando un Enter
		cout<<"Presiona ENTER para seguir "<< cuenta++ <<endl;
		cin.get();
	}
	destroyAllWindows();
	
	return 0;
}
