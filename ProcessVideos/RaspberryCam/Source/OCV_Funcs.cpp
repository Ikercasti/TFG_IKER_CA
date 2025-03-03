
#include "../Include/OCV_Funcs.hpp"

// default values for some functions on .hpp file

void FullCanny(InputArray img_input, OutputArray img_output, int _lowThreshold, int _ratio, int BlurKSize, int SobelKSize)
{
	// if it's an even number, make it odd
	if (BlurKSize % 2 == 0)
		BlurKSize += 1;
	if (SobelKSize % 2 == 0)
		SobelKSize += 1;

	Mat input = img_input.getMat().clone();
	Mat grayscale, edges;
	cvtColor(input, grayscale, COLOR_BGR2GRAY);
	medianBlur(grayscale, edges, BlurKSize);
	Canny(edges, edges, _lowThreshold, _lowThreshold * _ratio, SobelKSize);
	// Because of OutputArray, we don't need a return statement
	edges.copyTo(img_output);
	// imshow("FullCanny", img_output);
}

void FullCannySingleChannel(InputArray img_input, OutputArray img_output, int _lowThreshold, int _ratio, int BlurKSize, int SobelKSize)
{
	// if it's an even number, make it odd
	if (BlurKSize % 2 == 0)
		BlurKSize += 1;
	if (SobelKSize % 2 == 0)
		SobelKSize += 1;

	Mat input = img_input.getMat().clone();
	Mat edges(input.rows, input.cols, CV_8UC1, Scalar::all(0));
	;
	// if single channel we don't need grayscale
	medianBlur(input, edges, BlurKSize);
	try
	{
		Canny(edges, edges, _lowThreshold, _lowThreshold * _ratio, SobelKSize);
		edges.copyTo(img_output);
		// Because of OutputArray, we don't need a return statement
	}
	catch (Exception &ex)
	{
		cout << "FullCannySingleChannel: " << ex.what() << endl;
		exit(-1);
	}
}

void FullCannyHSVInput(InputArray img_input, OutputArray img_output, int _lowThreshold, int _ratio, int BlurKSize, int SobelKSize)
{
	Mat input = img_input.getMat(), inpBGR;
	try
	{
		cvtColor(input, inpBGR, COLOR_HSV2BGR);
	}
	catch (Exception &ex)
	{
		cout << ex.what() << endl;
		exit(-1);
	}
	FullCanny(inpBGR, img_output, _lowThreshold, _ratio, BlurKSize, SobelKSize);
}

void PaintCanny(InputOutputArray img_input, InputArray canny_mask, const Scalar &color, int thickness)
{
	Mat Color(img_input.getMat().rows, img_input.getMat().cols, CV_8UC3, color);
	Mat dilated_canny;
	if (thickness < 1)
		thickness = 1;
	if (thickness != 1)
	{
		Mat dilateKern = getStructuringElement(MORPH_RECT, Size(thickness, thickness));
		dilate(canny_mask, dilated_canny, dilateKern);
	}
	else
		dilated_canny = canny_mask.getMat().clone();
	Color.copyTo(img_input, canny_mask);
}

void ApplyROI(InputOutputArray img_input, const vector<Point> &maskVerts)
{
	cv::Mat mask(img_input.getMat().rows, img_input.getMat().cols, CV_8UC1, Scalar(0)); //crea imagen en negro
	cv::fillPoly(mask, maskVerts, 255);  //dibuja el trapecio en la imagen mask
	bitwise_and(img_input, mask, img_input);
}

void Apply_NormalROI(InputOutputArray img_input)
{
	vector<cv::Point> maskVerts;
	Mat inp = img_input.getMat();
	// ROI Mask
	// reserve 4 spots to avoid costly memory reallocation operations
	maskVerts.reserve(4);
	maskVerts.emplace_back(0, inp.rows);
	maskVerts.emplace_back(inp.cols / 9, inp.rows / 1.7f); // emplace contructs the object directly inside the vector instead of copying
	maskVerts.emplace_back(inp.cols / 9 * 8, inp.rows / 1.7f);
	maskVerts.emplace_back(inp.cols, inp.rows);
	ApplyROI(inp, maskVerts);
}

/**
 * Pinta la ROI sobre la imagen de color y
 * 		aplica una másrcara con la ROI sobre la imagen B&W
 * InputArray bin_draw: imagen a color a analizar
 * InputArray binImg: imagen B&W a enmascarar
 * const vector<Point>& maskVerts: los 4 puntos que forman la ROI
 */
// img_draw and img_bin must have the same size, maskverts is the contour of the ROI
void Apply_Draw_ROI(InputOutputArray img_draw, InputOutputArray img_bin, const vector<Point> &maskVerts)
{
	Draw_Contour(img_draw.getMat(), maskVerts, Scalar(155, 155, 0));
	ApplyROI(img_bin.getMat(), maskVerts);
}

// img_draw and img_bin must have the same size
void Apply_Draw_NormalROI(InputOutputArray img_draw, InputOutputArray img_bin)
{
	vector<cv::Point> maskVerts;
	Mat inp = img_draw.getMat();
	// ROI Mask
	// reserve 4 spots to avoid costly memory reallocation operations
	maskVerts.reserve(4);
	maskVerts.emplace_back(0, inp.rows);
	maskVerts.emplace_back(inp.cols / 9, inp.rows / 1.7f);
	maskVerts.emplace_back(inp.cols / 9 * 8, inp.rows / 1.7f);
	maskVerts.emplace_back(inp.cols, inp.rows);

	Apply_Draw_ROI(inp, img_bin.getMat(), maskVerts);
}

// void PaintROICanny(InputOutputArray img_input, InputArray canny_mask, int x_start, int y_start, const Scalar& color, int thickness){
// Mat Color(img_input.getMat().rows, img_input.getMat().cols, CV_8UC3, color);
// Mat dilated_canny;
// if(thickness<1) thickness=1;
// if(thickness!=1) {
// Mat dilateKern = getStructuringElement(MORPH_RECT, Size(thickness, thickness));
// dilate(canny_mask, dilated_canny, dilateKern);
//}
// else dilated_canny = canny_mask.getMat().clone();
// Mat littleimg_input = img_input.getMat()(cv::Rect(x_start,y_start,Color.cols, Color.rows));
// Color.copyTo(littleimg_input, canny_mask);
// littleimg_input.copyTo(img_input.getMat()(cv::Rect(x_start,y_start,Color.cols, Color.rows)));
//}

vector<Point2f> Get_Centroids(InputOutputArray inp, const vector<vector<cv::Point>> &contours, int min_area)
{
	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	vector<Point2f> mret;

	for (size_t i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i]);
		// add 1e-5 to avoid division by zero
		if (mu[i].m00 > min_area)
		{
			mc[i] = Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
							static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
			mret.push_back(mc[i]);
		}
	}
	return mret;
}

void Get_Draw_Centroids(InputOutputArray inp, const vector<vector<cv::Point>> &contours, int min_area, int roi_x, int roi_y)
{

	vector<Moments> mu(contours.size());
	vector<Point2f> mc(contours.size());
	vector<cv::Point> shifted_cont;

	for (size_t i = 0; i < contours.size(); i++)
	// for (const vector<cv::Point> single_cont : contours)
	{
		// size before to reduce reallocation
		shifted_cont.reserve(contours[i].size());
		for (size_t j = 0; j < contours[i].size(); j++)
		{
			shifted_cont[j].x = contours[i][j].x + roi_x;
			shifted_cont[j].y = contours[i][j].y + roi_y;
		}
		// contours[i] shifted
		mu[i] = moments(contours[i]);
		// add 1e-5 to avoid division by zero
		if (mu[i].m00 > min_area)
		{
			mc[i] = Point2f(static_cast<float>(mu[i].m10 / (mu[i].m00 + 1e-5)),
							static_cast<float>(mu[i].m01 / (mu[i].m00 + 1e-5)));
			circle(inp, mc[i], 10, cv::Scalar(0, 0, 255), -1);
		}
		shifted_cont.clear();
	}
}

/**
 * Pinta un circulo por cada punto del contorno
 * InputOutputArray inp: imagen sobre la que se pinta
 * vector<cv::Point>& contour
 * cv::Scalar& color: color de los puntos
 */
void Draw_Contour_Points(InputOutputArray inp, const vector<cv::Point> &contour, const cv::Scalar &color)
{
	// for( size_t j = 0; j < contour.size(); j++ ){ //recorremos los puntos de cada contorno
	for (const cv::Point &point : contour)
	{
		circle(inp, point, 10, color, -1);
	}
}

/**
 * Llama a pintar cada uno de los contornos con la funcion que está encima
 */
void Draw_Contours_Points(InputOutputArray inp, const vector<vector<cv::Point>> &contours, const cv::Scalar &color)
{
	// for( size_t i = 0; i < contours.size(); i++ ){ //recorremos los contornos de la imagen
	for (const vector<cv::Point> &contour : contours)
	{
		Draw_Contour_Points(inp, contour, color); // contours[i]
	}
}

vector<cv::Point> Get_One_Side_Contour(const vector<cv::Point> &contour)
{
	vector<cv::Point> one_side_dir1, one_side_dir2;
	cv::Point prev_point = contour[0];
	cv::Point current_point;
	int lowest_Point_Index = 0;

	for (size_t i = 1; i < contour.size(); i++)
	{
		current_point = contour[i];
		// si encontramos  un punto del contorno más bajo, lo guardamos
		if (current_point.y < prev_point.y)
		{
			prev_point = current_point;
			lowest_Point_Index = i;
		}
	}
	// reservamos el espacio necesario previamente
	one_side_dir1.reserve(contour.size() - (lowest_Point_Index));
	one_side_dir2.reserve(lowest_Point_Index + 1);
	// Incluimos el punto más bajo en el contorno nuevo
	one_side_dir1.push_back(prev_point);
	one_side_dir2.push_back(prev_point);

	if (lowest_Point_Index + 1 < contour.size())
	{
		// recorremos desde el punto más bajo en una dirección
		for (size_t i = lowest_Point_Index + 1; i < contour.size(); i++)
		{
			current_point = contour[i];
			// si vamos subiendo añadimos a la línea
			if (current_point.y >= prev_point.y)
			{
				one_side_dir1.push_back(current_point);
				prev_point = current_point;
				// si dejamos de subir, paramos
			}
			else
				break;
		}
	}
	if (lowest_Point_Index - 1 > 0)
	{
		// recorremos desde el punto más bajo en la otra dirección
		for (size_t i = lowest_Point_Index - 1; i > 0; i--)
		{
			current_point = contour[i];
			// s i vamos subiendo añadimos a la línea
			if (current_point.y >= prev_point.y)
			{
				one_side_dir2.push_back(current_point);
				prev_point = current_point;
				// si dejamos de subir, paramos
			}
			else
				break;
		}
	}

	if (one_side_dir1.size() >= one_side_dir2.size())
		return one_side_dir1;
	else
		return one_side_dir2;
}

vector<cv::Point> Get_Biggest_Contour(const vector<vector<cv::Point>> &contours)
{
	Moments m1, m2;
	vector<cv::Point> biggest_area_contour = contours[0];
	m1 = moments(contours[0]);
	// for( size_t i = 1; i < contours.size(); i++ ){
	for (const vector<cv::Point> &single_cont : contours)
	{
		m2 = moments(single_cont);
		// si el area es mayor
		if (m2.m00 > m1.m00)
		{
			biggest_area_contour = single_cont;
			m1 = moments(biggest_area_contour);
		}
	}
	return biggest_area_contour;
}

vector<cv::Point> Get_Biggest_Contour_One_Line(const vector<vector<cv::Point>> &contours)
{
	vector<cv::Point> biggest_area_contour = Get_Biggest_Contour(contours);
	return Get_One_Side_Contour(biggest_area_contour);
}

void Draw_Contour(InputOutputArray inp, const vector<cv::Point> &contour, const Scalar &color, int roi_x, int roi_y)
{
	vector<cv::Point> shifted_cont;
	// allocate necessary space from the start
	shifted_cont.reserve(contour.size());
	shifted_cont[0].x = contour[0].x + roi_x;
	shifted_cont[0].y = contour[0].y + roi_y;
	for (int i = 0; i < contour.size() - 1; i++)
	{
		shifted_cont[i + 1].x = contour[i + 1].x + roi_x;
		shifted_cont[i + 1].y = contour[i + 1].y + roi_y;
		cv::line(inp, shifted_cont[i], shifted_cont[i + 1], color, 5);
	}
}

void Draw_line(InputOutputArray inp, const vector<double> &x_values, const vector<double> &y_values, const cv::Scalar &color)
{
	// std::cout<<"inside drawlines";
	for (int i = 0; i < x_values.size() - 1; i++)
	{
		cv::line(inp, cv::Point(x_values[i], y_values[i]), cv::Point(x_values[i + 1], y_values[i + 1]), color, 5);
	}
}

void Get_Draw_Biggest_One_Line(InputOutputArray inp, const vector<vector<cv::Point>> &contours)
{
	vector<cv::Point> one_sided = Get_Biggest_Contour_One_Line(contours);
	std::cout << "Got biggest contour. Size: " << one_sided.size() << std::endl;
	if (one_sided.size() < 2)
	{
		return;
	}
	// for(int i = 0; i < one_sided.size()-1; i++){
	// cv::line(inp, one_sided[i], one_sided[i+1], Scalar(155, 0, 155), 2);
	//}
	Draw_Contour(inp, one_sided);
}
void Get_Draw_Biggest_Fitted_One_Line(InputOutputArray inp, const vector<vector<cv::Point>> &contours)
{
	vector<cv::Point> one_sided = Get_Biggest_Contour_One_Line(contours);
	std::cout << "Got biggest contour" << std::endl;
	vector<cv::Point> one_sided_simple;
	cv::approxPolyDP(one_sided, one_sided_simple, 0.1, false);
	Draw_Contour(inp, one_sided_simple);
}

void Get_Draw_One_Sided_Lines(InputOutputArray inp, const vector<vector<cv::Point>> &contours, int min_area, int roi_x, int roi_y)
{
	vector<Moments> mu(contours.size());

	for (size_t i = 1; i < contours.size(); i++)
	{
		// for (const vector<cv::Point>& single_cont : contours){
		const vector<cv::Point> &single_cont = contours[i];
		mu[i] = moments(single_cont); // contours[i]

		if (mu[i].m00 > min_area)
		{ // filtro por área
			Draw_Contour(inp, Get_One_Side_Contour(single_cont), roi_x, roi_y);
		}
	}
}

void Get_Draw_One_Sided_Fitted_Lines(InputOutputArray inp, const vector<vector<cv::Point>> &contours, int min_area, int roi_x, int roi_y)
{
	vector<Moments> mu(contours.size());

	for (size_t i = 1; i < contours.size(); i++)
	{
		// for (const vector<cv::Point>& single_cont : contours){
		const vector<cv::Point> &single_cont = contours[i];
		mu[i] = moments(single_cont);

		if (mu[i].m00 > min_area)
		{ // filtro por área
			vector<cv::Point> one_sided_simple;
			cv::approxPolyDP(Get_One_Side_Contour(single_cont), one_sided_simple, 0.1, false);
			Draw_Contour(inp, one_sided_simple);
		}
	}
}

void HueShift(InputArray img_hsv, OutputArray img_shift, int shift)
{
	std::vector<cv::Mat> channels;
	split(img_hsv.getMat(), channels);

	Mat &H = channels[0];
	Mat &S = channels[1];
	Mat &V = channels[2];

	Mat shiftedH = H.clone();
	for (int j = 0; j < shiftedH.rows; ++j)
		for (int i = 0; i < shiftedH.cols; ++i)
		{
			// in openCV hue values go from 0 to 180 (so have to be doubled to get to 0 .. 360)
			// because of byte range from 0 to 255
			shiftedH.at<unsigned char>(j, i) = (shiftedH.at<unsigned char>(j, i) + shift) % 180;
		}
	// namedWindow("NormalH", WindowFlags::WINDOW_NORMAL);
	// imshow("NormalH", H);
	H = shiftedH.clone();
	// namedWindow("ShiftedH", WindowFlags::WINDOW_NORMAL);
	// imshow("ShiftedH", H);
	merge(channels, img_shift);
}

void ParallelHueShift(InputArray img_hsv, OutputArray img_shift, int shift)
{
	std::vector<cv::Mat> channels;
	split(img_hsv.getMat(), channels);

	Mat &H = channels[0];
	Mat &S = channels[1];
	Mat &V = channels[2];

	Mat shiftedH = H.clone();
	// optimization (reduced performance by aprox 10ms, from aprox. 85ms to 75ms)
	parallel_for_(Range(0, shiftedH.rows * shiftedH.cols), [&](const Range &range)
				  {
		for (int r = range.start; r < range.end; r++)
		{
			//transformación unidimensional de r a bidimensional para columnas i y filas j
			int j = r / shiftedH.cols;
            int i = r % shiftedH.cols;
			shiftedH.at<unsigned char>(j,i) = (shiftedH.at<unsigned char>(j,i) + shift)%180; //para que no se pase del valor máximo de H (179)
		} });

	shiftedH.copyTo(H);
	merge(channels, img_shift);
}

// las y van hacia abajo en OpenCV
vector<cv::Point> SampleBinImgAtHeight(InputOutputArray img, int y)
{
	Mat inp = img.getMat();
	vector<cv::Point> positives;
	for (int i = 0; i < inp.cols; i++)
	{
		if ((int)(inp.at<uchar>(y, i)) == 255)
		{
			positives.emplace_back(i, y);
		}
	}
	return positives;
}

cv::Point SampleBinImgAtHeightFirstPoint(InputOutputArray img, const cv::Point &init, const bool &direction)
{
	Mat inp = img.getMat();
	int y = init.y;

	if (direction)
	{
		for (int i = init.x; i < inp.cols; i++)
		{
			if ((int)(inp.at<uchar>(y, i)) == 255)
			{
				return Point(i, y);
			}
		}
		return Point(0, 0);
	}
	else
	{
		for (int i = init.x; i >= 0; i--)
		{
			if ((int)(inp.at<uchar>(y, i)) == 255)
			{
				return Point(i, y);
			}
		}
		return Point(0, 0);
	}
}

vector<cv::Point> SampleBinImgAtHeightFirstPoints(InputOutputArray img, const cv::Point &init)
{
	std::vector<cv::Point> points;
	points.reserve(2);
	points[0] = SampleBinImgAtHeightFirstPoint(img, init, false);
	points[1] = SampleBinImgAtHeightFirstPoint(img, init, true);
	return points;
}

// DOESN'T ESCAPE, DON'T USE
bool WaitForFps_or_Esc(clock_t *startTime, int delay)
{
	clock_t processTime = clock() - *startTime;
	while (processTime < delay)
	{
		// if Esc key is pressed
		if (waitKey((int)(delay - processTime)) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			return true;
		}
		// update processTime just in case
		processTime = clock() - *startTime;
	}
	return false;
}

std::vector<int> CvPointVecExtract(const std::vector<cv::Point> &line, const bool x_or_y)
{
	std::vector<int> coordinates;
	coordinates.reserve(line.size());
	if (x_or_y)
	{
		for (int i = 0; i < line.size(); i++)
		{
			coordinates.emplace_back(line[i].x);
		}
	}
	else
	{
		for (int i = 0; i < line.size(); i++)
		{
			coordinates.emplace_back(line[i].y);
		}
	}
	return coordinates;
}

std::vector<int> CvPointVecExtractReverse(const std::vector<cv::Point> &line, const bool x_or_y)
{
	std::vector<int> coordinates;
	coordinates.reserve(line.size());
	if (x_or_y)
	{
		for (int i = (line.size() - 1); i >= 0; i--)
		{
			coordinates.emplace_back(line[i].x);
		}
	}
	else
	{
		for (int i = (line.size() - 1); i >= 0; i--)
		{
			coordinates.emplace_back(line[i].y);
		}
	}
	return coordinates;
}

std::vector<cv::Point> OrderByY(const std::vector<cv::Point> &line)
{
	int n = line.size();
	std::vector<cv::Point> line2 = line;
	for (int i = 0; i < n - 1; ++i)
	{
		for (int j = 0; j < n - i - 1; ++j)
		{
			if (line2[j].y < line2[j + 1].y)
			{
				std::swap(line2[j], line2[j + 1]);
			}
		}
	}
	return line2;
}

namespace LineProcess
{

	/**
	 * Calcula el histograma de la imagne dada, eliminando valores < 5
	 * InputArray binImg: imagene a analizar
	 * int bottomY: parte de abajo de la imagenl número más grande
	 * int currentY: altura que se analiza
	 * int thresh: si hist < thresh, se pone a 0
	 * const std::vector<int>& hist: el vector que contiene el histograma
	 */
	std::vector<int> GetHist(InputArray binImg, int bottomY, int currentY, int thresh)
	{
		if (bottomY < currentY)
		{ // si nos lo dan al revés le damos la vuelta
			int temp = bottomY;
			bottomY = currentY;
			currentY = temp;
		}

		// se hace un rectángulo en la imagen de la misma anchura,
		//    pero altura marcada por bottomY-currentY
		cv::Mat inp = binImg.getMat(); // topX, topY, width, height
		int height = (bottomY - currentY);
		cv::Mat subImg = inp(cv::Rect(0, currentY, (inp.cols), height));
		cout << "Presiona ENTER para seguir" << endl;

		std::vector<int> hist;
		hist.reserve(inp.cols);
		int value = 0;

		cout << "NEW HIST" << endl;
		// recorrer columnas
		for (int j = 0; j < (inp.cols); j++)
		{
			value = 0;
			// recorrer cada fila de la columna
			for (int i = 0; i < height; i++)
			{
				if ((int)(subImg.at<uchar>(i, j)) != 0)
				{
					value++; ////
				}
			}
			if (value >= thresh)
			{
				hist.emplace_back(value);
				cout << " " << value;
			}
			else
				hist.emplace_back(0);
		}
		cout << " " << endl;

		// Se queda esperando un Enter
		// cout<<"Presiona ENTER para seguir"<<endl;
		// cin.get();

		return hist;
	}

	int GetHistMaxVal(const std::vector<int> &hist)
	{
		int max = 0;
		for (int val : hist)
		{
			if (val > max)
			{
				max = val;
			}
		}
		return max;
	}

	/**
	 * Busca los máximos locales en un vector
	 * const std::vector<int>& hist: el vector que contiene el histograma
	 */
	std::vector<int> GetLocalMaxes(const std::vector<int> &hist)
	{
		int local_max = 0;
		bool up = true;
		int local_max_index_first = 0;
		std::vector<int> local_max_indexes;

		for (int i = 0; i < hist.size(); i++)
		{
			if (hist[i] > local_max)
			{
				up = true;
				local_max = hist[i];
				local_max_index_first = i;
			}
			else if (hist[i] < local_max)
			{
				if (up)
				{
					up = false;
					// si se mantiene un valor en 2 columnas seguidas, calcula la media
					local_max_indexes.push_back((i + local_max_index_first) / 2);
				}
				local_max = hist[i];
			}
		}
		return local_max_indexes;
	}

	/**
	 * Hace un histrograma de la imagen y busca los máximos locales, la altura de los punto devueltos es (bottomY+currentY)/2
	 * InputArray binImg: imagen que se va a tratar
	 * int bottomY: la parte de abajo de la imagen, que es el número más alto
	 * int currentY: la parte más alta de la imagen que se va a analizar
	 */
	std::vector<cv::Point> GetHistLocalMaxes(InputArray binImg, int bottomY, int currentY)
	{
		int midY = (bottomY + currentY) / 2;
		std::vector<int> hist = GetHist(binImg, bottomY, currentY, 5);

		std::vector<int> local_max_indexes = GetLocalMaxes(hist);
		// std::cout<<local_max_indexes.size()<<" is the number of local maximums"<<std::endl;
		// cv::waitKey(-1);

		std::vector<cv::Point> local_max_points;
		local_max_points.reserve(local_max_indexes.size());
		for (int i = 0; i < local_max_indexes.size(); i++)
		{
			local_max_points.emplace_back(local_max_indexes[i], midY);
		}
		return local_max_points;
	}

	void GetDrawHistLocalMaxes(InputOutputArray drawImg, InputArray binImg, int bottomY, int topY)
	{
		Draw_Contour_Points(drawImg.getMat(), GetHistLocalMaxes(binImg, bottomY, topY));
	}

	void GetDrawHistLocalMaxes_Periodically(InputOutputArray drawImg, InputArray binImg, int bottomY, int topY, int window_size)
	{
		// si nos lo dan al revés le damos la vuelta
		if (bottomY < topY)
		{
			int temp = bottomY;
			bottomY = topY;
			topY = temp;
		}

		const int height = bottomY - topY;
		int n_of_windows = (int)height / window_size;

		int current_bottomY = bottomY;

		for (int i = 0; i < n_of_windows; i++)
		{
			GetDrawHistLocalMaxes(drawImg.getMat(), binImg.getMat(), current_bottomY, (current_bottomY - window_size));
			current_bottomY -= window_size;
		}
		// if(height % window_size != 0) GetDrawHistLocalMaxes(drawImg, binImg, bottomY, current_topY);
	}

	/**
	 * Busca la linea solo en la primera altura
	 * InputArray binImg: imagen que se va a tratar
	 * int bottomY: la parte de abajo de la imagen, que es el número más alto
	 * int currentY: altura a la que se hace el analisis
	 * int laneX: es el pixel del eje X en el que empieza la búsqueda, mitad de la ROI
	 */
	std::vector<cv::Point> GetHistFirstLanePoints(InputArray binImg, int bottomY, int currentY, int laneX)
	{
		// si nos lo dan al revés le damos la vuelta
		if (bottomY < currentY)
		{
			int temp = bottomY;
			bottomY = currentY;
			currentY = temp;
		}

		// si menor = que 0, empezamos en mitad de la imagen
		int searchPoint = (laneX <= 0) ? binImg.getMat().cols / 2 : laneX;

		std::vector<cv::Point> points = GetHistLocalMaxes(binImg, bottomY, currentY);
		std::vector<cv::Point> lanePoints;
		lanePoints.reserve(2);

		if (points.size() == 0)
		{
			lanePoints.emplace_back(-1, -1);
			lanePoints.emplace_back(-1, -1);
			return lanePoints;
		}

		// ¿A partir de aquí qué hace? CAMBIAR
		// hace una comprobación de donde estan los puntos en la ROI
		// y se decide cual corresponde a la linea en base a una distancia
		// no se tiene en cuenta donde estuvieron en el frame anterior
		int leftLaneIdx = 0, rightLaneIdx = 0;
		int currentLeftMinDist = -searchPoint;
		int currentRightMinDist = searchPoint;
		for (int i = 0; i < points.size(); i++)
		{
			int pointDist = points[i].x - searchPoint;

			// possible right lane
			if (pointDist > 0)
			{

				if (pointDist < currentRightMinDist)
				{
					currentRightMinDist = pointDist;
					rightLaneIdx = i;
				}
			}
			// posible left lane
			else
			{

				if (pointDist > currentLeftMinDist)
				{
					currentLeftMinDist = pointDist;
					leftLaneIdx = i;
				}
			}
		}

		if ((leftLaneIdx == rightLaneIdx) && (points.size() >= 1))
		{
			rightLaneIdx++;
		}
		lanePoints.push_back(points[leftLaneIdx]);
		lanePoints.push_back(points[rightLaneIdx]);

		return lanePoints;
	}

	/**
	 * Búsquesda de puntos en las lineas a partir de la primera fila
	 * InputArray binImg: imagen que se va a tratar
	 * int bottomY: currentY en las alturas de búsqueda
	 * int topY: (currentY-window_size)
	 * int prevPointX: seedPoint.x, punto horizontal de la linea en la primera búsqueda
	 * int thresh: window_size*2
	 */
	cv::Point GetHistNextLinePoint(InputArray binImg, int bottomY, int topY, int prevPointX, int thresh)
	{

		// si nos lo dan al revés le damos la vuelta
		if (bottomY < topY)
		{
			int temp = bottomY;
			bottomY = topY;
			topY = temp;
		}

		std::vector<cv::Point> points = GetHistLocalMaxes(binImg, bottomY, topY);

		cv::Point lanePoint;

		if (points.size() == 0)
		{
			lanePoint = cv::Point(-1, -1);
			return lanePoint;
		}

		int laneIdx = 0;
		thresh = std::abs(thresh);
		int currentMinDist = thresh;

		for (int i = 0; i < points.size(); i++)
		{
			int pointDist = std::abs(points[i].x - prevPointX);

			if (pointDist < currentMinDist)
			{
				currentMinDist = pointDist;
				laneIdx = i;
			}
		}
		// no ha registrado una distancia menor
		if (currentMinDist == thresh)
		{
			lanePoint = cv::Point(-1, -1);

			return lanePoint;
		}

		lanePoint = points[laneIdx];

		return lanePoint;
	}

	/**
	 * Búsqueda de puntos en las lineas por alturas
	 * InputArray binImg: imagen que se va a tratar
	 * int bottomY: la parte de abajo de la imagen, que es el número más alto
	 * int topY: la parte más alta de la imagen que se va a analizar
	 * int window_size: cada cuanto se busca la linea
	 * bool leftLane: buscar izquierda
	 * int thresh: viene igualado a 0
	 * int laneX: es el pixel del eje X en el que empieza la búsqueda, fujado en (FRAME_SIZE_X/2)-400 para chamonix
	 */
	std::vector<cv::Point> GetHistLanePoints(InputArray binImg, int bottomY, int topY, int window_size, bool leftLane, int thresh, int laneX)
	{
		// std::cout<<"Entered GetHistLanePoints"<<std::endl;

		// si nos lo dan al revés le damos la vuelta
		if (bottomY < topY)
		{
			int temp = bottomY;
			bottomY = topY;
			topY = temp;
		}

		const int height = bottomY - topY;
		int n_of_windows = (int)height / window_size;

		int currentY = bottomY - window_size;
		// Busca los puntos de las lineas cerca de la cámara
		// Por qué la búsqueda de puntos es distinta para la primera y las demás?
		std::vector<cv::Point> laneSeeds = GetHistFirstLanePoints(binImg, bottomY, currentY, laneX);

		cv::Point seedPoint, nextPoint;
		std::vector<cv::Point> lane;
		// detecta si no se han encontrado lineas
		// ¿¿ habría que repetir las anteriores ??
		if (laneSeeds[0] == cv::Point(-1, -1))
		{
			lane.emplace_back(-1, -1);
			return lane;
		}
		// elegir izq o der
		if (leftLane)
		{
			seedPoint = laneSeeds[0];
		}
		else if (laneSeeds.size() >= 2)
		{
			seedPoint = laneSeeds[1];
		}
		else
		{
			lane.emplace_back(-1, -1);
			return lane;
		}
		lane.push_back(seedPoint);

		// thresh es por defecto 0
		if (thresh <= 0)
			thresh = window_size * 2;

		// se llama a buscar las lineas en cada altura
		for (int i = 1; i < n_of_windows; i++)
		{
			nextPoint = GetHistNextLinePoint(binImg, currentY, (currentY - window_size), seedPoint.x, thresh);
			// if point is valid, add and update seed
			if (nextPoint != cv::Point(-1, -1))
			{
				lane.push_back(nextPoint);
				seedPoint = nextPoint;
			}
			// si no, salimos
			else
				return lane;

			currentY -= window_size; // go to the next window
		}

		return lane;
	}

	/**
	 * Calcula el punto medio entre ambas lineas
	 */
	std::vector<cv::Point> CalculateMidLane(const std::vector<cv::Point> &leftLane, const std::vector<cv::Point> &rightLane)
	{
		int limit = (leftLane.size() <= rightLane.size()) ? leftLane.size() : rightLane.size();
		std::vector<cv::Point> midLane;
		midLane.reserve(limit);
		int mid_x;
		for (int i = 0; i < limit; i++)
		{
			mid_x = (leftLane[i].x + rightLane[i].x) / 2;
			midLane.emplace_back(mid_x, leftLane[i].y);
		}

		return midLane;
	}
}

/**
 */
std::vector<cv::Point> DavidLanePoints(InputArray binImg, int bottomY, int topY, int window_size, int threshold)
{
	int height = (bottomY - topY);
	int n_of_windows = (int)height / window_size;

	int currentY = bottomY - window_size;

	// se hace un rectángulo en la imagen de la misma anchura,
	//    pero altura marcada por bottomY-currentY
	cv::Mat imgMat = binImg.getMat(); // topX, topY, width, height
	// cv::Mat subImg = inp(cv::Rect(0, currentY, (inp.cols), height));

	int count = 0;
	std::vector<cv::Point> white_points;

	// recorrer columnas para cada altura
	for (int i = currentY; i > topY; i = i - window_size)
	{
		// cout<<"LINE "<< i << endl;
		count = 0;
		// TODO:NO SE SI LA BUSQUEDA DEBE EMPEZAR EN EL CENTRO DE LA IMAGEN, O EN EL CENTRO DE LA CALLE
		// recorrer columnas del medio a la izq
		// el primer blanco que encuentra es la linea
		for (int j = (imgMat.cols / 2); j > 0; j--)
		{
			// cout<< (int)(imgMat.at<uchar>(i, j)) << " " ;
			if ((int)(imgMat.at<uchar>(i, j)) > 0)
			{
				count++;
			}
			else
			{
				if (count > threshold)
				{
					white_points.push_back(cv::Point(j + (count / 2), i));
					count = 0;
					break;
				}
			}
		}
		// recorrer columnas del medio a la dch
    	// el primer blanco que encuentra es la linea
		for (int j = (imgMat.cols / 2); j < (imgMat.cols); j++)
		{
			// cout<< (int)(imgMat.at<uchar>(i, j)) << " " ;
			if ((int)(imgMat.at<uchar>(i, j)) > 0)
			{
				count++;
			}
			else
			{
				if (count > threshold)
				{
					white_points.push_back(cv::Point(j - (count / 2), i));
					count = 0;
					break;
				}
			}
		}
		// cout<<"END Line "<< i << endl << endl;
	}
	return white_points;
}

/**
 * Function to calculate distance from point P to line formed by points A and B
 * numerator: This is the absolute value of the linear equation for the line AB,
 * 			  here the terms are calculated using the coordinates of A, B, and P.
 * denominator: This is the Euclidean distance between points A and B,
 * 				which is used to normalize the result.
 *  ESTO NO VALE PARA LAS CURVAS
 */
double distancePointToLine(const cv::Point &A, const cv::Point &B, const cv::Point &P)
{
	// Return the distance
	double numerator = std::abs((B.y - A.y) * P.x - (B.x - A.x) * P.y + B.x * A.y - B.y * A.x);
	double denominator = std::sqrt(std::pow(B.y - A.y, 2) + std::pow(B.x - A.x, 2));
	return numerator / denominator;
}

// Function to compute the Euclidean distance between point P(x_p, y_p) and the curve y = ax^2 + bx + c
double distancePointToCurve(const cv::Point &P, const QuadraticCurve &curve)
{
	double x_p = P.x;
	double y_p = P.y;

	// Search range for x values (centered around x_p)
	double searchStart = x_p - 10; // Adjust range as needed
	double searchEnd = x_p + 10;

	double minDistance = std::numeric_limits<double>::max(); // Initialize to a large value
	double bestX = x_p;										 // To store the x with the minimum distance

	// Step through the x-values to find the closest point on the curve
	for (double x = searchStart; x <= searchEnd; x += 0.01)
	{
		double y_curve = curve.evaluate(x);
		double distance = std::sqrt(std::pow(x - x_p, 2) + std::pow(y_curve - y_p, 2));

		if (distance < minDistance)
		{
			minDistance = distance;
			bestX = x;
		}
	}

	return minDistance;
}

// Function to compute the squared distance function f(x)
double distanceSquared(const cv::Point &P, const QuadraticCurve &curve, double x)
{
	double x_p = P.x;
	double y_p = P.y;
	double y_curve = curve.evaluate(x);
	return std::pow(x - x_p, 2) + std::pow(y_curve - y_p, 2);
}

// Function to compute the gradient f'(x) for gradient descent
double distanceGradient(const cv::Point &P, const QuadraticCurve &curve, double x)
{
	double x_p = P.x;
	double y_p = P.y;
	double y_curve = curve.evaluate(x);
	double curve_derivative = curve.derivative(x);

	// Derivative of the distance function with respect to x
	return 2 * (x - x_p) + 2 * (y_curve - y_p) * curve_derivative;
}

// Gradient descent to minimize the distance between point P and the curve
double gradientDescent(const cv::Point &P, const QuadraticCurve &curve, double learning_rate, int max_iters, double tolerance)
{
	double x = P.x; // Start the search at x_p
	double prev_x = x;
	for (int i = 0; i < max_iters; ++i)
	{
		double grad = distanceGradient(P, curve, x);
		x = x - learning_rate * grad; // Update x based on the gradient

		// Check for convergence
		if (std::abs(x - prev_x) < tolerance)
		{
			break;
		}
		prev_x = x;
	}
	return x;
}

// Fit the parabola and get the coefficients (a, b, c)
//  cv::Vec3d coefficients = fitParabola(P1, P2, P3);
cv::Vec3d fitParabola(const cv::Point2d &P1, const cv::Point2d &P2, const cv::Point2d &P3)
{
	// Setup the system of equations
	cv::Mat A(3, 3, CV_64F); // Coefficient matrix
	cv::Mat Y(3, 1, CV_64F); // Y values

	// Fill the coefficient matrix A and Y matrix for the points
	A.at<double>(0, 0) = P1.x * P1.x;
	A.at<double>(0, 1) = P1.x;
	A.at<double>(0, 2) = 1;
	Y.at<double>(0, 0) = P1.y;

	A.at<double>(1, 0) = P2.x * P2.x;
	A.at<double>(1, 1) = P2.x;
	A.at<double>(1, 2) = 1;
	Y.at<double>(1, 0) = P2.y;

	A.at<double>(2, 0) = P3.x * P3.x;
	A.at<double>(2, 1) = P3.x;
	A.at<double>(2, 2) = 1;
	Y.at<double>(2, 0) = P3.y;

	// Solve the system A * [a, b, c]^T = Y
	cv::Mat coefficients;
	cv::solve(A, Y, coefficients);

	// Return the coefficients as (a, b, c)
	return cv::Vec3d(coefficients.at<double>(0, 0), coefficients.at<double>(1, 0), coefficients.at<double>(2, 0));
}

double eval_parabola(double a, double b, double c, double x)
{
	return a * x * x + b * x + c;
}

double distanceToParabola(double a, double b, double c, double x_p, double y_p)
{
	// Initial guess for the closest point on the parabola
	double x_guess = x_p;
	double learning_rate = 0.01;
	double tolerance = 1e-5;

	while (true)
	{
		// Compute the value of the parabola at x_guess
		double y_guess = eval_parabola(a, b, c, x_guess);

		// Compute the distance derivative (the gradient)
		double gradient = 2 * (x_guess - x_p) + 2 * (y_guess - y_p) * (2 * a * x_guess + b);

		// Update guess using a simple gradient descent step
		x_guess -= learning_rate * gradient;

		// Check for convergence
		if (std::abs(gradient) < tolerance)
			break;
	}

	// After finding the closest x, calculate the y coordinate on the parabola
	double y_closest = eval_parabola(a, b, c, x_guess);
	// Compute the distance to the point
	double distance = std::sqrt((x_guess - x_p) * (x_guess - x_p) + (y_closest - y_p) * (y_closest - y_p));

	return distance;
}