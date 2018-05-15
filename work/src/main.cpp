// std
#include <iostream>

// opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// project
#include "invert.hpp"


using namespace cv;
using namespace std;

cv::Mat nnf2img(cv::Mat nnf, cv::Size s, bool absolute);
cv::Mat propagateUp(int x, int y, Mat source, Mat sourceExtend, Mat nnf);
cv::Mat propagateLeft(int x, int y, Mat source, Mat sourceExtend, Mat nnf);
cv::Mat propagateRight(int x, int y, Mat source, Mat sourceExtend, Mat nnf);
cv::Mat propagateDown(int x, int y, Mat source, Mat sourceExtend, Mat nnf);
cv::Mat frontPropagate(Mat source, Mat sourceExtend, Mat target, Mat targetExtend, Mat nnf);
cv::Mat backPropagate(Mat source, Mat sourceExtend, Mat target, Mat targetExtend, Mat nnf);
cv::Mat randomSearch(Mat source, Mat sourceExtend, Mat target, Mat targetExtend, Mat nnf);


// main program
// 
int main( int argc, char** argv ) {

	// check we have exactly two additional argument
	// eg. res/vgc-logo.png
	if( argc != 3) {
		cout << "Usage: display_image ImageToLoadAndDisplay" << endl;
		return -1;
	}

	// read the file
	Mat source;
	Mat target;
	source = imread(argv[1], CV_LOAD_IMAGE_COLOR); 
	target = imread(argv[2], CV_LOAD_IMAGE_COLOR);

	// check for invalid input
	if(!source.data ) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	if(!target.data ) {
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// creates a nearest-neighbour field 
	//which references a position in the source image for a pixel in target
	Mat nnf(source.rows, source.cols, CV_32SC2);


	// randomly initialize the nnf
	srand(time(NULL));
	for (int i = 0; i < nnf.rows; i++) {
		for(int j = 0; j < nnf.cols; j++){
			int x = rand() % nnf.cols;
			int y = rand() % nnf.rows;
			nnf.at<Vec2i>(i,j)[0] = x;
			nnf.at<Vec2i>(i,j)[1] = y;
		}
	}

	Mat sourceExtend;
	Mat targetExtend;
	// adding a border for the edge cases
	// duplicates the edge values and extends it outward
	copyMakeBorder(source, sourceExtend, 3, 3, 3, 3, BORDER_REPLICATE);
	copyMakeBorder(target, targetExtend, 3, 3, 3, 3, BORDER_REPLICATE);

	
	for(int k = 0; k < 5; k++){
		nnf = frontPropagate(source, sourceExtend, target, targetExtend, nnf);
		nnf = backPropagate(source, sourceExtend, target, targetExtend, nnf);
		nnf = randomSearch(source, sourceExtend, target, targetExtend, nnf);
	}

	Mat reconstruction(target.rows, target.cols, CV_8UC3);
	for(int i = 0; i < target.rows; i++){
		for(int j = 0; j < target.cols; j++){
			Point p = nnf.at<Point>(i,j);

			Vec3b color = source.at<Vec3b>(p);

			reconstruction.at<Vec3b>(i,j) = color;
		}
	}

	Mat img2 = nnf2img(nnf, nnf.size(), false);

	string img2_display = "Reconstruction Display";
	namedWindow(img2_display, WINDOW_AUTOSIZE);
	imshow(img2_display, reconstruction);

	Mat img = nnf2img(nnf, nnf.size(), false);

	string img_display = "Image Display";
	namedWindow(img_display, WINDOW_AUTOSIZE);
	imshow(img_display, img);

	// wait for a keystroke in the window before exiting
	waitKey(0);
}

cv::Mat frontPropagate(Mat source, Mat sourceExtend, Mat target, Mat targetExtend, Mat nnf){
	// propagating from top left to bottom right
	for(int i = 0; i < nnf.rows; i++){
		for(int j = 0; j < nnf.cols; j++){
			// tells us if there is a neighbouring pixel above or to the left of current pixel
			bool upPropagate = (i == 0) ? false : true;
			bool leftPropagate = (j == 0) ? false : true;
			Mat patchT;
			Mat patchS;
			// find the location of the pixel in the target that is being referenced by nnf
			Point p = nnf.at<Point>(i,j);

			// create a patch with the extended border if iterating close enough to edges
			if(j < 3 || i < 3 || j >= nnf.cols-3 || i >= nnf.rows-3){
				Rect roi1(j, i, 7, 7);
				Mat b(targetExtend, roi1);
				patchT = b.clone();
			}
			// create a patch for target image
			else{
				Rect roi1(j-3, i-3, 7, 7);
				Mat b(target, roi1);	
				patchT = b.clone();
			}
			
			// create a patch with the extended border if iterating close enough to edges
			if(p.x < 3 || p.y < 3 || p.x >= nnf.cols-3 || p.y >= nnf.rows-3){
				Rect roi1(p.x, p.y, 7, 7);
				Mat b(sourceExtend, roi1);
				patchS = b.clone();
			}
			// create a patch for source image
			else{
				Rect roi1(p.x-3, p.y-3, 7, 7);
				Mat b(source, roi1);
				patchS = b.clone();
			}

			// the difference between the patches, the lower the value, the better the patch is
			float upPatchValue = std::numeric_limits<float>::infinity();
			float leftPatchValue = std::numeric_limits<float>::infinity();
			float currPatchValue = cv::norm(patchT, patchS);
			Mat upMat;
			Mat leftMat;
			if(upPropagate){upMat = propagateUp(j, i, source, sourceExtend, nnf); upPatchValue = cv::norm(patchT, upMat);}
			if(leftPropagate){leftMat = propagateLeft(j, i, source, sourceExtend, nnf); leftPatchValue = cv::norm(patchT, leftMat);}

			// if the mapping above has a better value, set the current value to map to the value above
			if(upPatchValue < currPatchValue && upPatchValue < leftPatchValue){
				Point p = nnf.at<Point>(i-1,j) + Point(0, 1);
				if(p.y >= source.rows){p += Point(0,-1);}
				nnf.at<Point>(i,j) = p;
			}
			// if the mapping to the left has a better value, set current value to map to the value to the left
			if(leftPatchValue < currPatchValue && leftPatchValue < upPatchValue){
				Point p = nnf.at<Point>(i,j-1) + Point(1, 0);
				if(p.x >= source.cols){p += Point(-1,0);}
				nnf.at<Point>(i,j) = p;
			}
		}
	}
	return nnf;
}

cv::Mat backPropagate(Mat source, Mat sourceExtend, Mat target, Mat targetExtend, Mat nnf){
for(int i = nnf.rows-1; i >= 0; i--){
		for(int j = nnf.cols-1; j >= 0; j--){
			// tells us if there is a neighbouring pixel above or to the left of current pixel
			bool downPropagate = (i == nnf.rows-1) ? false : true;
			bool rightPropagate = (j == nnf.cols-1) ? false : true;
			Mat patchT;
			Mat patchS;
			// find the location of the pixel in the target that is being referenced by nnf
			Point p = nnf.at<Point>(i,j);

			// create a patch with the extended border if iterating close enough to edges
			if(j < 3 || i < 3 || j >= nnf.cols-3 || i >= nnf.rows-3){
				Rect roi1(j, i, 7, 7);
				Mat b(targetExtend, roi1);
				patchT = b.clone();
			}
			// create a patch for target image
			else{
				Rect roi1(j-3, i-3, 7, 7);
				Mat b(target, roi1);	
				patchT = b.clone();
			}
			
			// create a patch with the extended border if iterating close enough to edges
			if(p.x < 3 || p.y < 3 || p.x >= nnf.cols-3 || p.y >= nnf.rows-3){
				Rect roi1(p.x, p.y, 7, 7);
				Mat b(sourceExtend, roi1);
				patchS = b.clone();
			}
			// create a patch for source image
			else{
				Rect roi1(p.x-3, p.y-3, 7, 7);
				Mat b(source, roi1);
				patchS = b.clone();
			}

			// the difference between the patches, the lower the value, the better the patch is
			float downPatchValue = std::numeric_limits<float>::infinity();
			float rightPatchValue = std::numeric_limits<float>::infinity();
			float currPatchValue = cv::norm(patchT, patchS);
			Mat downMat;
			Mat rightMat;
			if(downPropagate){downMat = propagateDown(j, i, source, sourceExtend, nnf); downPatchValue = cv::norm(patchT, downMat);}
			if(rightPropagate){rightMat = propagateRight(j, i, source, sourceExtend, nnf); rightPatchValue = cv::norm(patchT, rightMat);}

			// if the mapping above has a better value, set the current value to map to the value above
			if(downPatchValue < currPatchValue && downPatchValue < rightPatchValue){
				Point p = nnf.at<Point>(i+1,j) + Point(0,-1);
				if(p.y < 0){p += Point(0,1);}
				nnf.at<Point>(i,j) = p;
			}
			// if the mapping to the left has a better value, set current value to map to the value to the left
			if(rightPatchValue < currPatchValue && rightPatchValue < downPatchValue){
				Point p = nnf.at<Point>(i,j+1) + Point(-1, 0);
				if(p.x < 0){p += Point(1,0);}
				nnf.at<Point>(i,j) = p;
			}
		}
	}
	return nnf;
}


cv::Mat randomSearch(Mat source, Mat sourceExtend, Mat target, Mat targetExtend, Mat nnf){
// random search
	for(int i = 0; i < nnf.rows; i++){
		for(int j = 0; j < nnf.cols; j++){
			Mat patchT;
			Mat patchS;
			// find the location of the pixel in the target that is being referenced by nnf
			Point p = nnf.at<Point>(i,j);

			// create a patch with the extended border if iterating close enough to edges
			if(j < 3 || i < 3 || j >= nnf.cols-3 || i >= nnf.rows-3){
				Rect roi1(j, i, 7, 7);
				Mat b(targetExtend, roi1);
				patchT = b.clone();
			}
			// create a patch for target image
			else{
				Rect roi1(j-3, i-3, 7, 7);
				Mat b(target, roi1);	
				patchT = b.clone();
			}
			
			// create a patch with the extended border if iterating close enough to edges
			if(p.x < 3 || p.y < 3 || p.x >= nnf.cols-3 || p.y >= nnf.rows-3){
				Rect roi1(p.x, p.y, 7, 7);
				Mat b(sourceExtend, roi1);
				patchS = b.clone();
			}
			// create a patch for source image
			else{
				Rect roi1(p.x-3, p.y-3, 7, 7);
				Mat b(source, roi1);
				patchS = b.clone();
			}


			for(auto r = max(source.rows, source.cols); r > 1; r = r / 2){
				
				Mat randomPatch;

				// random point within the radius
				Point q = p + Point(rand() % (2 * r) - r, rand() % (2 * r) - r);

				// if the point q is outside the image, move on to next iteration
				if(q.x < 0 || q.y < 0 || q.x >= source.cols || q.y >= source.rows){
					continue;
				}

				// get the patch for the point q and calculate the norm against the target
				if(q.x < 3 || q.y < 3 || q.x >= source.cols-3 || q.y >= source.rows-3){
					Rect roi1(q.x, q.y, 7, 7);
					Mat b(sourceExtend, roi1);
					randomPatch = b.clone();
				}
				// create a patch for source image
				else{
					Rect roi1(q.x-3, q.y-3, 7, 7);
					Mat b(source, roi1);
					randomPatch = b.clone();
				}

				float currPatchValue = cv::norm(patchT, patchS);
				float randomPatchValue = cv::norm(patchT, randomPatch);

				// if the random patch is a better match, replace the mapping with the random patch
				if(randomPatchValue < currPatchValue){
					nnf.at<Point>(i,j) = q;
				}
			}
		}
	}
	return nnf;
}


cv::Mat nnf2img(cv::Mat nnf, cv::Size s, bool absolute) {
	cv::Mat nnf_img(nnf.rows, nnf.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Rect rect(cv::Point(0, 0), s);
	for (int r = 0; r < nnf.rows; r++) {
		auto in_row = nnf.ptr<cv::Point>(r);
		auto out_row = nnf_img.ptr<cv::Vec3b>(r);
		for (int c = 0; c < nnf.cols; c++) {
			int x = absolute ? in_row[c].x : in_row[c].x + c;
			int y = absolute ? in_row[c].y : in_row[c].y + r;
			if (!rect.contains(cv::Point(x, y))) {
			/* coordinate is outside the boundry, insert error of choice */
			}
			out_row[c][2] = int(x * 255.0 / s.width); // cols -> red
			out_row[c][1] = int(y * 255.0 / s.height); // rows -> green
			out_row[c][0] = 255 - max(out_row[c][2], out_row[c][1]);
		}
	}	
	return nnf_img;
}


cv::Mat propagateUp(int x, int y,
				  Mat source, Mat sourceExtend,
				  Mat nnf){
	Mat patchS;

	// find the location of the pixel in the target that is being referenced by nnf
	Point p = nnf.at<Point>(y-1,x) + Point(0,1);
	if(p.y == source.rows){p += Point(0,-1);}
	
	// create a patch with the extended border if iterating close enough to edges
	if(p.x < 3 || p.y < 3 || p.x >= nnf.cols-3 || p.y >= nnf.rows-3){
		Rect roi1(p.x, p.y, 7, 7);
		Mat b(sourceExtend, roi1);
		patchS = b.clone();
	}
	// create a patch for target image
	else{
		Rect roi1(p.x-3, p.y-3, 7, 7);
		Mat b(source, roi1);
		patchS = b.clone();
	}

	return patchS;
}


cv::Mat propagateLeft(int x, int y, 
					Mat source, Mat sourceExtend,
					Mat nnf) {
	Mat patchS;

	// find the location of the pixel in the target that is being referenced by nnf
	Point p = nnf.at<Point>(y,x-1);
	if(p.x == source.cols){p += Point(-1,0);}
	// create a patch with the extended border if iterating close enough to edges
	if(p.x < 3 || p.y < 3 || p.x >= nnf.cols-3 || p.y >= nnf.rows-3){
		Rect roi1(p.x, p.y, 7, 7);
		Mat b(sourceExtend, roi1);
		patchS = b.clone();
	}
	// create a patch for target image
	else{
		Rect roi1(p.x-3, p.y-3, 7, 7);
		Mat b(source, roi1);
		patchS = b.clone();
	}

	// the difference between the patches, the lower the value, the better the patch is
	return patchS;
}


cv::Mat propagateRight(int x, int y, 
					Mat source, Mat sourceExtend,
					Mat nnf) {
	Mat patchS;

	// find the location of the pixel in the target that is being referenced by nnf
	Point p = nnf.at<Point>(y,x+1) + Point(-1,0);
	if(p.x < 0){p += Point(1,0);}
	
	// create a patch with the extended border if iterating close enough to edges
	if(p.x < 3 || p.y < 3 || p.x >= nnf.cols-3 || p.y >= nnf.rows-3){
		Rect roi1(p.x, p.y, 7, 7);
		Mat b(sourceExtend, roi1);
		patchS = b.clone();
	}
	// create a patch for target image
	else{
		Rect roi1(p.x-3, p.y-3, 7, 7);
		Mat b(source, roi1);
		patchS = b.clone();
	}

	// the difference between the patches, the lower the value, the better the patch is
	return patchS;
}


cv::Mat propagateDown(int x, int y, 
					Mat source, Mat sourceExtend,
					Mat nnf) {
	Mat patchS;

	// find the location of the pixel in the target that is being referenced by nnf
	Point p = nnf.at<Point>(y+1,x) + Point(0,-1);
	if(p.y < 0){p += Point(0,1);}
	// create a patch with the extended border if iterating close enough to edges
	if(p.x < 3 || p.y < 3 || p.x >= nnf.cols-3 || p.y >= nnf.rows-3){
		Rect roi1(p.x, p.y, 7, 7);
		Mat b(sourceExtend, roi1);
		patchS = b.clone();
	}
	// create a patch for target image
	else{
		Rect roi1(p.x-3, p.y-3, 7, 7);
		Mat b(source, roi1);
		patchS = b.clone();
	}

	// the difference between the patches, the lower the value, the better the patch is
	return patchS;
}







