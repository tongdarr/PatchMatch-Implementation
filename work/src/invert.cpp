
// project
#include "invert.hpp"

using namespace cv;

Mat cgraInvertImage(const Mat& m) {
	// NOTE: this implemnetation is only suitable
	// for a 3-channel ubyte-format image

	Mat r = m.clone();

	// manually iterate over all pixels
	for (auto it = r.begin<Vec3b>(); it != r.end<Vec3b>(); ++it) {
		(*it)[0] = 255 - (*it)[0];
		(*it)[1] = 255 - (*it)[1];
		(*it)[2] = 255 - (*it)[2];
	}

	return r;
}