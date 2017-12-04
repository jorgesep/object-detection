#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <fstream>
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream, std::stringbuf
#include <cstdlib>
#include <sys/time.h>
#include "utils.h"
#include <time.h>

using namespace cv;



double get_current_timestamp()
{
  struct timeval curt;
  gettimeofday(&curt, NULL);
  return (double)curt.tv_sec + ((double)curt.tv_usec)/1000000.0;
}

void timestamp() {
  time_t _tm =time(NULL );

  struct tm * curtime = localtime ( &_tm );
  std::cout << asctime(curtime) << std::endl;
  return;
}



void print_mat(InputArray im, std::string name) {

  Mat img = im.getMat();

  std::cout << name << "= "
            << "size(width x height):"<< img.size() << " Rows X Cols=[" << img.rows << ":" << img.cols << "] "
            << " type:" << img.type() << " channels:" << img.channels()
            << " depth:" << img.depth() << " dims:" << img.dims << std::endl;

}


void image_show(InputArray _src, const std::vector<Rect>& rects){

  Mat img = _src.getMat();

  // number of region proposals to show
  int numShowRects = 100;
  // increment to increase/decrease total number
  // of reason proposals to be shown
  int increment = 50;

  // iterate over all the region proposals
  for(int i = 0; i < rects.size(); i++) {
      if (i < numShowRects) {
          rectangle(img, rects[i], Scalar(0, 255, 0));
      }
      else {
          break;
      }
  }

  imshow("OBJDET", img);

  // Stop window
  char key=0;
  int delay =25;
  key = (char)waitKey(delay);
  if( key == 27 )
      return ;

  // pause program in with space key
  if ( key == 32) {
      bool pause = true;
      while (pause)
      {
          key = (char)waitKey(delay);
          if (key == 32) pause = false;

          // save frame with return key
          if (key == 13) {
              std::stringstream str;
              str << "objdet.png" ;
              imwrite( str.str()  , img  );

              str.str("") ;
          }
      }
  }
}
