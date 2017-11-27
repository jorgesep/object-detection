#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <boost/program_options.hpp>
#include "opencv2/ximgproc/segmentation.hpp"

#include "imgreader.h"


using namespace std;
using namespace cv;
using namespace cv::dnn;
namespace po = boost::program_options;
namespace cvseg = cv::ximgproc::segmentation;

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb) {
  Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
  Point classNumber;
  minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
  *classId = classNumber.x;
}


static std::vector<String> readClassNames(const char *filename = "/home/courses/student48/project/dataset/dnn/synset_words.txt")
{
  std::vector<String> classNames;
  std::ifstream fp(filename);
  if (!fp.is_open())
  {
    std::cerr << "File with classes labels not found: " << filename << std::endl;
    exit(-1);
  }
  std::string name;
  while (!fp.eof())
  {
    std::getline(fp, name);
    if (name.length())
      classNames.push_back( name.substr(name.find(' ')+1) );
  }
  fp.close();
  return classNames;
}


void display_usage()
{
  cout << "------------------------------------------------------------" << endl;
  cout << "Experimenting HPC with image object detection ...           " << endl;
  cout << "------------------------------------------------------------" << endl;
  cout << "Output bounding boxes on identified objects.                " << endl;
  cout << "OpenCV Version : "  << CV_VERSION << endl << endl ;
  cout << "Example:                                                    " << endl;
  cout << "./obj_det -i <image|video file>                             " << endl << endl;
  cout << "------------------------------------------------------------" << endl <<endl;
}


int main(int argc, char **argv) {

  //Parse console parameters
  po::options_description desc("Image Object Detection. v1.0.0");
  desc.add_options()
    ("help,h", "Display this help message")
    ("input-files", po::value< std::vector<std::string> >(), "Input files")
    ("show,s","Show image window")
    ("version,v", "Display the version number");

  po::positional_options_description p;
  p.add("input-files", -1);
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);

  if ( vm.empty() || vm.count("help") ) {
      //display_usage();
      cout << desc << "\n";
      return 0;
  }



  vector<std::string> files;
  if(vm.count("input-files")){
//      std::vector<std::string> files = vm["input-files"].as< std::vector<std::string> >();
      files = vm["input-files"].as< vector<string> >();

#ifdef __debug__
      for(std::string file : files){
          std::cout << "Input file " << file << std::endl;
      }
      cout << "Input file size:" << files.size() << endl ;
#endif

  }

  bool show_window = false;
  if ( vm.count("show") ){
      show_window = true;
      /// Create Windows
      namedWindow("OBJDET", CV_WINDOW_NORMAL);
      moveWindow("OBJDET",50,50);
  }

  //Verify input name is a video file or directory with image files.
  FrameReader *input_frame;
  try {
      input_frame = FrameReaderFactory::create_frame_reader(files);
  } catch (...) {
      cout << "Invalid file name "<< endl;
      return 0;
  }

  // create Selective Search Segmentation Object using default parameters
  Ptr<cvseg::SelectiveSearchSegmentation> ss = cvseg::createSelectiveSearchSegmentation();

  Mat im;
  int cnt=0;

  // main loop
  for(;;)
  {
    // get sequence of frames to process.
    im = Scalar::all(0);
    input_frame->getFrame(im);
    if (im.empty()) break;

    cnt = input_frame->getFrameCounter();

    // set input image on which we will run segmentation
    ss->setBaseImage(im);

    // high recall but slow Selective Search method
    ss->switchToSelectiveSearchQuality();

    // run selective search segmentation on input image
    vector<Rect> rects;
    ss->process(rects);
    cout << "Total Number of Region Proposals: " << rects.size() << endl;


    // number of region proposals to show
    int numShowRects = 100;
    // increment to increase/decrease total number
    // of reason proposals to be shown
    int increment = 50;


  if (show_window) {
      Mat img ;
      im.convertTo(img, CV_8UC3);


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

      char key  = 0;
      int delay = 25;
      key = (char)waitKey(delay);
      // Exit program
      if( key == 27 ) break;


      if ( key == 32) {
          bool pause = true;
          while (pause)
          {
              key = (char)waitKey(delay);
              if (key == 32)
                  pause = false;
              // if (key == 112)
              //     cout << "Number of frame: " << i << "Files: " << fileName(gt_it->second) << " : " << fileName(fg_it->second) << endl;
          }
      }


  }

  if (cnt >= input_frame->getNFrames())
    break;

 
  }


  // if (inputName.empty()) {
  //   cmd.printMessage();
  //   return -1;
  // }

  //cout << "Input1 : " << inputName << endl;
  
  //cout << "Path: " << inputPath << endl;
  //cout << "Input:" << img << "|" << endl;


    // CV_TRACE_FUNCTION();
    // String modelTxt = "/home/courses/student48/project/dataset/dnn/bvlc_googlenet.prototxt";
    // String modelBin = "/home/courses/student48/project/dataset/dnn/bvlc_googlenet.caffemodel";
    // String imageFile = (argc > 1) ? argv[1] : "/home/courses/student48/project/dataset/dnn/space_shuttle.jpg";
    // Net net;
    // try {
    //     net = dnn::readNetFromCaffe(modelTxt, modelBin);
    // }
    // catch (cv::Exception& e) {
    //     std::cerr << "Exception: " << e.what() << std::endl;
    //     if (net.empty())
    //     {
    //         std::cerr << "Can't load network by using the following files: " << std::endl;
    //         std::cerr << "prototxt:   " << modelTxt << std::endl;
    //         std::cerr << "caffemodel: " << modelBin << std::endl;
    //         std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
    //         std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
    //         exit(-1);
    //     }
    // }
    // Mat img = imread(imageFile);
    // if (img.empty())
    // {
    //     std::cerr << "Can't read image from the file: " << imageFile << std::endl;
    //     exit(-1);
    // }
    // //GoogLeNet accepts only 224x224 BGR-images
    // Mat inputBlob = blobFromImage(img, 1.0f, Size(224, 224),
    //                               Scalar(104, 117, 123), false);   //Convert Mat to batch of images
    // Mat prob;
    // cv::TickMeter t;
    // for (int i = 0; i < 10; i++)
    // {
    //     CV_TRACE_REGION("forward");
    //     net.setInput(inputBlob, "data");        //set the network input
    //     t.start();
    //     prob = net.forward("prob");                          //compute output
    //     t.stop();
    // }
    // int classId;
    // double classProb;
    // getMaxClass(prob, &classId, &classProb);//find the best class
    // std::vector<String> classNames = readClassNames();
    // std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
    // std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    // std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
  return 0;
} //main






