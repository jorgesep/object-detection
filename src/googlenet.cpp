/*******************************************************************************
 * This file is part of libraries to evaluate performance of Background 
 * Subtraction algorithms.
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <iostream>
#include <fstream> 

using namespace cv;
//using namespace cv::dnn;


#include "googlenet.h"


GoogLeNetClassifier::GoogLeNetClassifier() {
  path_models = "";
  elapsedTime = 0;
  class_name  = "";
  class_id    = 0;
  class_prob  = 0;
}

void GoogLeNetClassifier::Initialization() {
}

std::string GoogLeNetClassifier::PrintParameters()
{
    std::stringstream str;
    str
    << "# " ;
    return str.str();

}



void GoogLeNetClassifier::LoadModel( std::string src_dir ) {

  if (! src_dir.empty() )
    path_models = src_dir + "/";


  std::string modelTxt = path_models + "bvlc_googlenet.prototxt";
  std::string modelBin = path_models + "bvlc_googlenet.caffemodel";

  try {
      net = dnn::readNetFromCaffe(modelTxt, modelBin);
  }
  catch (cv::Exception& e) {
      std::cerr << "Exception: " << e.what() << std::endl;
      if (net.empty())
      {
          std::cerr << "Can't load network by using the following files: " << std::endl;
          std::cerr << "prototxt:   " << modelTxt << std::endl;
          std::cerr << "caffemodel: " << modelBin << std::endl;
          std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
          std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
          exit(-1);
      }
  }

  // Load class labels
  readClassNames();


}

void GoogLeNetClassifier::Process(InputArray _src, OutputArray _out){

  Mat img = _src.getMat();

  _out.create(1,1000,CV_32F);
  prob = _out.getMat();

  //GoogLeNet accepts only 224x224 RGB-images
  //Convert Mat to batch of images
  Mat inputBlob = dnn::blobFromImage(img, 1.0f, Size(224, 224),
                                  Scalar(104, 117, 123),false);

  TickMeter t;
  for (int i = 0; i < 10; i++) {
        
    //CV_TRACE_REGION("forward");

    // Sets the new value for the layer output blob.
    net.setInput(inputBlob, "data"); 

    t.start();
    // Runs forward pass to compute output of layer with name outputName.
    prob = net.forward("prob");
    t.stop();
  }

  elapsedTime = (double)t.getTimeMilli() / t.getCounter();
  numberIterations = t.getCounter();

}


void GoogLeNetClassifier::GetClassProb(int* classId, double* classProb){

  getClassMaxProb(prob, classId, classProb);
  class_id   = *classId;
  class_prob = *classProb;

  class_name = classNames.at(class_id);

}

std::string GoogLeNetClassifier::ElapsedTimeAsString()
{
    std::stringstream elapsed;
    elapsed << elapsedTime;
    return elapsed.str();
}

void GoogLeNetClassifier::readClassNames() {

  if (net.empty()) {
    std::cerr << "Modules not loaded " << std::endl ;
    return ;
  }

  std::string filename = path_models + "synset_words.txt" ;

  std::ifstream fp( filename.c_str() );
  if (!fp.is_open()) {
      std::cerr << "File with classes labels not found: " << filename << std::endl;
      exit(-1);
  }

  std::string name;
  while (!fp.eof()) {
      std::getline(fp, name);
      if (name.length())
          classNames.push_back( name.substr(name.find(' ')+1) );
  }
  fp.close();

}

// Find best class for the blob (i. e. class with maximal probability)
void GoogLeNetClassifier::getClassMaxProb(InputArray blob, int *classId, double *class_prob) {
  
  //reshape the blob to 1x1000 matrix
  Mat probMat = blob.getMat().reshape(1,1);
  //Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix

  Point classNumber;

  // Finds the global minimum and maximum in an array.
  minMaxLoc(probMat, NULL, class_prob, NULL, &classNumber);
    
  *classId = classNumber.x;

}

void GoogLeNetClassifier::print_mat(InputArray im, std::string name) {

  Mat img = im.getMat();

  std::cout << name << ": "
            << "size:"<< img.size() << " Rows X Cols=[" << img.rows << ":" << img.cols << "] "
            << " type:" << img.type() << " channels:" << img.channels()
            << " depth:" << img.depth() << " dims:" << img.dims << std::endl;

}



Ptr<DeepNeuralNetworkAlgorithmI> createGoogLeNetClassifier() {
    Ptr<DeepNeuralNetworkAlgorithmI> c = makePtr<GoogLeNetClassifier>();
    return c;
}

