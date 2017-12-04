#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
using namespace cv;
using namespace cv::dnn;
#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

void print_mat(InputArray im, std::string name) {
  
  Mat img = im.getMat();

  std::cout << name << ": "  
            << "size:"<< img.size() << " Rows X Cols=[" << img.rows << ":" << img.cols << "] " 
            << " type:" << img.type() << " channels:" << img.channels() 
            << " depth:" << img.depth() << " dims:" << img.dims << std::endl; 
 
}

/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
    print_mat(probBlob, "probBlob");

    Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
    print_mat(probBlob, "probBlob");

    Point classNumber;
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);

    std::cout << "Class prob:" << *classProb << " Point: " 
              << classNumber.x << ":" << classNumber.y << std::endl ;

    *classId = classNumber.x;
}


static std::vector<String> readClassNames(const char *filename = "../models/synset_words.txt")
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

  

int main(int argc, char **argv)
{
    CV_TRACE_FUNCTION();
    String modelTxt = "../models/bvlc_googlenet.prototxt";
    String modelBin = "../models/bvlc_googlenet.caffemodel";
    String imageFile = (argc > 1) ? argv[1] : "../dataset/horseback_riding.jpg";
    Net net;
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
    Mat img = imread(imageFile);
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }


    //GoogLeNet accepts only 224x224 BGR-images
    Mat inputBlob = blobFromImage(img, 1.0f, Size(224, 224),
                                  Scalar(104, 117, 123), false);   //Convert Mat to batch of images

    print_mat(img, "img");
    std::cout << "Input blob size:"<< inputBlob.size()<< " RowsxCols=[" << inputBlob.rows << ":" << inputBlob.cols << "] " 
              << " type:" << inputBlob.type() << " channels:" << inputBlob.channels() 
              << " depth:" << inputBlob.depth() << " dims:" << inputBlob.dims << std::endl; 

    //std::cout << "inputBlob = " << std::endl << " " << inputBlob << std::endl << std::endl;
    
    

    std::cout << "inputBlob.dims = " << inputBlob.dims << " inputBlob.size = [";
for(int i = 0; i < inputBlob.dims; ++i) {
    if(i) std::cout << " X ";
    std::cout << inputBlob.size[i];
}
std::cout << "] inputBlob.channels = " << inputBlob.channels() << std::endl;




    Mat prob;
    cv::TickMeter t;
    for (int i = 0; i < 10; i++)
    {
        CV_TRACE_REGION("forward");
        net.setInput(inputBlob, "data");        //set the network input
        t.start();
        prob = net.forward("prob");                          //compute output
        t.stop();
    }

    print_mat(prob, "prob");
 

    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);//find the best class
    std::vector<String> classNames = readClassNames();
    std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
    return 0;
} //main






