#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <boost/program_options.hpp>
#include "opencv2/ximgproc/segmentation.hpp"
#include <sys/time.h>
#include <mpi.h>
#include "imgreader.h"
#include "googlenet.h"
#include "dnn_algorithmI.h"
#include "utils.h"
#include <vector>
#include <math.h>
#include <stddef.h>
#include <map>

#define send_data_tag 2001
#define return_data_tag 2002


using namespace std;
using namespace cv;
using namespace cv::dnn;
namespace po = boost::program_options;
namespace cvseg = cv::ximgproc::segmentation;

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



/* Find best class for the blob (i. e. class with maximal probability) */
static void getMaxClass(const Mat &probBlob, int *classId, double *classProb) {
  Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
  Point classNumber;
  minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
  *classId = classNumber.x;
}


static std::vector<String> readClassNames(const char *filename = "synset_words.txt")
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

//--------------------------------------------------------------------------
// Makes image classification of Rects received.
//--------------------------------------------------------------------------
void rectangle_classification(InputArray im, 
                              const std::vector<Rect> & rects, 
                              std::string model,
                              std::map<int,prob_t> & res ){

  Mat img = im.getMat();
  int num_rects = rects.size();

  // Creates object classifier
  Ptr<DeepNeuralNetworkAlgorithmI> net = createGoogLeNetClassifier();

  // Load Caffe model
  net->LoadModel(model);

  double t0,t1;

  for(int i=0; i<num_rects; i++) {

    t0 = get_current_timestamp();
    Mat roi = img(rects[i]);

    //  image classification using GoogLeNet 
    //  trained network from Caffe model zoo.
    Mat Prob;
    net->Process(roi,Prob);

    int classId;
    double classProb;
    net->GetClassProb(&classId, &classProb);

    t1 = get_current_timestamp();
    if (classProb > 0.25 ) {
      if( res.find(classId) == res.end() ) {
        // not found
        res.insert( std::pair<int,prob_t>(classId,prob_t(rects[i],classId,classProb)) );
      } else {
        // found
        prob_t temp_prob = res[classId];
        if(classProb > res[classId].Prob) {
          res[classId].rect = rects[i];
          res[classId].Prob = classProb;
        } 
      }
 
      std::cout << " Elapsed: " <<  (t1-t0)*1000.0 << ":" << net->ElapsedTimeAsString() 
                << " classId: " << classId << " clasProb:" 
                << classProb << " className: " << net->className() << '\n';

    }//End prob less than 25%
  }//End for Rects received
}


int main(int argc, char **argv) {

    //-------------------------------------------------------------------------
    // Parse console input parameters
    //-------------------------------------------------------------------------
    po::options_description desc("Image Object Detection. v1.0.0");
    desc.add_options()
      ("help,h", "Display this help message")
      ("input-files", po::value< std::vector<std::string> >(), "Input files")
      ("show,s","Show image window")
      ("model,m","Path to ddn models")
      ("version,v", "Display the version number");
  
    po::positional_options_description p;
    p.add("input-files", -1);
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
  
    if ( vm.empty() || vm.count("help") ) {
        cout << desc << "\n";
        return 0;
    }
  
    vector<std::string> files;
    if(vm.count("input-files")){
        files = vm["input-files"].as< vector<string> >();
  
    #ifndef NDEBUG 
        for(std::string file : files) {
            std::cout << "Input file " << file << std::endl;
        }
        cout << "Input file size:" << files.size() << endl ;
    #endif
    }
  
    string model_path("");
    if( vm.count("model") ){
      model_path = vm["model"].as<string>();
      #ifndef NDEBUG 
      std::cout << "Path to models " << model_path << std::endl;
      #endif
    }
  
    // Create Windows
    bool show_window = false;
    if ( vm.count("show") ){
        show_window = true;
        namedWindow("OBJDET", CV_WINDOW_NORMAL);
        moveWindow("OBJDET",50,50);
    }



  //---------------------------------------------------------------------------
  // MPI section.
  //---------------------------------------------------------------------------
  int iam = 0, np = 1;
  int avg_rects_per_process;
  int num_procs, rank;
  int procs_id;
  int name_len;
  int root_process=0;
  int i_proc, start_row, end_row, num_rows;
  int num_rects_to_send, num_rects_received, num_rects_to_receive;
  int ierr;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Status status;

  vector<Rect> rects_child_process;
  size_t elemsize;
  int sizes[3];


  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &procs_id);

  // Get the name of the processor
  MPI_Get_processor_name(processor_name, &name_len);


  //---------------------------------------------------------------------------
  // Struct to share messages between nodes.
  //---------------------------------------------------------------------------
  MPI_Datatype mpi_classification_type;
  int blen[2]               = {1,1};
  MPI_Datatype types[2]     = {MPI_INT, MPI_DOUBLE};
  MPI_Aint displacements[2] = { offsetof(classification_type, id), 
                                offsetof(classification_type, prob) };

  MPI_Type_struct( 2, blen, displacements, types, &mpi_classification_type );
  MPI_Type_commit(&mpi_classification_type);

  MPI_Datatype mpi_rect_type;
  int          mrect_len[4]    = {1,1,1,1};
  MPI_Datatype mrect_types[4]  = {MPI_INT, MPI_INT,MPI_INT,MPI_INT};
  MPI_Aint     mrect_offset[4] = { offsetof(rect_type, x), 
                                   offsetof(rect_type, y), 
                                   offsetof(rect_type, w), 
                                   offsetof(rect_type, h) };

  MPI_Type_struct( 4, mrect_len, mrect_offset, mrect_types, &mpi_rect_type );
  MPI_Type_commit(&mpi_rect_type);

  //---------------------------------------------------------------------------
  // Inside of MPI root process for reading the input images to be segmented.
  //---------------------------------------------------------------------------
  if(procs_id == root_process) {

    timestamp ( );
    std::cout << "\n";
    std::cout << "  Master process " << processor_name << std::endl;
    std::cout << "  The number of processes is " << num_procs << "\n";

    //-----------------------------------------------------------------------
    // Send models to child processes.
    //-----------------------------------------------------------------------
//    int msize = model_path.size();
//    for (i_proc = 1; i_proc < num_procs; i_proc++) {
//      ierr = MPI_Send( &msize, 1 , MPI_INT, i_proc, 
//                     send_data_tag, MPI_COMM_WORLD);
//      ierr = MPI_Send( model_path.c_str(), msize, MPI_CHAR, i_proc, 
//                     send_data_tag, MPI_COMM_WORLD);
//    }

    //-------------------------------------------------------------------------
    // OpenCV initialization
    //-------------------------------------------------------------------------
    FrameReader *input_frame;
    try {
        input_frame = FrameReaderFactory::create_frame_reader(files);
    } catch (...) {
        cout << "Invalid file name "<< endl;
        return 0;
    }
  
    // create Selective Search Segmentation Object using default parameters
    Ptr<cvseg::SelectiveSearchSegmentation> 
    ss = cvseg::createSelectiveSearchSegmentation();

    // local variables to process 0
    Mat im;
    int cnt=0;
    double t1,t2,t3,t10,t11;
    t1 = get_current_timestamp();

    // main loop
    for(;;){

      // timing main loop
      t2 = get_current_timestamp();

      // get single frame.
      im = Scalar::all(0);
      input_frame->getFrame(im);

      // no more frames to process.
      if (im.empty()) break;

      // get frame counter
      cnt = input_frame->getFrameCounter();

      // resize image
      int newHeight = 224;
      int newWidth = im.cols*newHeight/im.rows;
      resize(im, im, Size(newWidth, newHeight));

      //#ifndef NDEBUG 
      //Vec3b intensity = im.at<Vec3b>(210,310);
      //std::cout << "Master process: " << im.size() << " " 
      //          << "B:" << (int)intensity[0] << " " 
      //          << "G:" << (int)intensity[1] << " "
      //          << "R:" << (int)intensity[2] << '\n';
      //vector<int> compression_params;
      //compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
      //compression_params.push_back(9);
      //imwrite("master.png",im,compression_params);
      //print_mat(im, "Master IMG");
      //#endif


      // speed-up using multithreads
      //setUseOptimized(true);
      //setNumThreads(4);

      //-----------------------------------------------------------------------
      // Send image to child processes.
      //-----------------------------------------------------------------------
      sizes[0] = newHeight;
      sizes[1] = newWidth;
      sizes[2]=im.elemSize();
      for (i_proc = 1; i_proc < num_procs; i_proc++) {

        //std::cout << "  Master process:  " << processor_name 
        //          << " Sending img size " << sizes[0] << "X" << sizes[1] 
        //          << " to child process " << i_proc << '\n';

        ierr = MPI_Send( sizes, 3 , MPI_INT, i_proc, 
                       send_data_tag, MPI_COMM_WORLD);
        ierr = MPI_Send( im.data, newHeight*newWidth*3, MPI_CHAR, i_proc, 
                       send_data_tag, MPI_COMM_WORLD);
      }
      //-----------------------------------------------------------------------
      // Image Segmentation 
      //-----------------------------------------------------------------------

      // set input image on which we will run segmentation
      ss->setBaseImage(im);

      // high recall but slow Selective Search method
      //ss->switchToSelectiveSearchQuality();
      ss->switchToSelectiveSearchFast();

      // run selective search segmentation on input image
      vector<Rect> rects;
      ss->process(rects);

      int num_rects = 20 ;
      //int num_rects = rects.size() ;
      avg_rects_per_process = num_rects / num_procs;
      std::cout << "  Master process:  " << processor_name
                << " Number of Region Proposals: " << rects.size() 
                << " Number of Rects per process: " 
                << avg_rects_per_process <<  '\n';

      //-----------------------------------------------------------------------
      // distribute a portion of rects vector to each child process 
      //-----------------------------------------------------------------------
      for (i_proc = 1; i_proc < num_procs; i_proc++) {

        start_row = i_proc      * avg_rects_per_process + 1;
        end_row   = (i_proc + 1)* avg_rects_per_process;

        if ((num_rows - end_row) < avg_rects_per_process)
          end_row = num_rects - 1;
        num_rects_to_send = end_row - start_row + 1;

        std::cout << "  Master process:  " << processor_name 
                  << " sending to process " << i_proc  << " " 
                  << num_rects_to_send << " Rects " << "[" << start_row 
                  << ":" << end_row << "]\n";

        ierr = MPI_Send( &num_rects_to_send, 1 , MPI_INT, i_proc, 
                         send_data_tag, MPI_COMM_WORLD);

        ierr = MPI_Send( &rects[start_row], num_rects_to_send*sizeof(Rect), MPI_BYTE, i_proc, 
                         send_data_tag, MPI_COMM_WORLD);


        std::cout << "  Master process:  " << processor_name 
                  << "  Rect[" << start_row << "] " 
                  << " x="      << rects[start_row+2].x 
                  << " y="      << rects[start_row+2].y 
                  << " width="  << rects[start_row+2].width
                  << " heigth=" << rects[start_row+2].height << '\n' ;

      }//end distribute for 

      if (cnt >= input_frame->getNFrames())
        break;

    }//end main loop
    t3 = get_current_timestamp();
    std::cout << "  Master process:  " << processor_name
              << " Img frame " << cnt
              << " Time elapsed:" 
              << (t3-t1)*1000.0 << " " 
              << (t3-t2)*1000.0 << " ms "  << std::endl;
  }
  else {
    std::cout << "\n"
              << "  Child process  " << processor_name << std::endl
              << "  process id: " << procs_id << "/" << num_procs << std::endl;

    //--------------------------------------------------------------------------
    // Receives model path
    //--------------------------------------------------------------------------
//    int msize;
//    ierr = MPI_Recv( &msize, 1, MPI_INT,
//           root_process, send_data_tag, MPI_COMM_WORLD, &status);
//    char* rec_buf;
//    rec_buf = (char *) malloc(msize+1);
//    ierr = MPI_Recv( rec_buf, msize+1, MPI_CHAR,
//           root_process, send_data_tag, MPI_COMM_WORLD, &status);
//    std::string model_path(rec_buf);
//    free(rec_buf);
//    std::cout << "  Child process: " << procs_id << " "
//              << processor_name << " model_path: " << model_path << '\n';
    std::string model_path(""); 
    //--------------------------------------------------------------------------
    // Receives image from root process.
    //--------------------------------------------------------------------------
    Mat img;
    ierr = MPI_Recv( sizes, 3, MPI_INT,
           root_process, send_data_tag, MPI_COMM_WORLD, &status);
    //std::cout << "  Child process: " << procs_id << " "
    //          << processor_name << " Reading image of sizes[0]Xsizes[1] " 
    //          << sizes[0] << "X"  << sizes[1] << '\n';

    img.create(sizes[0],sizes[1],CV_8UC3);

    ierr = MPI_Recv( img.data, sizes[0]*sizes[1]*3, MPI_CHAR,
           root_process, send_data_tag, MPI_COMM_WORLD, &status);

    //std::cout << "  Child process: " << procs_id << " "
    //          << processor_name << " Reading image of " 
    //          << img.rows << "X" << img.cols << '\n';

    //#ifndef NDEBUG 
    //Vec3b intensity = img.at<Vec3b>(220,310);
    //std::cout << "Child process: " << procs_id << " " << img.size() << " "
    //          << "B:" << (int)intensity[0] << " " 
    //          << "G:" << (int)intensity[1] << " "
    //          << "R:" << (int)intensity[2] << '\n';
    //vector<int> compression_params;
    //compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    //compression_params.push_back(9);
    //imwrite("child.png",img,compression_params);
    //print_mat(img, "Child IMG");
    //#endif


    //--------------------------------------------------------------------------
    // Receives Rects from root process.
    //--------------------------------------------------------------------------
    ierr = MPI_Recv( &num_rects_to_receive, 1, MPI_INT, root_process, 
                     send_data_tag, MPI_COMM_WORLD, &status);

    // make space for rects num_rows_to_receive
    rects_child_process.resize(num_rects_to_receive);

    ierr = MPI_Recv( &rects_child_process[0], num_rects_to_receive*sizeof(Rect), MPI_BYTE, 
                     root_process, send_data_tag, MPI_COMM_WORLD, &status);

    num_rects_received = num_rects_to_receive;
    std::cout << "  Child process: " << procs_id << " "
              << processor_name << " Received " << num_rects_received 
              << " rectangles to process\n";

    int cnt_rect = 2;
    std::cout << "  Child process: " << procs_id << " "
              << processor_name 
              << "  Rect[0"  << cnt_rect << "] " 
              << " x="      << rects_child_process[cnt_rect].x 
              << " y="      << rects_child_process[cnt_rect].y 
              << " width="  << rects_child_process[cnt_rect].width
              << " heigth=" << rects_child_process[cnt_rect].height << '\n' ;

    //--------------------------------------------------------------------------
    // Makes image classification with received rectangles.
    //--------------------------------------------------------------------------
    std::map<int,prob_t> m_prob;
    double t0,t1;
    t0 = get_current_timestamp();
    rectangle_classification(img,rects_child_process, model_path, m_prob);  
    t1 = get_current_timestamp();
    std::cout << "  Child process: " << procs_id << " " << processor_name 
              << " Elapsed: " <<  (t1-t0)*1000.0 << "\n";
 

//    //--------------------------------------------------------------------------
//    // Makes image classification of Rects received.
//    //--------------------------------------------------------------------------
//
//    // Creates object classifier
//    Ptr<DeepNeuralNetworkAlgorithmI> net = createGoogLeNetClassifier();
//    // Load Caffe model
//    net->LoadModel(model_path);
//
//    double t0,t1;
//    for(int i=0; i<num_rects_received; i++) {
//      t0 = get_current_timestamp();
//      Mat roi = img(rects_child_process[i]);
//
//      //  image classification using GoogLeNet 
//      //  trained network from Caffe model zoo.
//      Mat Prob;
//      net->Process(roi,Prob);
//
//      int classId;
//      double classProb;
//      net->GetClassProb(&classId, &classProb);
//
//      t1 = get_current_timestamp();
//      if (classProb > 0.25 ) {
//        std::cout << "  Child process: " << procs_id << " " << processor_name 
//                  << " Elapsed: " <<  (t1-t0)*1000.0 << ":" << net->ElapsedTimeAsString() 
//                  << " classId: " << classId << " clasProb:" 
//                  << classProb << " className: " << net->className() << '\n';
//      }//End prob less than 25%
//    }//End for Rects received
  }// End child processes


  MPI_Type_free(&mpi_rect_type);
  MPI_Type_free(&mpi_classification_type);
  // Finalize the MPI environment.
  MPI_Finalize();


  return 0;
} //main






