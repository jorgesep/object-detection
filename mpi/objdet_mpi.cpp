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
#include <omp.h>

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
// Select higher probability of the class Id.
//--------------------------------------------------------------------------
//void select_max_class(const std::map<int, rectangle_t>& in, 
//                            std::map<int, rectangle_t>& out ){
//
//}

//--------------------------------------------------------------------------
// Join classifications.
//--------------------------------------------------------------------------
//void merge_process_classification(const std::vector<rectangle_t>& in, 
//                                        std::map<int, rectangle_t>& out ){
//
//  std::vector<rectangle_t>::const_iterator it = in.begin();
//  for(; it != in.end(); ++it) {
//
//      if( out.find(it->id) == out.end() ) {
//        // not found
//        out.insert( std::pair<int,rectangle_t>(it->id,*it)) ;
//      }
//      else {
//        if( it->prob > out[it->id].prob) {
//          out[it->id].rect = it->rect;
//          out[it->id].prob = it->prob;
//       }
//  }
//}
//}

std::string point_as_string(InputArray im, int x, int y) {

  Mat img = im.getMat();

  Vec3b intensity = img.at<Vec3b>(y, x);
  uchar blue  = intensity.val[0];
  uchar green = intensity.val[1];
  uchar red   = intensity.val[2];

  stringstream _point ;
  _point    << " Point(" << x << "," << y << ")= RGB[" 
            << (int)red << ":" << (int)blue << ":" << (int)green << "]";

  return _point.str();
}


//--------------------------------------------------------------------------
// Makes image classification of Rects received.
//--------------------------------------------------------------------------
void rectangle_classification(InputArray im, 
                              const Rect & rect, 
                              std::string model,
                              rectangle_t & processed_rect ){

  Mat img = im.getMat();
  // Creates object classifier

//  std::cout << "  function rectangle_classification: (" << getpid() << ") "
//            << " image debug : " << point_as_string(im,rect.x,rect.y)
//            << '\n';
 

  Ptr<DeepNeuralNetworkAlgorithmI> net = createGoogLeNetClassifier();

  // Load Caffe model
  net->LoadModel(model);

  double t0,t1;

  t0 = get_current_timestamp();

  Mat roi = img(rect);

  //  image classification using GoogLeNet 
  //  trained network from Caffe model zoo.
  Mat Prob;
  net->Process(roi,Prob);

  int classId;
  double classProb;
  net->GetClassProb(&classId, &classProb);

  t1 = get_current_timestamp();

  //std::cout << "  function rectangle_classification: "<< "(" << getpid() << ") "
  //          << " Classification result id:prob:rec " 
  //          <<  classId << ":" << classProb << ":" << rect  << "\n"; 
  
  if (classProb > 0.5 ) 
    processed_rect = rectangle_t(rect,classId,classProb);
}

//--------------------------------------------------------------------------
// Merge two overlapped regions with same object id
//--------------------------------------------------------------------------
void merge_two_rects(const rectangle_t &root, const rectangle_t &child, rectangle_t &dest){

  // union of rects.
  Rect rect     = root.rect | child.rect ;

  // save bigger probability.
  double u_prob = (root.prob > child.prob ? root.prob : child.prob);

  int  u_id     = root.id ;
  dest          = rectangle_t( rect, u_prob, u_id );
}


//--------------------------------------------------------------------------
// Makes image classification of Rects received.
//--------------------------------------------------------------------------
void object_classification(InputArray im, 
                           Rect rect, 
                           std::string model,
                           rectangle_t & res ){

  Mat img = im.getMat();

  // Creates object classifier
  Ptr<DeepNeuralNetworkAlgorithmI> net = createGoogLeNetClassifier();

  // Load Caffe model
  net->LoadModel(model);

  int id;
  double prob;
  net->GetClassProb(&id, &prob);

  res = rectangle_t(rect, id, prob);
}


//------------------------------------------------------------------------------
// Image Segmentation 
//------------------------------------------------------------------------------
static void image_segmentation(InputArray img, vector<Rect> & rects) {

  Mat im = img.getMat();

  //----------------------------------------------------------------------------
  // create Selective Search Segmentation Object using default parameters
  //----------------------------------------------------------------------------
  Ptr<cvseg::SelectiveSearchSegmentation> 
  ss = cvseg::createSelectiveSearchSegmentation();

  // set input image on which we will run segmentation
  ss->setBaseImage(im);

  // high recall but slow Selective Search method
  //ss->switchToSelectiveSearchQuality();
  ss->switchToSelectiveSearchFast();

  // run selective search segmentation on input image
  ss->process(rects);

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
  int num_procs, procs_id, name_len;
  int root_process=0;
  int i_proc, ierr;
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Status status;
  Mat child_img;
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

    // local variables to process 0
    Mat im;
    int cnt=0;
    double t1,t2,t3,t4,t1_root, t2_root;
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

        ierr = MPI_Send( sizes, 3 , MPI_INT, i_proc, 
                       send_data_tag, MPI_COMM_WORLD);
        ierr = MPI_Send( im.data, newHeight*newWidth*3, MPI_CHAR, i_proc, 
                       send_data_tag, MPI_COMM_WORLD);
      }

      //-----------------------------------------------------------------------
      // Image Segmentation 
      //-----------------------------------------------------------------------
      vector<Rect> rects;
      image_segmentation(im, rects);

      int num_rects = rects.size() ;
      int avg_rects_per_process = num_rects / num_procs;
      int remainder_rects_to_send = num_rects % num_procs;
      int num_rects_to_send = avg_rects_per_process;
      int num_rects_extra_to_send = avg_rects_per_process + 1;
      std::cout << "  Master process:  " << processor_name
                << " Number of Region Proposals: " << rects.size() 
                << " Number of Rects per process: " << avg_rects_per_process 
                <<  '\n';
      for (i_proc = 1; i_proc < num_procs; i_proc++) {
        // increase number of rectangles
        if(i_proc < remainder_rects_to_send )
          ierr = MPI_Send( &num_rects_extra_to_send, 1 , MPI_INT, i_proc,
                           send_data_tag, MPI_COMM_WORLD);
        else
          ierr = MPI_Send( &num_rects_to_send, 1 , MPI_INT, i_proc,
                           send_data_tag, MPI_COMM_WORLD);

     }


      /*Debug*/
      //for(int j=0; j<5; j++) {
      //    rectangle(im,rects[j], Scalar(0,0,255),1,8,0);
      //    rectangle(im,rects[num_rects-j], Scalar(255,0,0),1,8,0);
      //    putText(im, std::to_string(j),Point(rects[j].x, rects[j].y+rects[j].height), 
      //            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255), 1.0);
      //}
      //imwrite("image_debug.jpg",im);

      //-----------------------------------------------------------------------
      // Object recognition shared to other process. 
      //-----------------------------------------------------------------------
      std::vector<rectangle_t> vrects;

      t3 = get_current_timestamp();
      int checkpoint_cnt = 0;

      // loop over all rectangles.
      //================================================================== 
      for (int i_rects = 0; i_rects < num_rects; i_rects+= num_procs) {

        // delivers one rect for each process.
        //================================================================== 
        int number_process_per_iteration = num_procs;
        if ((num_rects - i_rects) < num_procs)
          number_process_per_iteration = (num_rects - i_rects);
        for (i_proc = 1; i_proc < number_process_per_iteration; i_proc++) {

           ierr = MPI_Send( &rects[(i_rects + i_proc)], sizeof(Rect), MPI_BYTE,
                           i_proc, send_data_tag, MPI_COMM_WORLD);
        }
//std::cout << "============= " << i_rects << "/"<< num_rects << " "  << i_rects+i_proc-1 << "======="<< max_number_processes << "\n";

        // root object detection.
        //================================================================== 
        t1_root = get_current_timestamp();
        rectangle_t root_rect ;
        rectangle_classification(im,rects[i_rects], model_path, root_rect);
        t2_root = get_current_timestamp();

        if ( root_rect != rectangle_type(Rect(0,0,0,0),0,0) ) {
          vrects.push_back(root_rect);

          std::cout << "  Master process:  " << processor_name
                    << " rectangle[" << i_rects << "] input="
                    << rects[i_rects] 
                    << " output=" << root_rect.id << ":" 
                    << root_rect.prob << ":" << root_rect.rect 
                    << " time elapsed: " <<  (t2_root-t1_root)*1000.0 << " ms"
                    <<  '\n';
        }

        // collect partial classification (child_rect) from the processes.
        //================================================================== 
        for (i_proc = 1; i_proc < number_process_per_iteration; i_proc++) {

          rectangle_t child_rect;
          ierr = MPI_Recv( &child_rect, sizeof(rectangle_t), MPI_BYTE, 
               MPI_ANY_SOURCE, return_data_tag, MPI_COMM_WORLD, &status);

          if ( child_rect != rectangle_type(Rect(0,0,0,0),0,0) ){

            rectangle_t local_rect = child_rect;
            rectangle_t prev_rect;
            stringstream del_rect("");

            // object id is already in the vector
            std::vector<rectangle_t>::iterator it_rect=vrects.begin() ;
            for(; it_rect!=vrects.end();) {
            
              // object inside vector and overlap
              if (( it_rect->id  == child_rect.id ) &&
                  ( it_rect->rect & child_rect.rect ).area() > 0){

                // merge it_rect in child_rect and remove it.
                child_rect += *it_rect;
                prev_rect   = *it_rect;
                
                // always returns the next valid iterator, 
                // if you erase the last element it will point to .end()
                it_rect = vrects.erase(it_rect);
                del_rect << " Deleted=" << prev_rect.id << ":" << prev_rect.prob << ":"  <<  prev_rect.rect ;
              }
              else 
                ++it_rect;
            }
            // re-insert child_rect
            vrects.push_back(child_rect);
            std::cout << "  Master process:  " << processor_name
                      << " Inserted rectangle from process " 
                      << status.MPI_SOURCE << " vector size: " << vrects.size() 
                      << " input=" << local_rect.id << ":" << local_rect.prob << ":"  <<  local_rect.rect 
                      << del_rect.str() 
                      << " output=" << child_rect.id << ":" << child_rect.prob << ":"  <<  child_rect.rect 
                      <<  '\n';

          }

//          // Finish iteration
//          Mat it1_mat = im;
//          Mat it2_mat = im;
//          int cnt_iteration = i_rects/num_procs;
//          vector<int> compression_params;
//          compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
//          compression_params.push_back(9);
//          stringstream str1, str2;
//          std::string file_iteration = files[cnt-1].substr ( files[cnt-1].rfind("/")+1 ,
//                                                             files[cnt-1].rfind(".")) + ".png";
//          str1  << cnt_iteration << "_1_" << i_rects << "-" << (i_rects+num_procs)<< "_" << file_iteration ;
//          str2  << cnt_iteration << "_2_" << i_rects << "-" << (i_rects+num_procs)<< "_" << file_iteration ;
//
//          for(int j=i_rects; j < i_rects+num_procs; j++){
//            rectangle(it1_mat, rects[j], Scalar(0,255,0), 1,8,0);
//            //rectangle(it2_mat, rects[j], Scalar(0,255,255), 1,8,0);
//          }
//          imwrite(str1.str(),it1_mat,compression_params);
//
//
//          std::cout << "----------------------------------------------------\n";
//          for(int j=0 ; j<vrects.size(); j++){
//            rectangle(it2_mat, vrects[j].rect, Scalar(0,0,255), 1,8,0);
//            
//                 putText(it2_mat, std::to_string(vrects[j].id), 
//                 Point(vrects[j].rect.x, 
//                       vrects[j].rect.y + vrects[j].rect.height*0.9), 
//                 FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255), 1.0);
//
//            std::cout << "  Master process:  " << processor_name
//                      << " VRECTS: " 
//            << vrects[j].id << ":" << vrects[j].prob << ":" << vrects[j].rect << ":" << vrects[j].rect.area() << '\n';
//          }
//          std::cout << "----------------------------------------------------\n";
//          imwrite(str2.str(),it2_mat,compression_params);

        }//end for: over all processes
//if (i_rects < num_procs*4)break;
        
        // save a checkpoint.
        int checkpoint = 6;
        if ((i_rects/num_procs)% checkpoint == 0) {
          Mat D;
          im.copyTo(D);
          for(int j=0 ; j<vrects.size(); j++){
            rectangle(D, vrects[j].rect, Scalar(0,0,255), 1,8,0);
            putText(D, std::to_string(vrects[j].id), 
            Point(vrects[j].rect.x, 
                  vrects[j].rect.y + vrects[j].rect.height*0.9), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255), 1.0);
          }
          std::string file_checkpoint = files[cnt-1].substr ( files[cnt-1].rfind("/")+1 ,
                                                              files[cnt-1].rfind(".")) + ".png";
          vector<int> compression_params;
          compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
          compression_params.push_back(9);
          stringstream str1;
          str1  << "checkpoint/" << checkpoint_cnt << "_" <<  i_rects << "_" << file_checkpoint ;
          imwrite(str1.str(),D,compression_params);
          checkpoint_cnt++;
        }
      }// end loop over all rectangles.

      // From this point I should have a vector with rectangles of objects.
      //================================================================== 

      // Sorting rectangle vectors.
      std::cout << "  Master process:  " << processor_name
                << " sorting vector of size " << vrects.size()
                <<  '\n';

      std::sort(vrects.begin(), vrects.end(), &rectangle_sorter);
      std::cout << "  Master process:  " << processor_name
                << " sort done vector of size " << vrects.size()
                <<  '\n';

      t4 = get_current_timestamp();

      std::cout << "  Master process:  " << processor_name 
                << " number of final rectangles: " << vrects.size()
                << " whole classification elapsed: " <<  (t4-t3)*1000.0 << "\n";
 
      std::vector<String> class_names = readClassNames();
      for(int j=0; j<vrects.size(); j++) {
      
        std::cout << "  Master process:  " << processor_name 
                  << " class : " << vrects[j].id << ":" << vrects[j].prob
                  << ":" << class_names.at(vrects[j].id) << " rect: "
                  << vrects[j].rect << '\n';

        if (j<6){
          std::string first_word_class_name = class_names.at(vrects[j].id).
                      substr(0, class_names.at(vrects[j].id).find_first_of(" "));
  
          stringstream class_name_label;
          class_name_label << first_word_class_name << ":" 
                   << fixed << setprecision(1) << vrects[j].prob*100 << '%'; 
          rectangle(im, vrects[j].rect, Scalar(0,0,255), 1,8,0);
          putText(im, class_name_label.str(), 
                   Point(vrects[j].rect.x, 
                         vrects[j].rect.y + vrects[j].rect.height*0.9), 
                   FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255), 1.0);
        }
      }

      //vector<int> compression_params;
      //compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
      //compression_params.push_back(9);

      stringstream str;
      std::string name = files[cnt-1].substr ( files[cnt-1].rfind("/")+1 ,
                                               files[cnt-1].rfind("."));

      str << "master." << cnt << "." << name ;
      imwrite(str.str(),im);
      std::cout << "  Master process:  " << processor_name 
                << " filename:" << name << " " << str.str() << '\n';

      t4 = get_current_timestamp();
      std::cout << "  Master process:  " << processor_name
                << " Img frame " << cnt
                << " Time elapsed:" 
                << (t4-t1)*1000.0 << " " 
                << (t4-t2)*1000.0 << " "
                << (t4-t3)*1000.0 << " ms "  << std::endl;


      if (cnt >= input_frame->getNFrames())
        break;

    }//end main loop
  }
  else {
    //std::cout << "\n"
    //          << "  Child process  " << processor_name << std::endl
    //          << "  process id: " << procs_id << "/" << num_procs << std::endl;

    //--------------------------------------------------------------------------
    // Receives model path
    //--------------------------------------------------------------------------
    std::string model_path(""); 

    //--------------------------------------------------------------------------
    // Receives image from root process.
    //--------------------------------------------------------------------------
    ierr = MPI_Recv( sizes, 3, MPI_INT,
           root_process, send_data_tag, MPI_COMM_WORLD, &status);

    child_img.create(sizes[0],sizes[1],CV_8UC3);

    ierr = MPI_Recv( child_img.data, sizes[0]*sizes[1]*3, MPI_CHAR,
           root_process, send_data_tag, MPI_COMM_WORLD, &status);

    //--------------------------------------------------------------------------
    // Receives Rects from root process.
    //--------------------------------------------------------------------------
    int num_rects_to_receive;
    ierr = MPI_Recv( &num_rects_to_receive, 1, MPI_INT, root_process,
                     send_data_tag, MPI_COMM_WORLD, &status);

    std::cout << "  Child process: " << procs_id << " "
              << processor_name << " Ack will receive " << num_rects_to_receive << " " 
              << " from root"
              <<'\n';

    //--------------------------------------------------------------------------
    // Receives single Rect from root process.
    //--------------------------------------------------------------------------
    for(int k=0 ; k<num_rects_to_receive; k++ ) {
      Rect rect_from_root;

      ierr = MPI_Recv( &rect_from_root, sizeof(Rect), MPI_BYTE, 
                       root_process, send_data_tag, MPI_COMM_WORLD, &status);

//      std::cout << "  Child process: " << procs_id << " "
//                << processor_name << " Received " << k << " " << rect_from_root
//                << " from root"
//                <<'\n';

      //--------------------------------------------------------------------------
      // Makes image classification with received rectangle.
      //--------------------------------------------------------------------------
      std::map<int,rectangle_t> m_prob;
      double t0,t1, t1_child, t2_child;
      t0 = get_current_timestamp();
      rectangle_t child_rect ;

      t1_child = get_current_timestamp();
      rectangle_classification(child_img,rect_from_root, model_path, child_rect);
      t2_child = get_current_timestamp();

      if ( child_rect != rectangle_type(Rect(0,0,0,0),0,0) ){
        std::cout << "  Child process: " << procs_id << " " << processor_name 
                  << " rectangle[" << (num_procs * k +procs_id) 
                  << "] input=" << rect_from_root 
                  << " output=" << child_rect.id << ":" << child_rect.prob << ":" << child_rect.rect 
                  << " time elapsed: " <<  (t2_child-t1_child)*1000.0 << " ms"
  //                << " debug [x:y:w:]"   << rect_from_root.x << ":" << rect_from_root.y << ":" << rect_from_root.width << ":" << rect_from_root.height
                  <<  '\n';
      }


      ierr = MPI_Send( &child_rect, sizeof(rectangle_t), MPI_BYTE, root_process,
                 return_data_tag, MPI_COMM_WORLD); 
    }
    std::cout << "  Child process: " << procs_id << " " << processor_name 
              << " Finished.\n";
  }// End child processes


  // Finalize the MPI environment.
  MPI_Finalize();


  return 0;
} //main






