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
#include "imgreader.h"
#include <boost/filesystem.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/regex.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>


using namespace std;
namespace bfs = boost::filesystem;
namespace ba = boost::adaptors;

ImgFiles::ImgFiles()
: frame_counter(0), delay(25), cols(0), rows(0),nchannels(0)
{
  getListFiles(".");
  getImageProperties();
}


ImgFiles::ImgFiles(string file)
: frame_counter(0), delay(25), cols(0), rows(0),nchannels(0)
{
  getListFiles(file);
  getImageProperties();
}

ImgFiles::ImgFiles(const vector<string>& files)
: frame_counter(0), delay(25), cols(0), rows(0),nchannels(0)
{
  getListFiles(files);
  getImageProperties();
}

void ImgFiles::
getListFiles(const vector<string> & files) {

  for(vector<string>::const_iterator it = files.begin(); it != files.end(); it++){

    bfs::path p(*it);
    if( bfs::is_regular_file(p) ){
     im_files.push_back((bfs::canonical(p).string())); 
   }
 }

#ifdef __debug__ 
 for(vector<string>::const_iterator sit = im_files.begin(); sit!=im_files.end(); sit++)
  cout<< *sit << endl;
#endif

}

void ImgFiles::
getListFiles(string file) {

  bfs::path p( file.c_str() );
  if ( bfs::exists(p) ) {
    if (bfs::is_directory(p)) {

      vector<bfs::path> v;
      copy(bfs::directory_iterator(p), bfs::directory_iterator(), back_inserter(v));

      // default: look for jpg or png files.
      boost::regex expression(".*\\.(jpg|png)$");

      for (vector<bfs::path>::const_iterator it(v.begin()), it_end(v.end()); it != it_end; ++it) {
        if ( boost::regex_match(it->filename().c_str(), expression) ) {
          im_files.push_back((bfs::canonical(*it).string()));
        }

      }
    }
    else if (bfs::is_regular_file(p))
      im_files.push_back((bfs::canonical(p).string()));

  }
  else {
    cout << __FUNCTION__ << ": " << p << " does not exist\n";
    throw -1;
  }

#ifdef __debug__
  for(vector<string>::const_iterator sit = im_files.begin(); sit!=im_files.end(); sit++)
    cout<< *sit << endl;
#endif

}

// void ImgFiles::
// getListFiles(string names) {

//   cout << __FUNCTION__ << " :" << names << endl;

//   bfs::path p( names.c_str() );

//   vector<bfs::path> v;

//   // default: look for jpg or png files.
//   boost::regex expression(".*\\.(jpg|png)$");

//   //if( p.parent_path() == "")
//   //  p = bfs::path( "./" + names) ;

//   cout << __FUNCTION__ << ": path:" << p << " parent_path: " << p.parent_path() << endl ; 
//   cout << "parent_path_exists:" << bfs::exists(p.parent_path())<< " p_exists:" << bfs::exists(p) << " is_regular_file:" << bfs::is_regular_file(p) << " is_directory:" << bfs::is_directory(p) << endl << endl ;

//   if ( bfs::exists(p.parent_path()) ) {

//     // if not regular contains '*' as wildcard. 
//     if ( ! bfs::is_regular_file(p) && ! bfs::is_directory(p) ) {

//       string filename = p.filename().c_str();

// cout << "NOT REGULAR FILE " << endl ;

//       // replace * for wildcard .* in regular expression
//       boost::regex re("\\*");

//       // replace original regular expression
//       expression = boost::regex_replace(filename, re, ".*") ;

//       // look everything in the parent path.
//       p = p.parent_path();

//     }

//     cout << "1)Copying files in directory ..." <<  endl ;
//     cout << "1)Expression:" << expression << endl ;

//     copy(bfs::directory_iterator(p), bfs::directory_iterator(), back_inserter(v));

//     cout << "2)Copying files in directory ..." <<  endl ;
//     cout << "2)Expression:" << expression << endl ;

//     for (vector<bfs::path>::const_iterator it(v.begin()), it_end(v.end()); 
//       it != it_end; ++it) {

//       cout << "Expression:" << expression << " Filename: " << it->filename() << endl ;

//       if ( boost::regex_match(it->filename().c_str(), expression) ) {

//         im_files.push_back((bfs::canonical(*it).string()));


//       }
//     }
//   }
//   else {
//     cout << __FUNCTION__ << ": " << p << " does not exist\n";
//     throw -1;
//   }

//   for(int i=0; i<im_files.size(); i++)
//     cout << im_files[i] << endl;
//   cout << "reg: " << expression << endl;



  // // checking directory path
  // if (bfs::is_directory(p)) {

  //   copy(bfs::directory_iterator(p), bfs::directory_iterator(), back_inserter(v));

  //   for (vector<bfs::path>::const_iterator it(v.begin()), it_end(v.end()); 
  //     it != it_end; ++it) {
  //     boost::cmatch what;
  //     if ( boost::regex_match(it->filename().c_str(), what, expression) ) {

  //                 //cout << "filename:" << it->filename().c_str() << " what_1:"<< atoi(what[1].first) << " what_3:" << what[3].first << ":" << what[3].second << endl ;
  //                 // Converts to an absolute path that has no symbolic link.
  //       im_files.push_back((bfs::canonical(*it).string()));

  //     }
  //   }

  // }







// Look for jpg and png files in directory
void ImgFiles::lookForImageFilesInDirectory(std::string directory)
{
  bfs::path _path ( directory.c_str() );

  if (bfs::is_directory(_path)) {
    vector<bfs::path> v;
    copy(bfs::directory_iterator(_path), bfs::directory_iterator(), back_inserter(v));

    for (vector<bfs::path>::const_iterator it(v.begin()), it_end(v.end()); it != it_end; ++it) {

      if ( (it->extension() == ".jpg") || (it->extension() == ".png") )
        im_files.push_back((bfs::canonical(*it).string()));
    }

    sort(im_files.begin(), im_files.end());

    length = im_files.size();

  }

}


void ImgFiles::getFrame(OutputArray frame, int color)
{
  Mat Image;

  Image = imread(im_files[frame_counter]);

  if (color)
    Image.copyTo(frame);
  else
    cvtColor(Image, frame, CV_BGR2GRAY);

  frame_counter++;

}

void ImgFiles::getFrame(OutputArray frame)
{
  Mat Image;

  Image = imread(im_files[frame_counter]);
  Image.copyTo(frame);

  frame_counter++;

}



void ImgFiles::getImageProperties()
{
  if (im_files.size() > 0) {

        //create video object.
    Mat Frame = imread(im_files[0]);

        // Check file has been opened sucessfully
    if (Frame.data == NULL )
      return ;

    cols          = Frame.cols;
    rows          = Frame.rows;
    int frameType = Frame.type();
    nchannels     = CV_MAT_CN(frameType);

  }

}

VideoFile::VideoFile(std::string filename)
:  delay(25),
cols(0),
rows(0),
nchannels(0),
frame_counter(0)

{
  bfs::path _path_to_file(filename.c_str());
  if (is_regular_file(_path_to_file))
    video.open(filename.c_str());

  videoname = filename;
  getImageProperties();

}

VideoFile::~VideoFile()
{
  if (video.isOpened())
    video.release();
}

void VideoFile::getFrame(OutputArray frame, int color)
{
  Mat Image;
  if (video.isOpened()) {
    video >> Image;
    if (color)
      Image.copyTo(frame);
    else
      cvtColor(Image, frame, CV_BGR2GRAY);

    frame_counter = video.get(CV_CAP_PROP_POS_FRAMES);
  }

}

void VideoFile::getFrame(OutputArray frame)
{
  Mat Image;
  if (video.isOpened()) {
    video >> Image;
    Image.copyTo(frame);

    frame_counter = video.get(CV_CAP_PROP_POS_FRAMES);
  }

}


void VideoFile::getImageProperties()
{
  VideoCapture video_prop(videoname.c_str());
  if (!video_prop.isOpened())
    return;

  Mat Frame;
  video_prop >> Frame;

  double rate= video_prop.get(CV_CAP_PROP_FPS);
  delay  = 1000/rate ;
  cols   = video_prop.get(CV_CAP_PROP_FRAME_WIDTH);
  rows   = video_prop.get(CV_CAP_PROP_FRAME_HEIGHT);
  length = video_prop.get(CV_CAP_PROP_FRAME_COUNT);
  int frameType = Frame.type();
  nchannels = CV_MAT_CN(frameType);

    //reset video to initial position.
    //video.set(CV_CAP_PROP_POS_FRAMES, 0);

  video_prop.release();


}

FrameReader 
*FrameReaderFactory :: 
create_frame_reader(const vector<string> & files) {

#ifdef __debug__
  cout << __FUNCTION__ << " Number of Input:" << files.size() << endl;
#endif

  if (! files.empty() ) {

    if(files.size() == 1) {

      string name = files[0];

      // regular expression to verify supported video files.
      const char* re = ".*\\.(avi|mp4)$";
      boost::regex expression(re);

      if ( boost::regex_match(name.c_str(), expression) )
        return new VideoFile(name);
      else
        return new ImgFiles(name);

    }
    else {
      // set of files
      return new ImgFiles(files);
    }
  }

  return NULL;

}

//string ImgFiles::getFileExtension(const string s) {
//
//    // default extension
//    string extension("jpg");
//
//    boost::regex expr(".*([A-Za-z]{3}$)");
//    boost::smatch what;
//
//    //string st("out*.jpg");
//    string st("out*jpg");
//    boost::regex re("\\*");
//    string fmt(".*");
//    string str_re1 = boost::regex_replace(st, re, fmt) ;
//    boost::regex re1( str_re1.c_str() );
//    cout << boost::regex_replace(st, re, fmt) << '\n';
//
//
//    //std::string s = "Boost Libraries";
//    //boost::regex expr{"\\w+\\s\\w+"};
//    cout << boolalpha << boost::regex_match(st, re1) << '\n';
//
//
//
//    if (boost::regex_search(s, what, expr))
//        return what[1].str();
//
//    return extension;
//
//}


