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
#ifndef _IM_READER_H
#define _IM_READER_H

#include <iostream>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>



using namespace std;
using namespace cv;

class FrameReader
{
public:
  virtual ~FrameReader() {}
  virtual void getFrame (OutputArray) = 0;
  virtual void getFrame (OutputArray, int) = 0;
  virtual int getFrameDelay() = 0;
  virtual int getNChannels() = 0;
  virtual int getNumberCols() = 0;
  virtual int getNumberRows() = 0;
  virtual int getNFrames() = 0;
  virtual int getFrameCounter () = 0;
}; 

/** This class encapsulate images from directory
 */ 
class ImgFiles : public FrameReader
{
public:
  ImgFiles();
  ~ImgFiles() {};
  ImgFiles(string); 
  ImgFiles(const vector<string>&); 
  virtual void getFrame(OutputArray frame);
  virtual void getFrame(OutputArray frame, int color = 1);
  virtual int getFrameDelay() { return delay; };
  virtual int getNChannels() { return nchannels; };
  virtual int getNumberCols() { return cols; };
  virtual int getNumberRows() { return rows; };
  virtual int getNFrames() { return length; };
  virtual int getFrameCounter() { return frame_counter; };

private:
  void getListFiles(const string);
  void getListFiles(const vector<string>&);
  void lookForImageFilesInDirectory(string);
  void getImageProperties();
  int frame_counter;
    //bool is_file();

  vector<string> im_files;

  int delay;
  int cols;
  int rows;
  int length;
  int nchannels;
};

class VideoFile : public FrameReader
{
public:
  VideoFile()    :  
  delay(25),
  cols(0),
  rows(0),
  nchannels(0),
  frame_counter(0) {};

  ~VideoFile();
  VideoFile(string); 
  virtual void getFrame(OutputArray frame);
  virtual void getFrame(OutputArray frame, int color = 1);
  virtual int getFrameDelay() { return delay; };
  virtual int getNChannels() { return nchannels; };
  virtual int getNumberCols() { return cols; };
  virtual int getNumberRows() { return rows; };
  virtual int getNFrames() { return length; };
  virtual int getFrameCounter() { return frame_counter; };

private:
  VideoCapture video;
  void getImageProperties();
  int delay;
  int cols;
  int rows;
  int length;
  int nchannels;
  string videoname;
  int frame_counter;

};

class FrameReaderFactory
{
public:
  static FrameReader* create_frame_reader(const vector<string> &);
};

#endif
