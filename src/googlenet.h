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

#ifndef _GOOGLENET_CLASSIFIER_H
#define _GOOGLENET_CLASSIFIER_H


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include "dnn_algorithmI.h"



using namespace cv;
//using namespace boost::filesystem;


class GoogLeNetClassifier : public DeepNeuralNetworkAlgorithmI
{
    
public:
    
    //default constructor
    GoogLeNetClassifier();
    ~GoogLeNetClassifier() {};
    void SetAlgorithmParameters()  {};
    void LoadConfigParameters(std::string) {};
    void SaveConfigParameters() {};
    void Initialization();
    void GetBackground(OutputArray) {};
    void GetForeground(OutputArray) {};
    void Update(InputArray, OutputArray) {};
    void LoadModel(std::string);
    void SaveModel() {};
    void Process(InputArray, OutputArray);
    void GetClassProb(int*, double*);
    std::string PrintParameters();
    const std::string Name() {return std::string("GoogLeNetClassifier"); };
    std::string ElapsedTimeAsString();
    std::string className() {return class_name;};
    int         classId()   {return class_id;};
    double      classProbability(){return class_prob;};
    double ElapsedTime() {return elapsedTime; } ;
    long Iterations() {return numberIterations;} ;

private:

    void readClassNames();
    void getClassMaxProb(InputArray, int*, double*);
    void print_mat(InputArray, std::string);
 
    double duration;

    //static const double DefaultAlpha;
    
    dnn::Net net;

    std::vector<String> classNames;
    std::string path_models ;

    Mat prob;

    double elapsedTime;
    long numberIterations;

    std::string class_name ;
    int class_id;
    double class_prob;

};

Ptr<DeepNeuralNetworkAlgorithmI> createGoogLeNetClassifier();

#endif
