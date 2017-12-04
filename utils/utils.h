//#include <opencv2/dnn.hpp>

//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/core/utils/trace.hpp>
//#include <fstream>
//#include <iostream>
//#include <cstdlib>
//#include <sys/time.h>

using namespace cv;


typedef struct class_struct {

  class_struct():id(0),prob(0){};
  class_struct(int i, double p):id(i),prob(p){};
  //copy constructor
  class_struct (const class_struct & rhs) { *this = rhs; };
  //Inequality operator
  bool operator !=(const class_struct &rhs) const
  {
      return ((id!=rhs.id)||(prob!=rhs.prob));
  };
  // assigment operator
  class_struct &operator =(const class_struct &rhs)
  {
      if (*this != rhs)
      {
          id = rhs.id; prob = rhs.prob;
      }
      return *this;
  };
  int    id;
  double prob;

} classification_type;

typedef struct prob_type {

  prob_type():rect(0,0,0,0),Id(0),Prob(0){};
  prob_type(Rect r, int i, double p):rect(r),Id(i),Prob(p){};
  //copy constructor
  prob_type (const prob_type & rhs) { *this = rhs; };
  //Inequality operator
  bool operator !=(const prob_type &rhs) const
  {
      return ((rect!=rhs.rect)||(Id!=rhs.Id)||(Prob!=rhs.Prob));
  };
  // assigment operator
  prob_type &operator =(const prob_type &rhs)
  {
      if (*this != rhs)
      {
          rect=rhs.rect; Id = rhs.Id; Prob = rhs.Prob;
      }
      return *this;
  };

  Rect   rect;
  int    Id;
  double Prob;

} prob_t;



typedef struct rect_struct {
  int x;
  int y;
  int w;
  int h;
} rect_type;


double get_current_timestamp();
void timestamp();
void print_mat(InputArray, std::string); 
void image_show(InputArray, const std::vector<Rect>&);
