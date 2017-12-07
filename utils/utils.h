//#include <opencv2/dnn.hpp>

//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/core/utils/trace.hpp>
//#include <fstream>
//#include <iostream>
//#include <cstdlib>
//#include <sys/time.h>

using namespace cv;



typedef struct rectangle_type {

  rectangle_type():rect(0,0,0,0),id(0),prob(0){};
  rectangle_type(Rect r, int i, double p):rect(r),id(i),prob(p){};

  //copy constructor
  rectangle_type (const rectangle_type & rhs) { *this = rhs; };

  //Inequality operator
  bool operator !=(const rectangle_type &rhs) const
  {
    return ((rect!=rhs.rect)||(id!=rhs.id)||(prob!=rhs.prob));
  };

  //Inequality operator
  bool operator ==(const rectangle_type &rhs) const
  {
    return ((rect==rhs.rect)&&(id==rhs.id)&&(prob==rhs.prob));
  };

  // assigment operator
  rectangle_type &operator =(const rectangle_type &rhs)
  {
      if (*this != rhs)
      {
          rect=rhs.rect; id = rhs.id; prob = rhs.prob;
      }
      return *this;
  };

  // overloaded += operator
  rectangle_type & operator +=(const rectangle_type  &rhs)
  {
    // union of rects.
    Rect u_rect   = rect | rhs.rect ;
    // save bigger probability.
    double u_prob = (prob>rhs.prob ? prob : rhs.prob);
    rect = u_rect;
    prob = u_prob;
    return *this;
  };

  const rectangle_type operator+(const rectangle_type &rhs) const
  {
     rectangle_type result(*this);
     return result += rhs;
  }

  Rect   rect;
  int    id;
  double prob;

} rectangle_t;


bool rectangle_sorter (rectangle_type const& lhs, rectangle_type const& rhs) {
    return lhs.prob > rhs.prob;
}



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

  //Inequality operator
  bool operator ==(const prob_type &rhs) const
  {
    return ((rect==rhs.rect)&&(Id==rhs.Id)&&(Prob==rhs.Prob));
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
