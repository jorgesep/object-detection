#include "imgreader.h"


using namespace std;


int main( int argc, char** argv ) {

    ImgFiles f = ImgFiles();

    string name("hola.txt");

    //cout << f.getFileExtension("png") << endl ;

    string directory("");
    //string wildcard("*.jpg");
    //string wildcard("2011_005523.jpg");
    string wildcard("2011_*.jpg");
    //f.getImageFiles(directory, wildcard);
    return 0;
}
