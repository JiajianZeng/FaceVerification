#include "mainwindow.h"
#include <QApplication>

using namespace std;
using namespace cv;
const int FRAME_LEN=1280*720*3;




int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

   w.showDefaultMonitor();
    return a.exec();
}
