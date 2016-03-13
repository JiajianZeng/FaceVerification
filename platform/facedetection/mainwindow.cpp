#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace cv;

const int FRAME_LEN=1280*720*3;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

QImage cvMat2QImage(const cv::Mat& mat)
{
    // 8-bits unsigned, NO. OF CHANNELS = 1
    if(mat.type() == CV_8UC1)
    {
        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
        // Set the color table (used to translate colour indexes to qRgb values)
        image.setColorCount(256);
        for(int i = 0; i < 256; i++)
        {
            image.setColor(i, qRgb(i, i, i));
        }
        // Copy input Mat
        uchar *pSrc = mat.data;
        for(int row = 0; row < mat.rows; row ++)
        {
            uchar *pDest = image.scanLine(row);
            memcpy(pDest, pSrc, mat.cols);
            pSrc += mat.step;
        }
        return image;
    }
    // 8-bits unsigned, NO. OF CHANNELS = 3
    else if(mat.type() == CV_8UC3)
    {
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        return image.rgbSwapped();
    }
    else if(mat.type() == CV_8UC4)
    {
        //qDebug() << "CV_8UC4";
        // Copy input Mat
        const uchar *pSrc = (const uchar*)mat.data;
        // Create QImage with same dimensions as input Mat
        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        return image.copy();
    }
    else
    {
        //qDebug() << "ERROR: Mat could not be converted to QImage.";
        return QImage();
    }
}

int MainWindow::showDefaultMonitor(){
    NET_DVR_Init();

    NET_DVR_SetConnectTime(2000, 1);
    NET_DVR_SetReconnect(10000, true);


    LONG lUserID;
    NET_DVR_DEVICEINFO_V30 struDeviceInfo;
    lUserID = NET_DVR_Login_V30("10.214.143.82", 8000, "admin", "1234abcd", &struDeviceInfo);
    if (lUserID < 0)
    {
        printf("Login error, %d\n", NET_DVR_GetLastError());
        NET_DVR_Cleanup();
        return -1;
    }


    LPNET_DVR_JPEGPARA JpegPara = new NET_DVR_JPEGPARA;
    JpegPara->wPicQuality = 0;
    JpegPara->wPicSize = 9;

    char * Jpeg = new char[FRAME_LEN];
    DWORD len = FRAME_LEN;
    LPDWORD Ret = 0;

    if(!NET_DVR_SetCapturePictureMode(BMP_MODE))
    {
        cout<<"Set Capture Picture Mode error!"<<endl;
        cout<<"The error code is "<<NET_DVR_GetLastError()<<endl;
    }

    vector<char> data(FRAME_LEN);
    while(1){
         bool capture = NET_DVR_CaptureJPEGPicture_NEW(lUserID,1,JpegPara,Jpeg,len,Ret);
         if(!capture)
          {
              printf("Error: NET_DVR_CaptureJPEGPicture_NEW = %d", NET_DVR_GetLastError());
              return -1;
          }

          for(int i=0;i<FRAME_LEN;i++)
              data[i] = Jpeg[i];
          Mat init_frame = imdecode(Mat(data),1);
          QImage image = cvMat2QImage(init_frame).scaled(ui->monitor->width(),ui->monitor->height(), Qt::KeepAspectRatioByExpanding);
          QGraphicsScene *scene = new QGraphicsScene;
          scene->addPixmap(QPixmap::fromImage(image));

          ui->monitor->setScene(scene);
          ui->monitor->show();
          waitKey(1);
    }
    return 0;
}
