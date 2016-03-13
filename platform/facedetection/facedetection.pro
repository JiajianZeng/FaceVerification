#-------------------------------------------------
#
# Project created by QtCreator 2016-03-13T15:25:19
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = facedetection
TEMPLATE = app

INCLUDEPATH += /usr/include/opencv\
                              /usr/include/opencv2\
                              /usr/include\
QMAKER_LIBDIR += /usr/lib/camera

LIBS += \
/usr/lib/x86_64-linux-gnu/libopencv_core.so\
/usr/lib/x86_64-linux-gnu/libopencv_highgui.so\
/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so\
-L /usr/lib/camera\
-lhcnetsdk -lHCCore -lhpr\
-lanalyzedata -lHCAlarm -lHCCoreDevCfg -lHCDisplay -lHCGeneralCfgMgr -lHCIndustry -lHCPlayBack\
-lHCPreview -lHCVoiceTalk -lStreamTransClient -lSystemTransform\

SOURCES += main.cpp\
        mainwindow.cpp

HEADERS  += mainwindow.h

FORMS    += mainwindow.ui
