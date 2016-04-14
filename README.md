#FaceRecognitionSystem
Subfolder 'algorithm' is for Cao and Zeng to develop their algorithms for face verification, etc. <br>
Subfolder 'platform' is for Zhang to develop his integrated platform.

#libdeepid2
libdeepid2 is a shared library which realizes the deepid2 algorithm in c++, its root directory is in algorithm/libdeepid2.
#install of libdeepid2
(1)download this project<br>
(2)in root directory of libdeepid2, run 'make clean'<br>
(3)in root directory of libdeepid2, run 'make'<br>
(3)after run the above two commands, you will get libdeepid2.so in the root directory. Then copy libdeepid2.so and lib/libcaffe.so into some system library directory(e.g. /usr/lib, you may need root permission).<br>
(4)forward to the system library directory, and run 'ldconfig'<br>
(5)copy ./include directory to your project root directory<br>
(6)modify your Makefile accordingly(e.g. modify include dir and link libdeepid2.so)<br>
(7)copy ./data directory into some position.<br>
(8)modify ./data/libdeepid2.yaml accordingly<br>

see my demo application in ./demo.cpp for more details.
#prerequisites
(1)opencv<br>
(2)boost<br>

plz refer to the above prerequisites's project page to install them

