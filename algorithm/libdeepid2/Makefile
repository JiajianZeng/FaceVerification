# compiler
CC = g++
# compile flags
CFLAGS = -fPIC -g
# linking flags
LDFLAGS = -shared
# rm command
RM = rm -fv
# target library
TARGET_LIB = libdeepid2.so

# source files
SRCS = $(shell find ./src -name "*.cpp")
OBJS = $(SRCS:.cpp=.o)

# include
INCLUDES =  -I/usr/local/include/opencv2 \
            -I/usr/local/include/opencv \
            -I/usr/local/cuda/include \
            -I./include \
            
# link library
LIBRARIES = -lcaffe \
            -lopencv_core \
            -lopencv_objdetect \
            -lopencv_imgproc \
            -lopencv_highgui \
            -lboost_system \
            -L./lib \
            -L . \

      
.PHONY: all
all:demo
demo:$(TARGET_LIB)
	$(CC) -o demo demo.cpp $(LIBRARIES) $(INCLUDES) -ldeepid2
$(TARGET_LIB):objs
	$(CC) $(LDFLAGS) -Wl,-soname,$(TARGET_LIB) -o $(TARGET_LIB) $(LIBRARIES) $(OBJS)

objs:$(SRCS) 
	$(CC) $(CFLAGS) -c $(SRCS) $(INCLUDES)
	mv *.o ./src        

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB) $(OBJS)

