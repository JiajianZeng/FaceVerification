# compiler
CC = g++
# compile flags
CFLAGS = -fPIC
# linking flags
LDFLAGS = -shared
# rm command
RM = rm -f
# target library
TARGET_LIB = libdeepid2.so

# source files
SRCS = $(shell find ./src -name "*.cpp")

# include
INCLUDES =  -I/usr/local/include/opencv2 \
            -I/usr/local/include/opencv \
            -I/usr/local/cuda/include \
            -I./include \
            
# link library
LIBRARIES = -lcaffe \
            -lopencv_core \
            -lboost_system \
            -L./lib \

.PHONY: all
all:$(TARGET_LIB)

$(TARGET_LIB):$(OBJS)
	$(CC) $(LDFLAGS) -o $(TARGET_LIB) $(LIBRARIES) $(INCLUDES) $(OBJS)

$(OBJS):$(SRCS) 
	$(CC) $(CFLAGS) -c $(SRCS) $(INCLUDES)

.PHONY: clean
clean:
	-$(RM) $(TARGET_LIB) $(OBJS)

