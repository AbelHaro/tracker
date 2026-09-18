CXX = g++
CPPFLAGS += -Iinclude -I/usr/include/eigen3
CXXFLAGS = -std=c++23 -Wall -Wextra

SRCS = $(wildcard src/*.cpp)
HEADERS = $(wildcard include/*.h include/*.hpp)
TARGET = build/bin/tracker

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRCS) $(LDFLAGS) $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET)
