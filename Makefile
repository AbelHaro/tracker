CXX = g++
CPPFLAGS += -Iinclude -I/usr/include/eigen3 -I/usr/lib/python3/dist-packages/pybind11/include -I/usr/include/python3.14
CXXFLAGS = -std=c++23 -O2 -Wall -Wextra

LIB_SRCS = $(filter-out src/main.cpp,$(wildcard src/*.cpp))
HEADERS = $(wildcard include/*.h include/*.hpp)
TARGET = build/bin/tracker
TEST_TARGETS = build/bin/kalman_filter_test build/bin/tracker_test
BENCH_TARGET = build/bin/association_benchmark
PYTHON ?= python3
UV ?= uv

PYTHON_EXT = $(shell $(PYTHON)-config --extension-suffix)

.PHONY: all run test benchmark plot clean bindings stubs

all: $(TARGET)

$(TARGET): src/main.cpp $(LIB_SRCS) $(HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/main.cpp $(LIB_SRCS) $(LDFLAGS) $(LDLIBS) -o $@

run: $(TARGET)
	./$(TARGET)

build/bin/%_test: tests/%_test.cpp $(LIB_SRCS) $(HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(LIB_SRCS) $(LDFLAGS) $(LDLIBS) -o $@

test: $(TEST_TARGETS)
	./build/bin/kalman_filter_test
	./build/bin/tracker_test

$(BENCH_TARGET): tests/association_benchmark.cpp src/AssociationCost.cpp $(HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< src/AssociationCost.cpp $(LDFLAGS) $(LDLIBS) -o $@

benchmark: $(BENCH_TARGET)
	./$(BENCH_TARGET)

plot:
	$(PYTHON) tests/plot_trajectory.py

clean:
	$(RM) $(TARGET) $(TEST_TARGETS) $(BENCH_TARGET)

bindings: build/python/tracker$(PYTHON_EXT)

stubs:
	$(UV) run --no-editable pybind11-stubgen --exit-code -o bindings tracker

build/python/tracker$(PYTHON_EXT): bindings/tracker.cpp $(LIB_SRCS) $(HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -shared -fPIC \
		$(shell $(PYTHON) -m pybind11 --includes) \
		$< $(LIB_SRCS) $(LDFLAGS) $(LDLIBS) -o $@
