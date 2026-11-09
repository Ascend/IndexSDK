-include ../../makefile.inc

TESTS_SRC = TestIndexIVFPQ.cpp TestAscendTensor.cpp TestAscendResources.cpp \
	TestDeviceVector.cpp TestSegmentedDeviceVector.cpp TestIndexFlat.cpp TestIndexIVFSQL2.cpp \
	TestLinearTransform.cpp TestIndexSQL2.cpp \
	TestIndexIVFFlat.cpp TestIndexFlatInt8CosTS.cpp

TESTS_OBJ = $(TESTS_SRC:.cpp=.o)
TESTS_BIN = $(TESTS_OBJ:.o=)

run: $(TESTS_BIN)
	# for t in $(TESTS_BIN); do ./$$t || exit; done

$(TESTS_OBJ): %.o: %.cpp gtest
	$(ASCENDCXX) $(ASCENDFLAGS) -o $@ -c $< -Igtest/include -I../../

$(TESTS_BIN): %: %.o ../../libascendfaissdaemon.a gtest/make/gtest_dev.a
	$(ASCENDCXX) -o $@ $^ $(ASCENDLDFLAGS) $(ASCENDLIBS)

HOSTCXX ?= g++
HOSTCXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
HOST_GTEST_FLAGS ?= -lgtest

# Standalone host-only test: no ACL, Ascend libraries, or device initialization.
TestParallelFutureUtils: TestParallelFutureUtils.cpp ../../ascenddaemon/utils/ParallelFutureUtils.h
	$(HOSTCXX) $(HOSTCXXFLAGS) -I../../ -o $@ $< $(HOST_GTEST_FLAGS) -pthread

host-parallel-future-test: TestParallelFutureUtils
	./TestParallelFutureUtils

clean:
	rm -f *.o $(TESTS_BIN) TestParallelFutureUtils


.PHONY: clean run host-parallel-future-test
