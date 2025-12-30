-include ../../makefile.inc

TESTS_SRC = TestAscendIndexFlat.cpp TestAscendIndexIVFPQ.cpp \
	TestAscendIndexIVFSQ.cpp \
	TestAscendIndexSQ.cpp TestAscendIndexSQSlim.cpp \
	TestAscendIndexInt8Flat.cpp TestAscendIndexInt8IVFFlat.cpp \
	TestAscendIndexInt8TS.cpp

TESTS_OBJ = $(TESTS_SRC:.cpp=.o)
TESTS_BIN = $(TESTS_OBJ:.o=)

run: $(TESTS_BIN)
	for t in $(TESTS_BIN); do ./$$t || exit; done

$(TESTS_OBJ): %.o: %.cpp gtest
	$(CXX) -fPIE $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -o $@ -c $< -Igtest/include -I../../ -I. -I../../common

$(TESTS_BIN): %: %.o ../../libascendfaiss.a gtest/make/gtest_host.a
	$(CXX) -o $@ $^ $(LDFLAGS) -pie $(LIBS)

clean:
	rm -f *.o $(TESTS_BIN)

.PHONY: clean run

