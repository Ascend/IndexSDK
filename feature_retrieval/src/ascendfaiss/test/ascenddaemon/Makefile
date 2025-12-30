-include ../../makefile.inc

TESTS_SRC = TestIndexIVFPQ.cpp TestAscendTensor.cpp TestAscendResources.cpp \
	TestDeviceVector.cpp TestIndexFlat.cpp TestIndexIVFSQL2.cpp \
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

clean:
	rm -f *.o $(TESTS_BIN)


.PHONY: clean run

