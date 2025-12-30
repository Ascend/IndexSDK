-include ../../makefile.inc

TESTS_SRC = TestIndexILFlat.cpp

TESTS_OBJ = $(TESTS_SRC:.cpp=.o)
TESTS_BIN = $(TESTS_OBJ:.o=)

run: $(TESTS_BIN)
	# for t in $(TESTS_BIN); do ./$$t || exit; done

$(TESTS_OBJ): %.o: %.cpp gtest
	$(ASCENDCXX) $(ASCENDFLAGS) -o $@ -c $< -Igtest/include -I../../ -I../../common -I../../ascenddaemon/impl_device

$(TESTS_BIN): %: %.o libascendfaiss_minios.so gtest/make/gtest_dev.a
	$(ASCENDCXX) -o $@ $^ $(ASCENDLDFLAGS) $(ASCENDLIBS)

clean:
	rm -f *.o $(TESTS_BIN)

.PHONY: clean run

