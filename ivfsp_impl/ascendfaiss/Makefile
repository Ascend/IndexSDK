# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

-include makefile.inc

HEADERS = $(wildcard ascend/*.h ascend/impl/*.h ascend/custom/*.h ascend/utils/*.h ascend/fv/*.h)
SRC     = $(wildcard ascend/*.cpp ascend/impl/*.cpp ascend/custom/*.cpp ascend/custom/impl/*.cpp ascend/utils/*.cpp ascend/rpc/*.cpp)
OBJ     = $(sort $(SRC:.cpp=.o))
INSTALL_DIR = $(FAISS_ROOT)
INSTALLDIRS = $(INSTALL_DIR)/lib $(INSTALL_DIR)/include/faiss

ASCEND_HEADERS = $(wildcard ascenddaemon/*.h ascenddaemon/impl/*.h ascenddaemon/impl_custom/*.h ascenddaemon/utils/*.h ascenddaemon/rpc/*.h)
ASCEND_SRC     = $(wildcard ascenddaemon/*.cpp ascenddaemon/impl/*.cpp ascenddaemon/impl_custom/*.cpp ascenddaemon/utils/*.cpp ascenddaemon/rpc/*.cpp ascenddaemon/rpc/handler/*.cpp $(COMMON_DIR)/utils/*.cpp)
ASCEND_OBJ     = $(sort $(ASCEND_SRC:.cpp=.o))
ASCEND_TARGET  = ascendfaissdaemon


#COMMON
COMMON_DIR          = common
COMMON_SRC          = $(wildcard $(COMMON_DIR)/*.cpp)
COMMON_OBJ          = $(addprefix ascend/rpc/, $(notdir $(COMMON_SRC:.cpp=.o)))
ASCEND_COMMON_OBJ   = $(addprefix ascenddaemon/rpc/, $(notdir $(COMMON_SRC:.cpp=.o)))

#THREADPOOL
THREADPOOL_DIR        = $(COMMON_DIR)/threadpool
THREADPOOL_SRC        = $(wildcard $(THREADPOOL_DIR)/*.cpp)
THREADPOOL_OBJ        = $(addprefix ascend/, $(notdir $(THREADPOOL_SRC:.cpp=.o)))
ASCEND_THREADPOOL_OBJ = $(addprefix ascenddaemon/, $(notdir $(THREADPOOL_SRC:.cpp=.o)))

# RPC
PROTOC              = $(PROTOBUF_ROOT)/bin/protoc
PROTO_DIR           = common
RPC_PROTO           = $(wildcard $(PROTO_DIR)/*.proto)
PROTO_CFLAGS        = `pkg-config --cflags protobuf`
PROTO_SRC           = $(RPC_PROTO:.proto=.pb.cc)
PROTO_HEADER        = $(PROTO_SRC:.cc=.h)

PROTO_OBJ           = $(addprefix ascend/rpc/, $(notdir $(PROTO_SRC:.cc=.o)))
ASCEND_PROTO_PATH   = ascenddaemon/rpc/
ASCEND_PROTO_OBJ    = $(addprefix $(ASCEND_PROTO_PATH), $(notdir $(PROTO_SRC:.cc=.o)))

CPPFLAGS += -I. $(PROTO_CFLAGS) -I/opt/OpenBLAS/include -I./common
LDFLAGS += -L./.hostlib/ -L/opt/OpenBLAS/lib -lopenblas
ASCENDFLAGS += -I.
ASCENDFLAGS += -I./ascenddaemon/impl_custom
ASCENDFLAGS += -I./common

############################
# Building

all: libascendfaiss.a libascendfaiss.$(SHAREDEXT) libascendfaissdaemon.a $(ASCEND_TARGET)

libascendfaiss.a: $(PROTO_OBJ) $(COMMON_OBJ) $(OBJ) $(THREADPOOL_OBJ)
	$(AR) r $@ $^

libascendfaiss.$(SHAREDEXT): $(PROTO_OBJ) $(COMMON_OBJ) $(OBJ) $(THREADPOOL_OBJ)
	mkdir -p .hostlib
	ln -sf $(PROTOBUF_ROOT)/lib/libprotobuf.so .hostlib/libprotobuf.so
	$(CXX) -o $@ $^ $(SHAREDFLAGS) $(LDFLAGS) $(LIBS)

$(OBJ): %.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c $< -o $@

$(COMMON_OBJ): $(COMMON_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c $< -o $@

$(PROTO_OBJ): $(PROTO_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c $< -o $@

$(THREADPOOL_OBJ): $(THREADPOOL_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c $< -o $@

$(ASCEND_TARGET): $(ASCEND_PROTO_OBJ) $(ASCEND_COMMON_OBJ) $(ASCEND_OBJ) $(ASCEND_THREADPOOL_OBJ)
	$(ASCENDCXX) -o $@ $^ $(ASCENDLDFLAGS) $(ASCENDLIBS)

libascendfaissdaemon.a: $(ASCEND_PROTO_OBJ) $(ASCEND_COMMON_OBJ) $(ASCEND_OBJ) $(ASCEND_THREADPOOL_OBJ)
	$(ASCENDAR) r $@ $^

$(ASCEND_OBJ): %.o: %.cpp
	$(ASCENDCXX) $(ASCENDFLAGS) $(PROTO_CFLAGS) -c $< -o $@

$(ASCEND_COMMON_OBJ): $(COMMON_SRC)
	$(ASCENDCXX) $(ASCENDFLAGS) $(PROTO_CFLAGS) -c $< -o $@

$(ASCEND_PROTO_OBJ): $(PROTO_SRC)
	$(ASCENDCXX) $(ASCENDFLAGS) $(PROTO_CFLAGS) -c $< -o $@

$(ASCEND_THREADPOOL_OBJ): $(THREADPOOL_SRC)
	$(ASCENDCXX) $(ASCENDFLAGS) $(PROTO_CFLAGS) -c $< -o $@

.PRECIOUS: %.pb.cc
$(PROTO_SRC): $(RPC_PROTO)
	$(PROTOC) --cpp_out=$(PROTO_DIR) --proto_path=$(PROTO_DIR) $<

clean:
	rm -f libascendfaiss.a libascendfaiss.$(SHAREDEXT) libascendfaissdaemon.a $(ASCEND_TARGET)
	rm -f $(OBJ)
	rm -f $(COMMON_OBJ)
	rm -f $(PROTO_OBJ)
	rm -f $(ASCEND_OBJ)
	rm -f $(ASCEND_COMMON_OBJ)
	rm -f $(ASCEND_PROTO_OBJ)
	rm -f $(PROTO_SRC) $(PROTO_HEADER)

############################
# Installing

install: libascendfaiss.a libascendfaiss.$(SHAREDEXT) installdirs
	cp libascendfaiss.a libascendfaiss.$(SHAREDEXT) $(INSTALL_DIR)/lib
	tar cf - $(HEADERS) | tar xf - -C $(INSTALL_DIR)/include/faiss/

installdirs:
	$(MKDIR_P) $(INSTALLDIRS)

uninstall:
	rm -f $(INSTALL_DIR)/lib/libascendfaiss.a \
	      $(INSTALL_DIR)/lib/libascendfaiss.$(SHAREDEXT)
	rm -rf $(INSTALL_DIR)/include/faiss/ascend

#############################
# Dependencies

-include depend

depend: $(SRC) $(COMMON_SRC) $(PROTO_SRC)
	for i in $^; do \
		$(CXXCPP) $(CPPFLAGS) -x c++ -MM $$i; \
	done > depend

-include ascend_depend
ascend_depend: $(ASCEND_SRC) $(COMMON_SRC) $(PROTO_SRC)
	for i in $^; do \
		$(ASCENDCXX) $(ASCENDFLAGS) $(PROTO_CFLAGS) -x c++ -MM $$i; \
	done > ascend_depend

#############################
# Python

py: libascendfaiss.a
	$(MAKE) -C python


#############################
# Tests
gtest: third_party/googletest/googletest
	$(MAKE) -C third_party/googletest/googletest/make CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" gtest.a
	mv third_party/googletest/googletest/make/gtest.a third_party/googletest/googletest/make/gtest_host.a
	rm -rf third_party/googletest/googletest/make/gtest-all.o
	ln -sf `pwd`/third_party/googletest/googletest test/ascend/gtest
	$(MAKE) -C third_party/googletest/googletest/make CXX="$(ASCENDCXX)" CXXFLAGS="$(ASCENDFLAGS)" gtest.a
	mv third_party/googletest/googletest/make/gtest.a third_party/googletest/googletest/make/gtest_dev.a
	rm -rf third_party/googletest/googletest/make/gtest-all.o
	ln -sf `pwd`/third_party/googletest/googletest test/ascenddaemon/gtest
	#ln -sf `pwd`/third_party/googletest/googletest test/ops/gtest

test: libascendfaiss.a py
	$(MAKE) -C tests run
	PYTHONPATH=./python/build/`ls python/build | grep lib` \
	$(PYTHON) -m unittest discover tests/ -v

test_ascend: libascendfaiss.a third_party/googletest/googletest
	$(MAKE) -C test/ascend run

test_daemon: libascendfaissdaemon.a
	$(MAKE) -C test/ascenddaemon run

test_aicoreops: libascendfaiss.a third_party/googletest/googletest
	$(MAKE) -C test/ops/st run

.PHONY: all clean install installdirs py test_ascend test_daemon test_aicoreops uninstall
