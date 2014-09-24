CXX=clang++
LD=clang++

CXX_FLAGS=-g -O3 -march=native -std=c++11 -Inice_svm/include -Imnist/include
LD_FLAGS=-lsvm

ifeq ($(CXX),clang++)
	CXX_FLAGS += -Qunused-arguments
endif

ifeq ($(CXX),clang++)
	WARNING_FLAGS=-Werror -Wextra -Wall -Wuninitialized -Wsometimes-uninitialized -Wno-long-long -Winit-self -Wdocumentation
else
	WARNING_FLAGS=-Werror -Wextra -Wall -Wuninitialized -Wno-long-long -Winit-self
endif

default: svm_mnist

release/src/svm_mnist.cpp.o: src/svm_mnist.cpp nice_svm/include/nice_svm.hpp
	@mkdir -p release/src/
	$(CXX) $(WARNING_FLAGS) $(CXX_FLAGS) -o $@ -c $<

release/bin/svm_mnist: release/src/svm_mnist.cpp.o
	@mkdir -p release/bin/
	$(LD) $(LD_FLAGS) -o $@  $^

svm_mnist: release/bin/svm_mnist

clean:
	rm -rf release