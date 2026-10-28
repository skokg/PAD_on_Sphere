# Builds the C++ shared library next to PAD_on_sphere.py, where the ctypes wrapper loads it from.
#   make                     build with g++ (default)
#   make CXX=clang++         another compiler
#   make CXXFLAGS=-O3        extra flags, added after the base ones (so -O3 wins over -O2)
#   make clean               remove the built library
#
# Site-specific settings go in an optional local.mk next to this file (not tracked by git), e.g. a
# PRE_BUILD command that sets up the compiler environment and is run, in the same shell, before compiling.

-include local.mk

CXX ?= g++
# Base flags, always used; CXXFLAGS from the command line or environment is appended
BASE_CXXFLAGS := -std=c++11 -O2 -Wall -Wno-unused-result
# -shared builds a shared library (.so) that ctypes can load, instead of an executable; -fPIC is required for it
LIB_FLAGS := -shared -fPIC

SRC_DIR := source_for_Cxx_shared_library
# The other .cc files are #included by this one, so it is the only translation unit
MAIN_SRC := $(SRC_DIR)/CC_PAD_on_sphere_python_lib.cc
SOURCES := $(wildcard $(SRC_DIR)/*.cc)
LIB := PAD_on_sphere_Cxx_shared_library.so

.PHONY: all clean

all: $(LIB)

$(LIB): $(SOURCES)
	$(PRE_BUILD) $(CXX) $(BASE_CXXFLAGS) $(CXXFLAGS) $(LIB_FLAGS) -o $@ $(MAIN_SRC)

clean:
	rm -f $(LIB)
