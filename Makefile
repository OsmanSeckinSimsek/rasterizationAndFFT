CXX=mpicxx
CXXFLAGS=-std=c++0x -O3 -g
LDLIBS=-L/home/seckin/lib_installed/heffte/lib -lheffte -lfftw3 -lfftw3f -isystem -march=native -lboost_filesystem -lboost_iostreams -lmpi
ILIBS=-I/home/seckin/lib_installed/heffte/include -I/usr/include/x86_64-linux-gnu/mpi

calculate_power_spectrum: 
	$(CXX) calculate_power_spectrum.cpp raster.hpp file_operations.hpp -o calculate_power_spectrum $(CXXFLAGS) $(ILIBS) $(LDLIBS)

clean:
	rm -f calculate_power_spectrum ./*.o ./*.gch