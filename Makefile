all: randomForests

randomForests: main.cpp Makefile
	g++-5 -O2 -std=c++11 main.cpp -o randomForests -g
