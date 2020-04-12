#include <cassert>
#include <complex>
#include <string>
#include <iostream>
#include <chrono>
#include <future>
#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <cuckoo/gpu/cuckoo.hpp>

void print_array(std::size_t N, int* a) {
	std::cout << "[";
	for (std::size_t i = 0; i < N; ++i) {
		if (i == N - 1) {
			std::cout << a[i];
		} else {
			std::cout << a[i] << ", ";
		}
	}
	std::cout << "]" << std::endl;
	
}

bool is_same(std::size_t N, int* a, int* b) {
	bool is_same = true;
	for (std::size_t i = 0; i < N; ++i) {
		if (a[i] != b[i]) {
			is_same = false;
		}
	}
	std::cout << "a: ";
	print_array(N, &a[0]);	
	std::cout << std::endl;
	
	std::cout << "b: ";
	print_array(N, &b[0]);	
	std::cout << std::endl;
	
	return is_same;
}

TEST_CASE("Test Add", "cuckoo") {
	CHECK(add());
}

TEST_CASE("Test Add New", "cuckoo") {
	CHECK(add_new());
}

TEST_CASE("Test Add Array", "cuckoo") {
	CHECK(add_array());
}


TEST_CASE("Test Creation", "cuckoo") {
	auto c = Cuckoo();
}

TEST_CASE("Test Insertion", "cuckoo") {
	int a[] = {1, 2, 3};
	int b[] = {4, 5, 6};
	int c[] = {7, 8, 9};
	auto cuckoo = Cuckoo();
	CHECK(cuckoo.set(3, &a[0], &b[0], &c[0]) == 0);
}
TEST_CASE("Test Retreival", "cuckoo") {
	int a[] = {1, 2, 3};
	int b[] = {4, 5, 6};
	int c[] = {7, 8, 9};
	int d[] = {0, 0, 0};
	auto cuckoo = Cuckoo();
	cuckoo.set(3, &a[0], &b[0], &c[0]);
	cuckoo.get(3, &a[0], &d[0]);
	CHECK(is_same(3, &b[0], &d[0]));
}
