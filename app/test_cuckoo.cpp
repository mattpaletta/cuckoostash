#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <cassert>
#include <complex>
#include <string>
#include <iostream>
#include <chrono>
#include <future>

#include <cuckoo/cuckoo.hpp>

void print_array(const std::size_t& N, const int* a) {
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

bool is_same(const std::size_t& N, const int* a, const int* b) {
	bool is_same = true;
	for (std::size_t i = 0; i < N; ++i) {
		if (a[i] != b[i]) {
			is_same = false;
		}
	}
	if (!is_same) {
		print_array(N, a);
		print_array(N, b);
	}
	return is_same;
}

TEST_CASE("Test Creation", "cuckoo") {
	auto c = Cuckoo<>();
}

TEST_CASE("Test Insertion", "cuckoo") {
	int a[] = {1, 2, 3};
	int b[] = {4, 5, 6};
	int c[] = {0, 0, 0};
	auto cuckoo = Cuckoo<>();
	CHECK(cuckoo.set(3, &a[0], &b[0], &c[0]) == 0);
}
TEST_CASE("Test Retreival", "cuckoo") {
	int a[] = {1, 2, 3};
	int b[] = {4, 5, 6};
	int c[] = {0, 0, 0};
	int d[] = {0, 0, 0};
	auto cuckoo = Cuckoo<>();
	CHECK(cuckoo.set(3, &a[0], &b[0], &c[0]) == 0);
	cuckoo.get(3, &a[0], &d[0]);
	CHECK(is_same(3, &b[0], &d[0]));
}
