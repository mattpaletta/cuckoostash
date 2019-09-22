#include <cassert>
#include <complex>
#include <string>
#include <iostream>
#include <chrono>
#include <future>
#define CATCH_CONFIG_MAIN

#include <catch2/catch.hpp>

#include <cuckoo/gpu/cuckoo.hpp>

TEST_CASE("Test Creation", "cuckoo") {
	auto c = Cuckoo();
}

TEST_CASE("Test Insertion", "cuckoo") {
	auto c = Cuckoo();
	c.set();
}
