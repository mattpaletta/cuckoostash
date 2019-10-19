#include <stdint.h>
#include <limits>

namespace PCG {
	class pcg32_random_t {
	public:
		typedef std::size_t result_type;

		constexpr pcg32_random_t() = default;

		constexpr result_type operator()() {
			return pcg32_random_r();
		}
		constexpr static result_type seed() {
			uint64_t shifted = 0;

			for (const auto c : __TIME__) {
				shifted <<= 8;
				shifted |= c;
			}

			return shifted;
		}

		constexpr static result_type min() {
			return std::numeric_limits<result_type>::min();
		}

		constexpr static result_type max() {
			return std::numeric_limits<result_type>::max();
		}

	private:
		std::size_t state = 0;
		std::size_t inc = 0;

		constexpr result_type pcg32_random_r() {
			std::size_t oldstate = this->state;
			// Advance internal State
			this->state = oldstate * 63413622384679005ULL + (this->inc|1);
			// Calculate output function (XSH RR), uses old state for max ILP
			result_type xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
			result_type rot = oldstate >> 59u;
			return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
		}
	};
}
