#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "spikebinning.h"


PYBIND11_MODULE(spikebinning_cpplib, m) {
    m.doc() = "Fast spike binning"; // optional module docstring

    m.def("bin_spikes_movie",
            &bin_spikes_movie,
            pybind11::return_value_policy::take_ownership,
            "Function that bin spikes in consecutive frames given trigger boundaries");

    m.def("bin_spikes_trials",
          &bin_spikes_trials,
          pybind11::return_value_policy::take_ownership,
          "Function that bins spikes into bins determined by trial structure");

    m.def("bin_spikes_consecutive_trials",
            &bin_spikes_consecutive_trials,
            pybind11::return_value_policy::take_ownership,
            "Function that bins spikes into bins determined by trial structure. Assumes that the trials are consecutive in time");

    m.def("merge_multiple_sorted_array",
            &merge_multiple_sorted_array<int64_t>,
            pybind11::return_value_policy::take_ownership,
            "Merges sorted spike vectors");
}
