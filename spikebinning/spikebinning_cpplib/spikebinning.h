#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdlib.h>
#include <stdint.h>
#include <queue>

#include "MergeWrapper.h"
#include "NDArrayWrapper.h"

namespace py=pybind11;

template<class T>
using ContigNPArray = py::array_t<T, py::array::c_style | py::array::forcecast>;


int64_t bin_spikes_single_cell(
        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> &spike_time_buffer,
        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> &bin_write_buffer,
        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> &bin_cutoff_times,
        int64_t offset_below) {

    const int64_t n_bin_edges = bin_cutoff_times.shape[0];
    const int64_t n_spikes_total = spike_time_buffer.shape[0];

    for (int64_t i = 0; i < n_bin_edges - 1; ++i) {
        int64_t start = bin_cutoff_times.valueAt(i);
        int64_t end = bin_cutoff_times.valueAt(i + 1);

        int64_t n_spikes_in_bin = 0;
        while (offset_below < n_spikes_total && spike_time_buffer.valueAt(offset_below) < start) ++offset_below;
        while (offset_below < n_spikes_total && spike_time_buffer.valueAt(offset_below) < end) {
            ++offset_below;
            ++n_spikes_in_bin;
        }

        bin_write_buffer.storeTo(n_spikes_in_bin, i);
    }
    return offset_below;
}


template<typename T>
int64_t search_index_backward(
        CNDArrayWrapper::StaticNDArrayWrapper<T, 1> &spike_time_buffer,
        T before_time,
        int64_t offset) {

    while (offset > 0 && spike_time_buffer.valueAt(offset) >= before_time) --offset;
    return offset;
}


template<typename T>
int64_t binary_search_index(
        CNDArrayWrapper::StaticNDArrayWrapper<T, 1> &spike_time_buffer,
        T time) {

    /*
     * Intended behavior: Finds the index of the element at or immediately
     *      after before_time
     *      Returns 0 if all elements occur after before_time
     *
     * Assumptions: the entries in spike_time_buffer are strictly increasing,
     *      since spike_time_buffer must be a valid spike train, and a cell can't
     *      spike twice in the same sample
     */

    int64_t arr_len = spike_time_buffer.shape[0];

    int64_t low = 0, high = arr_len - 1; // inclusive
    int64_t idx = (arr_len >> 1);
    while (low <= high) {
        T value = spike_time_buffer.valueAt(idx);
        if (value == time) {
            return idx;
        } else if (value > time) {
            high = idx - 1;
            idx = ((high - low) >> 1) + low;
        } else {
            low = idx + 1;
            idx = ((high - low) >> 1) + low;
        }
    }

    return idx;
}


ContigNPArray<int64_t> bin_spikes_trials(
        py::dict &spikes_by_cell_id,
        py::list &cell_order,
        ContigNPArray <int64_t> &trial_bin_cutoffs) {

    // figure out how many trials there are, and how many bins there are
    py::buffer_info bin_info = trial_bin_cutoffs.request();
    int64_t *bin_time_matrix_ptr = static_cast<int64_t *> (bin_info.ptr);
    const int64_t n_trials = bin_info.shape[0];
    const int64_t n_bin_cutoffs = bin_info.shape[1];

    const int64_t n_bins = n_bin_cutoffs - 1;

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 2> bin_time_wrapper(
            bin_time_matrix_ptr,
            {n_trials, n_bin_cutoffs});

    // figure out how many cells there are
    const int64_t n_cells = cell_order.size();

    // allocate the output binned times
    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(int64_t),     /* Size of one item */
            py::format_descriptor<int64_t>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {n_cells, n_trials, n_bins}, /* Number of elements for each dimension */
            {sizeof(int64_t) * n_bins * n_trials, sizeof(int64_t) * n_bins, sizeof(int64_t)}  /* Strides for each dim */
    );

    ContigNPArray <int64_t> binned_output = ContigNPArray<int64_t>(output_buffer_info);
    py::buffer_info output_info = binned_output.request();
    int64_t *output_data_ptr = static_cast<int64_t *> (output_info.ptr);

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 3> output_wrapper(
            output_data_ptr,
            {n_trials, n_cells, n_bins});

    /*
     * Algorithm: loop over each cell
     * For each trial, bin the spikes
     */
    for (int64_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {

        py::object cell_id_pykey = cell_order[cell_idx];
        ContigNPArray <int64_t> spikes_for_current_cell = py::cast < ContigNPArray < int64_t >> (
                spikes_by_cell_id[cell_id_pykey]);

        py::buffer_info spike_time_info = spikes_for_current_cell.request();

        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> spike_time_wrapper(
                static_cast<int64_t *>(spike_time_info.ptr),
                {spike_time_info.shape[0]});

        int64_t spike_offset = 0;
        for (int64_t trial_idx = 0; trial_idx < n_trials; ++trial_idx) {

            CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> output_bin_wrapper = output_wrapper.slice<1>(
                    CNDArrayWrapper::makeIdxSlice(trial_idx),
                    CNDArrayWrapper::makeIdxSlice(cell_idx),
                    CNDArrayWrapper::makeAllSlice());

            CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> bin_cutoff_wrapper = bin_time_wrapper.slice<1>(
                    CNDArrayWrapper::makeIdxSlice(trial_idx),
                    CNDArrayWrapper::makeAllSlice());

            int64_t trial_bin_start = bin_time_wrapper.valueAt(cell_idx, trial_idx);
            spike_offset = binary_search_index<int64_t>(spike_time_wrapper, trial_bin_start);

            spike_offset = bin_spikes_single_cell(spike_time_wrapper, output_bin_wrapper,
                                                  bin_cutoff_wrapper, spike_offset);

        }
    }

    return binned_output;
}


ContigNPArray <int64_t> bin_spikes_consecutive_trials(
        py::dict &spikes_by_cell_id,
        py::list &cell_order,
        ContigNPArray <int64_t> &trial_bin_cutoffs) {

    // figure out how many trials there are, and how many bins there are
    py::buffer_info bin_info = trial_bin_cutoffs.request();
    int64_t *bin_time_matrix_ptr = static_cast<int64_t *> (bin_info.ptr);
    const int64_t n_trials = bin_info.shape[0];
    const int64_t n_bin_cutoffs = bin_info.shape[1];

    const int64_t n_bins = n_bin_cutoffs - 1;

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 2> bin_time_wrapper(
            bin_time_matrix_ptr,
            {n_trials, n_bin_cutoffs});

    // figure out how many cells there are
    const int64_t n_cells = cell_order.size();

    // allocate the output binned times
    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(int64_t),     /* Size of one item */
            py::format_descriptor<int64_t>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {n_cells, n_trials, n_bins}, /* Number of elements for each dimension */
            {sizeof(int64_t) * n_bins * n_trials, sizeof(int64_t) * n_bins, sizeof(int64_t)}  /* Strides for each dim */
    );

    ContigNPArray <int64_t> binned_output = ContigNPArray<int64_t>(output_buffer_info);
    py::buffer_info output_info = binned_output.request();
    int64_t *output_data_ptr = static_cast<int64_t *> (output_info.ptr);

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 3> output_wrapper(
            output_data_ptr,
            {n_trials, n_cells, n_bins});

    /*
     * Algorithm: loop over each cell
     * For each trial, bin the spikes
     */
    for (int64_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {

        py::object cell_id_pykey = cell_order[cell_idx];
        ContigNPArray <int64_t> spikes_for_current_cell = py::cast < ContigNPArray < int64_t >> (
                spikes_by_cell_id[cell_id_pykey]);

        py::buffer_info spike_time_info = spikes_for_current_cell.request();

        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> spike_time_wrapper(
                static_cast<int64_t *>(spike_time_info.ptr),
                {spike_time_info.shape[0]});

        int64_t spike_offset = 0;
        for (int64_t trial_idx = 0; trial_idx < n_trials; ++trial_idx) {

            CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> output_bin_wrapper = output_wrapper.slice<1>(
                    CNDArrayWrapper::makeIdxSlice(trial_idx),
                    CNDArrayWrapper::makeIdxSlice(cell_idx),
                    CNDArrayWrapper::makeAllSlice());

            CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> bin_cutoff_wrapper = bin_time_wrapper.slice<1>(
                    CNDArrayWrapper::makeIdxSlice(trial_idx),
                    CNDArrayWrapper::makeAllSlice());

            int64_t trial_bin_start = bin_time_wrapper.valueAt(cell_idx, trial_idx);
            spike_offset = search_index_backward<int64_t>(spike_time_wrapper, trial_bin_start, spike_offset);

            spike_offset = bin_spikes_single_cell(spike_time_wrapper, output_bin_wrapper,
                                                  bin_cutoff_wrapper, spike_offset);

        }
    }

    return binned_output;
}

ContigNPArray <int64_t> bin_spikes_movie(
        py::dict &spikes_by_cell_id,
        py::list &cell_order,
        ContigNPArray <int64_t> &movie_bin_cutoffs) {

    // figure out how many how many bins there are
    py::buffer_info bin_info = movie_bin_cutoffs.request();
    int64_t *bin_time_matrix_ptr = static_cast<int64_t *> (bin_info.ptr);
    const int64_t n_bin_cutoffs = bin_info.shape[0];
    const int64_t n_bins = n_bin_cutoffs - 1;

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> bin_time_wrapper(
            bin_time_matrix_ptr,
            {n_bin_cutoffs,});

    // figure out how many cells there are
    const int64_t n_cells = cell_order.size();

    // allocate the output binned times
    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(int64_t),     /* Size of one item */
            py::format_descriptor<int64_t>::value, /* Buffer format */
            2,          /* How many dimensions? */
            {n_cells, n_bins}, /* Number of elements for each dimension */
            {sizeof(int64_t) * n_bins, sizeof(int64_t)}  /* Strides for each dim */
    );

    ContigNPArray <int64_t> binned_output = ContigNPArray<int64_t>(output_buffer_info);
    py::buffer_info output_info = binned_output.request();
    int64_t *output_data_ptr = static_cast<int64_t *> (output_info.ptr);

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 2> output_wrapper(
            output_data_ptr,
            {n_cells, n_bins});

    // loop over the cells
    // within each loop bin spikes for the corresponding cell
    for (int64_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {

        py::object cell_id_pykey = cell_order[cell_idx];
        ContigNPArray <int64_t> spikes_for_current_cell = py::cast < ContigNPArray < int64_t >> (
                spikes_by_cell_id[cell_id_pykey]);

        py::buffer_info spike_time_info = spikes_for_current_cell.request();

        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> spike_time_wrapper(
                static_cast<int64_t *>(spike_time_info.ptr),
                {spike_time_info.shape[0], });

        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> write_wrapper = output_wrapper.slice<1>(
                CNDArrayWrapper::makeIdxSlice(cell_idx),
                CNDArrayWrapper::makeAllSlice());

        bin_spikes_single_cell(spike_time_wrapper, write_wrapper, bin_time_wrapper, 0);
    }

    return binned_output;
}

template<class T>
ContigNPArray <T> merge_multiple_sorted_array(
        py::list &list_of_spike_trains) {

    std::priority_queue <T, std::vector<MergeWrapper<T>>, ComparePriority<T>> priorityQueue;

    int64_t total_size = 0;
    for (auto item : list_of_spike_trains) {

        // convert/cast item
        ContigNPArray <int64_t> spike_train = py::cast < ContigNPArray < int64_t >> (item);

        py::buffer_info array_info = spike_train.request();
        T *base_ptr = static_cast<T *> (array_info.ptr);
        int64_t current_size = array_info.shape[0];

        priorityQueue.push(MergeWrapper<T>(base_ptr, current_size));
        total_size = total_size + current_size;
    }

    // allocate the output binned times
    auto merged_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(T),     /* Size of one item */
            py::format_descriptor<T>::value, /* Buffer format */
            2,          /* How many dimensions? */
            {total_size}, /* Number of elements for each dimension */
            {sizeof(T)}  /* Strides for each dim */
    );

    ContigNPArray <T> merged_output = ContigNPArray<T>(merged_buffer_info);
    py::buffer_info output_info = merged_output.request();
    T *output_base_ptr = static_cast<T *>(output_info.ptr);

    int64_t write_offset = 0;
    while (!priorityQueue.empty()) {

        MergeWrapper<T> min_element = priorityQueue.top();
        priorityQueue.pop();
        T current_val = min_element.getCurrent();
        *(output_base_ptr + write_offset) = current_val;
        ++write_offset;

        min_element.increment();
        if (!min_element.atEnd()) {
            priorityQueue.push(min_element);
        }
    }

    return merged_output;
}

