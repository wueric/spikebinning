#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdlib.h>
#include <stdint.h>
#include <queue>
#include <memory>

#include "MergeWrapper.h"
#include "NDArrayWrapper.h"

namespace py=pybind11;

template<class T>
using ContigNPArray = py::array_t<T, py::array::c_style | py::array::forcecast>;


int64_t bin_spikes_single_cell(
        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> &spike_time_buffer,
        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> &bin_write_buffer,
        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> &bin_cutoff_times,
        int64_t offset_below_idx) {

    /*
     * Bins spikes for a cell whose spike times are contained in spike_time_buffer
     *      according to bin edge times in bin_cutoff_times. Binned spikes are written
     *      into bin_write_buffer
     *
     * All times are in units of samples
     *
     * @param spike_time_buffer: wrapper on an array of shape (n_spikes, )
     * @param bin_write_buffer: wrapper on an array of shape (n_bins, )
     * @param bin_cutoff_times: wrapper on an array of shape (n_bins + 1, )
     * @param offset_below: integer index into spike_time_buffer, corresponding to the
     *      index of any spike either at or earlier than the first spike that needs
     *      to be binned
     */

    const int64_t n_bin_edges = bin_cutoff_times.shape[0];
    const int64_t n_spikes_total = spike_time_buffer.shape[0];

    int64_t start, end, n_spikes_in_bin;
    start = bin_cutoff_times.valueAt(0);

    while (offset_below_idx < n_spikes_total &&
           spike_time_buffer.valueAt(offset_below_idx) < start)
        ++offset_below_idx;

    for (int64_t i = 0; i < n_bin_edges - 1; ++i) {
        start = bin_cutoff_times.valueAt(i);
        end = bin_cutoff_times.valueAt(i + 1);
        n_spikes_in_bin = 0;

        while (offset_below_idx < n_spikes_total && spike_time_buffer.valueAt(offset_below_idx) < end) {
            ++offset_below_idx;
            ++n_spikes_in_bin;
        }

        bin_write_buffer.storeTo(n_spikes_in_bin, i);
    }

    return offset_below_idx;
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
     *
     * @param spike_time_buffer: wrapper with shape (n_spikes, ), corresponding to the
     *      spike times of a single cel
     * @param time: the time to find
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

    return idx + 1;
}


using MulticellSpikeTrain = std::map<int64_t, ContigNPArray<int64_t>>;
using MultiDataset = std::tuple <MulticellSpikeTrain, std::vector<int64_t>, ContigNPArray<int64_t>>;


void _bin_spikes_into_buffer(
        MulticellSpikeTrain spikes_by_cell_id,
        std::vector<int64_t> cell_order,
        ContigNPArray<int64_t> trial_bin_cutoffs,
        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 3> output_wrapper) {

    // figure out how many trials there are, and how many bins there are
    py::buffer_info bin_info = trial_bin_cutoffs.request();
    int64_t *bin_time_matrix_ptr = static_cast<int64_t *> (bin_info.ptr);
    const int64_t n_trials = bin_info.shape[0];
    const int64_t n_bin_cutoffs = bin_info.shape[1];

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 2> bin_time_wrapper(
            bin_time_matrix_ptr,
            {n_trials, n_bin_cutoffs});

    // figure out how many cells there are
    const int64_t n_cells = cell_order.size();

    using Int64_1DWrapper = CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1>;
    std::map <int64_t, std::unique_ptr<Int64_1DWrapper>> spike_time_wrapper_map{};
    for (int64_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {

        int64_t cell_id = cell_order[cell_idx];

        ContigNPArray<int64_t> spikes_for_current_cell = spikes_by_cell_id[cell_id];

        py::buffer_info spike_time_info = spikes_for_current_cell.request();

        std::array<int64_t, 1> spike_shape = {spike_time_info.shape[0]};

        spike_time_wrapper_map[cell_id] = std::make_unique<Int64_1DWrapper>(
                static_cast<int64_t *>(spike_time_info.ptr),
                spike_shape);

    }

#pragma omp parallel for
    for (int64_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
        int64_t cell_id = cell_order[cell_idx];

        auto spike_time_wrapper = *spike_time_wrapper_map[cell_id];

        int64_t trial_idx = 0;
        int64_t spike_offset = binary_search_index<int64_t>(spike_time_wrapper,
                                                            bin_time_wrapper.valueAt(trial_idx, 0));
        for (; trial_idx < n_trials; ++trial_idx) {

            CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> output_bin_wrapper = output_wrapper.slice<1>(
                    CNDArrayWrapper::makeIdxSlice(trial_idx),
                    CNDArrayWrapper::makeIdxSlice(cell_idx),
                    CNDArrayWrapper::makeAllSlice());

            CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> bin_cutoff_wrapper = bin_time_wrapper.slice<1>(
                    CNDArrayWrapper::makeIdxSlice(trial_idx),
                    CNDArrayWrapper::makeAllSlice());

            int64_t trial_bin_start = bin_time_wrapper.valueAt(trial_idx, 0);
            spike_offset = binary_search_index<int64_t>(spike_time_wrapper, trial_bin_start);

            spike_offset = bin_spikes_single_cell(spike_time_wrapper, output_bin_wrapper,
                                                  bin_cutoff_wrapper, spike_offset);

        }
    }

}


ContigNPArray<int64_t> multidataset_bin_spikes_trials_parallel(
        std::vector<MultiDataset> multiple_datasets) {

    int64_t n_datasets = multiple_datasets.size();

    std::vector<int64_t> dataset_lengths { };
    std::vector<int64_t> dataset_offsets { };

    auto first_tup = multiple_datasets[0];
    auto first_trial_bin_cutoffs = std::get<2>(first_tup);
    py::buffer_info first_bin_info = first_trial_bin_cutoffs.request();
    int64_t n_bins = first_bin_info.shape[1] - 1;

    auto first_cell_order = std::get<1>(first_tup);
    int64_t n_cells = first_cell_order.size();

    int64_t n_trials = 0;
    for (int64_t i = 0; i < n_datasets; ++i) {
        auto tup = multiple_datasets[i];

        auto trial_bin_cutoffs = std::get<2>(tup);
        py::buffer_info bin_info = trial_bin_cutoffs.request();
        int64_t n_trials_dataset = bin_info.shape[0];

        dataset_offsets.push_back(n_trials);
        dataset_lengths.push_back(n_trials_dataset);

        n_trials += n_trials_dataset;
    }

    // allocate the output binned times
    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(int32_t),     /* Size of one item */
            py::format_descriptor<int64_t>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {n_trials, n_cells, n_bins}, /* Number of elements for each dimension */
            {sizeof(int64_t) * n_bins * n_cells, sizeof(int64_t) * n_bins, sizeof(int64_t)}  /* Strides for each dim */
    );

    ContigNPArray<int64_t> binned_output = ContigNPArray<int64_t>(output_buffer_info);
    py::buffer_info output_info = binned_output.request();
    int64_t *output_data_ptr = static_cast<int64_t *> (output_info.ptr);

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 3> total_output_wrapper(
            output_data_ptr,
            {n_trials, n_cells, n_bins});

    for (int64_t i = 0; i < n_datasets; ++i) {

        int64_t dataset_offset = dataset_offsets[i];
        int64_t dataset_length = dataset_lengths[i];

        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 3> output_wrapper = total_output_wrapper.slice<3>(
                CNDArrayWrapper::makeRangeSlice(dataset_offset, dataset_offset + dataset_length),
                CNDArrayWrapper::makeAllSlice(),
                CNDArrayWrapper::makeAllSlice());

        auto tup = multiple_datasets[i];
        auto spikes_by_cell_id = std::get<0>(tup);
        auto cell_order = std::get<1>(tup);
        auto trial_bin_cutoffs = std::get<2>(tup);

        _bin_spikes_into_buffer(spikes_by_cell_id, cell_order, trial_bin_cutoffs, output_wrapper);
    }

    return binned_output;
}


ContigNPArray<int64_t> bin_spikes_trials_parallel(
        std::map <int64_t, ContigNPArray<int64_t>> spikes_by_cell_id,
        std::vector <int64_t> cell_order,
        ContigNPArray<int64_t> trial_bin_cutoffs) {

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
            {n_trials, n_cells, n_bins}, /* Number of elements for each dimension */
            {sizeof(int64_t) * n_bins * n_cells, sizeof(int64_t) * n_bins, sizeof(int64_t)}  /* Strides for each dim */
    );

    ContigNPArray<int64_t> binned_output = ContigNPArray<int64_t>(output_buffer_info);
    py::buffer_info output_info = binned_output.request();
    int64_t *output_data_ptr = static_cast<int64_t *> (output_info.ptr);

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 3> output_wrapper(
            output_data_ptr,
            {n_trials, n_cells, n_bins});

    _bin_spikes_into_buffer(spikes_by_cell_id, cell_order, trial_bin_cutoffs, output_wrapper);

    return binned_output;
}


ContigNPArray<int64_t> bin_spikes_trials(
        py::dict &spikes_by_cell_id,
        py::list &cell_order,
        ContigNPArray<int64_t> &trial_bin_cutoffs) {

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
            {n_trials, n_cells, n_bins}, /* Number of elements for each dimension */
            {sizeof(int64_t) * n_bins * n_cells, sizeof(int64_t) * n_bins, sizeof(int64_t)}  /* Strides for each dim */
    );

    ContigNPArray<int64_t> binned_output = ContigNPArray<int64_t>(output_buffer_info);
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
        ContigNPArray<int64_t> spikes_for_current_cell = py::cast<ContigNPArray<int64_t >>(
                spikes_by_cell_id[cell_id_pykey]);

        py::buffer_info spike_time_info = spikes_for_current_cell.request();

        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> spike_time_wrapper(
                static_cast<int64_t *>(spike_time_info.ptr),
                {spike_time_info.shape[0]});

        int64_t trial_idx = 0;
        int64_t spike_offset = binary_search_index<int64_t>(spike_time_wrapper,
                                                            bin_time_wrapper.valueAt(trial_idx, 0));
        for (; trial_idx < n_trials; ++trial_idx) {

            CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> output_bin_wrapper = output_wrapper.slice<1>(
                    CNDArrayWrapper::makeIdxSlice(trial_idx),
                    CNDArrayWrapper::makeIdxSlice(cell_idx),
                    CNDArrayWrapper::makeAllSlice());

            CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> bin_cutoff_wrapper = bin_time_wrapper.slice<1>(
                    CNDArrayWrapper::makeIdxSlice(trial_idx),
                    CNDArrayWrapper::makeAllSlice());

            int64_t trial_bin_start = bin_time_wrapper.valueAt(trial_idx, 0);
            spike_offset = binary_search_index<int64_t>(spike_time_wrapper, trial_bin_start);

            spike_offset = bin_spikes_single_cell(spike_time_wrapper, output_bin_wrapper,
                                                  bin_cutoff_wrapper, spike_offset);

        }
    }

    return binned_output;
}


ContigNPArray<int64_t> bin_spikes_movie(
        py::dict &spikes_by_cell_id,
        py::list &cell_order,
        ContigNPArray<int64_t> &movie_bin_cutoffs) {

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

    ContigNPArray<int64_t> binned_output = ContigNPArray<int64_t>(output_buffer_info);
    py::buffer_info output_info = binned_output.request();
    int64_t *output_data_ptr = static_cast<int64_t *> (output_info.ptr);

    CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 2> output_wrapper(
            output_data_ptr,
            {n_cells, n_bins});

    // loop over the cells
    // within each loop bin spikes for the corresponding cell
    for (int64_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {

        py::object cell_id_pykey = cell_order[cell_idx];
        ContigNPArray<int64_t> spikes_for_current_cell = py::cast<ContigNPArray<int64_t >>(
                spikes_by_cell_id[cell_id_pykey]);

        py::buffer_info spike_time_info = spikes_for_current_cell.request();

        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> spike_time_wrapper(
                static_cast<int64_t *>(spike_time_info.ptr),
                {spike_time_info.shape[0],});

        CNDArrayWrapper::StaticNDArrayWrapper<int64_t, 1> write_wrapper = output_wrapper.slice<1>(
                CNDArrayWrapper::makeIdxSlice(cell_idx),
                CNDArrayWrapper::makeAllSlice());

        bin_spikes_single_cell(spike_time_wrapper, write_wrapper, bin_time_wrapper, 0);
    }

    return binned_output;
}

template<class T>
ContigNPArray<T> merge_multiple_sorted_array(
        py::list &list_of_spike_trains) {
    /*
     * Merges spike trains of oversplits
     * Uses classic min-heap algorithm to merge N sorted spike trains
     *
     * @param list_of_spike_trains
     */

    /*
     * Implementation note: std::priority_queue is a max heap
     *
     * We want a min heap because we want to merge the spike times
     * in increasing order, so the priority in MergeWrapper is set up
     * to be the negative of the first unread spike time
     */
    std::priority_queue < T, std::vector < MergeWrapper < T >>, ComparePriority < T >> priorityQueue;

    int64_t total_size = 0;
    for (auto item : list_of_spike_trains) {

        // convert/cast item
        ContigNPArray<T> spike_train = py::cast<ContigNPArray<T >>(item);

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
            1,          /* How many dimensions? */
            {total_size}, /* Number of elements for each dimension */
            {sizeof(T)}  /* Strides for each dim */
    );

    ContigNPArray<T> merged_output = ContigNPArray<T>(merged_buffer_info);
    py::buffer_info output_info = merged_output.request();
    T *output_base_ptr = static_cast<T *>(output_info.ptr);

    int64_t write_offset = 0;
    while (!priorityQueue.empty()) {

        MergeWrapper <T> min_element = priorityQueue.top();
        priorityQueue.pop();
        *(output_base_ptr + write_offset) = min_element.getCurrent();;
        ++write_offset;

        min_element.increment();
        if (!min_element.atEnd()) {
            priorityQueue.push(min_element);
        }
    }

    return merged_output;
}

