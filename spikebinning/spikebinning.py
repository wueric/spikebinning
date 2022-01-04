import numpy as np
from typing import Dict, List

from . import spikebinning_cpplib


def bin_spikes_movie(spikes_by_cell_id: Dict[int, np.ndarray],
                     cell_ordering: List[int],
                     movie_cutoff_times: np.ndarray) -> np.ndarray:
    '''
    Bins spikes for multiple cells into specified intervals for a
        continuous movie (the entire dataset is treated as a single
        super-long trial)

    :param spikes_by_cell_id: Dict[int, np.ndarray], spike trains for all of the
        cells that we are interested in binning

        key is integer cell id
        value is np.ndarray of spike times, shape (n_spike_times, )

        The spike times are assumed to be monotonically increasing, since cells
            cannot spike twice in the same sample.

    :param cell_ordering: List[int] where each entry is a cell id, corresponding
        to a key in spikes_by_cell_id

        Ordering of cell id for the output binned matrix.

    :param movie_cutoff_times: np.ndarray, shape (n_cutoff_times, ) corresponding
        to the cutoff times of each bin, in units of samples

        n_cutoff_times corresponds to (n_cutoff_times - 1) total bins, since every
        bin requires its start and end samples to be specified, and bins are assumed
        to be consecutive.
    :return: np.ndarray, shape (n_cells, n_cutoff_times - 1)

        binned spike train for multiple cells,
    '''

    if movie_cutoff_times.ndim != 1:
        raise ValueError("movie_cutoff_times must have dim 1")

    return spikebinning_cpplib.bin_spikes_movie(spikes_by_cell_id,
                                                cell_ordering,
                                                movie_cutoff_times)


def bin_spikes_trials(spikes_by_cell_id: Dict[int, np.ndarray],
                      cell_ordering: List[int],
                      trial_structured_bin_times: np.ndarray) -> np.ndarray:
    '''

    Bins spikes for multiple cells into the specified intervals
        Assumes a trial design where the output matrix has a dimension
        for the different trials

    The trials can be in any order.

    :param spikes_by_cell_id: Dict[int, np.ndarray], spike trains for all of the
        cells that we are interested in binning

        key is integer cell id
        value is np.ndarray of spike times, shape (n_spike_times, )

        The spike times are assumed to be monotonically increasing, since cells
            cannot spike twice in the same sample.

    :param cell_ordering: List[int] where each entry is a cell id, corresponding
        to a key in spikes_by_cell_id

        Ordering of cell id for the output binned matrix.

    :param trial_structured_bin_times: np.ndarray, shape (n_trials, n_bin_cutoffs),
        corresponding to the bin cutoff times for each trial.

        n_bin_cutoffs corresponds to (n_bin_cutoffs - 1) bins per trial, since every
        bin requires its start and end times to be specified, and we assume that the
        bins are consecutive.
    :return: np.ndarray, shape (n_trials, n_cells, n_bin_cutoffs - 1)

        Binned spike trains for every trial / cell
    '''

    if trial_structured_bin_times.ndim != 2:
        raise ValueError('trial_structured_bin_times should have dim 2')

    return spikebinning_cpplib.bin_spikes_trials(spikes_by_cell_id,
                                                 cell_ordering,
                                                 trial_structured_bin_times)


def bin_spikes_trials_parallel(spikes_by_cell_id: Dict[int, np.ndarray],
                               cell_ordering: List[int],
                               trial_structured_bin_times: np.ndarray) -> np.ndarray:
    '''

    Bins spikes for multiple cells into the specified intervals
        Assumes a trial design where the output matrix has a dimension
        for the different trials

    The trials can be in any order.

    :param spikes_by_cell_id: Dict[int, np.ndarray], spike trains for all of the
        cells that we are interested in binning

        key is integer cell id
        value is np.ndarray of spike times, shape (n_spike_times, )

        The spike times are assumed to be monotonically increasing, since cells
            cannot spike twice in the same sample.

    :param cell_ordering: List[int] where each entry is a cell id, corresponding
        to a key in spikes_by_cell_id

        Ordering of cell id for the output binned matrix.

    :param trial_structured_bin_times: np.ndarray, shape (n_trials, n_bin_cutoffs),
        corresponding to the bin cutoff times for each trial.

        n_bin_cutoffs corresponds to (n_bin_cutoffs - 1) bins per trial, since every
        bin requires its start and end times to be specified, and we assume that the
        bins are consecutive.
    :return: np.ndarray, shape (n_trials, n_cells, n_bin_cutoffs - 1)

        Binned spike trains for every trial / cell
    '''

    if trial_structured_bin_times.ndim != 2:
        raise ValueError('trial_structured_bin_times should have dim 2')

    return spikebinning_cpplib.bin_spikes_trials_parallel(spikes_by_cell_id,
                                                          cell_ordering,
                                                          trial_structured_bin_times)


def merge_multiple_sorted_array(spike_trains_to_merge: List[np.ndarray]) \
        -> np.ndarray:
    '''
    Merges multiple spike trains into a single spike train. Useful for merging
        oversplits.

    :param spike_trains_to_merge: List[np.ndarray], spike trains to merge
    :return:
    '''

    return spikebinning_cpplib.merge_multiple_sorted_array(spike_trains_to_merge)
