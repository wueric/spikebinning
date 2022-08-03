//
// Created by Eric Wu on 11/18/21.
//

#ifndef RECONSTRUCTION_V2_MERGEWRAPPER_H
#define RECONSTRUCTION_V2_MERGEWRAPPER_H

#include <limits>

template<class T>
struct MergeWrapper {
    T *base_ptr;
    int64_t dim0;
    int64_t curr_ix;


    MergeWrapper<T>(T *ptr_, int64_t dim0_) : base_ptr(ptr_), dim0(dim0_), curr_ix(0) {
    };

    MergeWrapper<T>(T *ptr_, int64_t dim0_, int64_t curr_ix_) : base_ptr(ptr_), dim0(dim0_), curr_ix(curr_ix_) {
    };

    T priority() {
        return -(*(base_idx + curr_ix));
    }

    T getCurrent() const {return *(base_idx + curr_ix);}

    bool atEnd() const {return curr_ix >= dim0;}

    void increment() {
        if (!atEnd()) ++curr_ix;
    }

    friend bool operator< (MergeWrapper<T> const& lhs, MergeWrapper<T> const& rhs) {
        return lhs.priority < rhs.priority;
    }
};

template<class T>
struct ComparePriority {
    bool operator()(MergeWrapper<T> const & p1, MergeWrapper<T> const & p2) {
        return p1.priority() < p2.priority();
    }
};

#endif //RECONSTRUCTION_V2_MERGEWRAPPER_H
