/*****************************************************************************
 * life.c
 * The original sequential implementation resides here.
 * Do not modify this file, but you are encouraged to borrow from it
 ****************************************************************************/
#include "life.h"
#include "util.h"
#include <sys/time.h>

/**
 * Swapping the two boards only involves swapping pointers, not
 * copying values.
 */


static double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
    return (double) tv.tv_usec/1000000.0 + tv.tv_sec ;
}

void compute_next_gen_bit(char* outboard, char* inboard, const int nrows, const int ncols)
{
    int ncols_bit = (ncols+7)/8;
    int num_bits_in_last_col = ncols % 8;
    if (num_bits_in_last_col == 0)
    {
        num_bits_in_last_col = 8;
    }
    memset(outboard, 0, nrows * ncols_bit * sizeof(char));

    unsigned int row = 0;
    unsigned int col = 0;
    char myself, up_left, up, up_right, left, right, down_left, down, down_right;
    unsigned int idx;
    char result;
    unsigned int bit;
    char neighbor_count;

    ////////////////////////////// FIRST ROW /////////////////////////////////
    ////////// first row, first col ///////////
    col = 0;
    idx = row * ncols_bit + col;
    up_left = inboard[(nrows - 1)*(ncols_bit) + ncols_bit - 1];
    up = inboard[(nrows - 1)*(ncols_bit)];
    up_right = inboard[(nrows - 1)*(ncols_bit) + 1];
    left = inboard[idx + ncols_bit - 1];
    myself = inboard[idx];
    right = inboard[idx + 1];
    down_left = inboard[idx + ncols_bit + ncols_bit - 1];
    down = inboard[idx + ncols_bit];
    down_right = inboard[idx + ncols_bit + 1];
    result = (char) 0;
    //loop through the 8 bits
    //left most bit
    bit = 0;
    neighbor_count = ((up_left & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
                     ((up & (0x01 << 7)) >> 7) + 
                     ((up & (0x01 << 6)) >> 6) + 
                     ((left & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
                     ((myself & (0x01 << 6)) >> 6) + 
                     ((down_left & (0x01 << (8 - num_bits_in_last_col))) >> num_bits_in_last_col) + 
                     ((down & (0x01 << 7)) >> 7) + 
                     ((down & (0x01 << 6)) >> 6);
    if ((((myself & (0x01 << 7)) >> 7) | neighbor_count) == 3)
    {
        result |= (0x01 << 7);
    }
    //middle bits, from right to left
    for (bit = 1; bit < 7; bit++)
    {
        neighbor_count = ((up & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((up & (0x01 << bit)) >> bit) + 
                          ((up & (0x01 << (bit + 1))) >> (bit + 1)) + 
                          ((myself & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((myself & (0x01 << (bit + 1))) >> (bit + 1)) + 
                          ((down & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((down & (0x01 << bit)) >> bit) + 
                          ((down & (0x01 << (bit + 1))) >> (bit + 1));
        if ((((myself & (0x01 << bit)) >> bit) | neighbor_count) == 3)
        {
            result |= (0x01 << bit);
        }
    }
    //right most bit
    bit = 7;
    neighbor_count = ((up & (0x01 << 1)) >> 1) + 
                     (up & (0x01)) + 
                     ((up_right & (0x01 << 7)) >> 7) + 
                     ((myself & (0x01 << 1)) >> 1) + 
                     ((right & (0x01 << 7)) >> 7) + 
                     ((down & (0x01 << 1)) >> 1) + 
                     (down & (0x01)) + 
                     ((down_right & (0x01 << 7)) >> 7);
    if (((myself & (0x01)) | neighbor_count) == 3)
    {
        result |= (0x01);
    }
    //store result in outboard
    outboard[idx] = result;

    ////////// first row, middle cols ///////////
    for (col = 1; col < ncols_bit - 1; col++)
    {
        idx = row * ncols_bit + col;
        up_left = inboard[(nrows - 1)*(ncols_bit) + col - 1];
        up = inboard[(nrows - 1)*(ncols_bit) + col];
        up_right = inboard[(nrows - 1)*(ncols_bit) + col + 1];
        left = inboard[idx - 1];
        myself = inboard[idx];
        right = inboard[idx + 1];
        down_left = inboard[idx + ncols_bit - 1];
        down = inboard[idx + ncols_bit];
        down_right = inboard[idx + ncols_bit + 1];
        result = (char) 0;
        //loop through the 8 bits
        //left most bit
        bit = 0;
        neighbor_count = (up_left & (0x01)) + 
                              ((up & (0x01 << 7)) >> 7) + 
                              ((up & (0x01 << 6)) >> 6) + 
                              (left & (0x01)) + 
                              ((myself & (0x01 << 6)) >> 6) + 
                              (down_left & (0x01)) + 
                              ((down & (0x01 << 7)) >> 7) + 
                              ((down & (0x01 << 6)) >> 6);
        if ((((myself & (0x01 << 7)) >> 7) | neighbor_count) == 3)
        {
            result |= (0x01 << 7);
        }
        //middle bits, from right to left
        for (bit = 1; bit < 7; bit++)
        {
            neighbor_count = ((up & (0x01 << (bit - 1))) >> (bit - 1)) + 
                                  ((up & (0x01 << bit)) >> bit) + 
                                  ((up & (0x01 << (bit + 1))) >> (bit + 1)) + 
                                  ((myself & (0x01 << (bit - 1))) >> (bit - 1)) + 
                                  ((myself & (0x01 << (bit + 1))) >> (bit + 1)) + 
                                  ((down & (0x01 << (bit - 1))) >> (bit - 1)) + 
                                  ((down & (0x01 << bit)) >> bit) + 
                                  ((down & (0x01 << (bit + 1))) >> (bit + 1));
            if ((((myself & (0x01 << bit)) >> bit) | neighbor_count) == 3)
            {
                result |= (0x01 << bit);
            }
        }
        //right most bit
        bit = 7;
        neighbor_count = ((up & (0x01 << 1)) >> 1) + 
                         (up & (0x01)) + 
                         ((up_right & (0x01 << 7)) >> 7) + 
                         ((myself & (0x01 << 1)) >> 1) + 
                         ((right & (0x01 << 7)) >> 7) + 
                         ((down & (0x01 << 1)) >> 1) + 
                         (down & (0x01)) + 
                         ((down_right & (0x01 << 7)) >> 7);
        if (((myself & (0x01)) | neighbor_count) == 3)
        {
            result |= (0x01);
        }
        //store result in outboard
        outboard[idx] = result;
    }

    ////////// first row, last col ///////////
    //col = ncols_bit - 1;
    idx = row * ncols_bit + col;
    up_left = inboard[(nrows - 1)*(ncols_bit) + ncols_bit - 2];
    up = inboard[(nrows - 1)*(ncols_bit) + ncols_bit - 1];
    up_right = inboard[(nrows - 1)*(ncols_bit)];
    left = inboard[idx - 1];
    myself = inboard[idx];
    right = inboard[idx - ncols_bit + 1];
    down_left = inboard[idx + ncols_bit - 1];
    down = inboard[idx + ncols_bit];
    down_right = inboard[idx + 1];
    result = (char) 0;
    //loop through the 8 bits
    //left most bit
    bit = 0;
    neighbor_count = ((up_left & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                     ((up & (0x01 << 7)) >> 7) + 
                     ((up & (0x01 << 6)) >> 6) + 
                     ((left & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                     ((myself & (0x01 << 6)) >> 6) + 
                     ((down_left & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                     ((down & (0x01 << 7)) >> 7) + 
                     ((down & (0x01 << 6)) >> 6);
    if ((((myself & (0x01 << 7)) >> 7) | neighbor_count) == 3)
    {
        result |= (0x01 << 7);
    }
    //middle bits, from right to left
    for (bit = (8 - num_bits_in_last_col + 1); bit < 7; bit++)
    {
        neighbor_count = ((up & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((up & (0x01 << bit)) >> bit) + 
                          ((up & (0x01 << (bit + 1))) >> (bit + 1)) + 
                          ((myself & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((myself & (0x01 << (bit + 1))) >> (bit + 1)) + 
                          ((down & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((down & (0x01 << bit)) >> bit) + 
                          ((down & (0x01 << (bit + 1))) >> (bit + 1));
        if ((((myself & (0x01 << bit)) >> bit) | neighbor_count) == 3)
        {
            result |= (0x01 << bit);
        }
    }
    //right most bit
    bit = 8 - num_bits_in_last_col;
    neighbor_count = ((up & (0x01 << (bit + 1))) >> (bit + 1)) + 
                     ((up & (0x01 << (bit))) >> (bit)) + 
                     ((up_right & (0x01 << 7)) >> 7) + 
                     ((myself & (0x01 << (bit + 1))) >> (bit + 1)) + 
                     ((right & (0x01 << 7)) >> 7) + 
                     ((down & (0x01 << (bit + 1))) >> (bit + 1)) + 
                     ((down & (0x01 << (bit))) >> (bit)) + 
                     ((down_right & (0x01 << 7)) >> 7);
    if ((((myself & (0x01 << bit)) >> bit) | neighbor_count) == 3)
    {
        result |= (0x01 << bit);
    }
    //store result in outboard
    outboard[idx] = result;


    ///////////////////////////////// MIDDLE ROWS /////////////////////////////////
    //#pragma omp parallel for
    for (row = 1; row < nrows - 1; row++)
    {
        unsigned int col_;
        char myself_, up_left_, up_, up_right_, left_, right_, down_left_, down_, down_right_;
        unsigned int idx_;
        char result_;
        unsigned int bit_;
        char neighbor_count_;
        ////////// middle rows, first col ///////////
        col_ = 0;
        idx_ = row * ncols_bit + col_;
        up_left_ = inboard[idx_ - 1];
        up_ = inboard[idx_ - ncols_bit];
        up_right_ = inboard[idx_ - ncols_bit + 1];
        left_ = inboard[idx_ + ncols_bit - 1];
        myself_ = inboard[idx_];
        right_ = inboard[idx_ + 1];
        down_left_ = inboard[idx_ + ncols_bit + ncols_bit - 1];
        down_ = inboard[idx_ + ncols_bit];
        down_right_ = inboard[idx_ + ncols_bit + 1];
        result_ = (char) 0;
        //loop through the 8 bits
        //left most bit
        bit_ = 0;
        neighbor_count_ = ((up_left_ & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
                         ((up_ & (0x01 << 7)) >> 7) + 
                         ((up_ & (0x01 << 6)) >> 6) + 
                         ((left_ & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
                         ((myself_ & (0x01 << 6)) >> 6) + 
                         ((down_left_ & (0x01 << (8 - num_bits_in_last_col))) >> num_bits_in_last_col) + 
                         ((down_ & (0x01 << 7)) >> 7) + 
                         ((down_ & (0x01 << 6)) >> 6);
        if ((((myself_ & (0x01 << 7)) >> 7) | neighbor_count_) == 3)
        {
            result_ |= (0x01 << 7);
        }
        //middle bits, from right to left
        for (bit_ = 1; bit_ < 7; bit_++)
        {
            neighbor_count_ = ((up_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                              ((up_ & (0x01 << bit_)) >> bit_) + 
                              ((up_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                              ((myself_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                              ((myself_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                              ((down_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                              ((down_ & (0x01 << bit_)) >> bit_) + 
                              ((down_ & (0x01 << (bit_ + 1))) >> (bit_ + 1));
            if ((((myself_ & (0x01 << bit_)) >> bit_) | neighbor_count_) == 3)
            {
                result_ |= (0x01 << bit_);
            }
        }
        //right most bit
        bit_ = 7;
        neighbor_count_ = ((up_ & (0x01 << 1)) >> 1) + 
                         (up_ & (0x01)) + 
                         ((up_right_ & (0x01 << 7)) >> 7) + 
                         ((myself_ & (0x01 << 1)) >> 1) + 
                         ((right_ & (0x01 << 7)) >> 7) + 
                         ((down_ & (0x01 << 1)) >> 1) + 
                         (down_ & (0x01)) + 
                         ((down_right_ & (0x01 << 7)) >> 7);
        if (((myself_ & (0x01)) | neighbor_count_) == 3)
        {
            result_ |= (0x01);
        }
        //store result in outboard
        outboard[idx_] = result_;

        ////////// middle rows, middle cols ///////////
        for (col_ = 1; col_ < ncols_bit - 1; col_++)
        {
            idx_ = row * ncols_bit + col_;
            up_left_ = inboard[idx_ - ncols_bit - 1];
            up_ = inboard[idx_ - ncols_bit];
            up_right_ = inboard[idx_ - ncols_bit + 1];
            left_ = inboard[idx_ - 1];
            myself_ = inboard[idx_];
            right_ = inboard[idx_ + 1];
            down_left_ = inboard[idx_ + ncols_bit - 1];
            down_ = inboard[idx_ + ncols_bit];
            down_right_ = inboard[idx_ + ncols_bit + 1];
            result_ = (char) 0;
            //loop through the 8 bits
            //left most bit
            bit_ = 0;
            neighbor_count_ = (up_left_ & (0x01)) + 
                                  ((up_ & (0x01 << 7)) >> 7) + 
                                  ((up_ & (0x01 << 6)) >> 6) + 
                                  (left_ & (0x01)) + 
                                  ((myself_ & (0x01 << 6)) >> 6) + 
                                  (down_left_ & (0x01)) + 
                                  ((down_ & (0x01 << 7)) >> 7) + 
                                  ((down_ & (0x01 << 6)) >> 6);
            if ((((myself_ & (0x01 << 7)) >> 7) | neighbor_count_) == 3)
            {
                result_ |= (0x01 << 7);
            }
            //middle bits, from right to left
            for (bit_ = 1; bit_ < 7; bit_++)
            {
                neighbor_count_ = ((up_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                                      ((up_ & (0x01 << bit_)) >> bit_) + 
                                      ((up_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                                      ((myself_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                                      ((myself_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                                      ((down_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                                      ((down_ & (0x01 << bit_)) >> bit_) + 
                                      ((down_ & (0x01 << (bit_ + 1))) >> (bit_ + 1));
                if ((((myself_ & (0x01 << bit_)) >> bit_) | neighbor_count_) == 3)
                {
                    result_ |= (0x01 << bit_);
                }
            }
            //right most bit
            bit_ = 7;
            neighbor_count_ = ((up_ & (0x01 << 1)) >> 1) + 
                             (up_ & (0x01)) + 
                             ((up_right_ & (0x01 << 7)) >> 7) + 
                             ((myself_ & (0x01 << 1)) >> 1) + 
                             ((right_ & (0x01 << 7)) >> 7) + 
                             ((down_ & (0x01 << 1)) >> 1) + 
                             (down_ & (0x01)) + 
                             ((down_right_ & (0x01 << 7)) >> 7);
            if (((myself_ & (0x01)) | neighbor_count_) == 3)
            {
                result_ |= (0x01);
            }
            //store result in outboard
            outboard[idx_] = result_;
        }

        ////////// middle rows, last col ///////////
        //col_ = ncols_bit - 1;
        idx_ = row * ncols_bit + col_;
        up_left_ = inboard[idx_ - ncols_bit - 1];
        up_ = inboard[idx_ - ncols_bit];
        up_right_ = inboard[idx_ -ncols_bit - ncols_bit + 1];
        left_ = inboard[idx_ - 1];
        myself_ = inboard[idx_];
        right_ = inboard[idx_ - ncols_bit + 1];
        down_left_ = inboard[idx_ + ncols_bit - 1];
        down_ = inboard[idx_ + ncols_bit];
        down_right_ = inboard[idx_ + 1];
        result_ = (char) 0;
        //loop through the 8 bits
        //left most bit
        bit_ = 0;
        neighbor_count_ = ((up_left_ & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                         ((up_ & (0x01 << 7)) >> 7) + 
                         ((up_ & (0x01 << 6)) >> 6) + 
                         ((left_ & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                         ((myself_ & (0x01 << 6)) >> 6) + 
                         ((down_left_ & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                         ((down_ & (0x01 << 7)) >> 7) + 
                         ((down_ & (0x01 << 6)) >> 6);
        if ((((myself_ & (0x01 << 7)) >> 7) | neighbor_count_) == 3)
        {
            result_ |= (0x01 << 7);
        }
        //middle bits, from right to left
        for (bit_ = (8 - num_bits_in_last_col + 1); bit_ < 7; bit_++)
        {
            neighbor_count_ = ((up_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                              ((up_ & (0x01 << bit_)) >> bit_) + 
                              ((up_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                              ((myself_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                              ((myself_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                              ((down_ & (0x01 << (bit_ - 1))) >> (bit_ - 1)) + 
                              ((down_ & (0x01 << bit_)) >> bit_) + 
                              ((down_ & (0x01 << (bit_ + 1))) >> (bit_ + 1));
            if ((((myself_ & (0x01 << bit_)) >> bit_) | neighbor_count_) == 3)
            {
                result_ |= (0x01 << bit_);
            }
        }
        //right most bit
        bit_ = 8 - num_bits_in_last_col;
        neighbor_count_ = ((up_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                         ((up_ & (0x01 << (bit_))) >> (bit_)) + 
                         ((up_right_ & (0x01 << 7)) >> 7) + 
                         ((myself_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                         ((right_ & (0x01 << 7)) >> 7) + 
                         ((down_ & (0x01 << (bit_ + 1))) >> (bit_ + 1)) + 
                         ((down_ & (0x01 << (bit_))) >> (bit_)) + 
                         ((down_right_ & (0x01 << 7)) >> 7);
        if ((((myself_ & (0x01 << bit_)) >> bit_) | neighbor_count_) == 3)
        {
            result_ |= (0x01 << bit_);
        }
        //store result in outboard
        outboard[idx_] = result_;
    }

    ///////////////////////////////// LAST ROW /////////////////////////////////
    row = nrows - 1;
    ////////// last row, first col ///////////
    col = 0;
    idx = row * ncols_bit + col;
    up_left_ = inboard[idx_ - 1];
    up_ = inboard[idx_ - ncols_bit];
    up_right_ = inboard[idx_ - ncols_bit + 1];
    left = inboard[idx + ncols_bit - 1];
    myself = inboard[idx];
    right = inboard[idx + 1];
    down_left = inboard[ncols_bit - 1];
    down = inboard[0];
    down_right = inboard[1];
    result = (char) 0;
    //loop through the 8 bits
    //left most bit
    bit = 0;
    neighbor_count = ((up_left & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
                     ((up & (0x01 << 7)) >> 7) + 
                     ((up & (0x01 << 6)) >> 6) + 
                     ((left & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
                     ((myself & (0x01 << 6)) >> 6) + 
                     ((down_left & (0x01 << (8 - num_bits_in_last_col))) >> num_bits_in_last_col) + 
                     ((down & (0x01 << 7)) >> 7) + 
                     ((down & (0x01 << 6)) >> 6);
    if ((((myself & (0x01 << 7)) >> 7) | neighbor_count) == 3)
    {
        result |= (0x01 << 7);
    }
    //middle bits, from right to left
    for (bit = 1; bit < 7; bit++)
    {
        neighbor_count = ((up & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((up & (0x01 << bit)) >> bit) + 
                          ((up & (0x01 << (bit + 1))) >> (bit + 1)) + 
                          ((myself & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((myself & (0x01 << (bit + 1))) >> (bit + 1)) + 
                          ((down & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((down & (0x01 << bit)) >> bit) + 
                          ((down & (0x01 << (bit + 1))) >> (bit + 1));
        if ((((myself & (0x01 << bit)) >> bit) | neighbor_count) == 3)
        {
            result |= (0x01 << bit);
        }
    }
    //right most bit
    bit = 7;
    neighbor_count = ((up & (0x01 << 1)) >> 1) + 
                     (up & (0x01)) + 
                     ((up_right & (0x01 << 7)) >> 7) + 
                     ((myself & (0x01 << 1)) >> 1) + 
                     ((right & (0x01 << 7)) >> 7) + 
                     ((down & (0x01 << 1)) >> 1) + 
                     (down & (0x01)) + 
                     ((down_right & (0x01 << 7)) >> 7);
    if (((myself & (0x01)) | neighbor_count) == 3)
    {
        result |= (0x01);
    }
    //store result in outboard
    outboard[idx] = result;

    ////////// last row, middle cols ///////////
    for (col = 1; col < ncols_bit - 1; col++)
    {
        idx = row * ncols_bit + col;
        up_left_ = inboard[idx_ - ncols_bit - 1];
        up_ = inboard[idx_ - ncols_bit];
        up_right_ = inboard[idx_ - ncols_bit + 1];
        left = inboard[idx - 1];
        myself = inboard[idx];
        right = inboard[idx + 1];
        down_left = inboard[col - 1];
        down = inboard[col];
        down_right = inboard[col + 1];
        result = (char) 0;
        //loop through the 8 bits
        //left most bit
        bit = 0;
        neighbor_count = (up_left & (0x01)) + 
                              ((up & (0x01 << 7)) >> 7) + 
                              ((up & (0x01 << 6)) >> 6) + 
                              (left & (0x01)) + 
                              ((myself & (0x01 << 6)) >> 6) + 
                              (down_left & (0x01)) + 
                              ((down & (0x01 << 7)) >> 7) + 
                              ((down & (0x01 << 6)) >> 6);
        if ((((myself & (0x01 << 7)) >> 7) | neighbor_count) == 3)
        {
            result |= (0x01 << 7);
        }
        //middle bits, from right to left
        for (bit = 1; bit < 7; bit++)
        {
            neighbor_count = ((up & (0x01 << (bit - 1))) >> (bit - 1)) + 
                                  ((up & (0x01 << bit)) >> bit) + 
                                  ((up & (0x01 << (bit + 1))) >> (bit + 1)) + 
                                  ((myself & (0x01 << (bit - 1))) >> (bit - 1)) + 
                                  ((myself & (0x01 << (bit + 1))) >> (bit + 1)) + 
                                  ((down & (0x01 << (bit - 1))) >> (bit - 1)) + 
                                  ((down & (0x01 << bit)) >> bit) + 
                                  ((down & (0x01 << (bit + 1))) >> (bit + 1));
            if ((((myself & (0x01 << bit)) >> bit) | neighbor_count) == 3)
            {
                result |= (0x01 << bit);
            }
        }
        //right most bit
        bit = 7;
        neighbor_count = ((up & (0x01 << 1)) >> 1) + 
                         (up & (0x01)) + 
                         ((up_right & (0x01 << 7)) >> 7) + 
                         ((myself & (0x01 << 1)) >> 1) + 
                         ((right & (0x01 << 7)) >> 7) + 
                         ((down & (0x01 << 1)) >> 1) + 
                         (down & (0x01)) + 
                         ((down_right & (0x01 << 7)) >> 7);
        if (((myself & (0x01)) | neighbor_count) == 3)
        {
            result |= (0x01);
        }
        //store result in outboard
        outboard[idx] = result;
    }

    ////////// last row, last col ///////////
    //col = ncols_bit - 1;
    idx = row * ncols_bit + col;
    up_left = inboard[(nrows - 1)*(ncols_bit) + ncols_bit - 2];
    up = inboard[(nrows - 1)*(ncols_bit) + ncols_bit - 1];
    up_right = inboard[(nrows - 1)*(ncols_bit)];
    left = inboard[idx - 1];
    myself = inboard[idx];
    right = inboard[idx - ncols_bit + 1];
    down_left = inboard[ncols_bit - 2];
    down = inboard[ncols_bit - 1];
    down_right = inboard[0];
    result = (char) 0;
    //loop through the 8 bits
    //left most bit
    bit = 0;
    neighbor_count = ((up_left & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                     ((up & (0x01 << 7)) >> 7) + 
                     ((up & (0x01 << 6)) >> 6) + 
                     ((left & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                     ((myself & (0x01 << 6)) >> 6) + 
                     ((down_left & (0x01 << num_bits_in_last_col)) >> num_bits_in_last_col) + 
                     ((down & (0x01 << 7)) >> 7) + 
                     ((down & (0x01 << 6)) >> 6);
    if ((((myself & (0x01 << 7)) >> 7) | neighbor_count) == 3)
    {
        result |= (0x01 << 7);
    }
    //middle bits, from right to left
    for (bit = (8 - num_bits_in_last_col + 1); bit < 7; bit++)
    {
        neighbor_count = ((up & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((up & (0x01 << bit)) >> bit) + 
                          ((up & (0x01 << (bit + 1))) >> (bit + 1)) + 
                          ((myself & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((myself & (0x01 << (bit + 1))) >> (bit + 1)) + 
                          ((down & (0x01 << (bit - 1))) >> (bit - 1)) + 
                          ((down & (0x01 << bit)) >> bit) + 
                          ((down & (0x01 << (bit + 1))) >> (bit + 1));
        if ((((myself & (0x01 << bit)) >> bit) | neighbor_count) == 3)
        {
            result |= (0x01 << bit);
        }
    }
    //right most bit
    bit = 8 - num_bits_in_last_col;
    neighbor_count = ((up & (0x01 << (bit + 1))) >> (bit + 1)) + 
                     ((up & (0x01 << (bit))) >> (bit)) + 
                     ((up_right & (0x01 << 7)) >> 7) + 
                     ((myself & (0x01 << (bit + 1))) >> (bit + 1)) + 
                     ((right & (0x01 << 7)) >> 7) + 
                     ((down & (0x01 << (bit + 1))) >> (bit + 1)) + 
                     ((down & (0x01 << (bit))) >> (bit)) + 
                     ((down_right & (0x01 << 7)) >> 7);
    if ((((myself & (0x01 << bit)) >> bit) | neighbor_count) == 3)
    {
        result |= (0x01 << bit);
    }
    //store result in outboard
    outboard[idx] = result;
    
    return;
}

char* bit_game_of_life (char* outboard, 
                        char* inboard,
                        const int nrows,
                        const int ncols,
                        const int gens_max)
{
    /* HINT: in the parallel decomposition, LDA may not be equal to
       nrows! */
    double timeStampA = getTimeStamp() ;

    //convert to bits representation
    int ncols_bit = (ncols+7)/8;
    char* bit_inboard = malloc(nrows * ncols_bit * sizeof(char));
    char* bit_outboard = malloc(nrows * ncols_bit * sizeof(char));

    #pragma omp parallel for
    for (unsigned int i = 0; i < nrows; i++)
    {
        char bit_result;
        unsigned int j;
        for (j = 0; j < ncols_bit - 1; j++)
        {
            bit_result = (char) 0;
            for (unsigned int bit_it = 0; bit_it < 7; bit_it++)
            {
                bit_result |= (inboard[i*ncols + j*8 + bit_it] & 0x01);
                bit_result <<= 1;
            }
            bit_result |= (inboard[i*ncols + j*8 + 7] & 0x01);
            bit_inboard[i*ncols_bit + j] = bit_result;
        }
        bit_result = (char) 0;
        for (unsigned int bit_it = 0; bit_it < 7; bit_it++)
        {
            if ((j*8 + bit_it) < ncols)
            {
                bit_result |= (inboard[i*ncols + j*8 + bit_it] & 0x01);
            }
            bit_result <<= 1;
        }
        if ((j*8 + 7) < ncols)
        {
            bit_result |= (inboard[i*ncols + j*8 + 7] & 0x01);
        }
        bit_inboard[i*ncols_bit + j] = bit_result;

    }

    int curgen;
    for (curgen = 0; curgen < gens_max; curgen++)
    {
        compute_next_gen_bit(bit_outboard, bit_inboard, nrows, ncols);
        //SWAP BOARDS
        char * temp = bit_inboard;
        bit_inboard = bit_outboard;
        bit_outboard = temp;
    }

    //convert back to char representation
    #pragma omp parallel for
    for (unsigned int i = 0; i < nrows; i++)
    {
        char bit_result;
        unsigned int j;
        for (j = 0; j < ncols_bit - 1; j++)
        {
            bit_result = bit_inboard[i*ncols_bit + j];
            for (unsigned int bit_it = 7; bit_it >= 0; bit_it--)
            {
                inboard[i*ncols + j*8 + bit_it] = bit_result & 0x01;
                bit_result >>= 1;
            }
        }
        bit_result = bit_inboard[i*ncols_bit + j];
        for (unsigned int bit_it = 7; bit_it >= 0; bit_it--)
        {
            if ((j*8 + bit_it) < ncols)
            {
                inboard[i*ncols + j*8 + bit_it] = bit_result & 0x01;
            }
            bit_result >>= 1;
        }
    }

    double timeStampD = getTimeStamp() ;
    double total_time = timeStampD - timeStampA;
    printf("CPU bit game_of_life: %.6f\n", total_time);
    /* 
     * We return the output board, so that we know which one contains
     * the final result (because we've been swapping boards around).
     * Just be careful when you free() the two boards, so that you don't
     * free the same one twice!!! 
     */

    return inboard;
}


