/*****************************************************************************
 * life.c
 * Parallelized and optimized implementation of the game of life resides here
 ****************************************************************************/
#include "life.h"
#include "util.h"
#include <string.h>
#include "pthread.h"
#include <sys/time.h>
#include <stddef.h>

#define LIFE_MASK 0x01 //0b00000001
#define ALIVE_MASK 0x01 //0b00000001
#define DEAD_MASK 0xfe //0b11111110
#define SET_ALIVE(board_cell) ((board_cell) |= ALIVE_MASK)
#define SET_DEAD(board_cell) ((board_cell) &= DEAD_MASK)
#define NEIGHBOR_MASK 0x1e //0b00011110
#define NEIGHBOR_SHIFT 1
#define INCREMENT_NEIGHBOR(board_cell) ((board_cell) += 2)
//#define GET_NEIGHBOR_COUNT(board_cell) (((board_cell) & NEIGHBOR_MASK) >> NEIGHBOR_SHIFT)
#define GET_NEIGHBOR_COUNT(board_cell) (((board_cell) & NEIGHBOR_MASK))
//NOTE: every increment/decrement of neighbor count means increment/decrement of 2 in actual board value.

//#define SHIRDEBUG 1

#include <stdio.h>

// returns time in seconds
static double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL) ;
    return (double) tv.tv_usec/1000000.0 + tv.tv_sec ;
}

pthread_mutex_t * row_locks = NULL;

/*****************************************************************************
 * Helper function definitions
 ****************************************************************************/
void compute_next_gen(char* outboard, char* inboard, const int nrows, const int ncols)
{
    //CLEAR OUT OUTBOARD. (Neibors = 0, live = 0 = dead)
    memset(outboard, 0, nrows * ncols * sizeof(char));

    unsigned int row = 0;
    unsigned int col = 0;
    unsigned int nrows_minus_one = nrows-1;
    unsigned int ncols_minus_one = ncols-1;
    unsigned int left, right, up, down;
    unsigned int row_nrows;
    unsigned int up_nrows;
    unsigned int down_nrows;

#ifdef SHIRDEBUG
    printf("initial inboard\n");
    for (unsigned int i = 0; i < nrows; i++)
    {
        for (unsigned int j = 0; j < ncols; j++)
        {
            printf("[%d]", inboard[i * nrows + j]);
        }
        printf("\n");
    }
#endif

////////////////////////////////////////////////////////////////////////////////////////
// 1) COMPUTE NEIGHBOR COUNT FOR NEXT GEN AND SET TO ALIVE IF WAS ALIVE IN CURRENT GEN
////////////////////////////////////////////////////////////////////////////////////////
    //compute first row ==========================
    row_nrows = row * nrows;
    up_nrows = nrows_minus_one * nrows; //up = nrows_minus_one;
    down_nrows = nrows; //down = 1;
    //compute first col
    col = 0;
    if (inboard[row_nrows + col] & LIFE_MASK)
    {
        SET_ALIVE(outboard[row_nrows + col]);
        left = ncols_minus_one;
        right = col + 1;
        INCREMENT_NEIGHBOR(outboard[up_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[up_nrows + col]);
        INCREMENT_NEIGHBOR(outboard[up_nrows + right]);
        INCREMENT_NEIGHBOR(outboard[row_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[row_nrows + right]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + col]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + right]);
    }
    //compute middle cols
    for (col = 1; col < ncols_minus_one; col++)
    {
        if (inboard[row_nrows + col] & LIFE_MASK)
        {
            SET_ALIVE(outboard[row_nrows + col]);
            left = col - 1;
            right = col + 1;
            INCREMENT_NEIGHBOR(outboard[up_nrows + left]);
            INCREMENT_NEIGHBOR(outboard[up_nrows + col]);
            INCREMENT_NEIGHBOR(outboard[up_nrows + right]);
            INCREMENT_NEIGHBOR(outboard[row_nrows + left]);
            INCREMENT_NEIGHBOR(outboard[row_nrows + right]);
            INCREMENT_NEIGHBOR(outboard[down_nrows + left]);
            INCREMENT_NEIGHBOR(outboard[down_nrows + col]);
            INCREMENT_NEIGHBOR(outboard[down_nrows + right]);
        }
    }
    //compute last col
    if (inboard[row_nrows + col] & LIFE_MASK)
    {
        SET_ALIVE(outboard[row_nrows + col]);
        left = col - 1;
        right = 0;
        INCREMENT_NEIGHBOR(outboard[up_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[up_nrows + col]);
        INCREMENT_NEIGHBOR(outboard[up_nrows + right]);
        INCREMENT_NEIGHBOR(outboard[row_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[row_nrows + right]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + col]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + right]);
    }
    //end of compute first row ====================

    //USE THREADS HERE
	#pragma omp parallel for private(row)
    //compute middle rows =========================
    for (unsigned int row_ = 1; row_ < nrows_minus_one; row_++)
    {
        register unsigned int row_nrows_ = row_ * nrows;
        register unsigned int row_minus_1 = row_ - 1;
        register unsigned int row_plus_1 = row_ + 1;
        register unsigned int up_nrows_ = (row_minus_1) * nrows; //up = row - 1;
        register unsigned int down_nrows_ = (row_plus_1) * nrows; //down = row + 1;
        //compute first col
        register unsigned int col_ = 0;
		register unsigned int left_, right_;
        if (inboard[row_nrows_ + col_] & LIFE_MASK)
        {
            left_ = ncols_minus_one;
            right_ = col_ + 1;
			pthread_mutex_lock(&row_locks[row_minus_1]);
	        INCREMENT_NEIGHBOR(outboard[up_nrows_ + left_]);
	        INCREMENT_NEIGHBOR(outboard[up_nrows_ + col_]);
	        INCREMENT_NEIGHBOR(outboard[up_nrows_ + right_]);
            pthread_mutex_unlock(&row_locks[row_minus_1]);

            pthread_mutex_lock(&row_locks[row_]);
            SET_ALIVE(outboard[row_nrows_ + col_]);
	        INCREMENT_NEIGHBOR(outboard[row_nrows_ + left_]);
	        INCREMENT_NEIGHBOR(outboard[row_nrows_ + right_]);
            pthread_mutex_unlock(&row_locks[row_]);

            pthread_mutex_lock(&row_locks[row_plus_1]);
	        INCREMENT_NEIGHBOR(outboard[down_nrows_ + left_]);
	        INCREMENT_NEIGHBOR(outboard[down_nrows_ + col_]);
	        INCREMENT_NEIGHBOR(outboard[down_nrows_ + right_]);
            pthread_mutex_unlock(&row_locks[row_plus_1]);
        }
        //compute middle cols
        for (col_ = 1; col_ < ncols_minus_one; col_++)
        {
            if (inboard[row_nrows_ + col_] & LIFE_MASK)
            {
                left_ = col_ - 1;
                right_ = col_ + 1;
                pthread_mutex_lock(&row_locks[row_minus_1]);
	            INCREMENT_NEIGHBOR(outboard[up_nrows_ + left_]);
	            INCREMENT_NEIGHBOR(outboard[up_nrows_ + col_]);
	            INCREMENT_NEIGHBOR(outboard[up_nrows_ + right_]);
                pthread_mutex_unlock(&row_locks[row_minus_1]);

                pthread_mutex_lock(&row_locks[row_]);
                SET_ALIVE(outboard[row_nrows_ + col_]);
	            INCREMENT_NEIGHBOR(outboard[row_nrows_ + left_]);
	            INCREMENT_NEIGHBOR(outboard[row_nrows_ + right_]);
                pthread_mutex_unlock(&row_locks[row_]);

                pthread_mutex_lock(&row_locks[row_plus_1]);
	            INCREMENT_NEIGHBOR(outboard[down_nrows_ + left_]);
	            INCREMENT_NEIGHBOR(outboard[down_nrows_ + col_]);
	            INCREMENT_NEIGHBOR(outboard[down_nrows_ + right_]);
                pthread_mutex_unlock(&row_locks[row_plus_1]);
            }
        }
        //compute last col
        if (inboard[row_nrows_ + col_] & LIFE_MASK)
        {
            left_ = col_ - 1;
            right_ = 0;
            pthread_mutex_lock(&row_locks[row_minus_1]);
	        INCREMENT_NEIGHBOR(outboard[up_nrows_ + left_]);
	        INCREMENT_NEIGHBOR(outboard[up_nrows_ + col_]);
	        INCREMENT_NEIGHBOR(outboard[up_nrows_ + right_]);
            pthread_mutex_unlock(&row_locks[row_minus_1]);

            pthread_mutex_lock(&row_locks[row_]);
            SET_ALIVE(outboard[row_nrows_ + col_]);
	        INCREMENT_NEIGHBOR(outboard[row_nrows_ + left_]);
	        INCREMENT_NEIGHBOR(outboard[row_nrows_ + right_]);
            pthread_mutex_unlock(&row_locks[row_]);

            pthread_mutex_lock(&row_locks[row_plus_1]);
	        INCREMENT_NEIGHBOR(outboard[down_nrows_ + left_]);
	        INCREMENT_NEIGHBOR(outboard[down_nrows_ + col_]);
	        INCREMENT_NEIGHBOR(outboard[down_nrows_ + right_]);
            pthread_mutex_unlock(&row_locks[row_plus_1]);
        }
    }
    //end of compute middle rows ==================
    //THREADS SHOULD MERGE HERE

    //compute last row ============================
	row = nrows_minus_one;
    row_nrows = row * nrows;
    up_nrows = (row - 1) * nrows; //up = row - 1;
    down_nrows = 0;////down = 0;
    //compute first col
    col = 0;
    if (inboard[row_nrows + col] & LIFE_MASK)
    {
        SET_ALIVE(outboard[row_nrows + col]);
        left = ncols_minus_one;
        right = col + 1;
        INCREMENT_NEIGHBOR(outboard[up_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[up_nrows + col]);
        INCREMENT_NEIGHBOR(outboard[up_nrows + right]);
        INCREMENT_NEIGHBOR(outboard[row_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[row_nrows + right]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + col]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + right]);
    }
    //compute middle cols
    for (col = 1; col < ncols_minus_one; col++)
    {
        if (inboard[row_nrows + col] & LIFE_MASK)
        {
            SET_ALIVE(outboard[row_nrows + col]);
            left = col - 1;
            right = col + 1;
            INCREMENT_NEIGHBOR(outboard[up_nrows + left]);
            INCREMENT_NEIGHBOR(outboard[up_nrows + col]);
            INCREMENT_NEIGHBOR(outboard[up_nrows + right]);
            INCREMENT_NEIGHBOR(outboard[row_nrows + left]);
            INCREMENT_NEIGHBOR(outboard[row_nrows + right]);
            INCREMENT_NEIGHBOR(outboard[down_nrows + left]);
            INCREMENT_NEIGHBOR(outboard[down_nrows + col]);
            INCREMENT_NEIGHBOR(outboard[down_nrows + right]);
        }
    }
    //compute last col
    if (inboard[row_nrows + col] & LIFE_MASK)
    {
        SET_ALIVE(outboard[row_nrows + col]);
        left = col - 1;
        right = 0;
        INCREMENT_NEIGHBOR(outboard[up_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[up_nrows + col]);
        INCREMENT_NEIGHBOR(outboard[up_nrows + right]);
        INCREMENT_NEIGHBOR(outboard[row_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[row_nrows + right]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + left]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + col]);
        INCREMENT_NEIGHBOR(outboard[down_nrows + right]);
    }
    //end of compute last row =====================
#ifdef SHIRDEBUG
    printf("end of calculating neighbors\n");
    printf("all value\n");
    for (unsigned int i = 0; i < nrows; i++)
    {
        for (unsigned int j = 0; j < ncols; j++)
        {
            printf("[%x]", outboard[i * nrows + j]);
        }
        printf("\n");
    }
#endif


#ifdef SHIRDEBUG
    printf("GET_NEIGHBOR_COUNT\n");
    for (unsigned int i = 0; i < nrows; i++)
    {
        for (unsigned int j = 0; j < ncols; j++)
        {
            printf("[%x]", GET_NEIGHBOR_COUNT(outboard[i * nrows + j]));
        }
        printf("\n");
    }
#endif

#ifdef SHIRDEBUG
    printf("was alive\n");
    for (unsigned int i = 0; i < nrows; i++)
    {
        for (unsigned int j = 0; j < ncols; j++)
        {
            printf("[%x]", outboard[i * nrows + j] & 0x01);
        }
        printf("\n");
    }
#endif


////////////////////////////////////////////////////////////////////////////////////////
// 2) COMPUTE LIFE IN NEXT GEN BASED ON NEIGHBOR AND PREV LIVENESS
////////////////////////////////////////////////////////////////////////////////////////
    //USE THREADS HERE
    // unsigned int num_of_neighbors;
    // char was_alive;
    #pragma omp parallel for
    for (unsigned int i = 0; i < nrows * ncols; i++)
    {
        register unsigned int num_of_neighbors;
        register char was_alive;
        num_of_neighbors = GET_NEIGHBOR_COUNT(outboard[i]);
        if (num_of_neighbors == 6)
        {
            SET_ALIVE(outboard[i]);
        }
        //else if (num_of_neighbors == 4)
        //{
			//doesn't need to do anything b/c alive > alive, dead -> dead
            //was_alive = (outboard[i] & 0x01);
            //if (!was_alive)
            //{
            //    SET_DEAD(outboard[i]);
            //}
            //else
            //{
            //    SET_DEAD(outboard[i]);
            //}
        //}
        else if (num_of_neighbors != 4)
        {
#ifdef SHIRDEBUG
            printf("reached SET_DEAD, num of neighbors != 2 and != 3\n");
            printf("outboard[i] before set dead = [%d]\n", outboard[i]);
#endif
            SET_DEAD(outboard[i]);
#ifdef SHIRDEBUG
            printf("outboard[i] after set dead = [%d]\n", outboard[i]);
#endif
        }
    }
    //THREADS SHOULD MERGE HERE 
#ifdef SHIRDEBUG
    printf("end of calculating new life\n");
    for (unsigned int i = 0; i < nrows; i++)
    {
        for (unsigned int j = 0; j < ncols; j++)
        {
            printf("[%x]", outboard[i * nrows + j]);
        }
        printf("\n");
    }
#endif

}

/*****************************************************************************
 * Game of life implementation
 ****************************************************************************/
char* game_of_life (char* outboard, char* inboard, const int nrows, const int ncols, const int gens_max)
{
    /* HINT: in the parallel decomposition, LDA may not be equal to
       nrows! */
    double timeStampA = getTimeStamp() ;
    register int curgen;

    row_locks = (pthread_mutex_t *) malloc(nrows * sizeof(pthread_mutex_t));

    register unsigned int i = 0;
    #pragma omp parallel for
    for (i = 0; i < nrows; ++i)
    {
        row_locks[i] = (pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
    }

    for (curgen = 0; curgen < gens_max; curgen++)
    {
        compute_next_gen(outboard, inboard, nrows, ncols);
        //SWAP BOARDS
        char * temp = inboard;
        inboard = outboard;
        outboard = temp;
    }
    /* 
     * We return the output board, so that we know which one contains
     * the final result (because we've been swapping boards around).
     * Just be careful when you free() the two boards, so that you don't
     * free the same one twice!!! 
     */

    //CLEAR the neighbor count in final result so we only keep life information
    register unsigned int j = 0;
    #pragma omp parallel for
    for (j = 0; j < nrows * ncols; j++)
    {
        inboard[j] &= 0x01;
    }
    double timeStampD = getTimeStamp() ;
    double total_time = timeStampD - timeStampA;
    printf("CPU optimized game_of_life: %.6f\n", total_time);
    return inboard;
}


//////////////////////////////////////////////////////////////////////////////////////////
//                         BIT VERSION 
//////////////////////////////////////////////////////////////////////////////////////////


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
    int bit;
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
                     ((down_left & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
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
    neighbor_count = ((up_left & (0x01))) + 
                     ((up & (0x01 << 7)) >> 7) + 
                     ((up & (0x01 << 6)) >> 6) + 
                     ((left & (0x01))) + 
                     ((myself & (0x01 << 6)) >> 6) + 
                     ((down_left & (0x01))) + 
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
    #pragma omp parallel for
    for (row = 1; row < nrows - 1; row++)
    {
        unsigned int col_;
        char myself_, up_left_, up_, up_right_, left_, right_, down_left_, down_, down_right_;
        unsigned int idx_;
        char result_;
        int bit_;
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
                         ((down_left_ & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
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
        neighbor_count_ = ((up_left_ & (0x01))) + 
                         ((up_ & (0x01 << 7)) >> 7) + 
                         ((left_ & (0x01))) + 
                         ((down_left_ & (0x01))) + 
                         ((down_ & (0x01 << 7)) >> 7);
        if (num_bits_in_last_col > 1)
        {
            neighbor_count_ += ((up_ & (0x01 << 6)) >> 6) + 
                               ((myself_ & (0x01 << 6)) >> 6) + 
                               ((down_ & (0x01 << 6)) >> 6);
        }
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
        //what if only 1 bit in last column .....
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
    up_left = inboard[idx - 1];
    up = inboard[idx - ncols_bit];
    up_right = inboard[idx - ncols_bit + 1];
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
                     ((down_left & (0x01 << (8 - num_bits_in_last_col))) >> (8 - num_bits_in_last_col)) + 
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
        up_left = inboard[idx - ncols_bit - 1];
        up = inboard[idx - ncols_bit];
        up_right = inboard[idx - ncols_bit + 1];
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
    up_left = inboard[idx - ncols_bit - 1];
    up = inboard[idx - ncols_bit];
    up_right = inboard[idx - ncols_bit - ncols_bit + 1];
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
    neighbor_count = ((up_left & (0x01))) + 
                     ((up & (0x01 << 7)) >> 7) + 
                     ((up & (0x01 << 6)) >> 6) + 
                     ((left & (0x01))) + 
                     ((myself & (0x01 << 6)) >> 6) + 
                     ((down_left & (0x01))) + 
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
            for (int bit_it = 7; bit_it >= 0; bit_it--)
            {
                inboard[i*ncols + j*8 + bit_it] = bit_result & 0x01;
                bit_result >>= 1;
            }
        }
        bit_result = bit_inboard[i*ncols_bit + j];
        for (int bit_it = 7; bit_it >= 0; bit_it--)
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
    printf("CPU game_of_life: %.6f\n", total_time);
    /* 
     * We return the output board, so that we know which one contains
     * the final result (because we've been swapping boards around).
     * Just be careful when you free() the two boards, so that you don't
     * free the same one twice!!! 
     */

    return inboard;
}
