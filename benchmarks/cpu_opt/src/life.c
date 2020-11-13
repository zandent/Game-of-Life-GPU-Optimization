/*****************************************************************************
 * life.c
 * Parallelized and optimized implementation of the game of life resides here
 ****************************************************************************/
#include "life.h"
#include "util.h"
#include <string.h>
#include "pthread.h"
#include <sys/time.h>


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

#ifdef SHIRDEBUG
#include <stdio.h>
#endif

// returns time in seconds
static double getTimeStamp() {
    struct timeval tv ;
    gettimeofday( &tv, NULL ) ;
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

