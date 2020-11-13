#ifndef _life_opt_h
#define _life_opt_h
char* game_of_life_gpu (char* outboard, char* inboard, const int nrows, const int ncols, const int gens_max);
char* sequential_game_of_life (char* outboard, char* inboard, const int nrows, const int ncols, const int gens_max);
#endif /* _life_opt_h */

