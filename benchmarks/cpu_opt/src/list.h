//shirley

typedef struct node {
	unsigned short row;
	unsigned short col;
	struct node * next;
} node;

void insert(char* board_cell, node** changelist);

void delete_list(node** changelist);