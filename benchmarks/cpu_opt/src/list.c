#include "list.h"

void insert(unsigned short row, unsigned short col, node** changelist)
{
	node* new_node = (node*)malloc(sizeof(node));
	new_node->row = row;
	new_node->col = col;
	new_node->next = changelist;
	*chagelist = new_node;
}

void delete_list(node** changelist)
{
	node* current = *changelist;
	node* next;
	while (current)
	{
		next = current->next;
		free(current);
		current = next;
	}
	*changelist = NULL;
}