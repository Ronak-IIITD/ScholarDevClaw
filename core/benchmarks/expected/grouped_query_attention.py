EXPECTED_SYMBOLS = ("GroupedQueryAttention", "group_queries")


def group_queries(query_heads, group_size):
    return [query_heads[index : index + group_size] for index in range(0, len(query_heads), group_size)]


class GroupedQueryAttention:
    def __call__(self, query_heads, group_size):
        return group_queries(query_heads, group_size)
