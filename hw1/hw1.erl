-module(hw1).

-export([nthtail/2, prefix/2, search/2]).

nthtail(0, List) when is_list(List) -> List;
nthtail(N, [_|T]) when (N > 0), (N =< length(T) + 1) -> nthtail(N-1, T).

prefix([], List) when is_list(List) -> true;
prefix(List, []) when is_list(List) -> false;
prefix([H|T], List) when is_list(List) ->
    [HList|TList] = List,
    case H =:= HList of
        true->
            prefix(T, TList);
        false->
            false
    end.

search(List1, List2) when is_list(List1), is_list(List2) -> helper(List1, List2, [], 0).
helper([], [], List0, N) -> lists:reverse([N+1 | List0]);
helper(_, [], List0, _) -> lists:reverse(List0);
helper(List1, List2, List0, N) ->
    [_|T2] = List2,
    case prefix(List1, List2) of
        true->
            helper(List1, T2, [N+1 | List0], N+1);
        false->
            helper(List1, T2, List0, N+1)
    end.

