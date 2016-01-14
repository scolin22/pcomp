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

search(List1, List2) when is_list(List1), is_list(List2) -> helper(List1, List2, [], 1).
helper([], [], List0, N) -> lists:reverse([N | List0]);
helper(_, [], List0, _) -> lists:reverse(List0);
helper(List1, List2, List0, N) ->

    case prefix(List1, List2) of
        true->
            Off = max(length(List1), 1),
            T2 = nthtail(Off, List2),
            helper(List1, T2, [N | List0], N+Off);
        false->
            [_|T2] = List2,
            helper(List1, T2, List0, N+1)
    end.

