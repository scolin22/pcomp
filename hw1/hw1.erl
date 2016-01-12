-module(hw1).

-export([nthtail/2, prefix/2]).

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
