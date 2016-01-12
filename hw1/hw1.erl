-module(hw1).

-export([nthtail/2]).

nthtail(0, List) when is_list(List) -> List;
nthtail(N, [_|T]) when (N > 0), (N =< length(T) + 1) -> nthtail(N-1, T).
