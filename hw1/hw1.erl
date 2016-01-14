% Worked with Kevin Hui 38604112
-module(hw1).

-export([nthtail/2, prefix/2, search/2, subtract/2]).
-export([time_sub/0]).

nthtail(0, List) when is_list(List) -> List;
nthtail(N, [_|T]) when (N > 0) -> nthtail(N-1, T).

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

subtract(L1, L2) ->
ML1 = lists:keysort(1, lists:zip(L1, lists:seq(1, length(L1)))),
ML2 = lists:keysort(1, lists:zip(L2, lists:seq(1, length(L2)))),
Sub_ML1 = tuple_subtract(ML1, ML2),
Rev_Sub_ML1 = lists:keysort(2, Sub_ML1),
{Ret, _} = lists:unzip(Rev_Sub_ML1),
Ret.

tuple_subtract([{K1,V1}|T1],     [{K2,_ }|_ ]=ML2) when K1 < K2 ->
    [{K1,V1}|tuple_subtract(T1, ML2)];
tuple_subtract([{K1,_ }|_ ]=ML1, [{K2,_ }|T2]    ) when K1 > K2 ->
    tuple_subtract(ML1, T2);
tuple_subtract([_|T1], [_|T2]) ->
    tuple_subtract(T1, T2);
tuple_subtract([], _) -> [];
tuple_subtract(ML1, []) -> ML1.

elapsed(T0, T1) ->
    1.0e-9*erlang:convert_time_unit(T1-T0, native, nano_seconds).

time_sub(N) when is_integer(N) ->
    L = lists:seq(1,N),
    T0 = erlang:monotonic_time(),
    R_sub = lists:subtract(L, lists:reverse(L)),
    T1 = erlang:monotonic_time(),
    R_csub = subtract(L, lists:reverse(L)),
    T2 = erlang:monotonic_time(),
    R_sub = R_csub, % make sure they match
    io:format("N = ~8B: time lists:subtract = ~10.6f, time lists:subtract = ~10.6f~n", [N, elapsed(T0, T1), elapsed(T1, T2)]),
    ok;
time_sub(L) when is_list(L) ->
    [time_sub(N) || N <- L],
    ok.
time_sub() -> time_sub([1000, 2000, 3000, 5000, 10000, 20000, 30000]).
