-module(mini3).

-export([is_lol/1, is_rect/1, transpose/1]).
-export([stop_while_your_ahead/2]).

is_lol([]) -> true;
is_lol([H|T]) when is_list(H) ->
    is_lol(T);
is_lol(_) -> false.

is_rect([]) -> true;
is_rect([H|T]) ->
    is_rect(T, length(H));
is_rect(_) -> false.

is_rect([], _) -> true;
is_rect([H|T], L) when length(H) == L ->
    is_rect(T, L);
is_rect(_, _) -> false.

transpose([]) -> [];
transpose(X) ->
    case is_rect(X) and is_lol(X) of
        true ->
            N2 = length(hd(X)),
            if
                N2 > 0 ->
                    [[lists:nth(N,R1) || R1 <- X] || N <- lists:seq(1, N2)];
                N2 == 0 -> []
            end;
        false ->
            erlang:error(bad_input)
    end.

goodness([]) -> 0;
goodness([good | Tl]) -> 1 + goodness(Tl);
goodness([bad | Tl]) -> -1 + goodness(Tl);
goodness([_ | Tl]) -> goodness(Tl);
goodness(X) when not is_list(X) -> goodness([X]).

stop_while_your_ahead(List, scan) ->
    W = wtree:create(length(List)),
    workers:update(W, raw_data, List),
    stop_while_your_ahead_scan(W, 0, raw_data, cooked_data),
    Res = lists:zip(workers:retrieve(W, cooked_data), lists:seq(1, length(List))),
    Max = lists:foldl(fun({M, A}, {MAcc, _}) when
    M > MAcc ->
      {M, A};
    (_, Acc) ->
      Acc
    end, {-1,0}, Res),
    Max;
stop_while_your_ahead(List, reduce) ->
    W = wtree:create(length(List)),
    workers:update(W, raw_data, List),
    stop_while_your_ahead_reduce(W, raw_data).

stop_while_your_ahead_scan(W, Acc, ListKey, GoodnessKey) ->
    wtree:scan(W,
        fun(ProcState) ->
            goodness(wtree:get(ProcState, ListKey))
        end,
        fun(ProcState, AccIn) ->
            wtree:put(ProcState, GoodnessKey, AccIn + goodness(wtree:get(ProcState, ListKey)))
        end,
        fun(Left, Right) ->
            Left + Right
        end,
        Acc
        ).

stop_while_your_ahead_reduce(W, ListKey) ->
    wtree:reduce(W,
        fun(ProcState) ->
            goodness(wtree:get(ProcState, ListKey))
        end,
        fun(Left, Right) ->
            Left + Right
        end
        ).
