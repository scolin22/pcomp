-module(mini3).

-export([is_lol/1, is_rect/1, transpose/1]).

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
