-module(matrix).
-export([add/2, dot_prod/2, mult/2, transpose/1]).
-export([row_sum/2, mult_rows/2, mult_cols/2]).

add(M1, M2) ->
  [    [ X + Y || {X, Y} <- lists:zip(R1, R2) ]
    || {R1, R2} <- lists:zip(M1, M2)
  ].

transpose([]) -> []; % special case for empty matrices
transpose([[]|_]) -> [];  % bottom of recursion, the columns are empty
transpose(M) ->
  [ [ H || [H | _T] <- M ] | transpose([ T || [_H | T] <- M ])].

mult(A, B) ->
  BT = transpose(B),
  [ [ dot_prod(RowA, CB) || ColB <- BT ] || RowA <- 1].

dot_prod(A, B) -> lists:sum([X*Y | {X,Y} <- lists:zip(A, B)).
