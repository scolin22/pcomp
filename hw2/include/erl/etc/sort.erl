-module(sort).
-export([qsort/1, qsortc/1, time/1, time/0]).
-export([qsort/2]).

%% @doc quicksort, without comprehensions
qsort(List) -> qsort(List, []).

qsort([X], Suffix) -> [X | Suffix];
qsort([Pivot | T], Suffix) -> 
  {Lo, Hi} = lists:partition(fun(X) -> X < Pivot end, T),
  qsort(Lo, [Pivot | qsort(Hi, Suffix)]);
qsort([], Suffix) -> Suffix.

%% @doc quicksort, without comprehensions, second attempt
qsortp(List) -> qsortp(List, []).

qsortp([X], Suffix) -> [X | Suffix];
qsortp([Pivot | T], Suffix) -> 
  {Lo, Hi} = partition(Pivot, T, {[], []}),
  qsortp(Lo, [Pivot | qsortp(Hi, Suffix)]);
qsortp([], Suffix) -> Suffix.

partition(_Pivot, [], {Lo, Hi}) -> {Lo, Hi};
partition(Pivot, [H | T], {Lo, Hi}) ->
  if
    H < Pivot -> partition(Pivot, T, {[H | Lo], Hi});
    true ->      partition(Pivot, T, {Lo, [H | Hi]})
  end.

%% @doc quicksort, version from comprehension example
qsortc([Pivot|T]) -> qsortc([ X || X <- T, X < Pivot]) ++
    [Pivot] ++
    qsortc([ X || X <- T, X >= Pivot]);
qsortc([]) -> [].


%% @doc time: compare the run times of the two implementations
time(N) ->
  R = misc:rlist(N, 1000000),
  TC = time_it:t(fun() -> qsortc(R) end),
  TQ = time_it:t(fun() -> qsortp(R) end),
  io:format("N = ~b~n", [N]),
  io:format("  with comprehensions: mean = ~10.4e, std = ~10.4e~n",
  	[ element(2, lists:keyfind('mean', 1, TC)),
  	  element(2, lists:keyfind('std', 1, TC)) ]),
  io:format("  plain quicksort:     mean = ~10.4e, std = ~10.4e~n",
  	[ element(2, lists:keyfind('mean', 1, TQ)),
  	  element(2, lists:keyfind('std', 1, TQ)) ]).
time() -> time(10000).
