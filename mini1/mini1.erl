-module(mini1).

-export([who_am_i/0, registration/0, flatten/1, reverse_hr/1, reverse_tr/1,
         time_rev/0, time_rev/1]).
-export([fib_hr/1, fib_tr/1, time_fib/1, time_fib/0]).

who_am_i() ->
  { "Mark Greenstreet",
    00000000,
    "mrg@cs.ubc.ca"
  }.

registration() -> enrolled.

flatten(X) ->
  if X == []        -> X;
     is_list(X)     -> flatten(hd(X)) ++ flatten(tl(X));
     not is_list(X) -> [X]
  end.

reverse_hr([]) -> [];
reverse_hr([H | T]) -> reverse_hr(T) ++ [H].

% write a tail-recursive version below
% reverse_tr(_) -> ok.
reverse_tr(L) -> reverse_tr(L, []).
reverse_tr([], R) -> R;
reverse_tr([H | T], R) -> reverse_tr(T, [H | R]).

elapsed(T0, T1) ->
  1.0e-9*erlang:convert_time_unit(T1-T0, native, nano_seconds).

time_rev(N) when is_integer(N) ->
  L = lists:seq(1,N),
  T0 = erlang:monotonic_time(),
  R_hr = reverse_hr(L),
  T1 = erlang:monotonic_time(),
  R_tr = reverse_tr(L),
  T2 = erlang:monotonic_time(),
  R_tr = R_hr, % make sure they match
  io:format("N = ~8B: time reverse_hr = ~10.6f, time reverse_tr = ~10.6f~n",
            [N, elapsed(T0, T1), elapsed(T1, T2)]),
  ok;
time_rev(L) when is_list(L) ->
  [time_rev(N) || N <- L],
  ok.
time_rev() -> time_rev([1000, 10000, 20000, 30000, 40000, 50000]).

% Let fib(N) denote the N^th fibonacci number.
% Here's a head-recursive implementation
fib_hr(0) -> 0;
fib_hr(N) -> element(1, fib_pair_hr(N)).

% fib_pair_hr(N) -> {fib(N), fib(N-1)}
fib_pair_hr(1) -> {1, 0};
fib_pair_hr(N) ->
  {F1, F2} = fib_pair_hr(N-1), % F1 = fib(N-1), F2 = fib(N-1)
  {F1 + F2, F1}.

% Now for the tail-recursive version.
fib_tr(N) -> ok.

time_fib(N) when is_integer(N) ->
  T0 = erlang:monotonic_time(),
  F_hr = fib_hr(N),
  T1 = erlang:monotonic_time(),
  F_tr = fib_tr(N),
  T2 = erlang:monotonic_time(),
  F_tr = F_hr, % make sure they match
  io:format("N = ~8B: time fib_hr = ~10.6f, time fib_tr = ~10.6f~n",
            [N, elapsed(T0, T1), elapsed(T1, T2)]),
  ok;
time_fib(L) when is_list(L) ->
  [time_fib(N) || N <- L],
  ok.
time_fib() -> time_fib([1000, 3000, 10000, 30000, 100000, 300000]).
