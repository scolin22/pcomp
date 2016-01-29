%% @doc Sequential implementation of count 3's from
%% <a href="http://www.amazon.ca/Principles-Parallel-Programming-Calvin-Lin/dp/0321487907">Lin &amp; Snyder</a>.
-module count3s.
-export [count3s/1, time_it/0, time_it/1, time_it/2].

%% @spec count3s(List) -> integer()
%% @doc Return the number of elements of List that are equal to the integer 3.
count3s([]) -> 0;
count3s([3 | Tail]) -> 1 + count3s(Tail);
count3s([_Other | Tail]) -> count3s(Tail).

%% @spec time_it(N,M) -> { N3S, T_elapsed }
%%   N3S = integer(),
%%   T_elapsed = float()
%% @doc Measure the elapsed time to run <code>count3s(N,M)</code>.
%%   This does not include the time to generate the random lists
%%   (which takes <b>much</b> longer than the <code>count3s</code> function).
time_it(N, M) ->
  R = misc:rlist(N, M),
  T0 = now(),
  N3S = count3s(R),
  T1 = now(),
  T_elapsed = 1.0e-6*timer:now_diff(T1, T0),
  {N3S, T_elapsed}.

%% @spec time_it(N) -> { N3S, T_elapsed }
%% @equiv time_it(N, 10)
time_it(N) -> time_it(N, 10).

%% @spec time_it() -> { N3S, T_elapsed }
%% @equiv time_it(1000000)
time_it() -> time_it(1000000).
