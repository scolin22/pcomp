%% @doc Parallel implemention count 3&#146;s from
%% <a href="http://www.amazon.ca/Principles-Parallel-Programming-Calvin-Lin/dp/0321487907">Lin &amp; Snyder</a>.
%% This version uses spawns a bunch of child processes, handing each of them
%% a piece of the list to be counted.  Because erlang <em>copies</em> parameters
%% when passing them to the child process, the extra overhead makes this version
%% <em>slower</em> than the sequential version in
%% {@link count3s:count3s/1. count3s:count3s}.
%% See {@link count3s:count3s/1. count3s_p2:count3s} for a parallel version
%% that actually runs faster than the sequential one.

-module count3s_p1.
-export [count3s/1, count3s/2, childProc/2, time_it/3].
-export [time_it/2, time_it/1, time_it/0].

% count3s: return the number of 3's in a list.
count3s(L0, _N0, 1, _MyPid) ->  % 1 processor
  count3s:count3s(L0);	      % just do it.
count3s(L0, N0, NProcs, MyPid) -> % more than one processor.
  % spawn a process to handle the first N/NProcs elements of L.
  % make a recursive call with NProcs-1 to handle the rest.
  N1 = N0 div NProcs,
  N2 = N0 - N1,
  {L1, L2} = lists:split(N1, L0),
  spawn(count3s_p1, childProc, [L1, MyPid]),
  C2 = count3s(L2, N2, NProcs-1, MyPid),
  receive  % get a value from a child process, and add it to C2.
    {count3s, C1} -> C1 + C2
  end.

% fill in default arguments.
%% @spec count3s(L, NProcs) -> integer()
%% @doc return the number of 3&#146;s in <code>L</code> using
%%   <code>NProcs</code> parallel processes to do the counting.
count3s(L, NProcs) -> count3s(L, length(L), NProcs, self()).

%% @spec count3s(List) -> integer()
%% @equiv count3s(List, Nsched)
%% @doc where <code>Nsched</code> is the number of sc
count3s(L) -> count3s(L, erlang:system_info(schedulers)).

% childProc(L, ParentPid):
%   count the number of threes in L and send the result to ParentPid.
childProc(L, ParentPid) -> ParentPid ! {count3s, count3s:count3s(L)}.

%% @spec time_it(Nworkers, Nelements, M) -> { N3S, T_elapsed }
%% @doc Measure the time to run <code>{@link count3s/2. count3s}</code>.
time_it(NProcs, Nelements, M) ->
  R = misc:rlist(Nelements, M),
  T0 = now(),
  N3S = count3s(R, NProcs),
  T1 = now(),
  Telapsed = 1.0e-6*timer:now_diff(T1, T0),
  {N3S, Telapsed}.

%% @spec time_it(Nelements, M) -> { N3S, T_elapsed }
%% @equiv time_it(Nsched, Nelements, M)
%% @doc Measure the time to run <code>{@link count3s/2. count3s}</code>.
%% <code>Nsched</code> is the number of schedulers in the erlang runtime.
%% (See <a href="http://www.erlang.org/doc/man/erlang.html#system_info-1">erlang:system_info</a>(schedulers).)
time_it(Nelements, M) -> time_it(erlang:system_info(schedulers), Nelements, M).

%% @spec time_it(Nelements) -> { N3S, T_elapsed }
%% @equiv time_it(Nelements, 10)
time_it(Nelements) -> time_it(Nelements, 10).

%% @spec time_it() -> { N3S, T_elapsed }
%% @equiv time_it(1000000)
time_it() -> time_it(1000000).
