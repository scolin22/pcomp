%% @doc Parallel implemention count 3&#146;s from
%% <a href="http://www.amazon.ca/Principles-Parallel-Programming-Calvin-Lin/dp/0321487907">Lin &amp; Snyder</a>.
%% This version uses a pool of worker processes to first create a distributed
%% list of random integers, and then has each process count it&#146;s
%% list in parallel.  The root process combines the results.
%%
%% Note: while this implementation gets good speed-ups for very long lists;
%% the communication overhead is still high enough to make the code impractical.
%% We'll look at better implementations in upcoming lectures.

-module count3s_p2.
-export [rlist/4, count3s/2, time_it/3, time_it/2, time_it/1, time_it/0].
% -export [rlist1/4, rlist2/4, rlist3/4].  % can uncomment for debugging

%% @spec rlist(W, N, M, Key) -> ok
%% @doc  Generate a pseudo-random list distributed amongst the workers of
%%   worker pool <code>W</code>.
%% Parameters:
%% <ul>
%%   <li> <code>W</code>: a pool of worker processes.
%%   </li>
%%   <li> <code>N</code>: the total number of elements on the list.  Each
%%      worker gets 
%%      <code>N/{@link workers:nworkers/1. workers:nworkers}(W)</code>
%%      elements (rounded up or down so the total is <code>N</code>).
%%   </li>
%%   <li> <code>M</code>: each element is uniformly distributed in 1..M.
%%   </li>
%%   <li> <code>Key</code>: associate the random list with <code>Key</code>
%%     in the state of each worker process.
%%   </li>
%% </ul>
rlist(W, N, M, Key) -> rlist1({W, length(W)}, N, M, Key).

% rlist1(W, N, M, Key):
%   Like rlist, we figure out how many elements the first worker should
%   get, send it the appropriate task, and then make a recursive call to
%   handle handle the other workers.
rlist1({[], 0}, _N, _M, _Key) -> ok;
rlist1({[W_head | W_tail], LW}, N, M, Key) ->
  N1 = N div LW,
  N2 = N - N1,
  W_head ! (fun(ProcState) -> rlist2(N1, M, ProcState, Key) end),
  rlist1({W_tail, LW-1}, N2, M, Key).

% rlist2(N, M, ProcState, Key)
%   The worker task for generating a random list.
%   It the worker already has a random-number generator state, use it.
%   Otherwise, create one based on our PID.  That gives each worker a
%   different random sequence.
%   We associate the random list that we create with Key in ProcState.
rlist2(N, M, ProcState, Key) ->
  R0 = fun() ->  % generate a process-specific random seed if no seed in ProcState
    X = erlang:phash2(self()),
    {X, X*(X+17), X*(X-42)*(X+18780101)}
  end,
  V = rlist3(N, M, [], workers:get(ProcState, randomState, R0)), % make a list
  workers:put(ProcState, lists:zip([Key, randomState], V)). % update ProcState

% rlist3(N, M, List, RandState0)
%   Generate a random list of N elements uniformly distributed in 1..M.
%   Prepend this randome list to List.  Use RandState 0 as the seed for
%   the random number generator.  Return the random list and the updated
%   random number generator state.
rlist3(0, _M, List, RandState0) -> [List, RandState0];
rlist3(N, M, List, RandState0) ->
  {RandInt, RandState1} = random:uniform_s(M, RandState0),
  rlist3(N-1, M, [RandInt | List], RandState1).

%% @spec count3s(W, Key) -> integer()
%% @doc: count the number of 3's in a list.
%% Parameters:
%% <ul>
%%   <li>  <code>W</code>: a worker pool.
%%   </li>
%%   <li>  <code>Key</code>: count the 3's in the list associated with
%%	<code>Key</code> in each process.
%%   </li>
%% </ul>
%% Result: the total number of 3's in the lists of all of the workers.
count3s(W, Key) ->
  lists:sum(workers:retrieve(W,
    fun(ProcState) ->
      X = lists:keyfind(Key, 1, ProcState),
      if
        is_tuple(X) -> count3s:count3s(element(2, X));
	true -> io_lib:format("~w: error -- ~w not defined", self(), Key)
      end
    end)).

%% @spec time_it(Nworkers, Nelements, M) -> { N3S, T_elapsed }
%% @doc Measure the time to run <code>{@link count3s/2. count3s}</code>.
%% Parameters:
%% <ul>
%%   <li> <code>Nworkers</code>: how many worker processes to us.
%%   </li>
%%   <li> <code>Nelements</code>: the total number of elements on the random list.
%%   </li>
%%   <li> <code>M</code>: the random elements are uniform in 1..M.
%%   </li>
%% </ul>
%% Result:
%% <ul>
%%   <li> <code>N3S</code>: The total number of 3's across all workers for
%%	the random list.
%%   </li>
%%   <li> <code>T_elapsed</code>: The elapsed time '(i.e. wall-clock time)'
%%	to count the 3&#146;s.  This <b>does not</b> include the time to spawn
%%      the worker processes, generate the random lists, or reap the processes.
%%   </li>
%% </ul>
%%   
time_it(Nworkers, Nelements, M) ->
  W = workers:create(Nworkers),
  rlist(W, Nelements, M, 'R'),
  workers:retrieve(W, fun(_) -> ok end),
  T0 = now(),
  N3S = count3s(W, 'R'),
  T1 = now(),
  T_elapsed = 1.0e-6*timer:now_diff(T1, T0),
  workers:reap(W),
  {N3S, T_elapsed}.

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
