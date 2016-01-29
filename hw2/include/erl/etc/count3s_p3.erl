%% @doc Parallel implemention count 3's from
%% Principles fo Parallel Programming, by C. Lin and L. Snyder.
%% This version uses a pool of worker processes to first create a distributed
%% list of random integers, and then has each process count it's
%% list in parallel.  The root process combines the results.
%%
%% This module is a copy of count3s_p2, with changes made to do more
%% extensive timing measurements.

-module(count3s_p3).
-export([rlist/4, count3s/2, test/2, test/1, test/0, to_matlab/1]).
-export([count3s_reduce/2, count3s_reduce_log/2, count3s_brute_force/2]).
-export([count3s_brute_log/2, test2/2]).

% -export [rlist1/4, rlist2/4, rlist3/4].  % can uncomment for debugging

%% @spec rlist(W, N, M, Key) -> ok
%% @doc  Generate a pseudo-random list distributed amongst the workers of
%%   worker pool W.  N is th etotal numbr of elements in the list, and
%%   each element is uniformly distributed in 1..M.
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

%% count3s(W, Key) -> integer()
%% count the number of 3's in a list.
%% Parameters:
%% <ul>
%%   <li>  W: a worker pool.
%%   </li>
%%   <li>  Key: count the 3's in the list associated with Key.
%%   </li>
%% </ul>
%% Result: the total number of 3's in the lists of all of the workers.
count3s_brute_force(W, Key) ->
  lists:sum(workers:retrieve(W,
    fun(ProcState) ->
      case workers:get(ProcState, Key) of
        undefined -> failed;
        X -> count3s:count3s(X)
      end
    end)).

count3s_brute_log(W, Key) ->
  Log0 = time_it:log("start"),
  {V, Log1} = wlog:retrieve(W,
    fun(ProcState) ->
      L0 = time_it:log("start"),
      N3S = case workers:get(ProcState, Key) of
	undefined ->
	  failed;
	X -> count3s:count3s(X)
      end,
      L1 = time_it:log(L0, "finished, N3S = ~w", [N3S]),
      {N3S, L1}
    end),
  {Counts, WorkerLogs} = lists:unzip(V),
  N3S = lists:sum(Counts),
  Log2 = time_it:log([Log0, Log1, WorkerLogs],
  	    "finished, total 3's = ~w", [N3S]),
  { N3S, Log2 }.


count3s_reduce(W, Key) ->
  workers:reduce(W,
    fun(ProcState) -> 
      case workers:get(ProcState, Key) of
        undefined ->
	  io_lib:format("~w: error -- ~w not defined", self(), Key),
	  failed;
        X -> count3s:count3s(X)
      end
    end,
    fun(Left, Right) -> Left + Right end
  ).

count3s_reduce_log(W, Key) ->
  wlog:reduce(W,
    fun(ProcState) -> 
      case workers:get(ProcState, Key) of
        undefined ->
	  io_lib:format("~w: error -- ~w not defined", self(), Key),
	  failed;
        X -> count3s:count3s(X)
      end
    end,
    fun(Left, Right) -> Left + Right end
  ).

count3s(W, Key) -> count3s_brute_force(W, Key).


test(N, NWorkers) when is_list(N) and is_list(NWorkers) ->
  M = 10,
  W = workers:create(lists:max(NWorkers)),
  T = [ [ test_help(W, NW, NN, M) || NN <- N ] || NW <- NWorkers ],
  workers:reap(W),
  T;
test(N, NWorkers) when is_integer(N) ->  test([N], NWorkers);
test(N, NWorkers) when is_integer(NWorkers) ->  test(N, [NWorkers]).

test(N) -> test(N, [0, 2, 4, 8]).
test() -> test(misc:logsteps([1,2,3,5,7], lists:seq(1,5)) ++ [1000000]).

test_help(W, NW, N, M) when NW > 0 ->
  WW = element(1, lists:split(NW, W)), % the first NW workers
  rlist(WW, N, M, 'R'),
  workers:retrieve(WW, fun(_) -> ok end), % sync after random lists created
  F1 = fun() -> count3s(WW, 'R') end,
  % Now, figure out how many times we need to execute F to get a runtime
  % between 1ms and 2ms.
  NTrials = time_it:how_many_runs(F1, {1.0e-3, 2.0e-3}),
  F2 = fun() -> time_it:perseverate(F1, NTrials) end,
  T = time_it:t(F2, 1.0),
  list_to_tuple([ {'array size', N}, {'number of workers', NW},
    {'ntrials', NTrials } | T]);

test_help(_W, 0, N, M) -> % sequential version
  RList = misc:rlist(N,M),
  F1 = fun() -> count3s:count3s(RList) end,
  NTrials = time_it:how_many_runs(F1, {1.0e-3, 2.0e-3}),
  F2 = fun() -> time_it:perseverate(F1, NTrials) end,
  T = time_it:t(F2, 1.0),
  list_to_tuple([ {'array size', N}, {'number of workers', 0},
    {'ntrials', NTrials } | T]).


to_matlab(TT) ->
  T = lists:flatten(TT),
  io:format("function T = count3s_p3_time()~n"),
  to_matlab_help("  T = [ ", hd(T)),
  [ to_matlab_help("        ", X) || X <- tl(T) ],
  io:format("  ];~n"),
  io:format("end % count3s_p3_time~n").

to_matlab_help(Prefix,
    { {'array size', N}, {'number of workers', NW}, {'ntrials', NT},
      {mean, M}, {std, SD}
    }) ->
  io:format("~s[ ~10.10B, ~3.10B, ~6.10B, ~12.6e, ~12.6e ]; ...~n",
  	     [Prefix, N, NW, NT, M, SD]).



test2(N, NWorkers) ->
  W = workers:create(NWorkers),
  rlist(W, N, 10, 'R'),
  workers:retrieve(W, fun(_) -> ok end), % sync after random lists created
  count3s_brute_log(W, 'R'),
  { N3S, Log } = count3s_brute_log(W, 'R'),
  workers:reap(W),
  { N3S, Log, W }.
