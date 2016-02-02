-module(hw2).

-export([pi/0, degree_to_radian/1, radian_to_degree/1, move_pos/2, move_par/4, combine/2]).
-export([test_move_par/0, test_move_pos/0, test_rle/0, rlep/1]).
-export([rle/1, longest_run/3, traverse/2, traverse_pos/2, combine_max/2, leaf/2, max_tup/3]).
-export([match_count/2, best_match/2, best_match/3, best_match_par/3, best_match_time/0, all_matches/2, combine_match/2, leaf_match/2]).

-import(fp, [fuzzy_match/3, floor/1, ceiling/1]).

-import(wtree, [get/2, put/3]).

% pi from Wolfram Alpha
pi() -> 3.1415926535897932384626433832795028841971693993751058.

degree_to_radian(Deg) -> Deg*pi()/180.0.

radian_to_degree(Rad) -> Rad*180/pi().

% move_pos(InitPos, Move):
%  InitPos = {X, Y, Dir} where X and Y are the coordinates of the initial
%                          position.  Dir is the direction that the
%                          traveler is facing (in degrees counter-clockwise
%                          from the x-axis).
%  Move = {Dir, Dist}:  The traveler turns Dir degrees counter-clockwise,
%                          and then move Dist units "forward".
% We return the final position.
% If Move is a list, then we perform the given moves, in head-to-tail order.
move_pos({X, Y, Dir}, {Turn, Dist}) ->
  NewDir = Dir+Turn,
  NewDirRad = degree_to_radian(NewDir),
  { X + Dist*math:cos(NewDirRad),
    Y + Dist*math:sin(NewDirRad),
    NewDir
  };
move_pos({X, Y, Dir}, []) -> {fp:ceiling(X), fp:ceiling(Y), fp:ceiling(Dir)};
move_pos(Pos, [MoveH | MoveTail]) ->
  move_pos(move_pos(Pos, MoveH), MoveTail).

% move_par: parallel version of move_pos.
%   W is a worker-tree, and MoveKey is a key for ProcState.
%   The list of moves should be distributed across the workers of W and
%   associated with MoveKey.  The traveler starts at the position
%   indicated by InitPos.  The position after each move is stored
%   in the list associated with PosKey.  move_par returns the final
%   position of the traveller.

test_move_par() ->
  W = wtree:create(4),
  workers:update(W, raw_data, [{90,1},{90,1},{90,1},{90,1}]),
  InitPos = {0,0,0},
  move_par(W, InitPos, raw_data, cooked_data),
  workers:retrieve(W, cooked_data).

test_move_pos() ->
  InitPos = {0,0,0},
  Moves = [{90,1},{90,1},{90,1},{90,1}],
  test_move_pos(InitPos, Moves).
test_move_pos(P, [H|T]) ->
  New_P = move_pos(P, H),
  io:format("~p~n", [New_P]),
  test_move_pos(New_P, T);
test_move_pos(_, []) ->
  ok.

combine({LX,LY,LA}, {RX,RY,RA}) ->
  Rad_LA = degree_to_radian(LA),
  X = RX*math:cos(Rad_LA) - RY*math:sin(Rad_LA),
  Y = RX*math:sin(Rad_LA) + RY*math:cos(Rad_LA),
  {LX+X,LY+Y,LA+RA}.

move_par(W, InitPos, MoveKey, PosKey) ->
  wtree:scan(W,
    fun(ProcState) ->
      move_pos({0,0,0}, wtree:get(ProcState, MoveKey))
    end,
    fun(ProcState, AccIn) ->
      wtree:put(ProcState, PosKey, move_pos(AccIn, wtree:get(ProcState, MoveKey)))
    end,
    fun(Left, Right) ->
      combine(Left, Right)
    end,
    InitPos
    ).

% rle: run-length encoding
%   Convert a list of values in to a list of {Value, Count} pairs.
%     such that Count consecutive occurrences of Value are replaced by
%     the tuple {Value, Count}.  For example,
%     rle([1,2,2,3,3,3,0,5,5]) -> [{1,1},{2,2},{3,3},{0,1},{5,2}]
test_rle() ->
  L = [1,2,2,3,3,3,0,5,5],
  rlep(L).


rle(L) when is_list(L) -> % stub
  lists:foldl(fun hw2:traverse/2, [], lists:reverse(L)).

traverse(A, []) -> [{A,1}];
traverse(A, [{A,N}|T]) -> [{A,N+1}|T];
traverse(A, C) -> [{A,1}|C].

% return the a description of the longest run of V's in the distributed
% list associated with Key in worker-pool W.
longest_run(W, Key, V) -> % stub
  {_,Result,_,_} = wtree:reduce(W,
    fun(ProcState) -> leaf(wtree:get(ProcState, Key), V) end,
    fun(Left, Right) -> combine_max(Left, Right) end),
  {_,Pos,Count} = Result,
  {Count,Pos}.

combine_max(Left, Right) ->
  {LLeft,LMax,LRight,LOffset} = Left,
  {RLeft,RMax,RRight,ROffset} = Right,

  {V,P,LRightA} = LRight,
  {_,_,RLeftA} = RLeft,
  MMax = {V,P,LRightA+RLeftA},

  {_,PRMax,NRMax} = RMax,
  {_,_,NLLeft} = LLeft,
  {_,_,NRRight} = RRight,
  OffRMax = {V,LOffset+PRMax,NRMax},
  New_Max = max_tup([LMax, MMax, OffRMax], V, {V,1,0}),
  if
    (NLLeft /= 0) and (LLeft == LMax) and (New_Max == MMax) ->
      New_Left = New_Max;
    true ->
      New_Left = LLeft
  end,
  if
    (NRRight /= 0) and (RRight == RMax) and (New_Max == MMax) ->
      New_Right = New_Max;
    true ->
      {_,PRRight,NRRight} = RRight,
      New_Right = {V,LOffset+PRRight,NRRight}
  end,
  {New_Left,New_Max,New_Right,LOffset+ROffset}.

leaf(L, V) when is_list(L) ->
  APN = rlep(L),
  [LTup|_] = APN,
  RTup = lists:last(APN),
  {LA,_,_} = LTup,
  if
    LA == V ->
      Left = LTup;
    true ->
      Left = {V,1,0}
  end,
  {RA,_,_} = RTup,
  if
    RA == V ->
      Right = RTup;
    true ->
      Right = {V,length(L),0}
  end,
  Max = max_tup(APN, V, {V,1,0}),
  {Left, Max, Right, length(L)};
leaf(L, V) ->
  leaf([L], V).

max_tup([H], V, Acc) ->
  {A,_,N} = H,
  {_,_,Max} = Acc,
  if
    (A == V) and (N >= Max) ->
      H;
    true ->
      Acc
  end;
max_tup([H|T], V, Acc) ->
  {A,_,N} = H,
  {_,_,Max} = Acc,
  if
    A == V ->
      if
        N > Max ->
          max_tup(T, V, H);
        true ->
          max_tup(T, V, Acc)
      end;
    true ->
      max_tup(T, V, Acc)
  end.

rlep(L) when is_list(L) -> % stub
  NList = lists:zip(L, lists:seq(1, length(L))),
  lists:reverse(lists:foldl(fun hw2:traverse_pos/2, [], NList)).

traverse_pos({A,P}, []) -> [{A,P,1}];
traverse_pos({A,_}, [{A,P,N}|T]) -> [{A,P,N+1}|T];
traverse_pos({A,P}, C) -> [{A,P,1}|C].
% match_count:
%   We return the number of values for I,
%   with 1 <= I <= min(length(L1), length(L2)), such that
%   lists:nth(I, L1) =:= lists:nth(I, L2).
%   A brute force way to do this would be
%     length([ok || I <- lists:seq(1, min(length(L1), length(L2))),
%                        lists:nth(I, L1) =:= lists:nth(I, L2)])
%   but that would take quadratic time in the length of the lists :(
%   This implementation is linear time.
match_count(L1, L2) when is_list(L1), is_list(L2) -> match_count(L1, L2, 0).

match_count([], _, C) -> C;
match_count(_, [], C) -> C;
match_count([H | T1], [H | T2], C) -> match_count(T1, T2, C+1);
match_count([_ | T1], [_ | T2], C) -> match_count(T1, T2, C).

% best_match(L1, L2) -> {MatchCount, Alignment}
%   Find the smallest value of Alignment, with
%   -length(L1) =< Alignment =< length(L2) that maximizes
%     MatchCount = if
%       Alignment <  0 -> match_count(lists:nthtail(-Alignment, L1), L2);
%       Alignment >= 0 -> match_count(L1, lists:nthtail(L2))
%     end
%   Examples:
%     best_match([1,2,3],[1,0,2,3,2,1,2,0]) -> {2,1}
%     best_match("banana", "ananas is the genus of pineapple") -> {5, -1}
%     best_match("banana", "bandanna") -> {3, 0}

best_match(L1, L2) when is_list(L1), is_list(L2) ->
  Res = [best_match(L1, L2, N) || N <- lists:seq(-length(L1)+1, length(L2)-1)],
  {MatchCount, Alignment} = lists:foldl(fun({M, A}, {MAcc, _}) when
    M > MAcc ->
      {M, A};
    (_, Acc) ->
      Acc
    end, {-1,0}, Res),
  {MatchCount, Alignment}.
best_match(L1, L2, Alignment) -> % stub
  MatchCount = if
    Alignment <  0 -> match_count(lists:nthtail(-Alignment, L1), L2);
    Alignment >= 0 -> match_count(L1, lists:nthtail(Alignment,L2))
  end,
  {MatchCount, Alignment}.

best_match_time(P, N) ->
  L1 = [1,2,3,4],
  W = wtree:create(P),
  wtree:rlist(W, N, 1000000, best_match_data),
  wtree:barrier(W),  % make sure the rlist computations have finished
  MyList = lists:append(workers:retrieve(W, best_match_data)),
  % ParTime = time_it:t(fun() -> sum(W, best_match_data) end),
  ParTime = time_it:t(fun() -> best_match(L1, MyList) end),
  % ParSum = sum(W, best_match_data),
  ParSum = best_match(L1, MyList),
  SeqTime = time_it:t(fun() -> best_match(L1, MyList) end),
  SeqSum = best_match(L1, MyList),
  Status = case ParSum of
    SeqSum ->
      io:format("best_match_time: passed.  The match is ~w~n", [ParSum]),
      io:format("  timing stats for parallel version: ~w~n", [ParTime]),
      io:format("  timing stats for sequential version: ~w~n", [SeqTime]),
      SpeedUp = element(2, lists:keyfind(mean, 1, SeqTime)) /
                element(2, lists:keyfind(mean, 1, ParTime)),
      io:format("  speed-up: ~6.3f~n", [SpeedUp]);
    _ ->
      io:format("best_match_time: FAILED.  Got sum of ~w.  Expected ~w~n", [ParSum, SeqSum]),
      fail
  end,
  % wtree:reap(W),  % clean-up: terminate the workers
  Status.

best_match_time() ->
  best_match_time( 4,     250),
  best_match_time( 4,     500),
  best_match_time( 4,    1000),
  best_match_time( 4,    2000),
  best_match_time( 4,    4000).

% best_match_par(W, Key1, Key2) -> {MatchCount, Alignment}
%   The parallel version of best_match.
%   best_match_par(W, Key1, Key2) should return the same value as
%   best_match(workers:retrieve(W, Key1), workers:retrieve(W, Key2))
%   but best_match should do it's work in parallel.
best_match_par(W, Key1, Key2) ->
  wtree:reduce(W,
    fun(ProcState) -> leaf_match(wtree:get(ProcState, Key1), wtree:get(ProcState, Key2)) end,
    fun(Left, Right) -> combine_match(Left, Right) end).

leaf_match(L1, L2) when is_list(L1), is_list(L2) ->
  Res = all_matches(L1, L2),
  Max = lists:foldl(fun({M, A}, {MAcc, _}) when
    M > MAcc ->
      {M, A};
    (_, Acc) ->
      Acc
    end, {-1,0}, Res),
  Left = lists:sublist(Res,1,length(L1)-1),
  Right = lists:nthtail(length(Res)-length(L1)+1, Res),
  {Left, Max, Right};
leaf_match(L1, L2) when is_list(L1) ->
  leaf_match(L1, [L2]);
leaf_match(L1, L2) when is_list(L2) ->
  leaf_match([L1], L2);
leaf_match(L1, L2) ->
  leaf_match([L1], [L2]).

combine_match(Left, Right) ->
  {LLeft, LMax, LRight} = Left,
  {RLeft, RMax, RRight} = Right,

  MMax = lists:zipwith(fun({LM,LA},{RM, _}) -> {LM+RM,LA} end,LRight,RLeft),

  New_Max = lists:foldl(fun({M, A}, {MAcc, _}) when
    M > MAcc ->
      {M, A};
    (_, Acc) ->
      Acc
    end, LMax, [LMax,RMax] ++ MMax),

  if
    RRight == [] ->
      New_Right = RRight,
      New_Left = LLeft;
    true ->
      {_,LOffset} = lists:last(LRight),
      {_,ROffset} = lists:last(RRight),
      Off_Right = lists:map(fun({M,A}) -> {M,A+LOffset+1} end, RRight),
      if
        (length(RRight) - (ROffset+1)) >= 0 ->
          RZip = lists:nthtail(ROffset+1, MMax),
          RUnzip = lists:nthtail(length(RRight)-(ROffset+1), Off_Right),
          New_Right = RZip ++ RUnzip;
        true ->
          New_Right = Off_Right
      end,
      if
        (length(LLeft)-(LOffset+1)) >= 0 ->
          LUnzip = lists:sublist(LLeft, LOffset+1),
          LZip = lists:sublist(MMax, length(LLeft)-(LOffset+1)),
          New_Left = LUnzip ++ LZip;
        true ->
          New_Left = LLeft
      end
  end,

  {New_Left,New_Max,New_Right}.

all_matches(L1, L2) when is_list(L1), is_list(L2) ->
  [best_match(L1, L2, N) || N <- lists:seq(-length(L1)+1, length(L2)-1)];
all_matches(L1, L2) when is_list(L1) ->
  all_matches(L1, [L2]);
all_matches(L1, L2) when is_list(L2) ->
  all_matches([L1], L2);
all_matches(L1, L2) ->
  all_matches([L1], [L2]).

% use(X) suppresses compiler warnings that X is unused.
%   I put it in here so the stubs will compile without warning.
%   You should remove it in the final version of your code.
% use(_) -> ok.
