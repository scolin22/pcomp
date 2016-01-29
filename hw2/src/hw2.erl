-module(hw2).

-export([pi/0, degree_to_radian/1, radian_to_degree/1, move_pos/2, move_par/4]).
-export([rle/1, longest_run/3]).
-export([match_count/2, best_match/2, best_match_par/3]).

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
%  Move = {Dist, Dir}:  The traveler turns Dir degrees counter-clockwise,
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

move_par(W, InitPos, MoveKey, PosKey) ->
  wtree:scan(W,
    fun(ProcState) ->
      move_pos({0,0,0}, wtree:get(ProcState, MoveKey))
    end,
    fun(ProcState, AccIn) ->
      wtree:put(ProcState, PosKey, move_pos(AccIn, wtree:get(ProcState, MoveKey)))
    end,
    fun({LX,LY,LA}, {RX,RY,RA}) ->
      X = RX*math:cos(LA) - RY*math:sin(LA),
      Y = RX*math:sin(LA) + RY*math:cos(LA),
      {LX+X,LY+Y,LA+RA}
    end,
    InitPos
    ).

% rle: run-length encoding
%   Convert a list of values in to a list of {Value, Count} pairs.
%     such that Count consecutive occurrences of Value are replaced by
%     the tuple {Value, Count}.  For example,
%     rle([1,2,2,3,3,3,0,5,5]) -> [{1,1},{2,2},{3,3},{0,1},{5,2}]
rle(L) when is_list(L) -> % stub
  use(L),
  Value = any_term,
  RunLen = 0,
  [{Value, RunLen}].

% return the a description of the longest run of V's in the distributed
% list associated with Key in worker-pool W.
longest_run(W, Key, V) -> % stub
  use([W, Key, V]),
  Length = 0,
  Position = 0,
  {Length, Position}.


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
best_match(L1, L2) when is_list(L1), is_list(L2) -> % stub
  use(L1), use(L2),
  Alignment = 0,  % -length(L1) =< Alignment =< length(L2)
  MatchCount = 0,
  {MatchCount, Alignment}.


% best_match_par(W, Key1, Key2) -> {MatchCount, Alignment}
%   The parallel version of best_match.
%   best_match_par(W, Key1, Key2) should return the same value as
%   best_match(workers:retrieve(W, Key1), workers:retrieve(W, Key2))
%   but best_match should do it's work in parallel.
best_match_par(W, Key1, Key2) -> % stub
  use(W), use(Key1), use(Key2),
  Alignment = 0,  % -length(L1) =< Alignment =< length(L2)
  MatchCount = 0,
  {MatchCount, Alignment}.


% use(X) suppresses compiler warnings that X is unused.
%   I put it in here so the stubs will compile without warning.
%   You should remove it in the final version of your code.
use(_) -> ok.
