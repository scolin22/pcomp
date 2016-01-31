-module(hw2_test).

-import(hw2, [move_pos/2, move_par/4]).
-include_lib("eunit/include/eunit.hrl").

-export([move_pos_test_cases/1, move_par_test_cases/1]).
-export([try_it/2, try_it/1, try_it/0]).

mini_mod() -> hw2.

-define(assertMember(Expects, Expr),
        begin
        ((fun (__X, __V) ->
            case lists:member(__V, __X) of
                true  -> ok;
                false -> erlang:error({ assertMember,
                                        [ {module, ?MODULE},
                                          {line, ?LINE},
                                          {expression, (??Expr)},
                                          {expected_any_of, __X},
                                          {value, __V}]})
            end
          end)((Expects), (Expr)))
        end).
-define(_assertMember(Expects,Expr), ?_test(?assertMember(Expects,Expr))).

move_pos_test_cases({MoveFn}) ->
  [ ?_assertEqual({0, 0, 0},   MoveFn({0, 0, 0}, [])),
    ?_assertEqual({0, 0, 360},   MoveFn({0, 0, 0}, [{90, 10}, {90, 10}, {90, 10}, {90, 10}]))
  ].

move_par_test_cases({MovePar, MovePos}) ->
  W = wtree:create(4),
  Moves = [{90,1},{90,1},{90,1},{90,1}],
  workers:update(W, raw_data, Moves),
  InitPos = {0,0,0},
  MovePar(W, InitPos, raw_data, cooked_data),
  Last = lists:last(workers:retrieve(W, cooked_data)),
  [
    ?_assertEqual(MovePos(InitPos, Moves), Last)
  ].

try_it(Fridge, Tests) ->
  MiniMod = mini_mod(),
  FridgeFuns = case Fridge of
    move_pos  -> { fun MiniMod:move_pos/2 };
    move_par  -> { fun MiniMod:move_par/4, fun MiniMod:move_pos/2 }
  end,
  TestFun = case Tests of
    move_pos  -> fun ?MODULE:move_pos_test_cases/1;
    move_par  -> fun ?MODULE:move_par_test_cases/1
  end,
  TestList = element(2, lists:unzip(TestFun(FridgeFuns))),
  run_test_list(TestList).

try_it(Fridge) -> try_it(Fridge, Fridge).
try_it() -> try_it(move_par).

run_test_list([]) -> done;
run_test_list([Test | Tail]) ->
  io:format("test result: ~p~n", [Test()]),
  run_test_list(Tail).
