% mini2_test -- a minimal test module for mini2.erl
%
%   To run these tests:
%     1.  Compile this module and the mini2 module.
%     2.  In the Erlang shell, give the command:
%             mini2_test:test().
%           Equivalently, from the unix command line, give the command
%               erl -noinput -run mini2_test test -run init stop
%             That tells the Erlang runtime to first execute the function
%             mini2_test:test.  This performs the tests.  Then, Erlang
%             executes the function init:stop which makes Erlang terminate.
%   The tests are written using the EUnint macros.  See
%     http://www.erlang.org/doc/apps/eunit/chapter.html
%
%   These tests are also handy for debugging.  See the try_it functions
%   At the end of this file.
%
% Copyright, Mark Greenstreet, University of British Columbia, 2016

-module(mini2_test).
-import(mini2, [fridge3a/1, fridge3/1, store3/2, take3/2, start3/1]).
-import(mini2, [start/1, store2/2, take2/2, fridge2/1]).
-import(mini2, [prepare/1, add_food/2, add_food/3]).
-import(lists, [sort/1]).
-include_lib("eunit/include/eunit.hrl").

-export([lyse_fridge_test_cases/1, fridge3a_test_cases/1, fridge3_test_cases/1]).
-export([try_it/2, try_it/1, try_it/0]).

mini_mod() -> mini2.

% I'll define ?assertMember and ?_assertMember macros so tests can accept
%   multiple possible outcomes.
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


% To test the refrigerator, we spawn a refrigerator process, then do a
%   sequence of store and take operations and check the results.  EUnit has
%   stuff for set-up and take-down of test case, but I'll try to keep it
%   simple by running a test for each case, and not worrying about the
%   processes that might get left behind if something fails.  This should
%   work because I'm not running huge tests; so we won't run out of  memory
%   or any bad thing like that.

lyse_fridge_test_cases({StartFn, StoreFn, TakeFn}) ->
  FridgePid = StartFn([rhubarb, dog, hot_dog]),
  [ ?_assertEqual({ok, dog},   TakeFn(FridgePid, dog)),    % The dog's ok -- I'm so relieved.
    ?_assertEqual(not_found,   TakeFn(FridgePid, dog)),    % no more dogs in the fridge
    ?_assertEqual(ok,          StoreFn(FridgePid, water)), % put some water in the fridge
    ?_assertEqual({ok, water}, TakeFn(FridgePid, water)),  % and take it out.
    ?_assertEqual(not_found,   TakeFn(FridgePid, juice)),  % no juice in the fridge
    ?_assertEqual(terminate,   FridgePid ! terminate)      % clean up
  ].


fridge2_test_() ->
  MiniMod = mini_mod(),
  Fridge2Funs = { fun MiniMod:start/1, fun MiniMod:store2/2, fun MiniMod:take2/2 },
  { inorder, lyse_fridge_test_cases(Fridge2Funs)}.

fridge3a_test_cases({StartFn, StoreFn, TakeFn}) ->
  FridgePid = StartFn([rhubarb, dog, hot_dog]),
  [ ?_assertEqual(ok,           StoreFn(FridgePid, [{water, 2}, {juice, 3}])),
    ?_assertEqual({ok, water},  TakeFn(FridgePid, water)),
    ?_assertEqual({ok, juice},  TakeFn(FridgePid, juice)),
    ?_assertEqual({ok, dog},    TakeFn(FridgePid, dog)),
    ?_assertEqual({ok, water},  TakeFn(FridgePid, water)),
    ?_assertEqual(not_found,    TakeFn(FridgePid, dog)),
    ?_assertEqual(not_found,    TakeFn(FridgePid, water)),
    ?_assertEqual({ok, juice},  TakeFn(FridgePid, juice)),
    ?_assertEqual(terminate,    FridgePid ! terminate)      % clean up
  ].

fridge3a_test_() ->
  MiniMod = mini_mod(),
  Fridge3aFuns = { fun MiniMod:start3a/1, fun MiniMod:store3/2, fun MiniMod:take2/2 },
  { inorder,
    lyse_fridge_test_cases(Fridge3aFuns) ++ fridge3a_test_cases(Fridge3aFuns)
  }.


fridge3_test_cases({StartFn, StoreFn, TakeFn}) ->
  FridgePid = StartFn([rhubarb, dog, hot_dog]),
  TakeFn(FridgePid, [dog, hot_dog]),
  Sandwich = [bread, banana, brontosaurus],
  SandwichExtraMeat = [bread, banana, {brontosaurus, 2}],
  [ ?_assertEqual(ok,                                  StoreFn(FridgePid, [{bread, 3}, {banana, 2}, {water,3}])),
    ?_assertEqual({ok, [bread,water]},                 TakeFn(FridgePid, [bread, water])),
    ?_assertMember([not_found, {not_found, juice}],    TakeFn(FridgePid, juice)),
    ?_assertMember([not_found, {not_found, Sandwich}], TakeFn(FridgePid, Sandwich)),
    ?_assertEqual(ok,                                  StoreFn(FridgePid, [{brontosaurus,2}])),
    ?_assertEqual({ok, Sandwich},                      TakeFn(FridgePid, Sandwich)),
    ?_assertMember([not_found,
                    {not_found, SandwichExtraMeat},
                    not_enough,
		    {not_enough, SandwichExtraMeat}],
                                                       TakeFn(FridgePid, SandwichExtraMeat)),
    ?_assertEqual(ok,                                  StoreFn(FridgePid, [{brontosaurus,2}])),
    ?_assertEqual({ok, SandwichExtraMeat},             TakeFn(FridgePid, SandwichExtraMeat)),
    ?_assertEqual(terminate,                           FridgePid ! terminate)      % clean up
  ].

fridge3_test_() ->
  MiniMod = mini_mod(),
  Fridge3Funs = { fun MiniMod:start3/1, fun MiniMod:store3/2, fun MiniMod:take3/2 },
  { inorder,
    lyse_fridge_test_cases(Fridge3Funs) ++
      fridge3a_test_cases(Fridge3Funs) ++
      fridge3_test_cases(Fridge3Funs)
  }.


% I found it *WAY* easier to debug my code when I could run the test cases
%   outside of EUnit.  That means I could put io:format statements in the
%   code, and see what they were printing out.  For example, adding
%       io:format("~p: fridge3(~p)~n", [self(), FoodList]),
%   at the entry of fridge3, along with io:format statements after each
%   receive pattern match made it much easier to see what was going on.
%   For example, after receiving a message tagged with 'store', I added
%        io:format("~p: received ~p~n", [self(), StoreMsg])
% To run these, I wrote the try_it function below.

% try_it(Fridge, Tests) -- run the Tests for Fridge
%   Fridge must be one of the following three atoms:
%     fridge2:  run the LYSE code;
%     fridge3a: run the code for the solution to Q1;
%     fridge3:  run the code for the solution to Q2.
%   Likewise, Tests must have be one of those same three atoms:
%     fridge2:  run the tests from lyse_refrigerator_test_cases_();
%     fridge3a: run the tests from fridge3a_test_cases_();
%     fridge3:  run the tests from fridge3_test_cases_().
%   By design, fridge3a can handle all of the test cases for the
%   original, fridge2 implementation.  Likewise, fridge3 can handle
%   all of the test cases from fridge3a (and thus those for fridge2
%   as well).  If you try running fridge3 tests on fridge2 or fridge3a,
%   some will fail.  Likewise if you try running fridge3a tests of fridge2.
try_it(Fridge, Tests) ->
  MiniMod = mini_mod(),
  FridgeFuns = case Fridge of
    fridge2  -> { fun MiniMod:start/1,   fun MiniMod:store2/2, fun MiniMod:take2/2 };
    fridge3a -> { fun MiniMod:start3a/1, fun MiniMod:store3/2, fun MiniMod:take2/2 };
    fridge3  -> { fun MiniMod:start3/1,  fun MiniMod:store3/2, fun MiniMod:take3/2 }
  end,
  TestFun = case Tests of
    fridge2  -> fun ?MODULE:lyse_fridge_test_cases/1;
    fridge3a -> fun ?MODULE:fridge3a_test_cases/1;
    fridge3  -> fun ?MODULE:fridge3_test_cases/1
  end,
  TestList = element(2, lists:unzip(TestFun(FridgeFuns))),
  run_test_list(TestList).

try_it(Fridge) -> try_it(Fridge, Fridge).
try_it() -> try_it(fridge3).

run_test_list([]) -> done;
run_test_list([Test | Tail]) ->
  io:format("test result: ~p~n", [Test()]),
  run_test_list(Tail).

