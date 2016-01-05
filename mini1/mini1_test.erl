% mini1_test -- a minimal test module for mini1.erl
%
%   To run these tests:
%     1.  Compile this module and the mini1 module.
%     2.  In the Erlang shell, give the command:
%             mini1_test:test()
%           Equivalently, from the unix command line, give the command
%               erl -noinput -run mini1_test test -run init stop
%             That tells the Erlang runtime to first execute the function
%             mini1_test:test.  This performs the tests.  Then, Erlang
%             executes the function init:stop which makes Erlang terminate.
%   The tests are written using the EUnint macros.  See
%     http://www.erlang.org/doc/apps/eunit/chapter.html
%   I've provided one test per function as an example.  This is definitely
%     not a sufficient set of tests.
%
% Copyright, Mark Greenstreet, University of British Columbia, 2016

-module(mini1_test).

-import(mini1, [who_am_i/0, registration/0, flatten/1, reverse_hr/1,
                reverse_tr/1, time_rev/0, time_rev/1, fib_hr/1, fib_tr/1]).

-include_lib("eunit/include/eunit.hrl").

is_string(S) ->
  is_list(S) and lists:all(fun(X) -> (32 =< X) and (X < 128) end, S).

who_am_i_test_() ->
  [ ?_assert(is_string(element(1, who_am_i()))),
    ?_assert(is_integer(element(2, who_am_i()))),
    ?_assert(is_string(element(3, who_am_i())))].

registration_test() ->
  ?assert(lists:member(registration(), [enrolled, wait, audit])).

flatten_test() ->
  ?assertEqual([x, y, z, 2, a, 5.2, 23, 14, 8],
               flatten([x, [[y, z], 2, [a, 5.2, [], 23]], 14, [[[8]]]])).

reverse_tr_test() ->
  ?assertEqual([3, 2, 1], reverse_tr([1, 2, 3])).

fib_tr_test_() ->
  [ ?_assertEqual(0, fib_tr(0)),
    ?_assertEqual(1, fib_tr(1)),
    ?_assertEqual(1, fib_tr(2)),
    ?_assertEqual(2, fib_tr(3)),
    ?_assertEqual(3, fib_tr(4)),
    ?_assertEqual(5, fib_tr(5)),
    ?_assertEqual(8, fib_tr(6)),
    ?_assertEqual(13, fib_tr(7)),
    ?_assertEqual(fib_hr(1000), fib_tr(1000)) ].
