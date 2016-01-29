-module(test).
-export([test/2]).

test(0, _Seed) -> [];
test(N, State0) when is_tuple(State0) ->
  { _RandFloat, State1 } = random:uniform_s(100, State0),
  [ State1 | test(N-1, State1) ].
