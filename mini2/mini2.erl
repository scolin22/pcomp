-module(mini2).

-export([fridge3a/1, start3a/1, fridge3/1, store3/2, take3/2, start3/1]).
-export([start/1, store2/2, take2/2, fridge2/1]).
-export([prepare/1]).
-export([removeFromList/3]).


% code for fridge2, store2, and take2 from LYSE is at the end of this file.
% You can use them as a starting point for your solution -- cut-and-paste to
% your heart's content.

% My revised strategy -- in mini2.erl, I sketched an idea of keeping track
% of the refrigerator contents using a list of tuples, where each tuple
% is of the form {Foot, HowMany}.  This has an advantage that if you try
% to store 5234861 packages of tofu in your refrigerator, then we can
% represent it with a single tuple -- a nice way to save space.  But,
% I found I did too much work handling the tuples.
%   So, I tried again.  This time, I'm staying closer to the LYSE code.
% If there are 5234861 packages of tofu in my refrigerator, there will
% be 5234861 tuples for them in the list.  Maybe it wastes space, but
% it makes the code simpler.
%
% I note that the assignment didn't say how the fridge process represents
% the contents of the refrigerator; so, either approach is a legal solution.

% prepare(Food) -- once again, I'll write a function that takes the
%   arguments to store3 and take3 and converts them to the form used by the
%   new fridge processes.  According to the assignment, we can have:
%     store3(Food) when is_atom(Food) -> ...
%     store3(FoodList) when each element of FoodList is a tuple of the
%       form {Food, Count}.  I also support an element of the form Food,
%       where Food is an atom, as being equivalent to {Food, 1}.
%   See the problem statement for examples.
% Accordingly,
%  prepare(Food) when is_atom(Food) produces a singleton list, [Food].
%  prepare(FoodList) produces a list with the given number of each
%    type of Food.  For example, prepare([{bread, 4}, banana]) returns
%    [bread, bread, bread, bread, banana].
prepare(Food) when is_atom(Food) -> [Food];
prepare([]) -> [];
prepare([Food | FoodTail]) when is_atom(Food) -> [Food | prepare(FoodTail)];
prepare([{Food, N} | FoodTail])
  when (is_atom(Food) and is_integer(N) and (N > 0)) ->
    [ Food || _ <- lists:seq(1,N) ] ++ prepare(FoodTail);
prepare(_) -> bad_food.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	 	 	 	 	 	 	 	 	  %
% Templates for question 1                                                %
% 	 	 	 	 	 	 	 	 	  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% I'm providing an implementation of start3 because I didn't ask you to
%   write that in the problem statement.  It's just like start, but we
%   execute fridge3a instead of fridge2.
start3a(FoodList) ->
  case prepare(FoodList) of
    L when is_list(L) -> spawn(?MODULE, fridge3a, [L]);
    bad_food -> bad_food
  end.

store3(Pid, Food) ->
  case prepare(Food) of
    L when is_list(L) ->
      Pid ! {self(), {store, L}},
      receive
        {Pid, Msg} -> Msg
        after 3000 ->
          timeout
      end;
    bad_food -> bad_food
  end.


% fridge3a is an intermediate step in the problem.
%   fridge3a handles 'store' messages with lists of Food tuples,
%     and 'take' requests for single Food atoms.
fridge3a(FoodList) ->
  io:format("~p: fridge3(~p)~n", [self(), FoodList]),
  receive
    _Msg = {From, {store, Food}} ->
      io:format("~p: received ~p~n", [self(), _Msg]),
      From ! {self(), ok},
      fridge3a(Food ++ FoodList);
    _Msg = {From, {take, Food}} ->
      io:format("~p: received ~p~n", [self(), _Msg]),
      case lists:member(Food, FoodList) of
        true ->
          From ! {self(), {ok, Food}},
          fridge3a(lists:delete(Food, FoodList));
        false ->
          From ! {self(), not_found},
          fridge3a(FoodList)
      end;
    terminate ->
      ok
  end.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	 	 	 	 	 	 	 	 	  %
% Templates for question 2                                                %
% 	 	 	 	 	 	 	 	 	  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% I'm providing an implementation of start3 because I didn't ask you to
%   write that in the problem statement.
start3(FoodList) ->
  case prepare(FoodList) of
    L when is_list(L) -> spawn(?MODULE, fridge3, [L]);
    bad_food -> bad_food
  end.

fridge3(FoodList) ->
  io:format("~p: fridge3(~p)~n", [self(), FoodList]),
  receive
    _Msg = {From, {store, Food}} ->
      io:format("~p: received ~p~n", [self(), _Msg]),
      From ! {self(), ok},
      fridge3(Food ++ FoodList);
    _Msg = {From, {take, Food}} when is_list(Food) ->
      io:format("~p: received ~p~n", [self(), _Msg]),
      {Found, L} = removeFromList(Food, FoodList, true),
      case Found of
        true ->
          From ! {self(), {ok, Food}},
          fridge3(L);
        false ->
          From ! {self(), not_found},
          fridge3(FoodList)
      end;
    terminate ->
      ok
  end.

removeFromList([], L, false) -> {false, L};
removeFromList([], L, B) -> {B, L};
removeFromList([H|T], L, B) ->
  case lists:member(H, L) of
    true ->
      removeFromList(T, lists:delete(H, L), B);
    false ->
      removeFromList(T, L, false)
  end.


take3(Pid, Food) ->
  case prepare(Food) of
    L when is_list(L) ->
      Pid ! {self(), {take, L}},
      receive
        {Pid, Msg} -> Msg
        after 3000 ->
          timeout
      end;
    bad_food -> badfood
  end.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	 	 	 	 	 	 	 	 	  %
% Code from Learn You Some Erlang (More on Multiprocessing)               %
% 	 	 	 	 	 	 	 	 	  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fridge2(FoodList) ->
  receive
    {From, {store, Food}} ->
      From ! {self(), ok},
      fridge2([Food|FoodList]);
    {From, {take, Food}} ->
      case lists:member(Food, FoodList) of
      	true ->
      	  From ! {self(), {ok, Food}},
      	  fridge2(lists:delete(Food, FoodList));
      	false ->
      	  From ! {self(), not_found},
      	  fridge2(FoodList)
      end;
    terminate ->
      ok
  end.

store2(Pid, Food) ->
  Pid ! {self(), {store, Food}},
  receive
    {Pid, Msg} -> Msg
    after 3000 ->
      timeout
  end.

take2(Pid, Food) ->
  io:format("take2(~p, ~p)~n", [Pid, Food]),
  Pid ! {self(), {take, Food}},
  receive
    {Pid, Msg} -> Msg
    after 3000 ->
      timeout
  end.

% I'll add start for good measure
start(FoodList) ->
  spawn(?MODULE, fridge2, [FoodList]).
