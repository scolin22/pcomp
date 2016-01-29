-module(hw1).
-export([palindrome/1, maxima/1, poly_eval/2, poly_sum/2, poly_prod/2]).
-export([rev/1, rev/2, pow/2]).  % make visible for debugging


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Question 1: palindrome                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% @spec palindrome(X) -> bool()
%%   X = term()
%% @doc Return true if X is a list that is a palindrome.
palindrome([]) -> true;
palindrome(L=[_|_]) L == rev(L);
palindrome(_) -> false.  % not a list -> not a palindrome

%% @hidden
%% @spec rev(L) -> R
%%   L = [ term()]
%%   R = [ term()]
%% @doc Return the list with the same elements as L but in reversed order.
rev(L) -> rev([], L).

%% @hidden
%% @spec rev(L1::List, L2::List) -> R::List
%% @doc a helper function for rev/1.
%%   If we're called by rev(L), then for every recursive call, rev(L1, L2),
%%   L = reverse(L1) ++ L2, where reverse(L1) is L1 in reversed order.
rev(L1, []) -> L1;
rev(L1, [H | T]) -> rev([H | L1], T).


%% @spec maxima(L) -> L2
%%   L = [ term()]
%%   L2 = [{Pos, Value}]
%%   Pos = integer()
%%   Value = term()
%% @doc L2 records the positions and values of the local maxima of L.
maxima([]) -> [];  % empty list, no maxima
maxima([X]) -> [{1,X}]; % singleton list, the element is maximal
maxima(L = [X, Y | _]) -> % longer list.
  MaxT = maxima(2,L),  % find the maxima of the tail
  if  % is the first element of the list maximal?
    X >= Y -> [{1,X} | MaxT];
    true   -> MaxT
  end.

maxima(YPos, [X, Y]) ->  % end of the list
  if  % is Y maximal?
    X =< Y -> [{Y, YPos}];
    true   -> []
  end;
maxima(YPos, [X | (Tail = [Y, Z | _])]) ->  % middle of the list
  MaxT = maxima(YPos+1, Tail),  % find the maxima of the tail
  if  %  is Y maximal?
    (X =< Y) and (Y >= Z) -> [{YPos,Y} | MaxT];
    true -> MaxT
  end.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Question 3: polynomial operations                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% @spec poly_eval(P,X) -> Z
%%   P = [number()]
%%   X,Z = number()
%% @doc P is a list representation of a polynomial.  Compute P(X).
poly_eval(P, X) ->
  lists:sum([
    C*pow(X,I) ||
    {C, I} <- lists:zip(P, lists:seq(0, length(P)-1))
  ]).

%% @hidden
%% @spec pow(X,Y) -> Z
%%   X,Y,Z = number()
%% @doc Compute X^Y.
%%   If is_integer(Y), do it with a bunch of multiplications
%%   to avoid round-off error.  Otherwise (is_float(Y)), use the pow function
%%   from the math library (lm from C).
pow(_X, 0) -> 1;
pow(X, 1) -> X;
pow(X, N) when is_integer(N) and (N > 0) ->
  N2 = N div 2,
  pow(X, N2) * pow(X, N-N2);
pow(X, Y) ->
  math:pow(X,Y).  % y is (had better be) a float

%% @spec poly_sum(P1,P2) -> PS
%%   P1,P2,PS = [number()]
%% @doc Compute the sum of the polynomials represented by P1 and P2.
poly_sum([H1 | T1], [H2  | T2]) -> [H1+H2 | poly_sum(T1, T2)];
poly_sum([], L2) -> L2;
poly_sum(L1, []) -> L1.

%% @spec poly_prod(P1,P2) -> PP
%%   P1,P2,PP = [number()]
%% @doc Compute the product of the polynomials represented by P1 and P2.
poly_prod([], _P2) -> [];
poly_prod(_P1, []) -> [];
poly_prod(P1, P2) ->
  P2R = rev(P2),
  prod2(P1, P2R, prod1(P1,P2R)).

prod0([H1 | T1], [H2 | T2]) -> H1*H2 + prod0(T1, T2);
prod0(_,_) -> 0.

prod1([], _P2R) -> [];
prod1(L1 = [_ | T1], P2R) ->
  [ prod0(L1, P2R) | prod1(T1, P2R) ].

prod2(P1, [_ | T2=[_|_]], PP) ->
  prod2(P1, T2, [ prod0(P1, T2) | PP]);
prod2(_, _, PP) -> PP.
