%% @spec reduce(W, Leaf, Combine, Root) -> term2()
%%    W = worker_pool()
%%    Leaf = fun((ProcState::worker_state) -> term1())
%%    Combine = fun((Left::term1(), Right::term1()) -> term1())
%%    Root = fun((term1()) -> term2())
%% @doc A generalized reduce operation.
%%   The <code>Leaf()</code> function is applied in each worker.
%%   The results of these are combined, using a tree,
%%   using <code>Combine</code>.
%%   The <code>Root</code> function is applied to the final result
%%   from the combine tree to produce the result of this function.
%%   <br/>
%%   <b>Note:</b> The workers are ordered.  In particular, if one were
%%   to invoke <code>update(W, 'WID', lists:seq(1:Nworkers)</code>
%%   then all of the workers contributing to the <code>Left</code>
%%   argument will have <code>'WID'</code> values less than those
%%   contributing to the <code>Right</code>.  This interface says
%%   nothing about whether or not the trees are balanced.  This means
%%   that to get deterministic results, <code>Combine</code> should
%%   be <a href="http://en.wikipedia.org/wiki/Associative_property">associative</a>
%%   function.
%% @todo Add an optional <code>Args</code> parameter so that
%%   <code>Leaf</code> can be an arity-2 function that is called with
%%   the worker process state and the element of <code>Args</code> for
%%   its process.
reduce(W, Leaf, Combine, Root) ->
  MyPid = self(),
  NW = length(W),
  W1 = hd(W),
  W1 ! fun(S) -> reduce(W, NW, MyPid, S, Leaf, Combine) end,
  receive
    {W1, reduce, V} -> Root(V)
    after debug_msg_timeout() ->
      misc:msg_dump([io_lib:format("{~w, reduce, V}", [W1])]),
      failed
  end.

%% @spec reduce(W, Leaf, Combine) -> term1()
%   W = worker_pool(),
%   Leaf = fun((ProcState::worker_state()) -> term1()),
%   Combine = fun((Left::term1(), Right::term1())->term1())
%% @doc equivalent to <code>reduce(W, Leaf, Combine, IdentityFn)</code>,
%%   where <code>IdentifyFn</code> is the identity function.
reduce(W, Leaf, Combine) ->  % default for Root is the identity function
  reduce(W, Leaf, Combine, fun(V) -> V end).

% reduce(W, NW, ParentPid, S, Leaf, Combine)
%   The reduce function invoked in worker processes.
%   This function is only used within this module.
%   W is a worker pool, and NW is the number of workers that
%   we are handling.  length(W) >= NW.  S is the process state.
%   The recursive construction of the reduce tree is done by reduce2.
%   We're just a wrapper for reduce2.
reduce(W, NW, ParentPid, S, Leaf, Combine) when (NW > 1) ->
  MyPid = hd(W),
  V = reduce2(W, NW, MyPid, S, Leaf, Combine),
  ParentPid ! {MyPid, reduce, V},
  S;

% if we only have one worker (ourself), just evaluate Leaf and send
% the result to our Parent.
reduce(W, 1, ParentPid, S, Leaf, _Combine) ->
  V = Leaf(S),
  ParentPid ! {hd(W), reduce, V },
  S.

% reduce2: evaulate a reduce operation using a tree of processes.
% W is a list of worker process pids, and NW is the number of workers to
% use (i.e. we use the first NW workers of W).
% Assumption: hd(W) = self().
% We split W into two pieces.
%   We process the first piece ourself (by a recursive call to reduce2).
%   We hand the second piece to the first worker of the second piece for
%     it to handle.
reduce2(W, NW, MyPid, S, Leaf, Combine) when (NW > 1) ->
  NW2 = NW div 2,
  {W1, W2} = lists:split(NW2, W),
  H2 = hd(W2),
  H2 ! fun(S2) -> reduce(W2, NW-NW2, MyPid, S2, Leaf, Combine) end,
  V1 = reduce2(W1, NW2, MyPid, S, Leaf, Combine),
  V2 = receive
    {H2, reduce, V} -> V
    after debug_msg_timeout() ->
      misc:msg_dump([io_lib:format("{~w, reduce, V}", [H2])]),
      failed
  end,
  if
    (V1 == failed) or (V2 == failed) -> failed;
    true -> Combine(V1, V2)
  end;
reduce2(_W, 1, _MyPid, S, Leaf, _Combine) -> Leaf(S).
