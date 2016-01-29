-module(doc).
-export([doc/0]).

%% @spec doc() -> ok
%% @doc Generate edoc documentation for the erlang modules in this directory.
doc() ->
  Packages = [""],	% the anonymous package in this directory
  Files = ["misc.erl", "stat.erl", "time_it.erl", "wlog.erl", "workers.erl",
  	"wtree.erl"],
  Options  = [ {dir, "doc"}, {packages, false} ],
  edoc:run(Packages, Files, Options).
