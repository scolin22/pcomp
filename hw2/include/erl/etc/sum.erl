-module(sum).
-export([hr/1,hr_time/1, hr_time/0, tr/1, tr_time/1, tr_time/0]).

%% @doc head-recursive implementation of the sum of the elements of a list.
hr([]) -> 0;
hr([Head | Tail]) -> Head + hr(Tail).


%% @doc tail-recursive implementation of the sum of the elements of a list.
tr(List) -> tr(List, 0).
tr([], Sum) -> Sum;
tr([Head | Tail], Sum) -> tr(Tail, Sum+Head).

%% @doc measure time to run <code>hr</code>.
%%   <ul>
%%     <li><code>hr_time(List) when is_islist(List)</code>  measures
%%	 the time to compute the sum of the elements of <code>List</code>.
%%     </li>
%%     <li><code>hr_time(N) when is_integer(N)</code> constructs a
%%	 random list (see <code>misc:rlist</code>) and then measures
%%	 the time to compute the sum of the elements of this list.
%%     </li>
%%     <li><code>hr_time() measures the time to compute the sum of the
%%       elemnts of a random list with 1000 elements.
%%   </ul>
hr_time(List) when is_list(List) -> time_it:t(fun() -> hr(List) end);
hr_time(N) when is_integer(N) -> hr_time(misc:rlist(N, 1000)).
hr_time() -> hr_time(10000).
  

%% @doc measure time to run <code>tr</code>.
%%   <ul>
%%     <li><code>tr_time(List) when is_islist(List)</code>  measures
%%	 the time to compute the sum of the elements of <code>List</code>.
%%     </li>
%%     <li><code>tr_time(N) when is_integer(N)</code> constructs a
%%	 random list (see <code>misc:rlist</code>) and then measures
%%	 the time to compute the sum of the elements of this list.
%%     </li>
%%     <li><code>tr_time() measures the time to compute the sum of the
%%       elemnts of a random list with 1000 elements.
%%   </ul>
tr_time(List) when is_list(List) -> time_it:t(fun() -> tr(List) end);
tr_time(N) when is_integer(N) -> tr_time(misc:rlist(N, 1000)).
tr_time() -> tr_time(10000).
  
