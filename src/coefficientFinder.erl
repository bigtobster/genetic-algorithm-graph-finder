%%%-------------------------------------------------------------------
%%% @author Toby Leheup - TL258
%%% @copyright (C) 2014, <University of Kent>
%%% @doc
%%%
%%% @end
%%% Created : 30. Mar 2014 14:45
%%%-------------------------------------------------------------------
-module(coefficientFinder).
-author("Toby Leheup").

%% API
-export([start/0, data/2]).
-export([generateRandomCoefficients/2]).

%% Tuple of a list of X coordinates and list of Y coordiantes
%% The lists match i.e. The Y of the X @ index 1 is at Y index 1, etc
%% {[X1, X2, ..., Xn], [Y1, Y2, ..., Yn]} format
-type coordinates() :: {list(float()), list(float())}.

%% Set of 5 potentially/probably different coefficient values
%% {coeff1, coeff2, coeff3, coeff4, coeff5} format
-type coefficientSet() :: {float(), float(), float(), float(), float(), float()}.

%% A set of coefficients and their associated Y positions when run on a set of X Values
%% {coefficients, [Y1, Y2, ..., Yn]} format
-type funcGraph() :: {coefficientSet(), list(float())}.

%% A Result is a funcGraph with an associated and evaluated fitness
%% {funcGraph, fitness} format
-type result() :: {funcGraph(), float()}.

%% A ResultSet is a list of ResultSets
%% list({funcGraph, fitness}) format
-type resultSet() :: list(result()).

%% Process which simply stores the key variables of the application
%% Acts as a global data store that is easy to configure...
%% EXPORTED - Independent Process
-spec data(coordinates(), resultSet()) -> any().
%% noinspection ErlangUnusedFunction
data(TargetCoords, Results) ->
	receive
		{init, Return} ->
			TempData = spawn(?MODULE, data, [TargetCoords, Results]),
			TempData ! {getDir, self()},
			receive
				{dir, Dir} ->
					Lines = readLinesFromFile(Dir),
					Coordinates = parseDumpToCoordinates(Lines),
					TempData ! {getPopulation, self()},
					receive
						{population, Pop} ->
							Return ! {ready, Pop},
							data(Coordinates, Results)
					end
			end;
		{getDir, Return} ->
			Return ! {dir, "./../res/datfile.dat"},
			data(TargetCoords, Results);
		{getPopulation, Return} ->
			Return ! {population, 150},
			data(TargetCoords, Results);
		{getTargetCoordinates, Return} ->
			Return ! {targetCoordinates, TargetCoords},
			data(TargetCoords, Results);
		{getTournamentSize, Return} ->
			Return ! {tournamentSize, 4},
			data(TargetCoords, Results);
		{getCrossoverProbability, Return} ->
			Return ! {crossoverProbability, 0.75},
			data(TargetCoords, Results);
		{getMutationDegree, Return} ->
			Severity = 0.2,
			{A, B, C} = now(),
			random:seed(A, B, C),
			case random:uniform(2) of
				1 ->
					Return ! {mutationDegree, 1 - Severity};
				2 ->
					Return ! {mutationDegree, 1 + Severity}
			end,
			data(TargetCoords, Results);
		{getMutationProbability, Return} ->
			Return ! {mutationProbability, 0.01},
			data(TargetCoords, Results);
		{getFitnessRange, Return} ->
			Return ! {fitnessRange, 1000},
			data(TargetCoords, Results)
	end.

%% Starts the application
%% EXPORTED - Main method
-spec start() -> atom().
start() ->
	init().

%% Initialises application
%% Reads and parses datfile
%% Generates initial load of random functions
%% Evaluates initial load of random functions
%% Calculates fitness of evaluated functions
-spec init() -> atom().
init() ->
	DataStore = spawn_link(?MODULE, data, [invalid, invalid]),
	DataStore ! {init, self()},
	DataStore ! {getFitnessRange, self()},
	receive
		{ready, Pop} ->
			receive
				{fitnessRange, FitnessRange} ->
					Coeffs = generateRandomCoefficients(Pop, FitnessRange),
%% 					io:format("Original Coeffs: ~p~n", [Coeffs]),
					DataStore ! {getTargetCoordinates, self()},
					receive
						{targetCoordinates, {TargetXs, TargetYs}} ->
							FuncGraphs = evaluateGraphs(Coeffs, TargetXs),
							Results = evaluateFitnesses(FuncGraphs, TargetYs),
							control(DataStore, Results)
					end
			end
	end.

%% Controls application in a loop that can be exited
%% Main controlling method of application once initialised
%% Gets cycles from user and runs geneticCycle N many times and returns
%% Updates User on progress
-spec control(pid(), resultSet()) -> atom().
control(DataStore, Results) ->
	Msg = "Enter number of genetic cycles to run (0 to quit): ",
	{ok, [X]} = io:fread(Msg, "~d"),
	case X of
		0 ->
			exit(normal);
		N ->
			NewResults = controlGeneticCycle(DataStore, N, Results),
			updateUser(NewResults),
			control(DataStore, NewResults)
	end.


%% Runs the geneticCycle for N many cycles.
%% Checks for success and exits if success has been found
-spec controlGeneticCycle(pid(), integer(), resultSet()) -> resultSet().
controlGeneticCycle(_, 0, Results) ->
	Results;
controlGeneticCycle(DataStore, N, Results) ->
	successCheck(Results),
	NewResults = geneticCycle(DataStore, Results),
	controlGeneticCycle(DataStore, N - 1, NewResults).

plotFitness(Datastore, Results) ->
	Datastore ! {getTargetCoordinates, self()},
	receive
		{targetCoordinates, {Xs, Ys}} ->
			file:write_file("./../res/test.dat", "hi")
	end.

%% Gets essential data from DataStore
%% Tournament Selects best functions
%% Crosses over Functions
%% Mutates functions
%% Evaluates Functions
%% Shuffles Functions
%% Returns functions
-spec geneticCycle(pid(), resultSet()) -> resultSet().
geneticCycle(DataStore, Results) ->
	DataStore ! {getTournamentSize, self()},
	DataStore ! {getCrossoverProbability, self()},
	DataStore ! {getTargetCoordinates, self()},
	DataStore ! {getPopulation, self()},
	receive
		{tournamentSize, TournamentSize} ->
			receive
				{population, Pop} ->
					EliteResults = tournamentSelect(Results, TournamentSize, Pop),
					EliteCoefficients = getCoefficientsFromResults(EliteResults),
					Best = tournamentSelect(EliteResults, length(EliteResults), length(EliteResults)),
%% 					io:format("Best: ~p~n", [Best]),
%% 					io:format("Length of EliteCoefficients: ~p~n", [length(EliteCoefficients)]),
					receive
						{crossoverProbability, CrossoverProbability} ->
							BredEliteCoefficients = crossoverCoefficients(EliteCoefficients, CrossoverProbability),
%% 							io:format("Length of BredEliteCoefficients: ~p~n", [length(BredEliteCoefficients)]),
							MutatedBredEliteCoefficients = mutateCoefficients(BredEliteCoefficients, DataStore),
%% 							io:format("Length of MutatedBredEliteCoefficients: ~p~n", [length(MutatedBredEliteCoefficients)]),
							receive
								{targetCoordinates, {TargetXs, TargetYs}} ->
									FuncGraphs = evaluateGraphs(MutatedBredEliteCoefficients, TargetXs),
%% 									io:format("Length of FuncGraphs: ~p~n", [length(BredEliteCoefficients)]),
									GreatestResults = shuffle(evaluateFitnesses(FuncGraphs, TargetYs)),
									plotFitness(DataStore, Results),
									GreatestResults
							end
					end
			end
	end.

%% Accessor method to run Tournament selection
%% Selects the best from a small group of Results
%% Keeps selecting until N many found
%% Duplicates permitted (and encouraged!)
-spec tournamentSelect(resultSet(), integer(), integer()) -> resultSet().
tournamentSelect(Results, TournamentSize, TargetPopulationSize) ->
	tournamentSelect(Results, Results, [], TournamentSize, TargetPopulationSize).

%% Selects the best from a small group of Results
%% Keeps selecting until N many found
%% Duplicates permitted (and encouraged!)
-spec tournamentSelect(resultSet(), resultSet(), resultSet(), integer(), integer()) -> resultSet().
tournamentSelect(_, _, Winners, _, 0) ->
	Winners;
tournamentSelect(Candidates, Results, Winners, TournamentSize, TargetPopulationSize) ->
	case length(Candidates) >= TournamentSize of
		true ->
			{Remainder, Challengers} = select(Candidates, TournamentSize),
			Winner = battle(Challengers),
			tournamentSelect(Remainder, Results, [Winner | Winners], TournamentSize, TargetPopulationSize - 1);
		false ->
			ShuffledResults = shuffle(Results),
			tournamentSelect(ShuffledResults, ShuffledResults, Winners, TournamentSize, TargetPopulationSize)
	end.

% Accessor function for select
% Selects N elements from A OR as much as it can up to N
-spec select(resultSet(), integer()) -> {resultSet(), resultSet()}.
select(A, N) ->
	select(A, [], N).


%% Removes a subset of N elements from Big and adds to Small and returns small
%% If not enough elements to satisfy N, returns as many elements as it can
%% output {big, small}
-spec select(resultSet(), resultSet(), integer()) -> {resultSet(), resultSet()}.
select(Big, Small, 0) ->
	{Big, Small};
select([], Small, _) ->
	{[], Small};
select([X | Xs], Small, N) ->
	select(Xs, [X | Small], N - 1).

%% Accessor method to battle/2
%% Finds the best coefficients from a list of ResultSets
-spec battle(resultSet()) -> result().
battle([A | Rest]) ->
	battle(Rest, A).

%% Finds the best coefficients from a list of ResultSets
-spec battle(list(result()), result()) -> result().
battle([], Best) ->
	Best;
battle([A | Xs], B) ->
	Fit1 = A,
	Fit2 = B,
	case Fit1 > Fit2 of
		true ->
			battle(Xs, B);
		false ->
			battle(Xs, A)
	end.

%% Accessor function
%% Produces approximately Population * P(Mutation) Functions
%% Based on genetically crossing over 2 parent coefficients
%% Change is based on another function's coefficients
%% The child MAY be a duplicate in Funcs (or elsewhere in application)
%% Note that this returns a modified set of the same size
-spec crossoverCoefficients(list(coefficientSet()), float()) -> list(coefficientSet()).
crossoverCoefficients(Coeffs, CrossoverProbability) ->
	breedFuncs(Coeffs, [], CrossoverProbability, round(length(Coeffs)/2)).

%% Produces approximately Population * P(Mutation) Functions
%% Based on genetically crossing over 2 parent coefficients
%% Change is based on another function's coefficients
%% The child MAY be a duplicate in Funcs (or elsewhere in application)
%% Note that this returns a modified set of the same size
-spec breedFuncs(list(coefficientSet()), list(coefficientSet()), float(),
                 integer()) -> list(coefficientSet()).
breedFuncs(_, TempCoeffs, _, 0) ->
	TempCoeffs;
breedFuncs([], TempCoeffs, Prob, Tests) ->
	breedFuncs(TempCoeffs, TempCoeffs, Prob, Tests);
breedFuncs(Coeffs, TempCoeffs, CrossoverProbability, RemainingBreedTests) ->
	{A, B, C} = now(),
	random:seed(A, B, C),
	[H1|Ts] = Coeffs,
	[H2|Ts2] = Ts,
	case random:uniform() =< CrossoverProbability of
		true ->
			% Breed
			SwapCounter = random:uniform(2),
			SwappingIndices = generateRandomIndices(SwapCounter),
			{NewCoeff1, NewCoeff2} = swapCoeffs(SwappingIndices, H1, H2),
%% 			io:format("BREEDING!~n"),
%% 			io:format("H1: ~p~n", [H1]),
%% 			io:format("H2: ~p~n", [H2]),
%% 			io:format("NewCoeff1: ~p~n", [NewCoeff1]),
%% 			io:format("NewCoeff2: ~p~n", [NewCoeff2]),
%% 			io:format("COEFFS~p~n", [Coeffs]),
			breedFuncs(Ts2, [NewCoeff2] ++ [NewCoeff1] ++ TempCoeffs, CrossoverProbability, RemainingBreedTests - 1);
		false ->
			% Don't breed
			breedFuncs(Ts, [H2] ++ [H1] ++ TempCoeffs, CrossoverProbability, RemainingBreedTests - 1)
	end.

%% Accessor Function
%% Returns Counter number of Random Indices values
%% Values may be duplicates
-spec generateRandomIndices(integer()) -> list(integer()).
generateRandomIndices(Counter) ->
	generateRandomIndices(Counter, []).

%% Returns Counter number of Random Indices values
%% Values may be duplicates
-spec generateRandomIndices(integer(), list(integer())) -> list(integer()).
generateRandomIndices(0, Indices) ->
	Indices;
generateRandomIndices(N, Indices) ->
	generateRandomIndices(N - 1, [random:uniform(6) | Indices]).

%% Swaps 2 coefficients using the indices in a list of indices
%% {IndicesList, Coefficient1, Coefficient2} format
%% {Child1, Child2} return format
-spec swapCoeffs(list(integer()), coefficientSet(), coefficientSet()) -> {coefficientSet(), coefficientSet()}.
swapCoeffs([], Coeff1, Coeff2) ->
	{Coeff1, Coeff2};
swapCoeffs([I | Is], Coeff1, Coeff2) ->
	{NewCoeff1, NewCoeff2} = swapCoeff(I, Coeff1, Coeff2),
	swapCoeffs(Is, NewCoeff1, NewCoeff2).

%% Swaps a Coeff at Index I from A with B and B with A to produce 2 new Coeff sets
%% Returns {A, B}
-spec swapCoeff(integer(), coefficientSet(), coefficientSet()) -> {coefficientSet(), coefficientSet()}.
swapCoeff(Index, {A1, B1, C1, D1, E1, F1}, {A2, B2, C2, D2, E2, F2}) ->
	case Index of
		1 ->
			{{A2, B1, C1, D1, E1, F1}, {A1, B2, C2, D2, E2, F2}};
		2 ->
			{{A1, B2, C1, D1, E1, F1}, {A2, B1, C2, D2, E2, F2}};
		3 ->
			{{A1, B1, C2, D1, E1, F1}, {A2, B2, C1, D2, E2, F2}};
		4 ->
			{{A1, B1, C1, D2, E1, F1}, {A2, B2, C2, D1, E2, F2}};
		5 ->
			{{A1, B1, C1, D1, E2, F1}, {A2, B2, C2, D2, E1, F2}};
		6 ->
			{{A1, B1, C1, D1, E1, F2}, {A2, B2, C2, D2, E2, F1}}
	end.

%% Accessor function
%% Mutates the coefficients of a set of functions
-spec mutateCoefficients(list(coefficientSet()), pid()) -> list(coefficientSet()).
mutateCoefficients(Coefficients, DataStore) ->
	mutateCoefficients(Coefficients, [], DataStore).

%% Mutates the coefficients of a set of coefficients
%% Returns a list of mutated coefficients
-spec mutateCoefficients(list(coefficientSet()), list(coefficientSet()), pid()) -> list(coefficientSet()).
mutateCoefficients([], Mutants, _) ->
	lists:reverse(Mutants);
mutateCoefficients([Coeff | Coeffs], Mutants, DataStore) ->
	DataStore ! {getMutationProbability, self()},
	{A, B, C} = now(),
	random:seed(A, B, C),
	receive
		{mutationProbability, MutationProbability} ->
			case random:uniform() =< MutationProbability of
				true ->
					DataStore ! {getMutationDegree, self()},
					DataStore ! {getFitnessRange, self()},
					MutateCounter = random:uniform(3),
					MutationIndices = generateRandomIndices(MutateCounter),
					receive
						{mutationDegree, MutationDegree} ->
							receive
								{fitnessRange, FitnessRange} ->
									NewCoeff = mutateCoefficientSet(MutationIndices, Coeff, MutationDegree, FitnessRange),
									mutateCoefficients(Coeffs, [NewCoeff | Mutants], DataStore)
							end
					end;
				false ->
					mutateCoefficients(Coeffs, [Coeff | Mutants], DataStore)
			end
	end.

%% Mutates a single set of coefficients
%% Mutates @ all the indices in the list of indices (which may contain duplicates
%% Returns a CoefficientSet
-spec mutateCoefficientSet(list(integer()), coefficientSet(), float(), integer()) -> coefficientSet().
mutateCoefficientSet([], Coeff, _, _) ->
	Coeff;
mutateCoefficientSet([I | Is], Coeff, MutationDegree, FitnessRange) ->
	NewCoeff = mutateCoeff(I, MutationDegree, Coeff, FitnessRange),
	mutateCoefficientSet(Is, NewCoeff, MutationDegree, FitnessRange).

%% Mutates ajn individual coefficient in a tuple of coefficients @ Index I
-spec mutateCoeff(integer(), float(), coefficientSet(), integer()) -> coefficientSet().
mutateCoeff(I, MutationDegree, {A, B, C, D, E, F}, FitnessRange) ->
	case I of
		1 ->
			{mutate(A, MutationDegree, FitnessRange), B, C, D, E, F};
		2 ->
			{A, mutate(B, MutationDegree, FitnessRange), C, D, E, F};
		3 ->
			{A, B, mutate(C, MutationDegree, FitnessRange), D, E, F};
		4 ->
			{A, B, C, mutate(D, MutationDegree, FitnessRange), E, F};
		5 ->
			{A, B, C, D, mutate(E, MutationDegree, FitnessRange), F};
		6 ->
			{A, B, C, D, E, mutate(F, MutationDegree, FitnessRange)}
	end.

%% Changes a value by MutationDegree amount (MutationDegree is a random variable from a limited range)
%% Formats to 2 dp and ensures result falls in a valid range
-spec mutate(float(), float(), integer()) -> float().
mutate(Val, MutationDegree, FitnessRange) ->
	checkMutatedValue(decimalRound(Val * MutationDegree, 2), FitnessRange).

%% Checks a mutated value falls in a valid range
%% Fixes any bad values to be the boundary they broke
-spec checkMutatedValue(float(), integer()) -> float().
checkMutatedValue(Val, FitnessRange) when Val < FitnessRange * -1 ->
	decimalRound(FitnessRange * -1, 2);
checkMutatedValue(Val, FitnessRange) when Val > FitnessRange ->
	decimalRound(FitnessRange, 2);
checkMutatedValue(Val, _) ->
	decimalRound(Val, 2).

%% Checks to see whether algorithm has found the correct function
%% Updates User and Quits if True
-spec successCheck(resultSet()) -> atom().
successCheck([]) ->
	false;
successCheck([{Coeffs, _, 0.0} | _]) ->
	io:format("Function found!~n"),
	PrettyFunc = prettifyFunc(Coeffs),
	io:format("Function is: ~p~n", [PrettyFunc]),
	io:format("Shutting down...~n"),
	exit(normal);
successCheck([_ | Rs]) ->
	successCheck(Rs).

%% Takes a function and returns string of function in a pretty, mathematical way
-spec prettifyFunc(coefficientSet()) -> nonempty_string().
prettifyFunc({A, B, C, D, E, F}) ->
	Printy1 = prettifyNum(float_to_list(A, [{decimals, 8}, compact])) ++ " + ",
	Printy2 = Printy1 ++ prettifyNum(float_to_list(B, [{decimals, 8}, compact])) ++ "X",
	Printy3 = Printy2 ++ addPrintedCoefficient(C, 2),
	Printy4 = Printy3 ++ addPrintedCoefficient(D, 3),
	Printy5 = Printy4 ++ addPrintedCoefficient(E, 4),
	Printy5 ++ addPrintedCoefficient(F, 5).

%% Turns a string number into a pretty string with good trailing 0s, dps, etc
-spec prettifyNum(string()) -> string().
prettifyNum(Num) ->
	string:strip(string:strip(Num, right, $0), right, $.).

%% Adds a coefficient to a Pretty printed function
-spec addPrintedCoefficient(float(), integer()) -> string().
addPrintedCoefficient(Coeff, Power) ->
	" + " ++ prettifyNum(float_to_list(Coeff, [{decimals, 6}, compact])) ++ "X" ++ "^" ++ integer_to_list(Power).

%% Accessor function
%% Calculates a fitness for all coefficients
-spec evaluateFitnesses(list(funcGraph()), list(float())) -> resultSet().
evaluateFitnesses(FuncGraphs, TargetYs) ->
	evaluateFitnesses(FuncGraphs, TargetYs, []).

%% Calculates a fitness for all coefficients
-spec evaluateFitnesses(list(funcGraph()), list(float), resultSet()) -> resultSet().
evaluateFitnesses([], _, Results) ->
	Results;
evaluateFitnesses([{Coeffs, Ys} | Graphs], TargetYs, Results) ->
	evaluateFitnesses(Graphs, TargetYs, [{{Coeffs, Ys}, evaluateFitness(TargetYs, Ys)} | Results]).

%% Accessor Function
%% Calculates the fitness of a single coefficient's Y results
-spec evaluateFitness(list(float()), list(float())) -> float().
evaluateFitness(TargetYs, CurrentYs) ->
	evaluateFitness(TargetYs, CurrentYs, 0).

%% Calculates the fitness of a single coefficient's Y results
-spec evaluateFitness(list(float()), list(float()), float()) -> float().
evaluateFitness([], [], Fitness) ->
	Fitness;
evaluateFitness([TargetY | TargetYs], [CurrentY | CurrentYs], TotalFitness) ->
	Fitness = decimalRound(abs(TargetY - CurrentY), 2),
	evaluateFitness(TargetYs, CurrentYs, Fitness + TotalFitness).

%% Accessor function
%% Runs the function f(x)=a + bx + cx^2 + dx^3 + ex^4 + fx^5 with inserted Xs and Coefficients
-spec evaluateGraphs(list(coefficientSet()), list(float())) -> list(funcGraph()).
evaluateGraphs(Coeffs, Xs) ->
	evaluateGraphs(Coeffs, Xs, []).

%% Accessor function
%% Runs the function f(x)=a + bx + cx^2 + dx^3 + ex^4 + fx^5 with inserted Xs and Coefficients
-spec evaluateGraphs(list(coefficientSet()), list(float()), list(funcGraph())) -> list(funcGraph()).
evaluateGraphs([], _, FuncGraphs) ->
	lists:reverse(FuncGraphs);
evaluateGraphs([CoeffSet | Coeffs], Xs, FuncGraphs) ->
	evaluateGraphs(Coeffs, Xs, [evaluateGraph(Xs, CoeffSet) | FuncGraphs]).

%% Finds a Y value for each X point in a graph
-spec evaluateGraph(list(float()), coefficientSet()) -> funcGraph().
evaluateGraph(Xs, CoeffSet) ->
	evaluateGraph(Xs, CoeffSet, []).

%% Finds a Y value for each X point in a graph
-spec evaluateGraph(list(float()), coefficientSet(), list(float())) -> funcGraph().
evaluateGraph([], Coeffs, Ys) ->
%% 	io:format(Coeffs, lists:reverse(Ys)),
	{Coeffs, lists:reverse(Ys)};
evaluateGraph([X | Xs], Coeffs, Ys) ->
	evaluateGraph(Xs, Coeffs, [evaluateFunc(X, Coeffs) | Ys]).

%% Evaluates a + bx + cx^2 + dx^3 + ex^4 + fx^5 using a set of Coefficients and X
-spec evaluateFunc(float(), coefficientSet()) -> float().
evaluateFunc(X, {A, B, C, D, E, F}) ->
	(A * math:pow(X, 0)) *
	(B * math:pow(X, 1)) *
		(C * math:pow(X, 2)) *
		(D * math:pow(X, 3)) *
		(E * math:pow(X, 4)) *
		(F * math:pow(X, 5)).

%% Accessor Method
%% Generates N many Random Coefficient Sets
-spec generateRandomCoefficients(integer(), integer()) -> list(coefficientSet()).
generateRandomCoefficients(N, Range) ->
	generateRandomCoefficients(N, Range, []).

%% Generates N many Random Coefficient Sets
-spec generateRandomCoefficients(integer(), integer(), list(coefficientSet())) -> list(coefficientSet()).
generateRandomCoefficients(0, _, Coefficients) ->
	Coefficients;
generateRandomCoefficients(N, Range, Coefficients) ->
	CoeffSet = generateRandomCoefficientSet(Range),
	generateRandomCoefficients(N - 1, Range, [CoeffSet] ++ Coefficients).

%% Returns a random coefficient set
-spec generateRandomCoefficientSet(integer()) -> coefficientSet().
generateRandomCoefficientSet(Range) ->
	{A, B, C} = now(),
	random:seed(A, B, C),
	{
		getRandomCoefficientValue(Range),
		getRandomCoefficientValue(Range),
		getRandomCoefficientValue(Range),
		getRandomCoefficientValue(Range),
		getRandomCoefficientValue(Range),
		getRandomCoefficientValue(Range)
	}.

%% Returns a random number between the application value range in positive and negative
-spec getRandomCoefficientValue(integer()) -> float().
getRandomCoefficientValue(Range) ->
	((random:uniform(Range + 1) - 1) - (random:uniform(Range + 1) - 1)) +
	decimalRound(random:uniform() - random:uniform(), 2).

%% Reads all lines from a text file
%% Closes file handle when done
-spec readLinesFromFile(string()) -> list(string()).
readLinesFromFile(File) ->
	{ok, Handle} = file:open(File, [read]),
	try
		getAllLines(Handle)
	after
		file:close(Handle)
	end.

%% Gets all lines from a File Handle to a Text File
-spec getAllLines(pid()) -> list(string()).
getAllLines(Handle) ->
	case io:get_line(Handle, "") of
		eof ->
			[];
		Line ->
			[Line | getAllLines(Handle)]
	end.

%% REFERENCE: http://www.codecodex.com/wiki/Round_a_number_to_a_specific_decimal_place#Erlang
%% Converts a float to Precision decimal places
-spec decimalRound(float(), integer()) -> float().
decimalRound(Number, Precision) ->
	P = math:pow(10, Precision),
	round(Number * P) / P.

%% Accessor function for parseDumpToCoordinates/3
%% Parses a graph dump into X, Y coordinates
-spec parseDumpToCoordinates(list(string())) -> {list(float()), list(float())}.
parseDumpToCoordinates(Lines) ->
	parseDumpToCoordinates(Lines, [], []).

%% Parses a graph dump into X, Y coordinates
-spec parseDumpToCoordinates(list(string()), list(float), list(float())) -> coordinates().
parseDumpToCoordinates([], Xs, Ys) ->
	{Xs, Ys};
parseDumpToCoordinates([Line | Lines], Xs, Ys) ->
	{X, Y} = parseLineToCoordinates(Line),
	parseDumpToCoordinates(Lines, [X] ++ Xs, [Y] ++ Ys).

%% Converts a string of 2 numbers into a pair construct
-spec parseLineToCoordinates(string()) -> {float(), float()}.
parseLineToCoordinates(Line) ->
	[A, B, _] = string:tokens(Line, " "),
	FixedA = floatFix(A),
	FixedB = floatFix(B),
	{Num1, _} = string:to_float(FixedA),
	{Num2, _} = string:to_float(FixedB),
	{Num1, Num2}.

%% Fixes a number by adding a 0 (assumes it's a decimal already)
%% Prevents a number like 126. from producing an error when parsing to float
-spec floatFix(string()) -> string().
floatFix(NearlyFloat) ->
	string:concat(NearlyFloat, "0").

%% Updates the user as to the best function and fitness
-spec updateUser(resultSet()) -> atom().
updateUser(Results) ->
	[BestResult|_] = tournamentSelect(Results, length(Results), length(Results)),
	{_, BestFitness} = BestResult,
	[BestCoeffs] = getCoefficientsFromResults([BestResult]),
	PrettyFunc = prettifyFunc(BestCoeffs),
	PrettyFitness = prettifyNum(float_to_list(BestFitness, [{decimals, 2}, compact])),
	io:format("~nCurrent Best Function is ~p with Fitness ~p~n", [PrettyFunc, PrettyFitness]).


%% REFERENCE: https://erlangcentral.org/wiki/index.php/RandomShuffle
%% Shuffles any list into a random order
%% Determine the log n portion then randomize the list.
-spec shuffle(list()) -> list().
shuffle(List) ->
	{A, B, C} = now(),
	random:seed(A, B, C),
	randomise(round(math:log(length(List)) + 0.5), List).

%% Shuffles a list into a random order
-spec randomise(integer(), list()) -> list().
randomise(1, List) ->
	randomise(List);
randomise(T, List) ->
	lists:foldl(fun(_E, Acc) ->
		randomise(Acc)
	            end, randomise(List), lists:seq(1, (T - 1))).

%% Shuffles a list by assigning random numbers to each element and sorting by number
-spec randomise(list()) -> list().
randomise(List) ->
	D = lists:map(fun(A) ->
		{random:uniform(), A}
	              end, List),
	{_, D1} = lists:unzip(lists:keysort(1, D)),
	D1.

%% Accessor function
%% Returns the list of Coefficients from a set of Results
-spec getCoefficientsFromResults(resultSet()) -> list(coefficientSet()).
getCoefficientsFromResults(Results) ->
	getCoefficientsFromResults(Results, []).

%% Returns the list of Coefficients from a set of Results
-spec getCoefficientsFromResults(list(resultSet()), list(coefficientSet())) -> list(coefficientSet()).
getCoefficientsFromResults([], Coeffs) ->
	lists:reverse(Coeffs);
getCoefficientsFromResults([{{LatestCoeffs, _}, _} | Results], Coeffs) ->
	getCoefficientsFromResults(Results, [LatestCoeffs | Coeffs]).
