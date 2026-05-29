(* ::Package:: *)

(* ============================================================
   PointGeneratorMathematica.m
   
   Addresses a potential coverage deficiency in GeneratePointsM.

   Background (see Appendix F of the accompanying thesis):
   When sampling on a CICY X = product(P^{n_f}) the sampling measure
     mu_X = (omega_1 ^ ... ^ omega_n)|_X
   is defined by n ample line bundles L_1,...,L_n with coverage sets
     S_i = {f | m_{if} > 0}.
   If any ambient factor P^{n_k} is absent from the total coverage
   S = union(S_i), the measure degenerates near the locus where
   T_pX ∩ T_p(P^{n_k}) != {0}, making Monte Carlo weights singular.

   In practice, the original GeneratePointsM assigns K = number-of-equations
   free parameters (one per equation, distributed across ambient factors).
   With the original single-assignment strategy, some ambient factors may
   never receive a parameter, causing the coverage issue.

   Fix in this file:
   Instead of fixing one parameter assignment (numParamsInPn), we enumerate
   ALL valid assignments — integer vectors (n_1,...,n_F) with
     sum(n_j) = K,  0 <= n_j <= dimPs[[j]],
     and for each equation i: sum_j conf[[i,j]] * n_j > 0
   — and generate Ceiling[numPts/M] points from each of the M valid
   assignments, rotating through all of them.

   Example — Tetra-quadric (P^1)^4 with one degree-(2,2,2,2) equation:
     K = 1 parameter, F = 4 factors, 4 valid assignments:
     {1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0,0,1}.
     The original code used only one (e.g. {0,0,0,1}); this file uses all 4,
     generating numPts/4 points from each.

   Return value of GeneratePointsM:
     {pointsOnCY, allAssignments}
   where allAssignments is the list of ALL M valid numParamsInPn vectors.
   NOTE: The original returned {points, singleAssignment}. The Python
   PointGeneratorMathematica class reads pts[1] as selected_t; with this
   file pts[1] becomes a list-of-lists. Update the Python side accordingly
   when integrating.

   GenerateToricPointsM and GetPointsOnCYToric are unchanged from the
   original — the analogous fix for toric varieties is deferred.
   ============================================================ *)


(* ---- Unchanged helpers ---- *)

SamplePointsOnSphere[dimP_, numPts_] := Module[{randomPoints}, (
  randomPoints = RandomVariate[NormalDistribution[], {numPts, dimP, 2}];
  randomPoints = randomPoints[[;;, ;;, 1]] + I randomPoints[[;;, ;;, 2]];
  randomPoints = Normalize /@ randomPoints;
  Return[randomPoints];
)];


PrintMsg[msg_, frontEnd_, verbose_] := Module[{}, (
  If[verbose > 0,
    If[frontEnd,
      Print[msg];
      ,
      ClientLibrary`SetInfoLogLevel[];
      ClientLibrary`info[msg];
      ClientLibrary`SetErrorLogLevel[];
    ];
  ];
)];


getPointsOnCY[varsUnflat_, numParamsInPn_, dimPs_, params_, pointsOnSphere_, eqns_, precision_:20] := Module[{subst, pts, i, j, a, b, res, maxPoss, absPts}, (
  subst = {};
  pts = {};
  For[j = 1, j <= Length[dimPs], j++,
    AppendTo[subst, Table[varsUnflat[[j, a]] -> Sum[params[[j, b]] pointsOnSphere[[j, b, a]], {b, Length[params[[j]]]}], {a, Length[varsUnflat[[j]]]}]];
  ];
  subst = Flatten[subst];
  res = FindInstance[Table[eqns[[i]] == 0, {i, Length[eqns]}] /. subst, Variables[Flatten[params]], Complexes, 1000, WorkingPrecision -> precision];
  pts = Chop[(varsUnflat /. subst) /. res];
  absPts = Abs[pts];
  For[i = 1, i <= Length[pts], i++,
    pts[[i]] = Chop[Flatten[Table[pts[[i, j]] / pts[[i, j, Ordering[absPts[[i, j]], -1][[1]]]], {j, Length[dimPs]}]]];
  ];
  Return[pts];
)];


(* ---- New helper: enumerate all valid parameter assignments ----

   EnumerateValidAssignments[numParamsTotal, maxPerFactor, confMatrix]

   Returns the list of all integer vectors (n_1,...,n_F) such that:
     (1) sum(n_j) = numParamsTotal       -- total = K (number of equations)
     (2) 0 <= n_j <= maxPerFactor[[j]]   -- at most dimPs[[j]] params in factor j
     (3) Min[confMatrix . vec] > 0       -- every equation has >= 1 covered factor
         i.e. for each eq i, exists j with conf[[i,j]]>0 and n_j>0

   For the tetra-quadric: numParamsTotal=1, maxPerFactor={1,1,1,1},
   confMatrix={{2,2,2,2}}  →  returns {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}}.
*)
EnumerateValidAssignments[numParamsTotal_, maxPerFactor_, confMatrix_] :=
  Module[{F, enumStep, allAssignments},
    F = Length[maxPerFactor];

    (* Recursive helper: returns all completions of a partial assignment.
       remaining = parameters still to distribute;  pos = current factor (1-based) *)
    enumStep[remaining_, pos_] :=
      If[pos > F,
        (* Base case: we have assigned all factors *)
        If[remaining == 0, {{}}, {}],
        (* Recursive case: try every allowed count for this factor *)
        Flatten[
          Table[
            Map[Prepend[#, j] &, enumStep[remaining - j, pos + 1]],
            {j, 0, Min[remaining, maxPerFactor[[pos]]]}
          ],
          1
        ]
      ];

    (* Filter by coverage: every equation must have at least one covered factor.
       conf . assignment gives a K-vector of "covered degrees"; Min > 0 iff all covered. *)
    allAssignments = Select[enumStep[numParamsTotal, 1], Min[confMatrix . #] > 0 &];
    Return[allAssignments];
  ];


(* ---- Modified GeneratePointsM ----

   Differences from the original:
   1. After computing the configuration matrix, calls EnumerateValidAssignments
      to find ALL valid numParamsInPn assignments (M total).
   2. Generates Ceiling[numPts/M] points from each assignment in a loop.
   3. Returns {pointsOnCY, allAssignments} instead of {pointsOnCY, singleAssignment}.

   When M=1 (e.g. the quintic or any CICY where only one assignment exists),
   behaviour is identical to the original except for the different return shape.
*)
GeneratePointsM[numPts_, dimPs_, coefficients_, exponents_, precision_:20, verbose_:0, frontEnd_:False] :=
  Module[{varsUnflat, vars, eqns, i, j, conf, start, col, totalDeg,
          numParamsInPn, numPoints, params, low, pointsOnSphere,
          pointsOnCY, numPtsPerSample, allAssignments, M,
          numPtsPerAssignment, assignmentPoints, aIdx, assignmentIndices}, (

  (* --- Build symbolic variables and polynomial equations --- *)
  varsUnflat = Table[Subscript[x, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
  vars = Flatten[varsUnflat];
  eqns = Table[
    Sum[coefficients[[i, j]] Times @@ (Power[vars, exponents[[i, j]]]),
        {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}];

  (* --- Reconstruct the configuration matrix ---
     conf[[i,j]] = degree of equation i in ambient factor j  (K x F matrix) *)
  conf = {};
  For[i = 1, i <= Length[coefficients], i++,
    start = 1;
    col = {};
    For[j = 1, j <= Length[dimPs], j++,
      totalDeg = Plus @@ exponents[[i, 1, start;;start + dimPs[[j]]]];
      AppendTo[col, totalDeg];
      start += dimPs[[j]] + 1;
    ];
    AppendTo[conf, col];
  ];
  PrintMsg["Configuration matrix: " <> ToString[Transpose[conf]], frontEnd, verbose];

  (* --- Enumerate ALL valid parameter assignments ---
     K = Length[eqns] free parameters distributed across F ambient factors,
     subject to:  sum = K,  n_j <= dimPs[[j]],  each equation has >= 1 covered factor. *)
  allAssignments = EnumerateValidAssignments[Length[eqns], dimPs, conf];
  M = Length[allAssignments];

  If[M == 0,
    PrintMsg["Error: no valid parameter assignments found. Check configuration matrix.",
             frontEnd, verbose];
    Return[{{}, {}}];
  ];

  PrintMsg["Found " <> ToString[M] <> " valid parameter assignment(s). " <>
           "Generating ~" <> ToString[Ceiling[numPts/M]] <> " points per assignment.",
           frontEnd, verbose];

  numPtsPerAssignment = Ceiling[numPts / M];
  pointsOnCY = {};
  assignmentIndices = {};

  (* --- Loop over every valid assignment --- *)
  For[aIdx = 1, aIdx <= M, aIdx++,
    numParamsInPn = allAssignments[[aIdx]];
    PrintMsg["Assignment " <> ToString[aIdx] <> "/" <> ToString[M] <>
             ": numParamsInPn = " <> ToString[numParamsInPn], frontEnd, verbose];

    (* -- Trial run: estimate how many points one intersection call yields -- *)
    numPoints = 1;
    Clear[t];
    params = Table[Join[{1}, Table[Subscript[t, j, k], {k, numParamsInPn[[j]]}]],
                   {j, Length[numParamsInPn]}];
    pointsOnSphere = ParallelTable[
      SamplePointsOnSphere[dimPs[[i]] + 1, numPoints (numParamsInPn[[i]] + 1)],
      {i, Length[dimPs]}, DistributedContexts -> Automatic];
    assignmentPoints = ParallelTable[
      getPointsOnCY[varsUnflat, numParamsInPn, dimPs, params,
        Table[pointsOnSphere[[i, p + (b - 1) numPoints]],
              {i, Length[pointsOnSphere]}, {b, 1 + numParamsInPn[[i]]}],
        eqns, precision],
      {p, numPoints}, DistributedContexts -> Automatic];
    assignmentPoints = Flatten[assignmentPoints, 1];
    numPtsPerSample = Length[assignmentPoints];

    If[numPtsPerSample == 0,
      PrintMsg["Warning: trial run for assignment " <> ToString[numParamsInPn] <>
               " returned 0 points — skipping this assignment.", frontEnd, verbose];
      Continue[];
    ];
    PrintMsg["Points per intersection call: " <> ToString[numPtsPerSample],
             frontEnd, verbose];

    (* -- Main generation for this assignment -- *)
    numPoints = Ceiling[numPtsPerAssignment / numPtsPerSample];

    Clear[x, t];
    varsUnflat = Table[Subscript[x, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
    params = Table[Join[{1}, Table[Subscript[t, j, k], {k, numParamsInPn[[j]]}]],
                   {j, Length[numParamsInPn]}];
    pointsOnSphere = ParallelTable[
      SamplePointsOnSphere[dimPs[[i]] + 1, numPoints (numParamsInPn[[i]] + 1)],
      {i, Length[dimPs]}, DistributedContexts -> Automatic];

    If[frontEnd,
      (* Front-end (notebook) path: show a progress bar *)
      assignmentPoints = {};
      low = 1;
      Monitor[
        For[j = 1, j <= 20, j++,
          assignmentPoints = Join[assignmentPoints,
            ParallelTable[
              getPointsOnCY[varsUnflat, numParamsInPn, dimPs, params,
                Table[pointsOnSphere[[i, p + (b - 1) numPoints]],
                      {i, Length[pointsOnSphere]}, {b, 1 + numParamsInPn[[i]]}],
                eqns, precision],
              {p, low, Floor[j numPoints / 20]},
              DistributedContexts -> Automatic]];
          low += Floor[numPoints / 20];
        ];
        ,
        Row[{"Assignment " <> ToString[aIdx] <> "/" <> ToString[M] <> "  ",
             ProgressIndicator[5(j - 1), {1, 100}],
             "  " <> ToString[5(j - 1)] <> "/100"}, "   "]
      ];
      ,
      (* Non-front-end (Python / WolframClient) path *)
      If[numPoints <= 20 * numPtsPerSample,
        (* Small batch: generate all at once *)
        assignmentPoints = ParallelTable[
          getPointsOnCY[varsUnflat, numParamsInPn, dimPs, params,
            Table[pointsOnSphere[[i, p + (b - 1) numPoints]],
                  {i, Length[pointsOnSphere]}, {b, 1 + numParamsInPn[[i]]}],
            eqns, precision],
          {p, numPoints}, DistributedContexts -> Automatic];
        ,
        (* Large batch: split into 20 chunks for progress feedback *)
        assignmentPoints = {};
        low = 1;
        For[j = 1, j <= 20, j++,
          PrintMsg["Assignment " <> ToString[aIdx] <> "/" <> ToString[M] <>
                   ": " <> ToString[5 (j - 1)] <> "% done", frontEnd, verbose];
          assignmentPoints = Join[assignmentPoints,
            ParallelTable[
              getPointsOnCY[varsUnflat, numParamsInPn, dimPs, params,
                Table[pointsOnSphere[[i, p + (b - 1) numPoints]],
                      {i, Length[pointsOnSphere]}, {b, 1 + numParamsInPn[[i]]}],
                eqns, precision],
              {p, low, Floor[j numPoints / 20]},
              DistributedContexts -> Automatic]];
          low = Floor[j numPoints / 20];
        ];
      ];
    ];

    assignmentPoints = Flatten[assignmentPoints, 1];
    pointsOnCY = Join[pointsOnCY, assignmentPoints];
    assignmentIndices = Join[assignmentIndices, Table[aIdx, {Length[assignmentPoints]}]];
    PrintMsg["Assignment " <> ToString[aIdx] <> "/" <> ToString[M] <>
             " done: " <> ToString[Length[assignmentPoints]] <> " points generated.",
             frontEnd, verbose];
  ];

  PrintMsg["All assignments done. Total points: " <> ToString[Length[pointsOnCY]] <> ".",
           frontEnd, verbose];
  If[Length[pointsOnCY] > numPts,
    pointsOnCY = pointsOnCY[[1;;numPts]];
    assignmentIndices = assignmentIndices[[1;;numPts]];
  ];

  (* Return all points, the full list of valid assignments, and per-point assignment
     indices (1-based, matching the position in allAssignments).
     The Python side converts to 0-based and uses each point's matching single-assignment
     det(g_m) as the weight denominator — identical variance to the old single-assignment
     approach while covering all M ambient factors. *)
  Return[{pointsOnCY, allAssignments, assignmentIndices}];
)];


(* ---- Toric helpers: unchanged from PointGeneratorMathematica.m ----
   TODO: apply the same multi-assignment fix to GetPointsOnCYToric /
         GenerateToricPointsM once the CICY version has been validated. *)

GetPointsOnCYToric[dimCY_,CYEqn_,vars_,sections_,patchMasks_,sectionCoords_,sectionMonoms_,GLSMcharges_,precision_]:=Module[{a,i,j,k,l,eq,dimPs,numEqnsInPn,coeffs,newEqn,sectionSol,sectionCoordsFlat,expMatrix,expMatrixPinv,sVals,logS,logX,xVals,toricVarSols,pts,patchConstraints,patchCoordsList,patchMats,patchMatInverses,patchCoords,logLambda,lambdas,scalings,tmpPts,timeoutResult,validIdx},(
dimPs=Table[Length[sections[[i]]],{i,Length[sections]}];
numEqnsInPn=Table[1,{i,Length[dimPs]}];

(*Precompute exponent matrix and its pseudoinverse for log-linear inversion sections->toric vars*)
sectionCoordsFlat=Flatten[sectionCoords];
expMatrix=Flatten[sections,1];
expMatrixPinv=PseudoInverse[expMatrix];

(*Precompute the per-patch charge matrices and their inverses for log-linear GLSM rescaling*)
patchCoordsList=Table[Flatten[Position[patchMasks[[i]],1]],{i,Length[patchMasks]}];
patchMats=Table[Transpose[GLSMcharges[[All,patchCoordsList[[i]]]]],{i,Length[patchMasks]}];
patchMatInverses=Table[
  Quiet[Check[Inverse[patchMats[[i]]],PseudoInverse[patchMats[[i]]]]],
  {i,Length[patchMasks]}
];

(*Build the equation system: CY + non-CI relations + random patch choices + random sections*)
patchConstraints=Table[RandomChoice[sectionCoords[[a]]]==1.,{a,Length[sectionCoords]}];
eq=Join[CYEqn,patchConstraints];

For[i=1,i<=dimCY,i++,
For[j=1,j<=Length[sections],j++,
If[numEqnsInPn[[j]]>=dimPs[[j]],Continue[];];
coeffs=RandomVariate[NormalDistribution[],{Length[sectionMonoms[[j]]],2}];
newEqn=Sum[(coeffs[[k,1]]+I coeffs[[k,2]]) sectionCoords[[j,k]],{k,Length[coeffs]}];
AppendTo[eq,newEqn==0];
numEqnsInPn[[j]]+=1;
Break[];
];
];

(*Solve for the section values, then invert to toric vars via log-linear algebra*)
timeoutResult=TimeConstrained[
  sectionSol=DeleteCases[Quiet[NSolve[eq]],{}];
  toricVarSols=Table[
    Module[{sv,lS,lX},
      sv=sectionCoordsFlat/.sectionSol[[i]];
      If[AnyTrue[sv,(#==0)||(Head[#]===Complex&&Abs[#]==0)&],
        Nothing,
        lS=Log[sv];
        lX=expMatrixPinv . lS;
        Chop[Exp[lX]]
      ]
    ],
    {i,Length[sectionSol]}
  ];
  pts=DeleteCases[toricVarSols,Nothing];
  "success"
,30,$Failed];

If[timeoutResult===$Failed,
  Return[{{},numEqnsInPn-Table[1,{i,Length[numEqnsInPn]}]}];
];

(*Move each point to its canonical patch (largest coordinate = 1) via log-linear GLSM rescaling*)
Do[
Do[
patchCoords=patchCoordsList[[i]];
(*Skip the patch if any coordinate we'd normalize against is zero*)
If[AnyTrue[pts[[l,patchCoords]],#==0&],Continue[];];
logLambda=patchMatInverses[[i]] . (-Log[pts[[l,patchCoords]]]);
scalings=Exp[logLambda . GLSMcharges];
tmpPts=scalings*pts[[l]];
If[Max[Abs[tmpPts]]<=1+10^-8,
pts[[l]]=Chop[tmpPts];
Break[];
];
,{i,1,Length[patchMasks]}];
,{l,1,Length[pts]}];
Return[{pts,numEqnsInPn-Table[1,{i,Length[numEqnsInPn]}]}];
)];

GenerateToricPointsM[numPts_,dimCY_,coefficients_,exponents_,sections_,sectionRelationCoeffs_,sectionRelationExps_,patchMasks_,GLSMcharges_,precision_:10,verbose_:0,frontEnd_:False]:=Module[{vars,CYeqn,i,j,k,sectionCoords,sectionCoordsFlat,expSectionsFlat,nonCIRelations,linEqCoeffs,lineqs,toricToSections,sectionMonoms,numPoints,pointsOnCY,numPtsPerSample,numEqnsInPn},(
vars=Table[Subscript[x,i],{i,Length[sections]+dimCY+1}];

Clear[s];
sectionCoords=Table[s[a-1,i-1],{a,Length[sections]},{i,Length[sections[[a]]]}];
sectionCoordsFlat=Flatten[sectionCoords];

nonCIRelations={};
For[i=1,i<=Length[sectionRelationCoeffs],i++,
AppendTo[nonCIRelations,0==Sum[sectionRelationCoeffs[[i,a]] Product[sectionCoordsFlat[[r]]^sectionRelationExps[[i,a,r]],{r,Length[sectionCoordsFlat]}],{a,Length[sectionRelationCoeffs[[i]]]}]];
];

expSectionsFlat=Flatten[sections,1];
linEqCoeffs=Table[Subscript[a,r],{r,Length[expSectionsFlat]}];
CYeqn=0;
For[i=1,i<=Length[exponents],i++,
lineqs=Table[linEqCoeffs[[r]]>=0,{r,Length[linEqCoeffs]}];
AppendTo[lineqs,exponents[[i]]==Sum[linEqCoeffs[[i]] expSectionsFlat[[i]],{i,Length[linEqCoeffs]}]];
toricToSections=FindInstance[lineqs,linEqCoeffs,Integers,1];
If[Length[toricToSections]==0,
 If[frontEnd,Print["Something is wrong. Cannot express anticanonical section through KC sections."];,ClientLibrary`error["Something is wrong. Cannot express anticanonical section through KC sections."]];
 Return[{},{0,0}]
 ];
CYeqn+=coefficients[[i]] Product[sectionCoordsFlat[[r]]^toricToSections[[1,r,2]],{r,Length[sectionCoordsFlat]}]
];
CYeqn=Join[{CYeqn==0},nonCIRelations];

sectionMonoms=Table[Table[Times@@(vars^sections[[i,j]]),{j,Length[sections[[i]]]}],{i,Length[sections]}];

(*Distribute the helper once so parallel calls do not re-serialize the context*)
DistributeDefinitions[GetPointsOnCYToric];
ParallelEvaluate[
  Off[ParallelDo::subpar, ParallelTable::subpar, 
      ParallelMap::subpar, ParallelCombine::subpar, ParallelSum::subpar]
];

{pointsOnCY,numEqnsInPn}=GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision];

pointsOnCY=ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,10},DistributedContexts->Automatic];
If[Length[DeleteCases[Table[Length[pointsOnCY[[i]]],{i,Length[pointsOnCY]}],0]]==0,
  PrintMsg["All trial runs returned 0 points; aborting.", frontEnd, verbose];
  Return[{{}, numEqnsInPn}]
];
numPtsPerSample=Min[DeleteCases[Table[Length[pointsOnCY[[i]]],{i,Length[pointsOnCY]}],0]];
PrintMsg["Number of points on CY from one ambient space intersection: "<>ToString[numPtsPerSample],frontEnd,verbose];
pointsOnCY=Flatten[pointsOnCY,1];
numPoints=Ceiling[numPts/numPtsPerSample];
PrintMsg["Now generating "<>ToString[numPts]<>" points...",frontEnd,verbose];

If[frontEnd,
    pointsOnCY={};
    Monitor[
    For[j=1,j<=20,j++,
    pointsOnCY=Join[pointsOnCY,ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,Ceiling[numPoints/20]},DistributedContexts->Automatic]];
    ];
    ,Row[{ProgressIndicator[5(j-1),{1,100}],ToString[5(j-1)]<>"/100"},"   "]
   ];
    ,
    If[numPoints<=20*numPtsPerSample,
    pointsOnCY=ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,numPoints},DistributedContexts->Automatic];
    ,
    pointsOnCY={};
    For[j=1,j<=20,j++,
    PrintMsg["Generated "<>ToString[5(j-1)]<>"% of points",frontEnd,verbose];
    pointsOnCY=Join[pointsOnCY,ParallelTable[GetPointsOnCYToric[dimCY,CYeqn,vars,sections,patchMasks,sectionCoords,sectionMonoms,GLSMcharges,precision][[1]],{p,Ceiling[numPoints/20]},DistributedContexts->Automatic]];
    ];
    ];
];
PrintMsg["done.",frontEnd,verbose];
pointsOnCY=Flatten[pointsOnCY,1];
If[Length[pointsOnCY]>numPts,pointsOnCY=pointsOnCY[[1;;numPts]]];
Return[{pointsOnCY,numEqnsInPn}];
)];
