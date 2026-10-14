(* Improved Point Sampling for CICYs developed by Eric Habjan *)
(* Based on Keller-Lukic 0907.1387 Section 3.2 *)


(* Sample points on sphere *)
SamplePointsOnSphere[dimP_, numPts_] := Module[{randomPoints}, (

    randomPoints = RandomVariate[NormalDistribution[], {numPts, dimP, 2}];
    randomPoints = randomPoints[[;;, ;;, 1]] + I randomPoints[[;;, ;;, 2]];
    randomPoints = Normalize /@ randomPoints;
    Return[randomPoints];

)];


PrintMsg[msg_, frontEnd_, verbose_] := Module[{}, (
    If[verbose > 0,
        If[frontEnd,
            Print[msg];,
            ClientLibrary`SetInfoLogLevel[];
            ClientLibrary`info[msg];
            ClientLibrary`SetErrorLogLevel[];
        ];
    ];
)];


(* Compute Fubini-Study metric with optional Hermitian matrix h *)
getFS[varsUnflat_, bvarsUnflat_, h_: {}, kval_: Automatic] := Module[
  {hh, kk, dimPs, result, i, j, a, b},
(
    dimPs = (Length /@ varsUnflat) - 1;
    hh = h;

    If[hh === {}, hh = Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]];

    kk = Which[
      kval === Automatic, ConstantArray[1, Length[dimPs]],
      NumberQ[kval], ConstantArray[kval, Length[dimPs]],
      ListQ[kval] && Length[kval] == Length[dimPs] && VectorQ[kval, NumericQ], kval,
      True,
        PrintMsg["ERROR: kval must be a number or a numeric list of length Length[dimPs].", frontEnd, 1];
        Return[$Failed]
    ];
    
    (* Build block diagonal metric for product of projective spaces *)
    result = Table[0, {Plus @@ (dimPs + 1)}, {Plus @@ (dimPs + 1)}];
    
    For[i = 1, i <= Length[dimPs], i++,
        For[a = 1, a <= dimPs[[i]] + 1, a++,
            For[b = 1, b <= dimPs[[i]] + 1, b++,
                result[[
                    Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + a,
                    Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + b
                ]] = 1/(\[Pi] kk[[i]]) D[D[Log[bvarsUnflat[[i]] . (hh[[i]] . varsUnflat[[i]])],
                    bvarsUnflat[[i, b]]], varsUnflat[[i, a]]];
            ];
        ];
    ];
    
    Return[result];
)];

(* getAbsMaxPos[alist_] := Module[{k, maxPos}, (
    maxPos = 1;
    For[k = 1, k <= Length[alist], k++,
        If[Abs[alist[[k]]] > Abs[alist[[maxPos]]], maxPos = k];
    ];
    Return[maxPos];
)]; *)

splitByBlocks[pt_, dimPs_] := TakeList[pt, dimPs + 1];

patchIndicesByBlock[pt_, dimPs_] := Module[{blocks, absBlocks},
  blocks = splitByBlocks[pt, dimPs];
  absBlocks = Abs /@ blocks;
  (Ordering[#, -1][[1]] & /@ absBlocks)
];

patchGlobalsFromLocal[patchLocal_, dimPs_] := Module[{offsets},
  offsets = Most @ Accumulate @ Prepend[dimPs + 1, 0];
  offsets + patchLocal
];

substRulesBlockwise[varsUnflat_, pt_, dimPs_, patchLocal_] := Module[
  {blocks, denoms},
  blocks = splitByBlocks[pt, dimPs];
  denoms = MapThread[#1[[#2]] &, {blocks, patchLocal}];
  Flatten@Table[
    Thread[varsUnflat[[i]] -> (blocks[[i]]/denoms[[i]])],
    {i, Length[dimPs]}
  ]
];

substRulesBlockwiseBar[bvarsUnflat_, bpt_, dimPs_, patchLocal_] := Module[
  {blocks, denoms},
  blocks = splitByBlocks[bpt, dimPs];
  denoms = MapThread[#1[[#2]] &, {blocks, patchLocal}];
  Flatten@Table[
    Thread[bvarsUnflat[[i]] -> (blocks[[i]]/denoms[[i]])],
    {i, Length[dimPs]}
  ]
];

(* Choose eliminated coordinates j_elim *)
FSBlockMetric[block_List, volj_?NumericQ, hBlock_: Automatic] := Module[
  {z = N@block, H, hz, s, n},
  n = Length[z];

  H = Which[
    hBlock === Automatic || hBlock === None || hBlock === {}, IdentityMatrix[n],
    MatrixQ[hBlock, NumericQ] && Dimensions[hBlock] === {n, n}, N@hBlock,
    True, Return[ConstantArray[Indeterminate, {n, n}]]
  ];

  hz = H . z;
  s  = Conjugate[z] . hz;

  If[!NumericQ[s] || Abs[s] < 10^-30,
    Return[ConstantArray[Indeterminate, {n, n}]]
  ];

  ((s Transpose[H] - Outer[Conjugate[#1] #2 &, hz, hz])/s^2) * (volj/Pi)
];

FSAmbientMetric[ptFlat_List, dimPs_List, voljs_List, hBlocks_: Automatic] := Module[
  {blocks, mats, hh},
  blocks = splitByBlocks[ptFlat, dimPs];

  hh = Which[
    hBlocks === Automatic || hBlocks === None || hBlocks === {},
      Table[Automatic, {Length[blocks]}],
    ListQ[hBlocks] && Length[hBlocks] == Length[blocks],
      hBlocks,
    True,
      Return[$Failed]
  ];

  mats = MapThread[FSBlockMetric, {blocks, voljs, hh}];
  BlockDiagonalMatrix[mats]
];

metricBlocksFromL[L_List] := Table[
  Chop[ConjugateTranspose[L[[k]]] . L[[k]]],
  {k, Length[L]}
];

prepareWeightEvaluatorCICY[
  varsFlat_, bvarsFlat_, dimPs_, eqns_, numParamsInPn_, hBlocks_: Automatic
] := Module[
  {hh, ts},
  hh = Which[
    hBlocks === Automatic || hBlocks === None || hBlocks === {},
      Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}],
    ListQ[hBlocks] && Length[hBlocks] == Length[dimPs],
      hBlocks,
    True,
      Return[$Failed]
  ];

  ts = Flatten[
    Table[
      ConstantArray[UnitVector[Length[dimPs], i], (dimPs - numParamsInPn)[[i]]],
      {i, Length[dimPs]}
    ],
    1
  ];

  <|
    "dimPs" -> dimPs,
    "nCoords" -> Length[varsFlat],
    "varsFlat" -> varsFlat,
    "numEqns" -> Length[eqns],
    "numParamsInPn" -> numParamsInPn,
    "jacEqns" -> D[eqns, {varsFlat}],
    "ts" -> ts,
    "hBlocks" -> hh
  |>
];


preparePointGeometryCICY[pre_, pt_, patchLocal_] := Module[
  {
    nCoords, numEqns, dimPs,
    patchGlobal, ptNorm,
    jacEvals, jElimGlobal, goodMask, goodCoordsIndexSet,
    Bmat, Amat, detB, dZ, Jpb,
    Omega
  },

  nCoords = pre["nCoords"];
  numEqns = pre["numEqns"];
  dimPs   = pre["dimPs"];

  patchGlobal = patchGlobalsFromLocal[patchLocal, dimPs];

  ptNorm = patchNormalizeFlatPoint[pt, dimPs];
  If[!VectorQ[ptNorm, NumericQ],
    Return[$Failed]
  ];

  jacEvals = pre["jacEqns"] /. Thread[pre["varsFlat"] -> ptNorm];
  If[!MatrixQ[jacEvals, NumericQ],
    Return[$Failed]
  ];

  (* Coordinates to eliminate via the implicit function theorem.

     For one equation, |det B| = |dQ/dz_j|, so the largest |dQ/dz_j| is
     optimal and that rule is kept unchanged.

     For several equations, choosing each equation's largest |dQ_i/dz_j| in
     turn maximizes the diagonal of B but not |det B|: the gradients can be
     nearly parallel on the chosen coordinates, making B ill-conditioned and
     Omega = 1/det(B) and the pullback blow up, so take the numEqns-subset of
     available coordinates maximizing |det B| directly.  The weights
     |Omega|^2/det(g_pb) do not depend on the choice, but Omega does, and the
     cymetric models pair their own det(g) with the |Omega|^2 returned here.
     So this must match _find_max_dQ_coords in the Python backends exactly:
     subsets in lexicographic order, the first maximum wins. *)
  jElimGlobal = Module[{available, cands, scores},
    available = ConstantArray[True, nCoords];
    available[[patchGlobal]] = False;
    If[numEqns == 1,
      {First @ Ordering[Abs[jacEvals[[1]]] * Boole[available], -1]},
      cands = Subsets[Pick[Range[nCoords], available], {numEqns}];
      scores = Abs[Det[jacEvals[[All, #]]]] & /@ cands;
      cands[[First @ Ordering[-scores, 1]]]
    ]
  ];

  goodMask = ConstantArray[True, nCoords];
  goodMask[[Join[patchGlobal, jElimGlobal]]] = False;
  goodCoordsIndexSet = Pick[Range[nCoords], goodMask];

  Bmat = jacEvals[[All, jElimGlobal]];
  Amat = jacEvals[[All, goodCoordsIndexSet]];

  If[!MatrixQ[Bmat, NumericQ],
    Return[$Failed]
  ];

  detB = Det[Bmat];
  If[!NumericQ[detB] || Abs[detB] < 10^-30,
    Return[$Failed]
  ];

  dZ = -LinearSolve[Bmat, Amat];

  Jpb = ConstantArray[0., {nCoords, Length[goodCoordsIndexSet]}];
  Do[
    Jpb[[goodCoordsIndexSet[[\[Mu]]], \[Mu]]] = 1.,
    {\[Mu], Length[goodCoordsIndexSet]}
  ];
  Do[
    Jpb[[jElimGlobal[[i]], \[Mu]]] = dZ[[i, \[Mu]]],
    {i, numEqns}, {\[Mu], Length[goodCoordsIndexSet]}
  ];

  (* Complex holomorphic volume-form factor Omega = 1/det(B). This complex
     value is what gets exported downstream, matching the omega convention of
     the other point generators. |Omega|^2 is derived (Abs[Omega]^2) only
     where the sampling weight needs it -- never store |Omega| or |Omega|^2
     under the name Omega. *)
  Omega = 1/detB;

  <|
    "ptRaw" -> pt,
    "ptNorm" -> ptNorm,
    "patchLocal" -> patchLocal,
    "patchGlobal" -> patchGlobal,
    "jacEvals" -> jacEvals,
    "jElimGlobal" -> jElimGlobal,
    "goodCoordsIndexSet" -> goodCoordsIndexSet,
    "Bmat" -> Bmat,
    "Amat" -> Amat,
    "detB" -> detB,
    "dZ" -> dZ,
    "Jpb" -> Jpb,
    "Omega" -> Omega
  |>
];

(* Returns {w, Omega, jElimGlobal}: w is the real IPS weight, Omega is the
   COMPLEX holomorphic volume-form factor 1/det(B) (not |Omega| or |Omega|^2). *)
scorePointGeometryWithMetric[pre_, geom_] := Module[
  {ptNorm, Jpb, Omega, AbsOmegaSquared, fsPbs, detgNorm, w},

  If[geom === $Failed || !AssociationQ[geom],
    Return[{Indeterminate, Indeterminate, {}}]
  ];

  ptNorm = geom["ptNorm"];
  Jpb = geom["Jpb"];
  Omega = geom["Omega"];

  fsPbs = Table[
    With[{gAmbT = FSAmbientMetric[ptNorm, pre["dimPs"], pre["ts"][[k]], pre["hBlocks"]]},
      If[!MatrixQ[gAmbT] || Dimensions[gAmbT] =!= {pre["nCoords"], pre["nCoords"]},
        Return[{Indeterminate, Indeterminate, {}}]
      ];
      Quiet[Check[Transpose[Jpb].gAmbT.Conjugate[Jpb], Indeterminate], Dot::rect]
    ],
    {k, Length[pre["ts"]]}
  ];

  If[!VectorQ[fsPbs, MatrixQ],
    Return[{Indeterminate, Indeterminate, {}}]
  ];

  detgNorm = Chop[LeviCivitaWedgeDet[fsPbs], 10^-30];

  If[!NumericQ[detgNorm] || Abs[detgNorm] < 10^-30,
    Return[{Indeterminate, Indeterminate, {}}]
  ];

  (* The weight needs |Omega|^2 = Omega OmegaBar; only here is the squared
     magnitude used. Numerically identical to the old Abs[1/detB]^2. *)
  AbsOmegaSquared = Abs[Omega]^2;
  w = Re[AbsOmegaSquared/detgNorm];

  {w, Omega, geom["jElimGlobal"]}
];

getWeightOmegasPrepared[pre_, pt_, bpt_, patchLocal_] := Module[
  {geom},
  geom = preparePointGeometryCICY[pre, pt, patchLocal];
  scorePointGeometryWithMetric[pre, geom]
];

attachMetricScoresToPointsCICY[pres_List, \[Kappa]s_List, pts_List] := Module[
  {scoredPts},

  scoredPts = ParallelTable[
    Module[
      {
        point, weight, omega, patchLocal, jElimGlobal, regionLabel,
        geom, metricResults, rawMetricWeights, metricScores, bestMetric, minScore
      },

      point = pts[[i, 1]];
      weight = pts[[i, 2]];
      omega = pts[[i, 3]];
      patchLocal = pts[[i, 4]];
      jElimGlobal = pts[[i, 5]];
      regionLabel = If[Length[pts[[i]]] >= 6, pts[[i, 6]], Missing["NotSpecified"]];

      geom = preparePointGeometryCICY[pres[[1]], point, patchLocal];

      (* For each metric j, compute the raw weight w_j(p) and the |kappa_j w_j - 1| score *)
      metricResults = Table[
        Module[{tmp},
          tmp = scorePointGeometryWithMetric[pres[[j]], geom];
          If[NumericQ[tmp[[1]]],
            {tmp[[1]], Abs[\[Kappa]s[[j]] tmp[[1]] - 1]},
            {Indeterminate, Infinity}
          ]
        ],
        {j, Length[pres]}
      ];
      rawMetricWeights = metricResults[[All, 1]];
      metricScores    = metricResults[[All, 2]];

      minScore = Min[metricScores];
      bestMetric = First @ Ordering[metricScores, 1];

      {
        point,
        weight,
        omega,
        patchLocal,
        jElimGlobal,
        regionLabel,
        geom,
        metricScores,
        bestMetric,
        minScore,
        rawMetricWeights
      }
    ],
    {i, Length[pts]},
    DistributedContexts -> Automatic,
    Method -> "CoarsestGrained"
  ];

  scoredPts
];

dropCachedMetricDataCICY[ptsWithScores_List] := ptsWithScores[[All, 1 ;; 6]];

selectPointsOwnedByMetricCICY[
  ptsWithScores_List, metricIndex_Integer, ownershipTol_: 10^-6
] := Module[
  {allScores, minScores},
  allScores = ptsWithScores[[All, 8]];
  minScores = ptsWithScores[[All, 10]];
  Table[
    allScores[[i, metricIndex]] <= minScores[[i]] + ownershipTol,
    {i, Length[ptsWithScores]}
  ]
];

PullbackMetric[Jpb_, gAmb_] := Transpose[Jpb].gAmb.Conjugate[Jpb];

LeviCivitaWedgeDet[gpbs_List] := Module[
  {n, perms, signs},

  If[!VectorQ[gpbs, MatrixQ], Return[Indeterminate]];
  n = Length[gpbs];
  If[n < 1, Return[Indeterminate]];

  (* Fast path for CY1 / torus *)
  If[n == 1 && Dimensions[gpbs[[1]]] == {1, 1},
    Return @ N[gpbs[[1, 1, 1]]];
  ];

  (* Fast path for CY3 *)
  If[n == 3 && And @@ (Dimensions[#] == {3, 3} & /@ gpbs),
    perms = Permutations[Range[3]];
    signs = Signature /@ perms;
    Return @ N @ Sum[
      signs[[a]] signs[[b]]
      gpbs[[1, perms[[a, 1]], perms[[b, 1]]]]
      gpbs[[2, perms[[a, 2]], perms[[b, 2]]]]
      gpbs[[3, perms[[a, 3]], perms[[b, 3]]]],
      {a, 6}, {b, 6}
    ];
  ];

  (* Fallback: fully general version *)
  Module[{lcL, lcR, tp, pairs, res},
    lcL = Normal @ LeviCivitaTensor[n];
    lcR = Normal @ LeviCivitaTensor[n];

    tp = TensorProduct @@ Join[gpbs, {lcL, lcR}];

    pairs = Join[
      Table[{2 k - 1, 2 n + k}, {k, 1, n}],
      Table[{2 k, 3 n + k}, {k, 1, n}]
    ];

    res = Quiet[
      Check[TensorContract[tp, pairs], Indeterminate],
      TensorContract::contr
    ];

    If[res === Indeterminate, Return[Indeterminate]];
    res = If[Head[res] === SparseArray, Normal[res], res];
    res = If[ArrayQ[res], First@Flatten[res], res];
    N @ res
  ]
];

(* Patch Normalization Function *)
patchNormalizeFlatPoint[pt_, dimPs_, denomTol_: 10^-30] := Module[
  {blocks, absBlocks, localMaxPos, denoms, normBlocks},
  blocks = splitByBlocks[pt, dimPs];
  absBlocks = Abs /@ blocks;
  localMaxPos = Ordering[#, -1][[1]] & /@ absBlocks;
  denoms = MapThread[#1[[#2]] &, {blocks, localMaxPos}];
  If[Min[Abs[denoms]] < denomTol, Return[{Indeterminate, Indeterminate, {}}]];
  normBlocks = MapThread[#1/#2 &, {blocks, denoms}];
  Chop @ Flatten[normBlocks]
];

(* Compute weights for sampled points on the CICY.
   Returns {w, Omega, jElimGlobal} with Omega the complex 1/det(B). *)
getWeightOmegas[
  varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_,
  g_, eqns_, beqns_, pt_, bpt_, patchLocal_, numParamsInPn_, \[Kappa]_: 1
] := Module[
  {hBlocks, pre},

  hBlocks = Which[
    ListQ[g] && Length[g] == Length[dimPs] && And @@ (MatrixQ[#, NumericQ] & /@ g),
      g,
    True,
      Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]
  ];

  pre = prepareWeightEvaluatorCICY[
    varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn, hBlocks
  ];

  getWeightOmegasPrepared[pre, pt, bpt, patchLocal]
];

(* --- Persistent weight cache primitives --------------------------------
   The cache stores raw w_j(p_i); kappa-multiplication happens at read time.
   weightCache  :: List of Lists, [n_points x n_metrics]
   ptsFlat      :: List of 6-element point records, aligned with rows
   regionLabels :: List of integers, source-region of each point
*)

addCacheRows[
  weightCache_, ptsFlat_, regionLabels_, newPts_, sourceRegion_, Ls_,
  varsFlat_, bvarsFlat_, dimPs_, eqns_, numParamsInPn_
] := Module[
  {nNew, nMetrics, preMetrics, newRows, newLabels},

  nNew = Length[newPts];
  nMetrics = Length[Ls];

  If[nNew == 0, Return[{weightCache, ptsFlat, regionLabels}]];

  preMetrics = Table[
    prepareWeightEvaluatorCICY[
      varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn,
      metricBlocksFromL[Ls[[j]]]
    ],
    {j, nMetrics}
  ];

  newRows = ParallelTable[
    Module[{point, patchLocal, row},
      point = newPts[[i, 1]];
      patchLocal = newPts[[i, 4]];
      row = Table[
        If[j == sourceRegion,
          newPts[[i, 2]],
          Module[{tmp},
            tmp = getWeightOmegasPrepared[
              preMetrics[[j]], point, Conjugate[point], patchLocal];
            If[NumericQ[tmp[[1]]], tmp[[1]], Indeterminate]
          ]
        ],
        {j, nMetrics}
      ];
      row
    ],
    {i, nNew},
    DistributedContexts -> Automatic,
    Method -> "CoarsestGrained"
  ];

  newLabels = ConstantArray[sourceRegion, nNew];

  {
    Join[weightCache, newRows],
    Join[ptsFlat, newPts],
    Join[regionLabels, newLabels]
  }
];

addCacheColumn[
  weightCache_, ptsFlat_, newL_,
  varsFlat_, bvarsFlat_, dimPs_, eqns_, numParamsInPn_
] := Module[
  {nExisting, preMetric, newCol},

  nExisting = Length[ptsFlat];

  If[nExisting == 0, Return[weightCache]];

  preMetric = prepareWeightEvaluatorCICY[
    varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn,
    metricBlocksFromL[newL]
  ];

  newCol = ParallelTable[
    Module[{point, patchLocal, tmp},
      point = ptsFlat[[i, 1]];
      patchLocal = ptsFlat[[i, 4]];
      tmp = getWeightOmegasPrepared[preMetric, point, Conjugate[point], patchLocal];
      If[NumericQ[tmp[[1]]], tmp[[1]], Indeterminate]
    ],
    {i, nExisting},
    DistributedContexts -> Automatic,
    Method -> "CoarsestGrained"
  ];

  MapThread[Append, {weightCache, newCol}]
];

dropCacheRows[weightCache_, ptsFlat_, regionLabels_, indicesToDrop_] := Module[
  {keepMask},

  If[indicesToDrop === {} || indicesToDrop === Null,
    Return[{weightCache, ptsFlat, regionLabels}]
  ];

  keepMask = ConstantArray[True, Length[ptsFlat]];
  Do[keepMask[[i]] = False, {i, indicesToDrop}];

  {
    Pick[weightCache, keepMask, True],
    Pick[ptsFlat, keepMask, True],
    Pick[regionLabels, keepMask, True]
  }
];

(* Find new Lambda matrices based on min/max weight points *)
GetNewLambdas[
  dimPs_, weightCache_, ptsFlat_, Ls_, \[Kappa]s_, dimCY_,
  frontEnd_: False, verbose_: 0
] := Module[
  {
    nPoints, nMetrics, kappaWeights, chosenIdx, chosenWeight,
    minIdx, maxIdx, minPoint, maxPoint,
    wIPSMin, wIPSMax, wRefMin, wRefMax, \[Epsilon]Min, \[Epsilon]Max,
    \[Lambda]Min, \[Lambda]Max, \[Lambda]MinInv, \[Lambda]MaxInv
  },

  nPoints = Length[ptsFlat];
  nMetrics = Length[Ls];

  PrintMsg["GetNewLambdas: Starting with " <> ToString[nPoints] <> " points (cache read)", frontEnd, verbose];
  PrintMsg["GetNewLambdas: Current number of metrics: " <> ToString[nMetrics], frontEnd, verbose];
  PrintMsg["GetNewLambdas: dimPs = " <> ToString[dimPs], frontEnd, verbose];
  PrintMsg["GetNewLambdas: dimCY = " <> ToString[dimCY], frontEnd, verbose];

  (* Each points per-metric chosen weight: pick j minimizing |kappa_j w_j - 1|. *)
  {chosenWeight, chosenIdx} = Transpose @ ParallelTable[
    Module[{kw, idx},
      kw = Table[
        If[NumericQ[weightCache[[i, j]]],
          \[Kappa]s[[j]] weightCache[[i, j]],
          Infinity
        ],
        {j, nMetrics}
      ];
      idx = Ordering[Abs[kw - 1], 1][[1]];
      {kw[[idx]], idx}
    ],
    {i, nPoints},
    DistributedContexts -> Automatic,
    Method -> "CoarsestGrained"
  ];

  PrintMsg["GetNewLambdas: Read chosen weights from cache for " <> ToString[nPoints] <> " points", frontEnd, verbose];

  minIdx = First @ Ordering[chosenWeight, 1];
  maxIdx = First @ Ordering[-chosenWeight, 1];

  wIPSMin = chosenWeight[[minIdx]];
  minPoint = ptsFlat[[minIdx, 1]];
  wRefMin = \[Kappa]s[[1]] weightCache[[minIdx, 1]];
  \[Epsilon]Min = wRefMin^(1/dimCY) - 1;

  wIPSMax = chosenWeight[[maxIdx]];
  maxPoint = ptsFlat[[maxIdx, 1]];
  wRefMax = \[Kappa]s[[1]] weightCache[[maxIdx, 1]];
  \[Epsilon]Max = wRefMax^(1/dimCY) - 1;

  PrintMsg["GetNewLambdas: IPS extremal weights: wIPSMin = " <> ToString[wIPSMin] <> ", wIPSMax = " <> ToString[wIPSMax], frontEnd, verbose];
  PrintMsg["GetNewLambdas: reference FS weights for epsilon: wRefMin = " <> ToString[wRefMin] <> ", wRefMax = " <> ToString[wRefMax], frontEnd, verbose];
  PrintMsg["GetNewLambdas: epsMin = " <> ToString[\[Epsilon]Min] <> ", epsMax = " <> ToString[\[Epsilon]Max], frontEnd, verbose];
  PrintMsg["GetNewLambdas: minPoint = " <> ToString[minPoint], frontEnd, verbose];
  PrintMsg["GetNewLambdas: maxPoint = " <> ToString[maxPoint], frontEnd, verbose];

  \[Lambda]Min = Table[
    Module[{minPtBlock, PxBlock},
      minPtBlock = minPoint[[Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + 1 ;;
                             Sum[dimPs[[k]] + 1, {k, 1, i}]]];
      PxBlock = KroneckerProduct[minPtBlock, ConjugateTranspose[minPtBlock]] /
                (ConjugateTranspose[minPtBlock] . minPtBlock);
      1/(1 + \[Epsilon]Min) (IdentityMatrix[dimPs[[i]] + 1] + \[Epsilon]Min PxBlock)
    ],
    {i, Length[dimPs]}
  ];

  \[Lambda]Max = Table[
    Module[{maxPtBlock, PxBlock},
      maxPtBlock = maxPoint[[Sum[dimPs[[k]] + 1, {k, 1, i - 1}] + 1 ;;
                             Sum[dimPs[[k]] + 1, {k, 1, i}]]];
      PxBlock = KroneckerProduct[maxPtBlock, ConjugateTranspose[maxPtBlock]] /
                (ConjugateTranspose[maxPtBlock] . maxPtBlock);
      1/(1 + \[Epsilon]Max) (IdentityMatrix[dimPs[[i]] + 1] + \[Epsilon]Max PxBlock)
    ],
    {i, Length[dimPs]}
  ];

  \[Lambda]MinInv = Table[
    Module[{inv},
      inv = Chop[Inverse[Chop[\[Lambda]Min[[i]]]]];
      Chop[1/2 (inv + ConjugateTranspose[inv])]
    ],
    {i, Length[\[Lambda]Min]}
  ];

  \[Lambda]MaxInv = Table[
    Module[{inv},
      inv = Chop[Inverse[Chop[\[Lambda]Max[[i]]]]];
      Chop[1/2 (inv + ConjugateTranspose[inv])]
    ],
    {i, Length[\[Lambda]Max]}
  ];

  Return[{
    Table[CholeskyDecomposition[\[Lambda]MinInv[[i]]], {i, Length[\[Lambda]MinInv]}],
    Table[CholeskyDecomposition[\[Lambda]MaxInv[[i]]], {i, Length[\[Lambda]MaxInv]}]
  }];
];


(* Sample points on CICY with given metric *)
getPointsOnCYIPS[varsUnflat_, numParamsInPn_, dimPs_, params_, pointsOnSphere_, 
    eqns_, L_, precision_: 20] := Module[
    {subst, pts, i, j, a, b, c, res, maxPoss, absPts, transformedParams, 
     transformedSphere, eqSystem, paramVars},
    (
    
    (* For each projective space, compute L^{-1} and apply to sphere points.
       NOTE: this must be L^{-1}, not L^{-T}.  The points span L^{-1}.V for a
       Haar-random V, so z in L^{-1}.V  <=>  L.z in V, making the random
       sections a = Transpose[L].b with b standard Gaussian, hence
       |L.z|^2 = z^dag (L^dag L) z.  The sampling density is then the FS
       measure of Lambda = L^dag L, which is exactly what getWeightOmegas and
       the ownership test use.  Using L^{-T} here instead samples from
       Conjugate[L L^dag] while the weights assume L^dag L; the two agree only
       for normal L, and for L = Id, which is why single-region runs looked
       correct while multi-region Euler numbers came out ~20% low. *)
    transformedSphere = Table[
      Module[{LInv},
          LInv = Inverse[L[[j]]];
          Table[
              Module[{transformed},
                  transformed = Table[
                      Sum[LInv[[a, c]] pointsOnSphere[[j, b, c]],
                          {c, 1, Length[pointsOnSphere[[j, 1]]]}],
                      {a, 1, Length[pointsOnSphere[[j, 1]]]}
                  ];
                  transformed
              ],
              {b, 1, Length[pointsOnSphere[[j]]]}
          ]
        ],
        {j, 1, Length[dimPs]}
    ];
    
    (* Build substitution using transformed sphere points *)
    subst = {};
    For[j = 1, j <= Length[dimPs], j++,
        AppendTo[subst, 
            Table[varsUnflat[[j, a]] -> 
                Sum[params[[j, b]] transformedSphere[[j, b, a]], 
                    {b, Length[params[[j]]]}],
                {a, Length[varsUnflat[[j]]]}
            ]
        ];
    ];
    subst = Flatten[subst];
    
    (* Solve for parameters that put us on the CICY *)
    eqSystem = Table[eqns[[i]] == 0, {i, Length[eqns]}] /. subst;
    eqSystem = SetPrecision[eqSystem, precision];

    paramVars = Flatten[Rest /@ params];

    res = NSolve[eqSystem, paramVars, WorkingPrecision -> precision];
    res = res /. HoldPattern[Null * r_Rule] :> r /. HoldPattern[r_Rule * Null] :> r;
    res = Which[
        res === {} || res === Null || res === $Failed, {},
        MatchQ[res, {{(_Rule | _RuleDelayed) ..} ..}], res,
        MatchQ[res, {(_Rule | _RuleDelayed) ..}], {res},
        True, {}
        ];

    If[res === {} || res === Null, Return[{}]];

    pts = Chop @ N[(varsUnflat /. subst) /. res, precision];
    pts = Map[Flatten, pts];
    pts = Select[pts, VectorQ[#, NumericQ] && Length[#] == Length@Flatten[varsUnflat] &];
    If[pts === {}, Return[{}]];
    
    (* Patch Normalization *)
    pts = Select[
      patchNormalizeFlatPoint[#, dimPs] & /@ pts,
      VectorQ[#, NumericQ] && Length[#] == Length@Flatten[varsUnflat] &
    ];
    If[pts === {}, Return[{}]];
    
    Return[pts];
)];

(* Sample points and compute weights *)
SamplePointsIPS[
  varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_,
  coefficients_, exponents_, L_, numPts_, dimCY_,
  kahlerModuli_: Automatic, \[Kappa]In_: 1, precision_: 20, regionLabel_: Missing["NotSpecified"],
  frontEnd_: False, verbose_: 0
] := Module[
  {
    eqns, beqns, numParamsInPn, params, pointsOnSphere, pts, i, j,
    w, Omega, allPts, \[Kappa], conf, start, col, totalDeg, numPoints,
    eqnTol, ptsGood, tries, maxTries, ptsBatch, residuals, mask, ptsKeep, nKeep,
    hBlocksNative, preNative, rootsPerCall, nRaw
  },

  eqns = Table[
    Sum[coefficients[[i, j]] Times @@ (Power[varsFlat, exponents[[i, j]]]),
      {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}
  ];

  beqns = Table[
    Sum[Conjugate[coefficients[[i, j]]] Times @@ (Power[bvarsFlat, exponents[[i, j]]]),
      {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}
  ];

  eqnTol = 10^(-Floor[precision/2]);

  eqnResidual[pt_List] /; VectorQ[pt, NumericQ] && Length[pt] == Length[varsFlat] := Module[{vals},
    vals = eqns /. Thread[varsFlat -> SetPrecision[pt, precision]];
    Max[Abs[N[vals, precision]]]
  ];
  eqnResidual[_] := Infinity;

  conf = {};
  For[i = 1, i <= Length[coefficients], i++,
    start = 1;
    col = {};
    For[j = 1, j <= Length[dimPs], j++,
      totalDeg = Plus @@ exponents[[i, 1, start ;; start + dimPs[[j]]]];
      AppendTo[col, totalDeg];
      start += dimPs[[j]] + 1;
    ];
    AppendTo[conf, col];
  ];

  numParamsInPn = Table[1, {i, Length[dimPs]}];
  For[i = 1, i <= Length[eqns] - Length[dimPs], i++,
    If[Union[numParamsInPn*conf[[i]]] === {0},
      numParamsInPn[[Ordering[conf[[i]], 1][[1]]]]++
    ];
  ];

  While[Length[eqns] > Plus @@ numParamsInPn,
    For[i = 1, i <= Length[eqns], i++,
      If[Length[eqns] == Plus @@ numParamsInPn, Break[];];
      numParamsInPn[[Ordering[conf[[i]], 1][[1]]]]++
    ];
  ];

  (* Reduce the parameter count to the number of equations, keeping NSolve
     square. A factor must never go below zero: the For restarts at i = 1, so
     without that guard every decrement lands on the first factor once more
     than one is needed, i.e. when (#factors - #equations) >= 2. The
     tetraquadric gave {-2,1,1,1} -- three parameters against one equation, so
     NSolve got an underdetermined system and never returned; it now gives
     {0,0,0,1}. Geometries with (#factors - #equations) <= 1 never went
     negative, so their numParamsInPn is unchanged. *)
  While[Length[eqns] < Plus @@ numParamsInPn,
    For[i = 1, i <= Length[numParamsInPn], i++,
      If[numParamsInPn[[i]] <= 0, Continue[]];
      numParamsInPn[[i]]--;
      If[Min[Transpose[conf . numParamsInPn]] == 0,
        numParamsInPn[[i]]++;,
        Break[];
      ];
    ];
  ];

  i = 1;
  While[i <= Length[numParamsInPn],
    If[numParamsInPn[[i]] > dimPs[[i]],
      For[j = 1, j <= Length[numParamsInPn], j++,
        If[numParamsInPn[[j]] >= dimPs[[j]],
          Continue[];,
          numParamsInPn[[j]]++; numParamsInPn[[i]]--; Break[];
        ];
      ];
      Continue[];
    ];
    i++;
  ];

  hBlocksNative = metricBlocksFromL[L];
  preNative = prepareWeightEvaluatorCICY[
    varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn, hBlocksNative
  ];

  Clear[t];
  params = Table[Join[{1}, Array[Unique["t"] &, numParamsInPn[[j]]]], {j, Length[numParamsInPn]}];

  ptsGood = {};
  tries = 0;
  maxTries = 200;
  rootsPerCall = $Failed;
  (* Initial guess for batch 1; adaptively refined from batch 2 onward
     once rootsPerCall is known *)
  numPoints = Ceiling[numPts/5];

  While[Length[ptsGood] < numPts && tries < maxTries,
    tries++;

    (* Adaptive sizing: from batch 2 onward, use the observed roots/call
       to size only whats still needed (with a 15% safety margin). *)
    If[tries > 1 && NumericQ[rootsPerCall] && rootsPerCall > 0,
      numPoints = Max[10, Ceiling[(numPts - Length[ptsGood]) / rootsPerCall * 1.15]];
    ];

    pointsOnSphere = ParallelTable[
      SamplePointsOnSphere[dimPs[[i]] + 1, numPoints (numParamsInPn[[i]] + 1)],
      {i, Length[dimPs]},
      DistributedContexts -> Automatic
    ];

    ptsBatch = ParallelTable[
      getPointsOnCYIPS[
        varsUnflat, numParamsInPn, dimPs, params,
        Table[
          pointsOnSphere[[i, p + (b - 1) numPoints]],
          {i, Length[pointsOnSphere]}, {b, 1 + numParamsInPn[[i]]}
        ],
        eqns, L, precision
      ],
      {p, numPoints},
      DistributedContexts -> Automatic,
      Method -> "CoarsestGrained"
    ];

    ptsBatch = Flatten[ptsBatch, 1];
    ptsBatch = Select[ptsBatch, MatchQ[#, {_?NumericQ ..}] && Length[#] == Length[varsFlat] &];
    nRaw = Length[ptsBatch];
    rootsPerCall = If[numPoints > 0, N[nRaw / numPoints], rootsPerCall];
    PrintMsg["SamplePointsIPS: Batch " <> ToString[tries] <> " produced " <> ToString[nRaw] <>
      " raw points (numPoints=" <> ToString[numPoints] <> ", rootsPerCall=" <> ToString[rootsPerCall] <> ").",
      frontEnd, verbose];

    residuals = eqnResidual /@ ptsBatch;
    mask = Thread[residuals < eqnTol];

    ptsKeep = Pick[ptsBatch, mask, True];
    nKeep = Length[ptsKeep];

    ptsGood = Join[ptsGood, ptsKeep];

    PrintMsg["SamplePointsIPS: Batch " <> ToString[tries] <> " kept " <> ToString[nKeep] <>
      " points (eqnTol=" <> ToString[eqnTol] <> "). Total kept = " <> ToString[Length[ptsGood]] <> ".",
      frontEnd, verbose];
  ];

  pts = ptsGood;
  If[Length[pts] > numPts, pts = RandomSample[pts, numPts]];
  PrintMsg["SamplePointsIPS: Final valid points = " <> ToString[Length[pts]] <> " (requested " <> ToString[numPts] <>
    ", trimmed from " <> ToString[Length[ptsGood]] <> ").", frontEnd, verbose];

  allPts = ParallelTable[
    Module[{point, patchLocal},
      point = pts[[i]];
      patchLocal = patchIndicesByBlock[point, dimPs];
      {w, Omega, jElimGlobal} =
        getWeightOmegasPrepared[preNative, point, Conjugate[point], patchLocal];
      (* Record field 3 is the COMPLEX Omega = 1/det(B), not |Omega|^2. *)
      Chop[{point, w, Omega, patchLocal, jElimGlobal, regionLabel}]
    ],
    {i, Length[pts]},
    DistributedContexts -> Automatic,
    Method -> "CoarsestGrained"
  ];

  allPts = Select[
    allPts,
    NumericQ[#[[2]]] && NumericQ[#[[3]]] &&
    Abs[#[[2]]] < Infinity && Abs[#[[3]]] < Infinity &
  ];

  If[Length[allPts] == 0,
    PrintMsg["ERROR: All weights invalid; returning empty set.", frontEnd, 1];
    Return[{{}, Indeterminate, numParamsInPn}];
  ];

  PrintMsg["SamplePointsIPS: Computed weights for " <> ToString[Length[allPts]] <> " points", frontEnd, verbose];
  PrintMsg["SamplePointsIPS: Number with valid weights: " <> ToString[Count[allPts[[;;, 2]], _?NumericQ]], frontEnd, verbose];

  \[Kappa] = \[Kappa]In;
  If[\[Kappa] == 1,
    \[Kappa] = 1/Mean[allPts[[;;, 2]]];
  ];

  Return[{allPts, \[Kappa], numParamsInPn}];
];


(* Independent kappa estimation for a single metric L.

   Draws a FRESH kappa-only sample of size kappaSampleSize from metric L,
   computes the raw IPS weights w for that sample, and returns
   kappa = 1/Mean[w]. The sampled points are discarded afterward: this
   routine never writes to regionPts, adaptWeightCache, or prodWeightCache.

   This makes the kappa estimate statistically independent from both the
   adaptation sample (used to build Lambda matrices) and the production
   sample (used for the final estimate). It is implemented on top of
   SamplePointsIPS with \[Kappa]In = 1, reusing its fresh-draw +
   kappa = 1/Mean[w] logic, and simply returns the kappa while dropping
   the points. *)
EstimateKappaCICY[
  varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_,
  coefficients_, exponents_, L_, kappaSampleSize_, dimCY_,
  kahlerModuli_: Automatic, precision_: 20, regionLabel_: Missing["NotSpecified"],
  frontEnd_: False, verbose_: 0
] := Module[
  {tmpPts, tmpKappa, tmpNumParams},
  {tmpPts, tmpKappa, tmpNumParams} = SamplePointsIPS[
    varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
    dimPs, coefficients, exponents, L,
    kappaSampleSize, dimCY, kahlerModuli, 1, precision, regionLabel,
    frontEnd, verbose
  ];
  (* Discard the kappa-only sample; it must never enter any cache. *)
  tmpPts =.;
  Return[tmpKappa];
];


(* Rejection Sampling -- cache-aware.
   On entry, weightCache/ptsFlat/cacheRegionLabels describe ALL points
   sampled so far (across all regions) and their kappa-multiplied raw weights
   against every metric in Ls. On exit, the rows for region LPos in those
   structures have been replaced by the new region-LPos sample.
*)
SamplePointsWithRejectionCICY[
  varsFlat_, bvarsFlat_, varsUnflat_, bvarsUnflat_, dimPs_,
  coefficients_, exponents_, Ls_, pres_, LPos_, numPts_, dimCY_, numParamsInPn_,
  \[Kappa]s_, weightCache_, ptsFlat_, cacheRegionLabels_,
  kahlerModuli_: Automatic, precision_: 20,
  numSampledInitial_: 0, frontEnd_: False, verbose_: 0
] := Module[
  {
    newPts, pts, ptsScored, resampleCounter, numSampled, numAccepted,
    numToSample, tmpPts, tmpKappa, tmpNumParams, keepMask, keptScored,
    ownershipTol, nNeeded, estAcceptance, safetyFactor,
    minBatch, maxBatch, acceptedThisPass,
    cache, pfCache, lblCache,
    existingIdx, existingScores, ownershipMaskFirst,
    keptIdx, droppedIdx, keptFreshPts, keptFreshRows
  },

  ownershipTol = 10^-6;
  resampleCounter = 0;
  numSampled = numSampledInitial;
  numAccepted = 0;
  newPts = {};

  cache    = weightCache;
  pfCache  = ptsFlat;
  lblCache = cacheRegionLabels;

  safetyFactor = 1.25;
  minBatch = 250;
  maxBatch = 20000;

  While[Length[newPts] < numPts,

    nNeeded = numPts - Length[newPts];

    If[resampleCounter == 0,
      (* FIRST PASS: read cached weights for existing region-LPos points;
         keep the ones owned by LPos, drop the rest from the cache. *)
      existingIdx = Flatten @ Position[lblCache, LPos];

      If[Length[existingIdx] > 0,
        existingScores = Table[
          Module[{rawRow, kw},
            rawRow = cache[[idx]];
            kw = Table[
              If[NumericQ[rawRow[[j]]], \[Kappa]s[[j]] rawRow[[j]], Infinity],
              {j, Length[Ls]}
            ];
            Abs[kw - 1]
          ],
          {idx, existingIdx}
        ];

        ownershipMaskFirst = Table[
          (First @ Ordering[existingScores[[k]], 1]) == LPos,
          {k, Length[existingIdx]}
        ];

        keptIdx    = Pick[existingIdx, ownershipMaskFirst, True];
        droppedIdx = Pick[existingIdx, ownershipMaskFirst, False];

        keptFreshPts = pfCache[[keptIdx]];

        (* Drop not-owned rows from cache before continuing *)
        {cache, pfCache, lblCache} =
          dropCacheRows[cache, pfCache, lblCache, droppedIdx];

        acceptedThisPass = Length[keptFreshPts];
        newPts = Join[newPts, keptFreshPts];
        numAccepted += acceptedThisPass;

        PrintMsg[
          "Region " <> ToString[LPos] <> ": kept " <> ToString[Length[newPts]] <>
          "/" <> ToString[numPts] <> " points after pass 1 (from cache; " <>
          "accepted = " <> ToString[acceptedThisPass] <>
          ", dropped = " <> ToString[Length[droppedIdx]] <> ").",
          frontEnd, verbose
        ];
      ];

      ,
      (* SUBSEQUENT PASS: sample fresh from L_LPos, score against all metrics,
         keep only the L_LPos-owned ones and add them (with their full
         raw-weight rows) to the cache. Rejected fresh points never enter
         the cache. *)
      estAcceptance = If[numSampled > 0 && numAccepted > 0,
        N[numAccepted/numSampled],
        1./Max[1, Length[Ls]]
      ];
      estAcceptance = Max[10^-3, estAcceptance];

      numToSample = Ceiling[safetyFactor * nNeeded / estAcceptance];
      numToSample = Max[minBatch, numToSample];
      numToSample = Min[maxBatch, numToSample];

      {tmpPts, tmpKappa, tmpNumParams} = SamplePointsIPS[
        varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
        dimPs, coefficients, exponents, Ls[[LPos]],
        numToSample, dimCY, kahlerModuli, \[Kappa]s[[LPos]], precision, LPos,
        frontEnd, verbose
      ];

      pts = tmpPts;
      numSampled += Length[pts];

      ptsScored = attachMetricScoresToPointsCICY[pres, \[Kappa]s, pts];
      keepMask = selectPointsOwnedByMetricCICY[ptsScored, LPos, ownershipTol];
      keptScored = Pick[ptsScored, keepMask, True];
      acceptedThisPass = Length[keptScored];
      numAccepted += acceptedThisPass;

      keptFreshPts  = dropCachedMetricDataCICY[keptScored];
      keptFreshRows = keptScored[[All, 11]];  (* raw w_j(p) per metric *)

      cache    = Join[cache, keptFreshRows];
      pfCache  = Join[pfCache, keptFreshPts];
      lblCache = Join[lblCache, ConstantArray[LPos, acceptedThisPass]];

      newPts = Join[newPts, keptFreshPts];

      PrintMsg[
        "Region " <> ToString[LPos] <> ": kept " <> ToString[Length[newPts]] <>
        "/" <> ToString[numPts] <> " points after rejection pass " <>
        ToString[resampleCounter + 1] <>
        " (accepted this pass = " <> ToString[acceptedThisPass] <>
        ", estimated acceptance ~ " <>
        ToString[N[If[numSampled > 0, numAccepted/numSampled, 0], 4]] <> ").",
        frontEnd, verbose
      ];
    ];

    resampleCounter++;
  ];

  PrintMsg[
    "Region " <> ToString[LPos] <> ": acceptance = " <>
    ToString[numAccepted] <> "/" <> ToString[numSampled] <> ".",
    frontEnd, verbose
  ];

  Return[{newPts, cache, pfCache, lblCache, numAccepted, numSampled}];
];

(* Main function for IPS on CICYs *)
GeneratePointsMCICYIPS[
  TotalNumPts_, NumRegions_, dimPs_, coefficients_, exponents_,
  kahlerModuli_: Automatic, precision_: 20, verbose_: 0, frontEnd_: False,
  kappaSampleSize_: 25000, lambdaSearchSampleMin_: 20000
] := Module[
  {
    NumPts, AdaptNumPts, varsUnflat, bvarsUnflat, varsFlat, bvarsFlat, eqns, beqns,
    Ls, allPts, regionPts, newPts, adaptPts, pres,
    \[Kappa], \[Kappa]s, numParamsInPn, dimCY,
    r, i, Acceptance, NumSample, Acceptances, NumSamples, acceptanceRates,
    regionLabels, newLambdaPair, nextRegionIndex,
    adaptWeightCache, adaptPtsFlat, adaptCacheRegionLabels,
    prodWeightCache, prodPtsFlat, prodCacheRegionLabels
  },

  If[!frontEnd, ClientLibrary`SetInfoLogLevel[]];

  NumPts = Ceiling[TotalNumPts / NumRegions];
  (* Adaptation / Lambda-search samples use at least lambdaSearchSampleMin points,
     independent of the production sample size NumPts. *)
  AdaptNumPts = Max[NumPts, lambdaSearchSampleMin];
  PrintMsg[
    "Generating " <> ToString[NumPts] <> " points in each of the " <>
    ToString[NumRegions] <> " regions for a total of " <> ToString[TotalNumPts] <> " points.",
    frontEnd, verbose
  ];

  varsUnflat = Table[Subscript[x, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
  bvarsUnflat = Table[Subscript[bx, i, a], {i, Length[dimPs]}, {a, 0, dimPs[[i]]}];
  varsFlat = Flatten[varsUnflat];
  bvarsFlat = Flatten[bvarsUnflat];

  dimCY = Plus @@ dimPs - Length[coefficients];

  eqns = Table[
    Sum[coefficients[[i, j]] Times @@ (Power[varsFlat, exponents[[i, j]]]),
      {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}
  ];

  beqns = Table[
    Sum[Conjugate[coefficients[[i, j]]] Times @@ (Power[bvarsFlat, exponents[[i, j]]]),
      {j, Length[coefficients[[i]]]}],
    {i, Length[coefficients]}
  ];

  Ls = {Table[IdentityMatrix[dimPs[[i]] + 1], {i, Length[dimPs]}]};

  PrintMsg["Configuration matrix: " <> ToString[Transpose[
    Table[
      Module[{start, col, totalDeg},
        start = 1;
        col = {};
        For[i = 1, i <= Length[dimPs], i++,
          totalDeg = Plus @@ exponents[[r, 1, start ;; start + dimPs[[i]]]];
          AppendTo[col, totalDeg];
          start += dimPs[[i]] + 1;
        ];
        col
      ],
      {r, Length[coefficients]}
    ]
  ]], frontEnd, verbose];

  regionPts = {};
  Acceptances = {};
  NumSamples = {};

  (* Adaptation cache: used only while constructing Lambda matrices. *)
  adaptWeightCache = {};
  adaptPtsFlat = {};
  adaptCacheRegionLabels = {};

  PrintMsg["Processing region 1", frontEnd, verbose];
  (* Estimate kappa from a fresh, independent kappa-only sample. *)
  \[Kappa] = EstimateKappaCICY[
    varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
    dimPs, coefficients, exponents, Ls[[1]],
    kappaSampleSize, dimCY, kahlerModuli, precision, 1,
    frontEnd, verbose
  ];
  \[Kappa]s = {\[Kappa]};
  PrintMsg["Estimated kappa = " <> ToString[\[Kappa]] <>
    " from " <> ToString[kappaSampleSize] <> " independent points.", frontEnd, verbose];
  If[NumRegions > 1,
    (* Multi-region: draw an adaptation sample (>= lambdaSearchSampleMin points)
       used only to seed the Lambda-search cache. Adaptation samples are kept out
       of regionPts, which holds only final production samples. *)
    {adaptPts, \[Kappa], numParamsInPn} = SamplePointsIPS[
      varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
      dimPs, coefficients, exponents, Ls[[1]],
      AdaptNumPts, dimCY, kahlerModuli, \[Kappa], precision, 1,
      frontEnd, verbose
    ];

    (* Seed cache with region 1 rows (one column at this point, source weight reused). *)
    {adaptWeightCache, adaptPtsFlat, adaptCacheRegionLabels} = addCacheRows[
      adaptWeightCache, adaptPtsFlat, adaptCacheRegionLabels, adaptPts, 1, Ls,
      varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn
    ];
    ,
    (* Single region: no Lambda search needed. Draw the final production sample
       directly (sized NumPts) into regionPts. *)
    {newPts, \[Kappa], numParamsInPn} = SamplePointsIPS[
      varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
      dimPs, coefficients, exponents, Ls[[1]],
      NumPts, dimCY, kahlerModuli, \[Kappa], precision, 1,
      frontEnd, verbose
    ];
    regionPts = {newPts};
    Acceptances = {Length[newPts]};
    NumSamples = {Length[newPts]};
  ];

  nextRegionIndex = 2;

  While[nextRegionIndex <= NumRegions,

    newLambdaPair = GetNewLambdas[
      dimPs, adaptWeightCache, adaptPtsFlat, Ls, \[Kappa]s, dimCY,
      frontEnd, verbose
    ];

    (* First new metric in the pair *)
    Ls = Append[Ls, newLambdaPair[[1]]];
    (* Extend adaptation cache with column for new L_min across existing points. *)
    adaptWeightCache = addCacheColumn[
      adaptWeightCache, adaptPtsFlat, Ls[[-1]],
      varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn
    ];

    PrintMsg["Processing region " <> ToString[nextRegionIndex], frontEnd, verbose];
    (* Estimate kappa for the new metric from a fresh, independent sample. *)
    \[Kappa] = EstimateKappaCICY[
      varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
      dimPs, coefficients, exponents, Ls[[-1]],
      kappaSampleSize, dimCY, kahlerModuli, precision, nextRegionIndex,
      frontEnd, verbose
    ];
    AppendTo[\[Kappa]s, \[Kappa]];
    PrintMsg["Estimated kappa = " <> ToString[\[Kappa]] <>
      " from " <> ToString[kappaSampleSize] <> " independent points.", frontEnd, verbose];
    (* Adaptation sample (>= lambdaSearchSampleMin points), using the precomputed
       kappa. Kept out of regionPts; feeds only the Lambda-search cache. *)
    {adaptPts, \[Kappa], numParamsInPn} = SamplePointsIPS[
      varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
      dimPs, coefficients, exponents, Ls[[-1]],
      AdaptNumPts, dimCY, kahlerModuli, \[Kappa], precision, nextRegionIndex,
      frontEnd, verbose
    ];

    (* Add new region rows to the adaptation cache. *)
    {adaptWeightCache, adaptPtsFlat, adaptCacheRegionLabels} = addCacheRows[
      adaptWeightCache, adaptPtsFlat, adaptCacheRegionLabels, adaptPts, nextRegionIndex, Ls,
      varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn
    ];

    nextRegionIndex++;

    (* Second new metric in the pair, only if needed *)
    If[nextRegionIndex <= NumRegions,
      Ls = Append[Ls, newLambdaPair[[2]]];
      adaptWeightCache = addCacheColumn[
        adaptWeightCache, adaptPtsFlat, Ls[[-1]],
        varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn
      ];

      PrintMsg["Processing region " <> ToString[nextRegionIndex], frontEnd, verbose];
      (* Estimate kappa for the new metric from a fresh, independent sample. *)
      \[Kappa] = EstimateKappaCICY[
        varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
        dimPs, coefficients, exponents, Ls[[-1]],
        kappaSampleSize, dimCY, kahlerModuli, precision, nextRegionIndex,
        frontEnd, verbose
      ];
      AppendTo[\[Kappa]s, \[Kappa]];
      PrintMsg["Estimated kappa = " <> ToString[\[Kappa]] <>
        " from " <> ToString[kappaSampleSize] <> " independent points.", frontEnd, verbose];
      (* Adaptation sample (>= lambdaSearchSampleMin points), using the precomputed
         kappa. Kept out of regionPts; feeds only the Lambda-search cache. *)
      {adaptPts, \[Kappa], numParamsInPn} = SamplePointsIPS[
        varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
        dimPs, coefficients, exponents, Ls[[-1]],
        AdaptNumPts, dimCY, kahlerModuli, \[Kappa], precision, nextRegionIndex,
        frontEnd, verbose
      ];

      {adaptWeightCache, adaptPtsFlat, adaptCacheRegionLabels} = addCacheRows[
        adaptWeightCache, adaptPtsFlat, adaptCacheRegionLabels, adaptPts, nextRegionIndex, Ls,
        varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn
      ];

      nextRegionIndex++;
    ];
  ];

  pres = Table[
    prepareWeightEvaluatorCICY[
      varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn,
      metricBlocksFromL[Ls[[j]]]
    ],
    {j, Length[Ls]}
  ];


  If[Length[Ls] > 1,
    pres = Table[
      prepareWeightEvaluatorCICY[
        varsFlat, bvarsFlat, dimPs, eqns, numParamsInPn,
        metricBlocksFromL[Ls[[j]]]
      ],
      {j, Length[Ls]}
    ];

    PrintMsg["Now revisiting all regions with full rejection sampling.", frontEnd, verbose];

    (* Production cache starts empty after Ls are fixed; adaptation rows never enter it. *)
    prodWeightCache = {};
    prodPtsFlat = {};
    prodCacheRegionLabels = {};

    (* regionPts holds only final production samples. Adaptation no longer populated
       it, so size the per-region containers to Length[Ls] before the production loop. *)
    regionPts = ConstantArray[{}, Length[Ls]];
    Acceptances = ConstantArray[0, Length[Ls]];
    NumSamples = ConstantArray[0, Length[Ls]];

    For[r = 1, r <= Length[Ls], r++,
      PrintMsg["Reprocessing region " <> ToString[r], frontEnd, verbose];

      {newPts, prodWeightCache, prodPtsFlat, prodCacheRegionLabels, Acceptance, NumSample} =
        SamplePointsWithRejectionCICY[
          varsFlat, bvarsFlat, varsUnflat, bvarsUnflat,
          dimPs, coefficients, exponents,
          Ls, pres, r, NumPts, dimCY, numParamsInPn, \[Kappa]s,
          prodWeightCache, prodPtsFlat, prodCacheRegionLabels,
          kahlerModuli, precision,
          0,
          frontEnd, verbose
        ];

      regionPts[[r]] = newPts;
      Acceptances[[r]] = Acceptance;
      NumSamples[[r]] = NumSample;
    ];
  ];

  regionPts = Table[
    Module[{nHere = Length[regionPts[[i]]], take},
      take = Min[NumPts, nHere];
      If[nHere <= take,
        regionPts[[i]],
        RandomSample[regionPts[[i]], take]
      ]
    ],
    {i, Length[regionPts]}
  ];

  regionLabels = Flatten@Table[
    ConstantArray[r, Length[regionPts[[r]]]],
    {r, Length[regionPts]}
  ];

  allPts = Flatten[regionPts, 1];

  acceptanceRates = N[Acceptances / NumSamples];

  PrintMsg["done.", frontEnd, verbose];

  Return[{
    allPts[[;;, 1]],
    MapThread[acceptanceRates[[#2]] #1 &, {allPts[[;;, 2]], regionLabels}],
    (* Field 3: complex Omega = 1/det(B) per point, matching the omega
       convention of the other point generators (NOT |Omega| or |Omega|^2). *)
    allPts[[;;, 3]],
    allPts[[;;, 4]],
    allPts[[;;, 5]],
    regionLabels,
    \[Kappa]s,
    Acceptances,
    NumSamples,
    {dimCY},
    Ls
  }];
];
