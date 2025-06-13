from pyscipopt import Sepa, LP, SCIP_RESULT, SCIP_LPSOLSTAT
from separators.lptransformer import *

class LiftProjectSimplex(Sepa):
    def sepainit(self):
        pass

    def __init__(self):
        self.ncuts = 0

    def solveCGLP(self, stdlp, stdcolsols, candcolid, down_ub, up_lb):
        # Initialize cut coefficients and right-hand side for the cut

        # Retrieve the SCIP model instance
        scip = self.model

        # Build the cut generation LP (CGLP) with the objective of minimizing the cut
        cglp = LP(name="CGLP", sense="minimize")

        # Number of gamma variables corresponding to the number of rows in the standard LP
        ngamma = stdlp.ncsr_cols

        # Set up the cut coefficients (gamma) for the LP
        gammaobjs = stdcolsols + [1.0]  # Objective coefficients for gamma
        gammaentries = [[]] * ngamma  # Initialize entries for gamma columns
        assert len(gammaobjs) == ngamma
        lbs = [-cglp.infinity()] * ngamma  # Set lower bounds to negative infinity
        cglp.addCols(entrieslist=gammaentries, objs=gammaobjs, lbs=lbs, ubs=None)  # Add gamma columns to the LP

        # Determine the number of zeta variables
        nzeta = stdlp.ncsr_rows
        # 0 for down disjunction, 1 for up disjunction
        indexmap = {0: [], 1:[]}

        zeta_entries = [[]] * nzeta
        # add cols of aggregation coefficients (zeta'_h_eq_pos, zeta'_h_eq_neg, zeta'_h_noneq)
        col_start = ngamma
        zeta_start = ngamma
        for h in range(2):
            # add noneq
            cglp.addCols(entrieslist = zeta_entries, objs = None, lbs = None, ubs = None)
            indexmap[h].extend(range(col_start, col_start + nzeta))
            col_start += nzeta

        col_end = col_start
        tranposedcsr = stdlp.getTransposedCSR()
        lhss = [0.0] * (ngamma -1) + [-cglp.infinity()]
        rhss = [0.0] * ngamma

        # add rows: C^t_h zeta_h - gamma <= 0
        for h in range(2):
            mat_entrieslist = []
            assert  tranposedcsr.shape[0] == ngamma
            for stdlp_col in range(ngamma):
                start = tranposedcsr.indptr[stdlp_col]
                end = tranposedcsr.indptr[stdlp_col + 1]
                # Get stdlp transposed column indices and corresponding values
                row_indices = tranposedcsr.indices[start:end]
                values = tranposedcsr.data[start:end]
                # add entry for gamma at stdlp_col
                entries = [(stdlp_col, -1)]
                for row, value in zip(row_indices, values):
                    # get cglp col index
                    cglp_col = indexmap[h][row]
                    assert row >= 0 and row < nzeta
                    std_col = stdlp.tranposedcsr_stdcols[row]
                    # if corresponding to beta, and corresponding to variable bounds
                    if stdlp_col == ngamma - 1 and std_col is not None:
                        # get stdcol index and set correct value in down and up disjunction
                        # lower bound col and up disjunction
                        assert (std_col < 0 and value == 0.0) or (std_col > 0 and value == stdlp.stdcols[std_col - 1].ub)
                        if std_col < 0 and -std_col - 1 == candcolid and h == 1:
                            #print(std_col, cglp_col, row, candcolid, h, value,  -up_lb)
                            value = -up_lb
                        # upper bound col and down disjunction
                        elif std_col > 0 and std_col - 1 == candcolid and h == 0:
                            #print(std_col, cglp_col, row, candcolid, h, value, down_ub)
                            value = down_ub
                    entries += [(cglp_col, value)]
                mat_entrieslist.append(entries)

            cglp.addRows(entrieslist = mat_entrieslist, lhss = lhss, rhss = rhss)

        # add normalization condition: sum_h zeta'_h_noneq + sum_h zeta'_h_eq_pos + sum_h zeta'_h_eq_neg <= 1
        entries = [(i, 1.0) for i in range(zeta_start, col_end)]
        hs = col_end - zeta_start

        cglp.addRow(entries = entries, lhs = hs, rhs = hs)
        print(f"stlp, #rows:{stdlp.ncsr_rows}, #cols:{stdlp.ncsr_cols}. cglp #cols:{col_end}, #zeta:{nzeta}. solving...\n")
        #cglp.writeLP(b"cglp.cip")
        cglp.solve(dual = True)
        if cglp.isPrimalFeasible():
            primalsols = cglp.getPrimal()

            alpha, beta = stdlp.getOriginalCut(primalsols[0:ngamma - 1], primalsols[ngamma - 1])

            #print(alpha, len(alpha), "\n", beta, "\n", primalsols, len(primalsols),"\n")
            cutefficiency = 1.0 * primalsols[len(stdcolsols)]
            for i in range(len(stdcolsols)):
                #print(primalsols[i])
                cutefficiency += stdcolsols[i] * primalsols[i]
            print(f"cut effiency: {cutefficiency} \n")

            return alpha, -beta, primalsols[ngamma: ngamma + nzeta], primalsols[ngamma + nzeta : ngamma + 2 * nzeta]

        else:
            print("cglp not feasible\n")
            return None, None, None, None

    def sepaexeclp(self):
        result = SCIP_RESULT.DIDNOTRUN
        scip = self.model

        # This heuristic does not run if the LP status is not optimal
        lpsolstat = scip.getLPSolstat()

        if lpsolstat != SCIP_LPSOLSTAT.OPTIMAL:
            return {"result": result}

        # get LP data
        cols = scip.getLPColsData()
        rows = scip.getLPRowsData()

        # exit if LP is trivial
        if len(cols) == 0 or len(rows) == 0:
            return {"result": result}

        stdlp, stdcolsols, colsols = getStandardLP(scip, rows, cols)
        candcolid, down_ub, up_lb, _, _ = selectCandidate(stdlp.stdcols, stdcolsols)
        result = SCIP_RESULT.DIDNOTFIND

        # get cut's coefficients
        cutcoefs, cutlhs, _, _ = self.solveCGLP(stdlp, stdcolsols, candcolid, down_ub, up_lb)

        if cutcoefs is None:
            return {"result": result}

        cuteffiency = -cutlhs
        for i in range(len(cutcoefs)):
            cuteffiency += cutcoefs[i] * colsols[i]
        # add cut
        cut = scip.createEmptyRowSepa(self, "liftproject%d"%(self.ncuts), lhs = cutlhs, rhs = None)
        scip.cacheRowExtensions(cut)

        for j in range(len(cutcoefs)):
            if scip.isZero(cutcoefs[j]): # maybe here we need isFeasZero
                continue
            #print(cutcoefs[j])
            scip.addVarToRow(cut, cols[j].getVar(), cutcoefs[j])

        if cut.getNNonz() == 0 and scip.isFeasNegative(-cutlhs):
            print("cutoff\n")
            return {"result": SCIP_RESULT.CUTOFF}

        print(f"normalized efficiency:{scip.getCutEfficacy(cut)} original cut efficiency: {cuteffiency} \n")

        # Only take efficacious cuts, except for cuts with one non-zero coefficient (= bound changes)
        # the latter cuts will be handeled internally in sepastore.
        if cut.getNNonz() == 1 or scip.isCutEfficacious(cut):

            # flush all changes before adding the cut
            scip.flushRowExtensions(cut)

            infeasible = scip.addCut(cut, forcecut=True)
            self.ncuts += 1

            if infeasible:
                result = SCIP_RESULT.CUTOFF
            else:
                result = SCIP_RESULT.SEPARATED
        scip.releaseRow(cut)

        return {"result": result}