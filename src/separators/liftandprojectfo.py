from pyscipopt import Sepa, LP, SCIP_RESULT, SCIP_LPSOLSTAT
from separators.lptransformer import *
from utils.utils import *
from separators.liftandprojectspx import LiftProjectSimplex
import random
import torch

class ModuleParams:
    """Stores module-specific hyperparameters for the lift-and-project matrix"""
    def __init__(self, nrowAin, uplbinds, downubinds, updlbs, downdubs, ncolC, lbs, ubs, device, dtype):
        """
        Args:
            numaggrs: Number of aggregation rows
            nrowC: Number of rows in constraint matrix C
            nrowA: Number of rows in matrix A
            uplbinds, downubinds: Indices for bound constraints
            updlbs, downdubs: Values for bound constraints
            device: Computation device (CPU/GPU)
        """
        self.nlbs = len(lbs)
        self.nubs = len(ubs)
        self.nrowA = nrowAin
        self.nrowC = nrowAin + self.nlbs + self.nubs
        self.uplbinds = torch.tensor(uplbinds, device=device, requires_grad=False)
        self.downubinds = torch.tensor(downubinds, device=device, requires_grad=False)
        self.updlbs = torch.tensor(updlbs, device=device, dtype = dtype, requires_grad=False)
        self.downdubs = torch.tensor(downdubs, device=device, dtype = dtype, requires_grad=False)
        self.ncolC = ncolC
        self.lbs = torch.tensor(lbs, device=device,  requires_grad=False, dtype = dtype)
        self.ubs = torch.tensor(ubs, device=device,  requires_grad=False, dtype = dtype)


def pwlfunc(modparam: ModuleParams, Abin, zeta):
    """
    Forward pass of the lift-and-project module
    Args:
        Abin: Input tensor of shape [nrowA, ncolC+1]
        zeta: tensor variable of shape [2, nrowC]
    """

    assert( Abin.size(0) == modparam.nrowAin )
    zeta = zeta
    # Lift-and-project layer
    if Abin.is_sparse_csr:
        zeta_reshaped = zeta[:,:, 0:modparam.nrowA].view(-1, modparam.nrowA)  # Reshape to match Abin's dimensions

        # Perform sparse multiplication
        result = torch.sparse.mm(zeta_reshaped, Abin)

        Ab2 = result.view(modparam.numliftprojs, 2, Abin.shape[1])
    else:
        Ab2 = torch.tensordot(zeta[:,:, 0:modparam.nrowA], Abin, dims=([2],[0]))

    # Handle bounds
    zetalb = zeta[:, modparam.nrowA : (modparam.nrowA + modparam.nlbs) ]
    zetaub = zeta[:, (modparam.nrowA + modparam.nlbs) : (modparam.nrowA + modparam.nlbs + modparam.nubs) ]
    Ab2[:, 0:-1] += zetalb
    Ab2[:, 0:-1] -= zetaub
    Ab2[:, -1] -= torch.tensordot(zetalb, modparam.lbs, dims=([1],[0]))
    Ab2[:, -1] += torch.tensordot(zetaub, modparam.ubs, dims=([1],[0]))

    # Apply conjunction constraints
    #print(Ab2[:, 0, -1].shape, modparam.downubinds.shape, (zetaub[:, 0, modparam.downubinds] * modparam.downdubs).squeeze().shape)
    Ab2[0, -1] += zetaub[0, modparam.downubinds] * modparam.downdubs
    Ab2[1, -1] -= zetalb[1, modparam.uplbinds] * modparam.updlbs
    Ab3, _ = torch.max(Ab2, dim = 0)

    Ab4 = Ab3.reshape(modparam.ncolC)
    return Ab4

class LiftProjectFO(Sepa):
    def sepainit(self):
        pass

    def __init__(self, options):
        self.ncuts = 0
        self.options = options

    def solveFO(self, stdlp, stdcolsols, candcolid, down_ub, up_lb, int_ids, int_probs):
        # Initialize cut coefficients and right-hand side for the cut

        # Retrieve the SCIP model instance
        scip = self.model

        # Set up the cut coefficients (gamma) for the LP
        objs = stdcolsols + [1.0]  # Objective coefficients for gamma
        ncolC = stdlp.ncsr_cols
        assert len(objs) == ncolC
        lbstdcols = [-stdlp.csr_origins[rowid]-1 for rowid in stdlp.csr_typerows[RowType.VLB]]
        ubstdcols = [-stdlp.csr_origins[rowid]-1 for rowid in stdlp.csr_typerows[RowType.VUB]]
        # get lower and upper bounds for bound rows
        lbs = [0 for stdcolid in lbstdcols]
        ubs = [stdlp.stdcols[stdcolid].ub for stdcolid in ubstdcols]

        # find lower and upper bounds for the candidate in two disjunctions
        uplbind = stdlp.stdcol_rowpos[0][candcolid]
        downubind = stdlp.stdcol_rowpos[1][candcolid]
        assert uplbind is not None and downubind is not None
        sol = stdcolsols[candcolid]
        lbs = [0.0]
        ubs = [stdlp.stdcols[candcolid].ub]
        assert sol >= lbs[0] and sol <= ubs[0]
        uplbinds = [uplbind]
        downubinds = [downubind]
        updlbs = [math.ceil(sol) -lbs[0]]
        downdubs = [math.floor(sol) -ubs[0]]

        # should we use the spx's results for debug?
        if self.options.spxinit:
            liftprojspx = LiftProjectSimplex()
            dbgalpha, dbgbeta, dbgzetadown, dbgzetaup = liftprojspx.solveCGLP(stdlp, stdcolsols, candcolid, down_ub, up_lb)
            print(np.linalg.norm(dbgzetadown, ord = 1), np.linalg.norm(dbgzetaup, ord = 1))

        # assemble data matrices and parameters for the module
        # C is implicitly encoded by A and lbs and ubs
        nrowA0 = len(stdlp.csr_typerows[RowType.INEQ])
        start = stdlp.csr_rowstarts[0]
        end = stdlp.csr_rowstarts[nrowA0]
        row_indices = stdlp.csr_rowstarts[0:nrowA0+1]
        col_indices = stdlp.csr_cols[start:end]
        values = stdlp.csr_data[start:end]
        modparam = ModuleParams(nrowA0, uplbinds, downubinds, updlbs, downdubs, ncolC, lbs, ubs, self.options.device, self.options.dtype)
        torch.manual_seed(self.options.seed)
        # init values to the tensor data
        Abin = torch.sparse_csr_tensor(row_indices, col_indices, values, size=(nrowA0, ncolC),
            requires_grad=False, device= self.options.device, dtype = self.options.dtype)
        if self.options.denseinput:
            Ain = Abin.to_dense()
        # set zeta: either the optimal value from spx or all pnes
        zeta = torch.ones((2, ncolC), device=self.options.device, dtype = self.options.dtype)
        if self.options.spxinit:
            if dbgzetadown is not None:
                zeta.data[0,:] = torch.tensor(dbgzetadown, device=self.options.device, dtype = self.options.dtype)
            if dbgzetaup is not None:
                zeta.data[1,:] = torch.tensor(dbgzetaup, device=self.options.device, dtype = self.options.dtype)

        # do your customized optimization

        # try evaluate the solution and get a numpy array
        output = pwlfunc(ModuleParams, Abin, zeta)
        objs_ = torch.tensor(objs, device=self.options.device, dtype = self.options.dtype, requires_grad=False)
        loss = torch.sum(output * objs_)
        print(f"Loss: {loss:.7f}")
        primalsols = output.cpu().numpy().flatten()

        # construct the original solution
        ngamma = stdlp.ncsr_cols
        alpha, beta = stdlp.getOriginalCut(primalsols[0:ngamma - 1], primalsols[ngamma - 1])
        cutefficiency = 1.0 * primalsols[len(stdcolsols)]
        for i in range(len(stdcolsols)):
            cutefficiency += stdcolsols[i] * primalsols[i]
        print(f"cut effiency: {cutefficiency} \n")
        return alpha, -beta

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
        candcolid, down_ub, up_lb, int_ids, int_probs = selectCandidate(stdlp.stdcols, stdcolsols)
        result = SCIP_RESULT.DIDNOTFIND

        # get cut's coefficients
        cutcoefs, cutlhs = self.solveFO(stdlp, stdcolsols, candcolid, down_ub, up_lb, int_ids, int_probs)

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

