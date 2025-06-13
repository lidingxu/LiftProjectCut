import numpy as np
import math
from scipy import sparse
from enum import Enum


class RowType(Enum):
    INEQ = 1
    VLB = 2
    VUB = 3

class STDCol:
    # the originl columun at lppos equals to slope * x + shift, and x has a new upper bound ub
    def __init__(self, lppos, isintegral, slope, shift, ub = None):
        self.lppos = lppos
        self.isintegral = isintegral
        self.slope = slope
        self.shift = shift
        self.ub = ub

class StandardLP:
    # sparse data structures of the standard LP
    def __init__(self, csr_data, csr_rows, csr_cols, csr_typerows, csr_origins, csr_rowstarts, stdcols):
        self.csr_data = csr_data
        self.csr_rows = csr_rows
        self.csr_cols = csr_cols
        self.csr_typerows = csr_typerows
        self.csr_origins = csr_origins
        self.csr_rowstarts = csr_rowstarts
        self.stdcols = stdcols
        self.ncsr_rows = len(csr_origins)
        self.ncsr_cols = len(stdcols) + 1
        self.csr = None
        self.stdcol_rowpos = [[None] * len(stdcols), [None] * len(stdcols)]
        self.tranposedcsr = None
        self.tranposedcsr_stdcols = [None] * self.ncsr_rows # map rows to corresponding lb and ub constraint
        start_lb = len(self.csr_typerows[RowType.INEQ])
        start_ub = start_lb + len(self.csr_typerows[RowType.VLB])
        for i in range(start_lb, start_ub):
            assert self.csr_origins[i] < 0
            # col index of lb constraint is negative and shifted by 1
            self.stdcol_rowpos[0][-self.csr_origins[i] - 1] = i - start_lb
            self.tranposedcsr_stdcols[i] = self.csr_origins[i]
        for i in range(start_ub, self.ncsr_rows):
            assert self.csr_origins[i] < 0
            # col index of ub constraint is positive and shifted by 1
            self.stdcol_rowpos[1][-self.csr_origins[i] - 1] = i - start_ub
            self.tranposedcsr_stdcols[i] = -self.csr_origins[i]


    def getCSR(self):
        if self.csr is None:
            self.csr = sparse.csr_matrix((self.csr_data, (self.csr_rows, self.csr_cols)), shape=(self.ncsr_rows, self.ncsr_cols))
        return self.csr

    def getTransposedCSR(self):
        if self.tranposedcsr is None:
            self.tranposedcsr = self.getCSR().transpose().tocsr()
        return self.tranposedcsr

    def getOriginalCut(self, alpha, beta):
        beta_ = beta
        alpha_ = alpha
        assert len(alpha) == len(self.stdcols)
        # Adjust alpha and beta based on the slope and shift of each column
        # For each element in alpha, update alpha_ using the formula:
        # alpha_[i] = alpha[i] / slope[i]
        # Update beta_ using the formula:
        # beta_ = beta_ - (alpha[i] * shift[i] / slope[i])
        for i, a in enumerate(alpha):
            alpha_[i] = a / self.stdcols[i].slope
            beta_ -= a * self.stdcols[i].shift / self.stdcols[i].slope
        return alpha_, beta_


    # transform an LP rowlhs <= rows <= rowrhs, and collbs <= cols <= colubs, into a problem of the form Dx -d = 0, Ax - b >= 0, 0<= x <= ub
def getStandardLP(scip, rows, cols):
    # CSR (Compressed Sparse Row) Matrix data structures for storing the transformed LP
    # _csr_data: Stores the coefficient values for the constraint matrix
    # _csr_rows: Stores the row indices for each nonzero entry
    # _csr_cols: Stores the column indices for each nonzero entry
    # _csr_typerows[rowtype]: Stores indices of row type
    # _csr_origins: Maps transformed rows back to original constraint indice
    _csr_data = []
    _csr_rows = []
    _csr_rowstarts = []
    _csr_cols = []
    _csr_origins = []
    _csr_typerows = {
        RowType.INEQ: [],
        RowType.VLB: [],
        RowType.VUB: []
    }

    currowid = 0
    currowstart = 0

    # Initialize lists to store standard columns and their solutions
    stdcols = []
    stdcolsols = []
    colsols = []
    ncols = len(cols)  # Get the number of columns

    # Iterate over each column to transform it into a standard form
    for i in range(ncols):
        col = cols[i]
        lppos = col.getLPPos()  # Get the LP position of the column

        # Debugging information to ensure the index matches the LP position
        if i != lppos:
            print(f"Debug: Mismatch at index {i}, expected lppos {i}, but got {lppos}")
        assert i == lppos, f"Assertion failed: index {i} does not match lppos {lppos}"

        # Retrieve upper and lower bounds of the column
        ub = col.getUb()
        lb = col.getLb()

        # Check if the bounds are valid (not infinite)
        validlb = not scip.isInfinity(-lb)
        validub = not scip.isInfinity(ub)

        # Ensure that at least one bound is valid
        assert (validlb or validub), f"Assertion failed: column indexed at {lppos} has no bounds"

        # Check if the column is integral
        isintegral = col.isIntegral()


        # Slope translation based on the validity of the lower bound
        if validlb:
            # Append a new standard column with slope 1.0 and shift lb
            stdcols.append(STDCol(lppos, isintegral, 1.0, lb, ub - lb if validub else None))
            # Update CSR data structures for variable lower bound
            _csr_data += [1, -0.0]
            _csr_rows += [currowid, currowid]
            _csr_cols += [lppos, ncols]
            _csr_typerows[RowType.VLB].append(currowid)
            _csr_origins.append(-1 - i)
            _csr_rowstarts.append(currowstart)
            currowstart += 2
            currowid += 1

            # If the upper bound is valid, update CSR data structures for variable upper bound
            if validub:
                _csr_data += [-1, stdcols[-1].ub]
                _csr_rows += [currowid, currowid]
                _csr_cols += [lppos, ncols]
                _csr_typerows[RowType.VUB].append(currowid)
                _csr_origins.append(-1 - i)
                _csr_rowstarts.append(currowstart)
                currowstart += 2
                currowid += 1
        elif validub:
            # Complement upper bound. Append a new standard column with slope -1.0 and shift ub
            stdcols.append(STDCol(lppos, isintegral, -1.0, ub, None))
            # Update CSR data structures for variable lower bound
            _csr_data += [1, -0.0]
            _csr_rows += [currowid, currowid]
            _csr_cols += [lppos, ncols]
            _csr_typerows[RowType.VLB].append(currowid)
            _csr_origins.append(-1 - i)
            _csr_rowstarts.append(currowstart)
            currowstart += 2
            currowid += 1


        # Calculate and store the solution for the standard column
        colsol = col.getPrimsol()
        #print(i, lppos, lb, colsol, ub, isintegral)
        stdcolsols.append((colsol - stdcols[-1].shift) / stdcols[-1].slope)
        colsols.append(colsol)

    # Iterate over each row in the LP rows
    for i in range(len(rows)):
        row = rows[i]

        # Retrieve the constant term of the row and adjust the lhs and rhs by subtracting it
        constant = row.getConstant()
        lhs = row.getLhs() - constant
        rhs = row.getRhs() - constant

        # Get the norm of the row for normalization purposes
        norm = row.getNorm() + 1e-6

        # Get the number of non-zero elements in the row
        nlpnonz = row.getNLPNonz()

        # Retrieve the columns and their corresponding values for the current row
        rowcols = row.getCols()
        vals = row.getVals()

        # Initialize variables to store the sum of shifts and the normalized row values
        sumshift = 0.0
        rowvals = [0.0] * nlpnonz
        rowcolids = [0] * (nlpnonz + 1)

        # Calculate the sum of shifts and normalize the row values
        for j in range(nlpnonz):
            rowcollppos = rowcols[j].getLPPos()
            sumshift += vals[j] * stdcols[rowcollppos].shift
            rowcolids[j] = rowcollppos
            rowvals[j] = vals[j] / norm * stdcols[rowcollppos].slope

        # Add an extra column index for the hand side variable
        rowcolids[nlpnonz] = ncols

        # Subtract the shift from lhs and rhs, then normalize them
        lhs_ = (lhs - sumshift) / norm
        rhs_ = (rhs - sumshift) / norm

        # Check if the lhs and rhs are valid (not infinite)
        validlhs = not scip.isInfinity(-lhs)
        validrhs = not scip.isInfinity(rhs)


        # If the row is an inequality, check and add valid lhs and rhs to the CSR data structures
        if validlhs:
            _csr_data += rowvals + [-lhs_]
            _csr_rows += [currowid] * (nlpnonz + 1)
            _csr_cols += rowcolids
            _csr_typerows[RowType.INEQ].append(currowid)
            _csr_origins.append(i)
            _csr_rowstarts.append(currowstart)
            currowstart += nlpnonz + 1
            currowid += 1
        if validrhs:
            _csr_data += [-val for val in rowvals] + [rhs_]
            _csr_rows += [currowid] * (nlpnonz + 1)
            _csr_cols += rowcolids
            _csr_typerows[RowType.INEQ].append(currowid)
            _csr_origins.append(i)
            _csr_rowstarts.append(currowstart)
            currowstart += nlpnonz + 1
            currowid += 1

    # Append the final row start position to the list
    _csr_rowstarts.append(currowstart)

    # reoragnize rows into blocks, equality rows, inequaliy rows, var lowerbounds, var upper bounds
    # Initialize CSR (Compressed Sparse Row) data structures with zeros
    csr_data = [0.0] * currowstart  # Array to store non-zero values of the matrix
    csr_rows = [0] * currowstart    # Array to store row indices corresponding to each value in csr_data
    csr_cols = [0] * currowstart    # Array to store column indices corresponding to each value in csr_data

    # Initialize CSR type rows for different row types with zeros
    csr_typerows = {
        RowType.INEQ: [0] * len(_csr_typerows[RowType.INEQ]), # Inequality constraints
        RowType.VLB: [0] * len(_csr_typerows[RowType.VLB]),   # Variable lower bounds
        RowType.VUB: [0] * len(_csr_typerows[RowType.VUB])    # Variable upper bounds
    }

    # Initialize arrays to track the origin of each row and the start of each row in csr_data
    csr_origins = [0] * currowid
    csr_rowstarts = [0] * (currowid + 1)

    # Reset current row ID and start position
    currowid = 0
    currowstart = 0

    # Iterate over each row type to reorganize rows into blocks
    for rowtype in RowType:
        # Iterate over each row in the current row type
        for typerowid, row in enumerate(_csr_typerows[rowtype]):
            trowstart = _csr_rowstarts[row]  # Start index of the current row in _csr_data
            trowsend = _csr_rowstarts[row + 1]  # End index of the current row in _csr_data
            nlpnonz = trowsend - trowstart  # Number of non-zero elements in the current row

            # Calculate the next start position for the current row in csr_data
            nextcurrowstart = currowstart + nlpnonz

            # Copy data, row indices, and column indices from temporary CSR to final CSR
            csr_data[currowstart : nextcurrowstart] = _csr_data[trowstart : trowsend]
            csr_rows[currowstart : nextcurrowstart] = [currowid] * nlpnonz
            csr_cols[currowstart : nextcurrowstart] = _csr_cols[trowstart : trowsend]

            # Update the type row mapping with the current row ID
            csr_typerows[rowtype][typerowid] = currowid

            # Record the origin and start position of the current row
            csr_origins[currowid] = _csr_origins[row]
            assert (( rowtype == RowType.INEQ) and _csr_origins[row] >= 0) or ((rowtype == RowType.VLB or rowtype == RowType.VUB) and _csr_origins[row] < 0)
            csr_rowstarts[currowid] = currowstart

            # Update the current row start position and ID for the next iteration
            currowstart = nextcurrowstart
            currowid += 1


    stdlp = StandardLP(csr_data, csr_rows, csr_cols, csr_typerows, csr_origins, csr_rowstarts, stdcols)
    return stdlp, stdcolsols, colsols


def selectCandidate(stdcols, stdcolsols):
    # Initialize the best candidate index and the best fractional part
    bestcandidate = -1
    bestfrac = 0.0

    # Iterate over the standard columns to find the best candidatei
    int_ids = []
    int_probs = []
    sum_probs = 0.0
    for i in range(len(stdcols)):  # Corrected to use range for iteration
        # Check if the current column is integral
        if stdcols[i].isintegral and stdcols[i].ub is not None:
            f = stdcolsols[i]  # Get the solution value for the current column
            frac_part = f - math.floor(f)  # Calculate the fractional part of the solution
            frac = min(frac_part, 1 - frac_part)  # Get the minimum of the fractional part and its complement
            #print(i, frac, bestfrac, bestcandidate)
            # Update the best candidate if the current fraction is better
            int_ids.append(i)
            int_probs.append(frac)
            sum_probs += frac
            if frac > bestfrac:
                bestfrac = frac  # Update the best fractional part
                bestcandidate = i  # Update the best candidate index

    # Calculate the lower and upper bounds based on the best fractional part
    down_ub = math.floor(stdcolsols[bestcandidate])  # Upper bound for down direction
    up_lb = math.ceil(stdcolsols[bestcandidate])    # Lower bound for up direction

    #print(down_ub, up_lb)

    return bestcandidate, down_ub, up_lb, int_ids, int_probs


def candidatePoolBound(stdcols, stdcolsols, stdcol_rowpos, candidates):
    uplbinds = []
    downubinds = []
    downdubs = []
    updlbs = []
    for i in candidates:
        f = stdcolsols[i]
        assert stdcols[i].ub is not None
        assert stdcol_rowpos[i][0] is not None
        assert stdcol_rowpos[i][1] is not None
        downdubs.append(math.floor(f) - stdcols[i].ub)
        downubinds.append(stdcol_rowpos[i][1])
        updlbs.append(math.ceil(f) - stdcols[i].lb)
        uplbinds.append(stdcol_rowpos[i][0])