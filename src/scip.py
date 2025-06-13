#   python3 src/scip.py --datadir data --filename app1-2.mps --statdir results --sepa s --device cpu
import argparse
from pyscipopt import Model
from utils.utils import Options
from separators.liftandprojectspx import LiftProjectSimplex
from separators.liftandprojectfo import LiftProjectFO

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run SCIP with lift-and-project separators.")
parser.add_argument("--statdir", type=str, required=True, help="Path to the problem stats.")
parser.add_argument("--sepa", type=str, required=True, help="separator (char \"f\" for fo and \"s\" for spx).")
parser.add_argument("--datadir", type=str, required=True, help="Path to the data directory.")
parser.add_argument("--filename", type=str, required=True, help="Name of the file.")
parser.add_argument("--device", type=str, required=False, help="device.")
args = parser.parse_args()

scip = Model()
scip.setParam("limits/time", 3600)
scip.setParam("limits/nodes", 1)
scip.readProblem(filename=args.datadir + "/" + args.filename)
options = Options(sepamethod=args.sepa, device=args.device)
sepamethod = options.sepamethod
if sepamethod == "s":
    sepa = LiftProjectSimplex()
    scip.includeSepa(sepa, "py_liftprjectspx", "generates lift and project cuts by spx", priority=10000, freq=0)
elif sepamethod == "f":
    sepa = LiftProjectFO(options)
    scip.includeSepa(sepa, "py_liftprjectfo", "generates lift and project cuts by fo", priority=10000, freq=0)
else:
    raise Exception("sepamethod should be either s or f")

scip.optimize()
scip.writeStatistics(filename=args.statdir + "/" + args.filename + "." + args.sepa)







