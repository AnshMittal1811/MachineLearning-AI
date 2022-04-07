import json
from .utils import ( load_sigmoid )
import torch.nn.functional as F

# Load sets the modules param file
def load(args):
  global current
  if args.param_file is None or args.param_file == "": return
  params = json.load(open(args.param_file))
  current = HyperParameters(empty=False, name=args.param_file,params=params)

supported_kinds = {
  "softplus": F.softplus,
  "leaky_relu": F.leaky_relu,
  "relu": F.relu,
  # TODO add other activation kinds here
}

def load_act(kind):
  if kind in supported_kinds: return supported_kinds[kind]
  else: return load_sigmoid(kind)

# HyperParameters is a convenience object for assigning
# hyper parameters, from a file, warning if there is a parameter which wasn't assigned.
class HyperParameters():
  def __init__(
    self,
    name: str = "",
    empty: bool = True,
    params: dict = {}
  ):
    self.name = name
    self.empty = empty
    self.params = params
  def get(self, key, kind, default):
    val = default
    if not self.empty:
      curr = self.params;
      for arg in key.split(":"):
        if hasattr(curr, arg): curr = curr[arg]
        else:
          print(f"[warning]: missing {arg} from {self.name} in path {key} of kind {kind}.")
          curr = default
          break
      val = curr
    ...
    if kind == "act":
      val = load_act(val)
    elif kind == "uint":
     val = int(val)
     assert(val > 0)
    elif kind == "float": val = float(val)
    return val







# initialize global config to empty
current = HyperParameters()
