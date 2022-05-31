from pyqubo import Binary, AndConst, OrConst
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.embedding import embed_qubo
import neal



MYTOKEN = "DEV-5d38d7ec19c2b5bd630867985d4849f014164e48"
MYSOLVER = "Advantage_system6.1"

SOLVE_WITH_DWAVE = False

#Solving with Simulated Annealing
def simulateda_solution(model):
    sampler = neal.SimulatedAnnealingSampler()
    bqm = model.to_bqm()
    sampleset = sampler.sample(bqm, num_reads=10)
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample = min(decoded_samples, key=lambda x: x.energy)
    return best_sample

def dwave_sol(model):
    qubo, qubo_offset = model.to_qubo()
    sampler_kwargs = {"num_reads": 100, "annealing_time": 20, "num_spin_reversal_transforms": 4, "auto_scale": True, "chain_strength": 2.0, "chain_break_fraction": True}
    dw_sampler = DWaveSampler(endpoint="https://cloud.dwavesys.com/sapi", token=MYTOKEN, solver=MYSOLVER)
    sampler = EmbeddingComposite(dw_sampler)
    sampleset = sampler.sample_qubo(qubo)
    decoded_samples = model.decode_sampleset(sampleset)
    best_sample = min(decoded_samples, key=lambda x: x.energy)
    return best_sample

n=8
variables=["V"+str(i+1) for i in range(n)]
H=0
x={v: Binary(v) for v in variables}
y=Binary("Y")
target=y
for i in range(n):
    H+=OrConstr()

model = H.compile()

best_sample = None
if SOLVE_WITH_DWAVE:
    best_sample = dwave_sol(model)
else:  
    best_sample = simulateda_solution(model)
