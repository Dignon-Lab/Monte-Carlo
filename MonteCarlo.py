import os
import copy
import time
import scipy
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from predictor import *                                   # predictor.py must be in the same directory as this script
import joblib
plt.rcParams['pdf.fonttype'] = 42

# scikit-learn gives an improper version warning
warnings.filterwarnings('ignore', category=UserWarning)
# Seq_Comp gives a deprecation warning for the way it is solved, however, it gives the right values
warnings.filterwarnings('ignore', category=DeprecationWarning)

def delta_G(seq,features,residues,nu_file):
    delG_start = time.perf_counter()
    X = X_from_seq(seq,features,residues=residues,charge_termini=CHARGE_TERMINI,nu_file=nu_file)
    ys = models['dG'].predict(X)
    ys_m = np.mean(ys)

    delG_end = time.perf_counter()
    delG_elapsed = delG_end - delG_start
    return ys_m, delG_elapsed #kT

def SCD_calc(Seq):
    SCD_start = time.perf_counter()
    Length = len(Seq)
    ## Calculate charge for each residue
    q_Sequence = np.array([charge[Seq[i]] for i in range(Length)])

    ## From sequences, calculate the sequence charge decoration (SCD)
    SCD = 0
    for i in range(1, Length):
        SCD += ((q_Sequence[i:] * q_Sequence[:-i]) * np.sqrt(i)).sum()
    SCD /= Length

    SCD_end = time.perf_counter()
    SCD_elapsed = SCD_end - SCD_start
    return SCD, SCD_elapsed

def SHD_calc(Seq):
    SHD_start = time.perf_counter()
    Length = len(Seq)
    ## Calculate hydropathy for each residue
    h_Sequence = np.array([hydropathy[Seq[i]] for i in range(Length)])

    ## From sequences, calculate the sequence hydrophobicity decoration (SHD)
    SHD = 0
    for i in range(1, Length):
        SHD += ((h_Sequence[i:] + h_Sequence[:-i]) / i).sum()
    SHD /= Length

    SHD_end = time.perf_counter()
    SHD_elapsed = SHD_end - SHD_start
    return SHD, SHD_elapsed

def SAD_calc(Seq):
    SAD_start = time.perf_counter()
    Length = len(Seq)
    ## Calculate aromaticity for each residue
    a_Sequence = np.array([bAromatic[Seq[i]] for i in range(Length)])

    ## From sequences, calculate the sequence aromatic decoration (SAD)
    SAD = 0
    for i in range(1, Length):
        SAD += ((a_Sequence[i:] * a_Sequence[:-i]) / i).sum()
    SAD /= Length

    SAD_end = time.perf_counter()
    SAD_elapsed = SAD_end - SAD_start
    return SAD, SAD_elapsed

# Function that takes a sequence and returns all desired parameters
def param_calc(Seq):
    global SCD_time, SHD_time, SAD_time, delG_time
    Length = len(Seq)

    # Define SCD, SHD, SAD, delta(G), and composition RMSD of the sequence
    if InputSCD == 'None':
        SCD = 'None'
    else:
        [SCD, SCD_upd] = SCD_calc(Seq)
        SCD_time += SCD_upd

    if InputSHD == 'None':
        SHD = 'None'
    else:
        [SHD, SHD_upd] = SHD_calc(Seq)
        SHD_time += SHD_upd

    if InputSAD == 'None':
        SAD = 'None'
    else:
        [SAD, SAD_upd] = SAD_calc(Seq)
        SAD_time += SAD_upd

    if InputdelG == 'None':
        delG = 'None'
    else:
        [delG, delG_upd] = delta_G(Seq,features,residues,nu_file)
        delG_time += delG_upd

    # Define Composition RMSD
    if InputComp == 'None':
        Comp_RMSD = 'None'
    else:
        # Calculate fraction of each amino in the sequence tested
        Seq_Comp = {amino: ((len(np.where(np.array(Seq) == amino)[0].tolist())) / (Length)) for amino in Amino_1}
        Comp_RMSD = 0
        for amino_name, frac_amino in Composition.items():
            # Uses common dictionary keys (Amino 1-Letter Abbrev.) to compare the fraction of aminos in ideal vs input sequence
            Comp_RMSD += (Seq_Comp[amino_name] - frac_amino)**2

        Comp_RMSD = (np.sqrt(Comp_RMSD))/len(Amino_1)

    params = {'SCD': SCD,
              'SHD': SHD,
              'SAD': SAD,
              'delG': delG,
              'Comp': Comp_RMSD}
    return params

# Calculate the "energy" of a single sequence by first calling the "params" function to determine parameters
def energy_func(Seq):
    # Params is array with following format: [SCD, SHD, SAD, delG]
    params = param_calc(Seq)
    energy = 0
    for key, (goal, weight) in Goals.items():
        if params[key] != 'None':
            energy += weight * (abs(params[key] - goal))

    # create temporary array that stores fraction of energy
    fract_energy = {}
    tot_energy = {}
    for key, (goal, weight) in Goals.items():
        if params[key] != 'None':
            fract_energy[key] = (weight * (abs(params[key] - goal))) / energy
            tot_energy[key] = (weight * (abs(params[key] - goal)))

    return energy, fract_energy, tot_energy

# Calculate Metropolis criterion by taking two sequences and using the "energy_func" function to calculate their energy
def Metropolis(Seq0, SeqMut):
    k_B = 1
    T = 10**(-2)
    energy0 = energy_func(Seq0)[0]
    energyMut = energy_func(SeqMut)[0]
    deltaE = energyMut - energy0
    Metrop = np.exp(-deltaE/(k_B*T))
    return Metrop

####################  Type of alterations  ####################
def seq_mut(Seq):
    # determines place in chain where mutation will occur
    idx = np.random.randint(0, len(Seq))  # integer that will be the index for mutated residue in sequence
    Seq_New = copy.deepcopy(Seq[:])  # Create new sequence to not affect original input sequence
    # create new Amino list that doesn't include residue currently there
    Amino_1_New = [res for res in Amino_1 if res != Seq[idx]]
    Seq_New[idx] = np.random.choice(Amino_1_New)  # assign random residue to new sequence in determined spot
    return Seq_New

def seq_swap(Seq):
    idx1 = np.random.randint(0, len(Seq))  # integer that will be the index for the first swapped residue in the sequence
    Amino_idx1 = Seq[idx1]
    idx2 = np.random.randint(0, len(Seq))  # integer that will be the index for the second swapped residue in the sequence
    Amino_idx2 = Seq[idx2]
    Seq_New = copy.deepcopy(Seq[:])  # Create new sequence to not affect original input sequence
    # Swap residues at determined locations
    Seq_New[idx1] = Amino_idx2
    Seq_New[idx2] = Amino_idx1
    return Seq_New

# Function that will shuffle a variable length of the sequence ranging from (3, N)
def seq_shuf(Seq):
    # Input sequence MUST be a numpy array containing a list. This is because the sequences will be indexed with another list and
    # this functionality is only possible with a numpy array
    copyShuf = copy.deepcopy(Seq[:])
    lenseq = len(Seq)
    shuf_size = np.random.randint(3, lenseq)
    shuf_start = np.random.randint(0, lenseq)
    shuf_indices = np.linspace(shuf_start, shuf_start+shuf_size-1, shuf_size)
    indices = []
    # conditional that makes indices periodic by catching out of bounds indices
    for ind in shuf_indices:
        if ind > (lenseq-1):
            indices.append(int(ind)-int(lenseq))
        else:
            indices.append(int(ind))
    Seq_New = copy.deepcopy(copyShuf[:])
    Seq_New[indices] = np.random.permutation(copyShuf[indices])
    return Seq_New
###############################################################


#################### Choosing Alteration ######################
def alteration(mutAmt, swpAmt, shfAmt, Seq):
    lowbound = 0
    upbound = mutAmt + swpAmt + shfAmt
    coin_flip = np.random.randint(lowbound, upbound)
    if (coin_flip >= lowbound) & (coin_flip < (lowbound + mutAmt)):
        New_Seq = seq_mut(Seq)
        seqIdentifier = 0  # variable used to identify what moveset will be applied
    elif (coin_flip >= (lowbound + mutAmt)) & (coin_flip < (lowbound + mutAmt + swpAmt)):
        New_Seq = seq_swap(Seq)
        seqIdentifier = 1
    elif (coin_flip >= (lowbound + mutAmt + swpAmt)) & (coin_flip < (lowbound + mutAmt + swpAmt + shfAmt)):
        New_Seq = seq_shuf(Seq)
        seqIdentifier = 2
    return New_Seq, seqIdentifier
###############################################################

##################### Param Weights Calc #######################
def SCD_WeightFunc(Seq, Length):
    q_Sequence = np.array([charge[Seq[i]] for i in range(Length)])
    q_Net = q_Sequence.sum()
    NCPR = q_Net / Length
    fplus = (q_Sequence > 0).mean()
    fminus = (q_Sequence < 0).mean()
    FCR = fplus + fminus
    # Fitted functions for estimating the mean and standard deviation SCD for the sequences fixed-composition ensemble
    est_meanSCD = (0.25425) * (Length**(1.50780)) * (NCPR**2)
    est_stdSCD = (0.05306 * (Length**(1.12164))) * FCR * ((NCPR**2) - (1.36401*(NCPR**4)) + (Length**(-0.39905)) )
    WtSCD = 1/est_stdSCD
    return WtSCD

def SHD_WeightFunc(Seq, Length):
    h_Sequence_orig = np.array([hydropathy[Seq[i]] for i in range(Length)])
    mean_hyd = np.mean(h_Sequence_orig)
    std_hyd = np.std(h_Sequence_orig)
    # Fitted functions for estimating the mean and standard deviation SHD for the sequences fixed-composition ensemble
    est_meanSHD = (2.7610) * (mean_hyd) * (Length**(0.2354))
    est_stdSHD = (0.65 * (Length ** (-0.45922))) * std_hyd
    WtSHD = 1/est_stdSHD
    return WtSHD

def SAD_WeightFunc(Seq, Length):
    a_Sequence = np.array([bAromatic[Seq[i]] for i in range(Length)])
    FAR = (a_Sequence > 0).mean()
    std_arom = np.std(a_Sequence)
    # Fitted functions for estimating the mean and standard deviation SAD for the sequences fixed-composition ensemble
    est_meanSAD = (1.6188) * (FAR**(2.0390)) * (Length**(0.2110))
    est_stdSAD = (1.4177) * (std_arom**(2.4598)) * (Length**(-0.4404))
    WtSAD = 1/est_stdSAD
    return WtSAD
################################################################

# Time counters for analyzing optimization of functions
SCD_time, SHD_time, SAD_time, delG_time = 0, 0, 0, 0

## Import and define relevant data/constants
Amino_1 = ['A','R','N','D','C','Q','E','G','H','I','L','K','M',
           'F','P','S','T','W','Y','V']

#########################  User Inputs  ###########################

# Insert reference sequence or "None" if looking to generate a new sequence based on parameters alone
Ideal_Seq = list('MESNQSNNGGSGNAALNRGGRYVPPHLRGGDGGAAAAASAGGDDRRGGAGGGGYRRGGGNSGGGGGGGYDRGYNDNRDDRDNRG'
                 'GSGGYGRDRNYEDRGYNGGGGGGGNRGYNNNRGGGGGGYNRQDRGDGGSSNFSRGGYNNRDEGSDNRGSGRSYNNDRRDNGGDG')  #  LAF-1

## 3 length variables in this code. Inp_SeqLength is the length of the input sequence that a user enters and wants their
## sequence to be compositionally similar to. DesLength is the optional length they've input for how long they want their
## final sequence to be. Length is the local variable used in parameter calculation functions and is the length of
## whatever sequence is passed into the function. In this way, this length will be relative to whatever sequence is
## put into the function
Inp_SeqLength = len(Ideal_Seq)

# Insert desired length of generated protein sequence
DesLength = 'None'

if DesLength == 'None':
    DesLength = Inp_SeqLength

# Define Comp_Goal and composition of sequence depending on whether there is a reference sequence
if Ideal_Seq == 'None':
    Composition = 'None'
    Comp_Goal = 'None'
else:
    # Create a dictionary composition by finding the fraction of each amino acid in the reference sequence (n_amino/n_total)
    Composition = {amino: (len(np.where(np.array(Ideal_Seq) == amino)[0].tolist()))/(Inp_SeqLength) for amino in Amino_1}
    Comp_Goal = 0

# Desired values of parameters
SCD_Goal, SHD_Goal, SAD_Goal, delG_Goal = -5, 5.327, 0.031, -8.5
# -8.5 delG
Goal_Array = [SCD_Goal, SHD_Goal, SAD_Goal, delG_Goal, Comp_Goal]

# Desired ratio of weights of importance per parameter
InputSCD, InputSHD, InputSAD, InputdelG, InputComp = 1, 1, 1, 10, 1
InputWeights = [InputSCD, InputSHD, InputSAD, InputdelG, InputComp]
InputWeightDictionary = {'SCD': InputSCD, 'SHD': InputSHD, 'SAD': InputSAD, 'delG': InputdelG, 'Comp': InputComp}

# Desired ratio of mutations to swaps to shuffles
mutRatio = 1
swpRatio = 1
shfRatio = 1

# Maximum amount of Monte Carlo moves
desired_cycles = int(10000)

# Residue information
hydropathy = np.loadtxt('AA_properties/Urry.dat', dtype=object)
hydropathy[:, 1] = hydropathy[:, 1].astype(float)
hydropathy = dict(hydropathy)
charge = np.loadtxt('AA_properties/charges.dat', dtype=object)
charge[:, 1] = charge[:, 1].astype(float)
charge = dict(charge)
mass = np.loadtxt('AA_properties/masses.dat', dtype=object)
mass[:, 1] = mass[:, 1].astype(float)
mass = dict(mass)
bAromatic = np.loadtxt('AA_properties/aromaticity.dat', dtype=object)
bAromatic[:, 1] = bAromatic[:, 1].astype(int)
bAromatic = dict(bAromatic)


####################################### Delta G Calculation #######################################
residues = pd.read_csv('residues.csv').set_index('one')   # residue.csv must be in the same directory as this script
nu_file = 'svr_model_nu.joblib'                           # svr_model_nu.joblib must be in the same directory as this script
features = ['mean_lambda', 'faro', 'shd', 'ncpr', 'fcr', 'scd', 'ah_ij','nu_svr']
models = {}
models['dG'] = joblib.load(f'model_dG.joblib')          # model_dG.joblib must be in the same directory as this script
SEQUENCE = Ideal_Seq.copy()
SEQUENCE = str(SEQUENCE)
SEQUENCE = SEQUENCE.replace('[','').replace(']','').replace(',','').replace('\'','').replace(' ','')
CHARGE_TERMINI = True # @param {type:'boolean'}
seq = SEQUENCE
if " " in seq:
    seq = ''.join(seq.split())
    print('Blank character(s) found in the provided sequence. Sequence has been corrected, but check for integrity.')



######################  Include recommended weights based on calculations  ######################
# Calculated Weights
SCD_Weight, SHD_Weight, SAD_Weight, delG_Weight, Comp_Weight = (SCD_WeightFunc(Ideal_Seq, Inp_SeqLength), SHD_WeightFunc(Ideal_Seq, Inp_SeqLength),
                                                                SAD_WeightFunc(Ideal_Seq, Inp_SeqLength), 1/10, 1/np.sqrt(2))

CalcWeights = [SCD_Weight, SHD_Weight, SAD_Weight, delG_Weight, Comp_Weight]
##########################################################################################################

# Initialize final weights
Weights = [0, 0, 0, 0, 0]
# Use input weight ratios to balance the calculated and input weights & define Goals + Intervals
for count, inp in enumerate(InputWeights):
    if inp == 'None':
        Weights[count] = 0
    else:
        Weights[count] = inp*CalcWeights[count]

# Set confidence intervals
Intervals = [0, 0, 0, 0, 0]
for count, goal in enumerate(Goal_Array):
    if goal == 0:
        Intervals[count] = 0.0005
    elif goal == "None":
        Intervals[count] = "None"
    else:
        Intervals[count] = abs(goal*0.0005)

Intervals_Dict = {
    'SCD': Intervals[0],
    'SHD': Intervals[1],
    'SAD': Intervals[2],
    'delG': Intervals[3],
    'Comp': Intervals[4]}
# Save keys where Interval != None
saved_Keys = []
for key, inte in InputWeightDictionary.items():
    if inte != "None":
        saved_Keys.append(key)

# Place parameter goals and their associated weights in a dictionary format
Goals = {
    'SCD': [SCD_Goal, Weights[0]],
    'SHD': [SHD_Goal, Weights[1]],
    'SAD': [SAD_Goal, Weights[2]],
    'delG': [delG_Goal, Weights[3]],
    'Comp': [Comp_Goal, Weights[4]]}

# Set counters for number of specific types of alterations and their respective accepted changes.
mutCount, mutAccepted, swpCount, swpAccepted, shfCount, shfAccepted, mnCount = 0, 0, 0, 0, 0, 0, 0
acceptedIntervals = []

# Determine starting sequence by minimizing the energy function out of 100 randomly generated sequences
# Initialize array for saving pairs of sequences and their respective "energies"
# ** If user wants no mutations, skip this step
if (mutRatio == 0):
    Seq = Ideal_Seq
else:
    MinEnergySeq = np.empty(shape=[100, 2], dtype=object)
    for i in range(100):
        SeqRand = np.random.choice(Amino_1, DesLength)
        energyRand = energy_func(SeqRand)
        MinEnergySeq[i] = ["".join(SeqRand), energyRand]
        MinEnergySeq[i] = ["".join(SeqRand), energyRand]
    # Sequence with the minimum energy
    Seq = np.array(list(MinEnergySeq[np.argmin(MinEnergySeq[:, 1], axis=0), 0]))

# Set starting sequence's parameters and initialize the iterations variable and output array for tracking param movement
params = param_calc(Seq)
Seq_Comp = {amino: ((len(np.where(np.array(Seq) == amino)[0].tolist())) / (DesLength)) for amino in Amino_1}
SCD, SHD, SAD, delG, Comp = params['SCD'], params['SHD'], params['SAD'], params['delG'], params['Comp']

# Will be used to store sequence and all other desired parameters
Movement = np.empty(shape=[(desired_cycles)+1, 6], dtype=object)
acceptedFrac = np.empty(shape=[(desired_cycles)+1], dtype=object)
iterations = 0

# Store fraction of energy
energyFrac = np.empty(shape=[(desired_cycles)+1, 5], dtype=float)
energyTot = np.empty(shape=[(desired_cycles)+1, 5], dtype=float)

# store change in contributions to energy function
energyDiff = np.empty(shape=[desired_cycles, 5], dtype=float)
meanDiff = np.empty(shape=[(desired_cycles//10), 5], dtype=float)

def check_conditions():
    # Determine form of conditionals
    if len(saved_Keys) == 1:
        idx1 = saved_Keys[0]
        cond = (abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]
    elif len(saved_Keys) == 2:
        idx1, idx2 = saved_Keys[0], saved_Keys[1]
        cond = (((abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]) or
                ((abs(params[idx2] - Goals[idx2][0])) > Intervals_Dict[idx2]))
    elif len(saved_Keys) == 3:
        idx1, idx2, idx3 = saved_Keys[0], saved_Keys[1], saved_Keys[2]
        cond = ((((abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]) or
                ((abs(params[idx2] - Goals[idx2][0])) > Intervals_Dict[idx2])) or
                ((abs(params[idx3] - Goals[idx3][0])) > Intervals_Dict[idx3]))
    elif len(saved_Keys) == 4:
        idx1, idx2, idx3, idx4 = saved_Keys[0], saved_Keys[1], saved_Keys[2], saved_Keys[3]
        cond = (((((abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]) or
                 ((abs(params[idx2] - Goals[idx2][0])) > Intervals_Dict[idx2])) or
                ((abs(params[idx3] - Goals[idx3][0])) > Intervals_Dict[idx3])) or
                ((abs(params[idx4] - Goals[idx4][0])) > Intervals_Dict[idx4]))
    elif len(saved_Keys) == 5:
        idx1, idx2, idx3, idx4, idx5 = saved_Keys[0], saved_Keys[1], saved_Keys[2], saved_Keys[3], saved_Keys[4]
        cond = ((((((abs(params[idx1] - Goals[idx1][0])) > Intervals_Dict[idx1]) or
                  ((abs(params[idx2] - Goals[idx2][0])) > Intervals_Dict[idx2])) or
                 ((abs(params[idx3] - Goals[idx3][0])) > Intervals_Dict[idx3])) or
                ((abs(params[idx4] - Goals[idx4][0])) > Intervals_Dict[idx4])) or
                ((abs(params[idx5] - Goals[idx5][0])) > Intervals_Dict[idx5]))
    return cond

# Set combination of conditionals based on what parameters are fed into the script
print(f"Original Sequence: {''.join(Seq)}\n")
# Loop that runs until ALL parameters are within their proper interval OR iterations reaches the desired # of cycles
while (check_conditions() & (iterations <= desired_cycles)):

    # Update iteration for non-divisibles of 10
    if desired_cycles % 10 != 0:
        if desired_cycles <= 100:
            if iterations % 10 == 0:
                print(f"Current iteration: {iterations}\n")
        elif desired_cycles <= 500:
            if iterations % 25 == 0:
                print(f"Current iteration: {iterations}\n")
        else:
            if iterations % 100 == 0:
                print(f"Current iteration: {iterations}\n")
    else:
        # Update what iteration you are on every 1/10th of the maximum amount of cycles
        if iterations % (desired_cycles / 10) == 0:
            print(f"Current iteration: {iterations}\n")

    Movement[iterations] = ["".join(Seq), SCD, SHD, SAD, delG, Comp]
    acceptedFrac[iterations] = (mutAccepted + swpAccepted + shfAccepted) / (iterations+1)

    # Perform sequence alteration
    Seq_New, chngIdent = alteration(mutRatio, swpRatio, shfRatio, Seq)

    acceptedBool = False
    # Determine Metropolis criterion of this sequence change
    Metrop = Metropolis(Seq, Seq_New)
    if Metrop >= 1:
        # Accept the change by making the "Seq" variable identical to "Seq_New"
        Seq = copy.deepcopy(Seq_New)
        # Since alteration is accepted, we must recalculate new parameters to update variables in loops conditionals.
        params = param_calc(Seq)
        SCD, SHD, SAD, delG, Comp = params['SCD'], params['SHD'], params['SAD'], params['delG'], params['Comp']
        acceptedBool = True
    elif Metrop < 1:
        # generate another random number between 0 and 1 to determine if sequence mutation/swap is accepted
        coin_flip2 = np.random.uniform(0, 1)
        # If this number is less than or equal to the Metropolis (less than one), accept the sequence alteration
        if coin_flip2 <= Metrop:
            Seq = copy.deepcopy(Seq_New)
            # Since alteration is accepted, we must recalculate new parameters to update variables in loops conditionals.
            params = param_calc(Seq)
            SCD, SHD, SAD, delG, Comp = params['SCD'], params['SHD'], params['SAD'], params['delG'], params['Comp']
            acceptedBool = True
    if (chngIdent == 0):
        mutCount += 1
        if (acceptedBool):
            mutAccepted += 1
    elif (chngIdent == 1):
        swpCount += 1
        if (acceptedBool):
            swpAccepted += 1
    elif (chngIdent == 2):
        shfCount += 1
        if (acceptedBool):
            shfAccepted += 1

    if (acceptedBool):
        acceptedIntervals.append(iterations)

    [fracEn, totEn] = [energy_func(Seq)[1], energy_func(Seq)[2]]
    # totEn = energy_func(Seq_New)[2]
    energyFrac[iterations] = [fracEn["SCD"], fracEn["SHD"], fracEn["SAD"], fracEn["delG"], fracEn["Comp"]]
    energyTot[iterations] = [totEn["SCD"], totEn["SHD"], totEn["SAD"], totEn["delG"], totEn["Comp"]]

    # Count iterations for total number of Monte Carlo moves in script
    iterations += 1

print(f"Final, optimized sequence: {''.join(Seq)}\n")
# Output array to save final sequence and resultant parameters: [Seq, SCD, SHD, SAD, delG, Composition RMSD]
output = ["".join(Seq), params['SCD'], params['SHD'], params['SAD'], params['delG'], params['Comp']]

# Create directory if it doesn't exist
if os.path.exists('Run_Data/Data') == False:
    os.makedirs('Run_Data/Data')

file_save1 = 'Coordinates'
new_save1 = copy.deepcopy(file_save1)
count = 0
while os.path.exists(new_save1):
    count += 1
    new_save1 = f"Run_Data/Data/{file_save1}{count}"
file_save1 = copy.deepcopy(new_save1)
np.savetxt(file_save1, output, fmt='%s', delimiter=',')

# Save all trajectory information
suffix = count if count else ''
file_save = 'MonteCarloData'
new_save = f"Run_Data/Data/{file_save}{suffix}"
file_save = copy.deepcopy(new_save)
np.savetxt(file_save, Movement, fmt='%s', delimiter=',')

# Create directory if it doesn't exist
if os.path.exists('Run_Data/Data/Energy') == False:
    os.makedirs('Run_Data/Data/Energy')
np.savetxt(f"Run_Data/Data/Energy/EnergyFrac{suffix}_Shufs", energyFrac, fmt='%s', delimiter=',')
np.savetxt(f"Run_Data/Data/Energy/EnergyTot{suffix}_Shufs", energyTot, fmt='%s', delimiter=',')


print(f"The number of iterations: {iterations}\n\n")
print(f"Output Array [Seq, SCD, SHD, SAD, delG, Composition RMSD]:\n {output[1:6]}")

for i in range(1, 6):
    if Movement[0, i] != "None":
        Movement[:, i] = Movement[:, i].astype(float)

total_Moves = mutAccepted + swpAccepted + shfAccepted

Steps = np.linspace(0, len(Movement[:, 1]), len(Movement[:, 1]))
print(f"Movements: {len(Movement[:, 1])}\n")
print(f"MutAtt: {mutCount}\n MutAcc: {mutAccepted}\n")
print(f"SwpAtt: {swpCount}\n SwpAcc: {swpAccepted}\n")
print(f"ShfAtt: {shfCount}\n ShfAcc: {shfAccepted}\n")
print(f"Moves Attempted: {iterations}\nMoves Accepted: {total_Moves}")

# Create directory if it doesn't exist
if os.path.exists('Run_Data/Plots') == False:
    os.makedirs('Run_Data/Plots')

######################## *Start* Plotting Movement of Parameters ########################
if Movement[0, 1] != "None":
    plt.figure(1)
    plt.plot(Steps, Movement[:, 1], color='k')
    plt.title(f"SCD")
    plt.xlabel('Iterations')
    plt.savefig(f'Run_Data/Plots/SCD{suffix}.pdf')
    plt.show()

if Movement[0, 2] != "None":
    plt.figure(2)
    plt.plot(Steps, Movement[:, 2], color='k')
    plt.title(f"SHD")
    plt.xlabel('Iterations')
    plt.savefig(f'Run_Data/Plots/SHD{suffix}.pdf')
    plt.show()

if Movement[0, 3] != "None":
    plt.figure(3)
    plt.plot(Steps, Movement[:, 3], color='k')
    plt.title(f"SAD")
    plt.xlabel('Iterations')
    plt.savefig(f'Run_Data/Plots/SAD{suffix}.pdf')
    plt.show()

if Movement[0, 4] != "None":
    plt.figure(4)
    plt.plot(Steps, Movement[:, 4], color='k')
    plt.title(f"deltaG")
    plt.xlabel('Iterations')
    plt.savefig(f'Run_Data/Plots/deltaG{suffix}.pdf')
    plt.show()

if Movement[0, 5] != "None":
    plt.figure(5)
    plt.plot(Steps, Movement[:, 5], color='k')
    plt.title(f"Composition RMSD")
    plt.xlabel('Iterations')
    plt.savefig(f'Run_Data/Plots/CompRMSD{suffix}.pdf')
    plt.show()
######################## *End* Plotting Movement of Parameters ########################