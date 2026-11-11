## 1. Variables

x_j is the beam strength of an individual beamlet
x_j must be greater than or equal to 0 because the beamlet is either on and producing a beam (greater than 0) or off and not producing a beam (zero)

## 2. Dose Rule

A_ij is the efficiency of beamlet j for voxel i, how much a beamlet delivers to voxel i per unit of strength. A_ij is bigger when the beamlet and voxel intersect close to where the beam enters, smaller when the voxel is deeper, and near zero when they don't intersect
d_i is the total dose at voxel i - the sum of all beamlets' effeciencies multiplied by their strength 

## 3. Objective

1. The objective is to minimize the average dosage to the healthy tissue and organ 

2. Organ voxels count 3 times as much because the organ is healthy tissue that must be protected the most. A dose that lands in the organ is 3 times as harmful as the same dose in healthy tissue

## 4. Constraints

1. Every tumor voxel must receive a dose of at least 60 Gy because using average could hide untreated voxels that fall below the 60 Gy threshold

2. Every organ voxel must receive a dose of no more than 30 Gy because this is the safety thresshold and an average dosage could sheild the errors we are looking for.

3. Every healthy voxel must receive a dose of no more than 65 Gy because using an average could hide an area getting too much, without this rule the model put 88.4 Gy into a healthy voxel while the average appeared to be stable. The cap of 65 Gy is just under the baseline plan's worst healthy voxel (65.1 Gy), so that the optimized plan doesn't create a worse hot spot and still remains feasible. 

## 5. Assumptions

1. The body is a 2D slice
2. Doses add linearly and beamlets don't interact: double a beamlet's strength and you double it dose everywhere
3. Doses fade as depth increases in a fixed manner
4. Doses are in Gray (Gy), the real unit of radiation dose — the prescription is 60 Gy, a typical curative dose. The geometry and physics are still simplified
5. There's exactly one tumor and one organ
