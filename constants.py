
import numpy as np
import itertools

######################################################
##  Diagram of a tessaract for reference.           ##
##                                                  ##
##  Left cube is the ana cell of the tessaract.     ##
##  Right cube is the kata cell of the tessaract.   ##
##  Eight edges are hidden in the W dimension.      ##
##                                                  ##
##           E---------F          M---------N       ##
##           :\        :\         :\        :\      ##
##    z      : G=========H        : O=========P     ##
##    |      : |       : |        : |       : |     ##
##    *---x  A-|-------B |  ---w  I-|-------J |     ##
##     \      \|        \|         \|        \|     ##
##      y      C=========D          K=========L     ##
##                                                  ##
######################################################


# Vertices
################### [x,y,z,w]
VERTEX_A = np.array([0,0,0,0])
VERTEX_B = np.array([1,0,0,0])
VERTEX_C = np.array([0,1,0,0])
VERTEX_D = np.array([1,1,0,0])
VERTEX_E = np.array([0,0,1,0])
VERTEX_F = np.array([1,0,1,0])
VERTEX_G = np.array([0,1,1,0])
VERTEX_H = np.array([1,1,1,0])
VERTEX_I = np.array([0,0,0,1])
VERTEX_J = np.array([1,0,0,1])
VERTEX_K = np.array([0,1,0,1])
VERTEX_L = np.array([1,1,0,1])
VERTEX_M = np.array([0,0,1,1])
VERTEX_N = np.array([1,0,1,1])
VERTEX_O = np.array([0,1,1,1])
VERTEX_P = np.array([1,1,1,1])

# Edges aligned to the X dimension
EDGE_AB = np.array([VERTEX_A, VERTEX_B]) 
EDGE_CD = np.array([VERTEX_C, VERTEX_D])
EDGE_EF = np.array([VERTEX_E, VERTEX_F])
EDGE_GH = np.array([VERTEX_G, VERTEX_H])
EDGE_IJ = np.array([VERTEX_I, VERTEX_J])
EDGE_KL = np.array([VERTEX_K, VERTEX_L])
EDGE_MN = np.array([VERTEX_M, VERTEX_N])
EDGE_OP = np.array([VERTEX_O, VERTEX_P])
# Edges aligned to the Y dimension
EDGE_AC = np.array([VERTEX_A, VERTEX_C])
EDGE_BD = np.array([VERTEX_B, VERTEX_D])
EDGE_EG = np.array([VERTEX_E, VERTEX_G])
EDGE_FH = np.array([VERTEX_F, VERTEX_H])
EDGE_IK = np.array([VERTEX_I, VERTEX_K])
EDGE_JL = np.array([VERTEX_J, VERTEX_L])
EDGE_MO = np.array([VERTEX_M, VERTEX_O])
EDGE_NP = np.array([VERTEX_N, VERTEX_P])
# Edges aligned to the Z dimension
EDGE_AE = np.array([VERTEX_A, VERTEX_E])
EDGE_BF = np.array([VERTEX_B, VERTEX_F])
EDGE_CG = np.array([VERTEX_C, VERTEX_G])
EDGE_DH = np.array([VERTEX_D, VERTEX_H])
EDGE_IM = np.array([VERTEX_I, VERTEX_M])
EDGE_JN = np.array([VERTEX_J, VERTEX_N])
EDGE_KO = np.array([VERTEX_K, VERTEX_O])
EDGE_LP = np.array([VERTEX_L, VERTEX_P])
# Edges aligned to the W dimension
EDGE_AI = np.array([VERTEX_A, VERTEX_I])
EDGE_BJ = np.array([VERTEX_B, VERTEX_J])
EDGE_CK = np.array([VERTEX_C, VERTEX_K])
EDGE_DL = np.array([VERTEX_D, VERTEX_L])
EDGE_EM = np.array([VERTEX_E, VERTEX_M])
EDGE_FN = np.array([VERTEX_F, VERTEX_N])
EDGE_GO = np.array([VERTEX_G, VERTEX_O])
EDGE_HP = np.array([VERTEX_H, VERTEX_P])

# Edges of the unit tessaract, in an order that I find
# aesthetically pleasing when represented as above.
ALL_EDGES_ORDERED_UT = np.array([
    EDGE_AC, EDGE_CD, EDGE_BD, EDGE_AB,
    EDGE_AE, EDGE_CG, EDGE_DH, EDGE_BF,
    EDGE_EG, EDGE_GH, EDGE_FH, EDGE_EF,
    EDGE_AI, EDGE_CK, EDGE_DL, EDGE_BJ,
    EDGE_EM, EDGE_GO, EDGE_HP, EDGE_FN,
    EDGE_IK, EDGE_KL, EDGE_JL, EDGE_IJ,
    EDGE_IM, EDGE_KO, EDGE_LP, EDGE_JN,
    EDGE_MO, EDGE_OP, EDGE_NP, EDGE_MN,
])

# Edges of the centered tessaract, in the same order.
ALL_EDGES_ORDERED_CT = (ALL_EDGES_ORDERED_UT * 2) - 1


# Identity matrix
IDENTITY_MATRIX = np.array([[ 1, 0, 0, 0],
                            [ 0, 1, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1]])

###
# List of matrices describing the symmetries of a tessaract.
#
# This list has 384 items. Each item is a 4x4 matrix describing
# an element of the hyperoctahedral group of order 4. This includes
# the identity transformation plus all rotations and reflections that
# preserve the symmetry of a tessaract.
#
# Multiplying any one of these matrices by some vertex of the centered
# tessaract will always return another vertex of the centered tessaract.
# The same is NOT true for the unit tessaract, because rotations and
# reflections are centered at the origin.
###
TRANSFORMATION_MATRICES = np.array([
    permutation * inversion
        for permutation in np.array(list(
            itertools.permutations(IDENTITY_MATRIX)))
        for inversion in np.array([
            (a,b,c,d)
                for a in (-1,1)
                for b in (-1,1)
                for c in (-1,1)
                for d in (-1,1) ])])


###
### The items below are unused but left here for reference purposes.
###


# Convert between "unit" tessaract and "centered" tessaract.
# The unit tessaract has coordinates spanning (0,0,0,0) to (1,1,1,1), while
# The centered tessaract has coordinates spanning (-1,-1,-1,-1) to (1,1,1,1).
# Input `a` can be any numpy array.
def ut_to_ct(a):
    return (a * 2) - 1
def ct_to_ut(a):
    return (a + 1) // 2

# Identity matrix
MATRIX_ID = np.array([[ 1, 0, 0, 0],
                      [ 0, 1, 0, 0],
                      [ 0, 0, 1, 0],
                      [ 0, 0, 0, 1]])

# Transform a vector v by a matrix m.
# Input `v` should be a 4-element numpy array.
# Input `m` should be a 4x4-element numpy array.
def transform(v,m):
    return m.dot(v)


