
import numpy as np
import itertools

######################################################
##  Diagram of a tesseract for reference.           ##
##                                                  ##
##  Left cube is the ana cell of the tesseract.     ##
##  Right cube is the kata cell of the tesseract.   ##
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
VERTEX_A = (0,0,0,0)
VERTEX_B = (1,0,0,0)
VERTEX_C = (0,1,0,0)
VERTEX_D = (1,1,0,0)
VERTEX_E = (0,0,1,0)
VERTEX_F = (1,0,1,0)
VERTEX_G = (0,1,1,0)
VERTEX_H = (1,1,1,0)
VERTEX_I = (0,0,0,1)
VERTEX_J = (1,0,0,1)
VERTEX_K = (0,1,0,1)
VERTEX_L = (1,1,0,1)
VERTEX_M = (0,0,1,1)
VERTEX_N = (1,0,1,1)
VERTEX_O = (0,1,1,1)
VERTEX_P = (1,1,1,1)

EDGE_AB = ( VERTEX_A, VERTEX_B ) 
EDGE_CD = ( VERTEX_C, VERTEX_D )
EDGE_EF = ( VERTEX_E, VERTEX_F )
EDGE_GH = ( VERTEX_G, VERTEX_H )
EDGE_IJ = ( VERTEX_I, VERTEX_J )
EDGE_KL = ( VERTEX_K, VERTEX_L )
EDGE_MN = ( VERTEX_M, VERTEX_N )
EDGE_OP = ( VERTEX_O, VERTEX_P )
EDGE_AC = ( VERTEX_A, VERTEX_C )
EDGE_BD = ( VERTEX_B, VERTEX_D )
EDGE_EG = ( VERTEX_E, VERTEX_G )
EDGE_FH = ( VERTEX_F, VERTEX_H )
EDGE_IK = ( VERTEX_I, VERTEX_K )
EDGE_JL = ( VERTEX_J, VERTEX_L )
EDGE_MO = ( VERTEX_M, VERTEX_O )
EDGE_NP = ( VERTEX_N, VERTEX_P )
EDGE_AE = ( VERTEX_A, VERTEX_E )
EDGE_BF = ( VERTEX_B, VERTEX_F )
EDGE_CG = ( VERTEX_C, VERTEX_G )
EDGE_DH = ( VERTEX_D, VERTEX_H )
EDGE_IM = ( VERTEX_I, VERTEX_M )
EDGE_JN = ( VERTEX_J, VERTEX_N )
EDGE_KO = ( VERTEX_K, VERTEX_O )
EDGE_LP = ( VERTEX_L, VERTEX_P )
EDGE_AI = ( VERTEX_A, VERTEX_I )
EDGE_BJ = ( VERTEX_B, VERTEX_J )
EDGE_CK = ( VERTEX_C, VERTEX_K )
EDGE_DL = ( VERTEX_D, VERTEX_L )
EDGE_EM = ( VERTEX_E, VERTEX_M )
EDGE_FN = ( VERTEX_F, VERTEX_N )
EDGE_GO = ( VERTEX_G, VERTEX_O )
EDGE_HP = ( VERTEX_H, VERTEX_P )

# Edges of the unit tesseract, in an order that I find
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

# Edges of the centered tesseract, in the same order.
ALL_EDGES_ORDERED_CT = (ALL_EDGES_ORDERED_UT * 2) - 1

# Make them immutable/hashable
ALL_EDGES_ORDERED_CT = tuple([frozenset((tuple(e[0]), tuple(e[1])))
        for e in ALL_EDGES_ORDERED_CT])
ALL_EDGES_ORDERED_UT = tuple([frozenset((tuple(e[0]), tuple(e[1])))
        for e in ALL_EDGES_ORDERED_UT])

# Identity matrix
IDENTITY_MATRIX = np.array([[ 1, 0, 0, 0],
                            [ 0, 1, 0, 0],
                            [ 0, 0, 1, 0],
                            [ 0, 0, 0, 1]])

###
# List of matrices describing the symmetries of a tesseract.
#
# This list has 384 items. Each item is a 4x4 matrix describing
# an element of the hyperoctahedral group of order 4. This includes
# the identity transformation plus all rotations and reflections that
# preserve the symmetry of a tesseract.
#
# Multiplying any one of these matrices by some vertex of the centered
# tesseract will always return another vertex of the centered tesseract.
# The same is NOT true for the unit tesseract, because rotations and
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
# Make it immutable/hashable
TRANSFORMATION_MATRICES = frozenset(
       (tuple(mat[0]),
        tuple(mat[1]),
        tuple(mat[2]),
        tuple(mat[3]))
        for mat in TRANSFORMATION_MATRICES)


###
### The items below are unused but left here for reference purposes.
###


# Convert between "unit" tesseract and "centered" tesseract.
# The unit tesseract has coordinates spanning (0,0,0,0) to (1,1,1,1), while
# The centered tesseract has coordinates spanning (-1,-1,-1,-1) to (1,1,1,1).
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


