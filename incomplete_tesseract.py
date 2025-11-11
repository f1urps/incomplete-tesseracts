
import numpy as np
import constants
import functools
import itertools
from dataclasses import dataclass

### CONSTANTS ###

# Minimum value of a packed representation.
MIN = 0x00000000

# Maximum value of a packed representation.
MAX = 0xffffffff

###
# A list of all 32 edges in a full tesseract, arranged in
# preferential order.
#
# This is used as the canonical definition for which bits of the
# packed representation map to which edges of the unit tesseract.
#
# UT refers to the "unit" tesseract, which is the tesseract
# that spans (0,0,0,0) to (1,1,1,1).
#
# CT refers to the "centered" tesseract, which is the tesseract
# that spans (-1,-1,-1,-1) to (1,1,1,1).
###
ALL_EDGES_ORDERED_UT = constants.ALL_EDGES_ORDERED_UT
ALL_EDGES_ORDERED_CT = constants.ALL_EDGES_ORDERED_CT

###
# List of all 384 symmetrical transformations of a tesseract,
# represented as 4x4 matrices.
###
TRANSFORMATION_MATRICES = constants.TRANSFORMATION_MATRICES

@dataclass(order=True, frozen=True)
class IncompleteTesseract:

    # Packed bit representation.
    # Every 32-bit integer represents an IncompleteTesseract,
    # because a tesseract has 32 edges.
    packed: int
    
    
    ### Instance Methods ###

    ###
    # Get all the edges that compose this IncompleteTesseract.
    #
    # The returned set will be a subset of ALL_EDGES_ORDERED_CT,
    # where the nth edge is included iff the nth least significant
    # bit of the packed representation is a 1.
    #
    # Each edge will be returned in immutable form, as a frozenset containing two 4-tuples.
    ###
    @functools.cache
    def edges(self, unit=False):
        all_edges = (ALL_EDGES_ORDERED_UT if unit
                else ALL_EDGES_ORDERED_CT)
        return frozenset([self._make_unordered_edge(all_edges[i])
            for i in range(0,32)
            if (self.packed >> i) & 0x1])

    ###
    # Get a list of all symmetrical transformations of this IncompleteTesseract.
    # The returned list will contain exactly 384 new IncompleteTesseract objects,
    # and includes the identity transformation.
    ###
    @functools.cache
    def transformations(self):
        return [self.transform(transformation_matrix)
            for transformation_matrix in TRANSFORMATION_MATRICES]

    ###
    # Get another IncompleteTesseract that is identical to this one
    # transformed by the given matrix.
    ###
    def transform(self, transformation_matrix):
        return IncompleteTesseract(self._pack_edges((
            self._transform_edge(edge, transformation_matrix)
                for edge in self.edges(unit=False)), unit=False))

    ###
    # Return true iff the given edge is included in this IncompleteTesseract.
    #
    # The given edge argument can be any iterable containing exactly two four-length iterables.
    ###
    def has_edge(self, edge, unit=False):
        return (self._make_unordered_edge(edge) in self.edges(unit=unit))

    ###
    # Get another IncompleteTesseract that is identical to this one
    # but with the given edge added. If the given edge is already included,
    # the new object will just be identical.
    ###
    def with_edge(self, edge, unit=False):
        return IncompleteTesseract(self._pack_edges(
            self.edges(unit=unit).union({self._make_unordered_edge(edge)}),
            unit=unit))

    ###
    # Get a graph representation of this IncompleteTesseract.
    # Returns a tuple (v, e) where v is a frozenset of vertices,
    # each vertex being a 4-tuple of coordinates, and e is a frozenset
    # of edges, each edge itself being a frozenset of two vertices.
    ###
    @functools.cache
    def graph(self, unit=False):
        e = self.edges(unit=unit)
        v = frozenset(itertools.chain.from_iterable(e))
        return v, e

    ###
    # str method
    ###
    def __str__(self):
        return self.ascii_drawing()
    
    ###
    # Return a string which, when printed, renders a diagram of
    # this IncompleteTesseract as an ASCII drawing.
    ###
    @functools.cache
    def ascii_drawing(self,
            indent_spaces           = 5,
            gap_width_spaces        = 6,
            horizontal_edge_length  = 10,
            vertical_edge_length    = 4,
            diagonal_edge_length    = 2 ):

        assert indent_spaces >= 0
        assert gap_width_spaces >= 0
        assert diagonal_edge_length >= 1
        assert horizontal_edge_length >= diagonal_edge_length + 1
        assert vertical_edge_length >= diagonal_edge_length + 1

        # Display characters to use for the drawing.
        # These can be more than one actual character, as long as they're
        # all the same length.
        lvf = '|'  # Vertical line (front)
        lvb = ':'  # Vertical line (back)
        lhf = '='  # Horizontal line (front)
        lhb = '-'  # Horizontal line (back)
        lsf = '#'  # Spissitudinal line (front)
        lsb = '+'  # Spissitudinal line (back)
        ldg = '\\' # Diagonal line
        vxf = '@'  # Vertex (front)
        vxb = '*'  # Vertex (back)
        spc = ' '  # Blank space

        # Assert all strings above should be the same length.
        assert (len({len(c) for c in
            [lvf,lvb,lhf,lhb,lsf,lsb,ldg,vxf,vxb,spc]}) == 1)

        # List of bools, where the nth bool is true iff the nth edge
        # is present in this IncompleteTesseract.
        has = [self.has_edge(e, unit=False)
                for e in ALL_EDGES_ORDERED_CT]
        
        # Set a display character for each edge.
        # If the edge is present, use a line character to show it,
        # otherwise set it to an empty space.
        # The mapping of indices to edge names implicitly depends on
        # the order of ALL_EDGES_ORDERED_CT.
        ac = ldg if has[ 0] else spc
        cd = lhf if has[ 1] else spc
        bd = ldg if has[ 2] else spc
        ab = lhb if has[ 3] else spc
        ae = lvb if has[ 4] else spc 
        cg = lvf if has[ 5] else spc
        dh = lvf if has[ 6] else spc
        bf = lvb if has[ 7] else spc
        eg = ldg if has[ 8] else spc
        gh = lhf if has[ 9] else spc
        fh = ldg if has[10] else spc
        ef = lhb if has[11] else spc
        ai = lsb if has[12] else spc
        ck = lsf if has[13] else spc
        dl = lsf if has[14] else spc
        bj = lsb if has[15] else spc
        em = lsb if has[16] else spc
        go = lsf if has[17] else spc
        hp = lsf if has[18] else spc
        fn = lsb if has[19] else spc
        ik = ldg if has[20] else spc
        kl = lhf if has[21] else spc
        jl = ldg if has[22] else spc
        ij = lhb if has[23] else spc
        im = lvb if has[24] else spc
        ko = lvf if has[25] else spc
        lp = lvf if has[26] else spc
        jn = lvb if has[27] else spc
        mo = ldg if has[28] else spc
        op = lhf if has[29] else spc
        np = ldg if has[30] else spc
        mn = lhb if has[31] else spc

        # Set a display character for each vertex.
        # A vertex is only shown if any of its adjacent edges
        # are present, otherwise set it to an empty space.
        va = vxb if (f'{ab}{ac}{ae}{ai}'.strip()) else spc
        vb = vxb if (f'{ab}{bd}{bf}{bj}'.strip()) else spc
        vc = vxf if (f'{ac}{cd}{cg}{ck}'.strip()) else spc
        vd = vxf if (f'{bd}{cd}{dh}{dl}'.strip()) else spc
        ve = vxb if (f'{ae}{ef}{eg}{em}'.strip()) else spc
        vf = vxb if (f'{bf}{ef}{fh}{fn}'.strip()) else spc
        vg = vxf if (f'{cg}{eg}{gh}{go}'.strip()) else spc
        vh = vxf if (f'{dh}{fh}{gh}{hp}'.strip()) else spc
        vi = vxb if (f'{ai}{ij}{ik}{im}'.strip()) else spc
        vj = vxb if (f'{bj}{ij}{jl}{jn}'.strip()) else spc
        vk = vxf if (f'{ck}{ik}{kl}{ko}'.strip()) else spc
        vl = vxf if (f'{dl}{jl}{kl}{lp}'.strip()) else spc
        vm = vxb if (f'{em}{im}{mn}{mo}'.strip()) else spc
        vn = vxb if (f'{fn}{jn}{mn}{np}'.strip()) else spc
        vo = vxf if (f'{go}{ko}{mo}{op}'.strip()) else spc
        vp = vxf if (f'{hp}{lp}{np}{op}'.strip()) else spc

        w       = horizontal_edge_length
        v       = vertical_edge_length
        d       = diagonal_edge_length
        w1      = d
        w2      = w-d-1
        indent  = spc * indent_spaces
        gap     = spc * gap_width_spaces
        gapd    = spc * d
        gapw    = spc * w
        gapw2   = spc * w2
        gapdv   = spc * (d+1)
        gapmid  = spc * (d+w+3)

        top_back_line =     (f'{indent}{ve}{ef*w}{vf}{gapdv}' +
                             f'{gap}{em}{gapw}{fn}{gapdv}' +
                             f'{gap}{vm}{mn*w}{vn}' )

        top_diag_lines =   [(f'{indent}{ae}{spc*ofs}{eg}{spc*(d-ofs)}' +
                             f'{gapw2}{bf}{spc*ofs}{fh}{spc*(d-ofs)}' +
                             f'{gap}{gapmid}' +
                             f'{gap}{im}{spc*ofs}{mo}{spc*(d-ofs)}' +
                             f'{gapw2}{jn}{spc*ofs}{np}' )
                            for ofs in range(0,d)]

        top_front_line =    (f'{indent}{ae}{gapd}{vg}{gh*w}{vh}' +
                             f'{gap}{gapdv}{go}{gapw}{hp}' +
                             f'{gap}{im}{gapd}{vo}{op*w}{vp}' )

        middle_lines =     [(f'{indent}{ae}{gapd}{cg}{gapw2}{bf}{gapd}{dh}' +
                             f'{gap}{gapmid}' +
                             f'{gap}{im}{gapd}{ko}{gapw2}{jn}{gapd}{lp}' )
                            for _ in range(0, v-d-1)]

        btm_back_line =     (f'{indent}{va}{ab*w1}{cg}{ab*w2}{vb}{gapd}{dh}' +
                             f'{gap}{ai}{spc*w}{bj}{gapdv}' +
                             f'{gap}{vi}{ij*w1}{ko}{ij*w2}{vj}{gapd}{lp}' )

        btm_diag_lines =   [(f'{indent}{spc*ofs}{ac}{spc*(d-ofs)}{cg}' +
                             f'{gapw2}{spc*ofs}{bd}{spc*(d-ofs)}{dh}' +
                             f'{gap}{gapmid}' +
                             f'{gap}{spc*ofs}{ik}{spc*(d-ofs)}{ko}' +
                             f'{gapw2}{spc*ofs}{jl}{spc*(d-ofs)}{lp}' )
                            for ofs in range(1,d+1)]

        btm_front_line =    (f'{indent}{gapdv}{vc}{cd*w}{vd}' +
                             f'{gap}{gapdv}{ck}{spc*w}{dl}' +
                             f'{gap}{gapdv}{vk}{kl*w}{vl}' )

        all_lines = ([top_back_line] + top_diag_lines +
                     [top_front_line] + middle_lines +
                     [btm_back_line] + btm_diag_lines +
                     [btm_front_line])

        return "\n".join(all_lines)



    ### Hidden Methods ###

    # Transform a single edge by the given matrix.
    # This consists of transforming each vertex individually.
    def _transform_edge(self, edge, transformation_matrix):
        assert len(transformation_matrix) == 4
        assert all(len(row) == 4 for row in transformation_matrix)
        assert len(edge) == 2
        assert all(len(v) == 4 for v in edge)
        edge = np.array(list(edge))
        return np.array([
            transformation_matrix.dot(edge[0]),
            transformation_matrix.dot(edge[1]) ])

    # Take a list of edges and return a packed bit representation.
    # This may throw ValueError if any of the included_edges are not valid.
    def _pack_edges(self, included_edges, unit=False):
        all_edges = (ALL_EDGES_ORDERED_UT if unit
                else ALL_EDGES_ORDERED_CT)
        all_edges_immutable = [self._make_unordered_edge(e)
                for e in all_edges]
        included_edges_immutable = (self._make_unordered_edge(e)
                for e in included_edges)
        bit_indices = (all_edges_immutable.index(e)
                for e in included_edges_immutable)
        bit_masks = (0x1 << i
                for i in bit_indices)
        return functools.reduce(
                (lambda x,y: x | y), bit_masks, 0x0)

    # Take an edge and convert it to a frozenset of tuples.
    # The edge given may be represented as any iterable of length 2,
    # containing two sub-iterables of length 4.
    def _make_unordered_edge(self, e):
        assert len(e) == 2
        assert all(len(v) == 4 for v in e)
        return frozenset([tuple(v) for v in e])
        


