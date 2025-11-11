
from incomplete_tesseract import IncompleteTesseract


### MAIN ###

def main():

    n = 2

    # Load previously calculated results with n edges
    previous = load_previous_results(n)

    # Get all possible extensions with n+1 edges
    t = generate_all_options_with_one_more_edge(previous)

    # Filter for uniqueness by symmetry
    t = remove_symmetrical_copies(t)
    
    # Filter out any tesseracts that aren't properly 4d
    t = remove_non_4d(t)

    # Write out the results for n+1
    write_results(t, n+1)


### CONSTANTS ###

# Set of all vertices and edges in the complete tesseract.
ALL_VERTICES, ALL_EDGES = IncompleteTesseract(0xffffffff).graph()

# Set of all edges that point in the W direction.
_, W_EDGES = IncompleteTesseract(0x0000ff00).graph()

# Read the results file for n edges, and return a list of
# IncompleteTesseract objects whose numbers are in the file.
def load_previous_results(n:int):
    filename = f'results/results_with_{n}_edges'
    with open(filename, 'r') as f:
        return [IncompleteTesseract(int(line.strip()))
            for line in f.readlines()]

# Write a results file with a list of IncompleteTesseract objects
# that all have n edges.
def write_results(tesseracts, n:int)
    filename = f'results/results_with_{n}_edges'
    with open(filename, 'w+') as f:
        for t in tesseracts:
            f.write(f'{t.packed}\n')
            f.flush()

# Take a collection of IncompleteTesseract objects with n edges and generate
# all the ways of adding one more connected edge to each one.
# Yields IncompleteTesseract objects with n+1 edges each.
def generate_all_options_with_one_more_edge(tesseracts):
    for t in tesseracts:
        v, e = t.graph()
        for new_edge in ALL_EDGES:
            if ((new_edge not in e)
                    and any(new_vertex in v for new_vertex in new_edge)):
                yield t.with_edge(new_edge)

# Take a collection of IncompleteTesseract objects and yield only
# the ones that are unique up to symmetrical transformations.
def remove_symmetrical_copies(tesseracts):
    results_seen = {}
    for t in tesseracts:
        min_transformation = min(t.transformations())
        if min_transformation not in results_seen:
            results_seen.add(min_transformation)
            yield min_transformation

# Take a collection of IncompleteTesseract objects and yield only
# the ones that are "properly" 4-dimensional. A tesseract is properly 4d
# iff it has at least one W-pointing edge in all transformations.
# I.e. it should not be possible to rotate it out of the fourth dimension.
def remove_non_4d(tesseracts):
    for t in tesseracts:
        if all(frozenset(tt.edges()).intersection(W_EDGES)
                for tt in t.transformations()):
            yield t

if __name__ == "__main__":
    main()


