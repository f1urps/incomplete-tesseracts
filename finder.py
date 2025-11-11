#! /usr/bin/python3

from incomplete_tesseract import IncompleteTesseract
import sys
from datetime import datetime


log_messages_enabled = True
log_diagrams_enabled = True


def main(n):

    stats = init_stats()

    # Load previously calculated results with n-1 edges
    previous = load_results(n-1)

    # Get all possible extensions with n edges
    t = generate_extensions(previous, stats)

    # Filter for uniqueness by symmetry
    t = remove_symmetrical_copies(t, stats)

    # Write out the results for n
    write_results(t, n, stats)

    # Print summary of findings
    finalize_stats(stats)
    print_final_report(load_results(n), stats, n)



### CONSTANTS ###

# Set of all vertices and edges in the complete tesseract.
ALL_VERTICES, ALL_EDGES = IncompleteTesseract(0xffffffff).graph(unit=True)

# Set of all edges that point in the W direction.
_, W_EDGES = IncompleteTesseract(0x000ff000).graph(unit=True)

# Read the results file for n edges, and return a list of
# IncompleteTesseract objects whose numbers are in the file.
def load_results(n:int):
    filename = f'results/results_with_{n}_edges'
    with open(filename, 'r') as f:
        return [IncompleteTesseract(int(line.strip()))
            for line in f.readlines()]



# Write a results file with a list of IncompleteTesseract objects
# that all have n edges.
def write_results(tesseracts, n:int, stats):
    filename = f'results/results_with_{n}_edges'
    with open(filename, 'w') as f:
        for t in tesseracts:
            log(f'Writing result #{t.packed} to "{filename}".')
            f.write(f'{t.packed}\n')
            f.flush()
            stats.results_written_to_file += 1


# Take a collection of IncompleteTesseract objects with n edges and generate
# all the ways of adding one more connected edge to each one.
# Yields IncompleteTesseract objects with n+1 edges each.
def generate_extensions(tesseracts, stats):
    for t in tesseracts:
        log(f'Generating extensions of input tesseract #{t.packed}.')
        log_diagram(t)
        stats.inputs_processed += 1

        v, e = t.graph(unit=True)

        for new_edge in ALL_EDGES:
            if ((new_edge not in e) and any(new_vertex in v for new_vertex in new_edge)):

                extension = t.with_edge(new_edge, unit=True)

                log(f'Extended #{t.packed} with edge ' +
                    f'{list(new_edge)} to create #{extension.packed}.')
                log_diagram(extension)
                stats.extensions_created += 1

                yield extension


# Take a collection of IncompleteTesseract objects and yield only
# the ones that are unique up to symmetrical transformations.
def remove_symmetrical_copies(tesseracts, stats):
    results_seen = set()
    for t in tesseracts:
        log(f'Checking #{t.packed} for symmetrical uniqueness.')
        stats.uniqueness_checks += 1

        tt = min(t.transformations())

        log(f'Converted #{t.packed} to minimum transformation #{tt.packed}.')
        log_diagram(tt)

        if tt not in results_seen:
            results_seen.add(tt)

            log(f'#{tt.packed} is new!')
            log(f'results_seen={[t.packed for t in results_seen]}')
            stats.uniqueness_checks_accepted += 1

            yield tt

        else:
            log(f'#{tt.packed} has been seen before.')
            stats.uniqueness_checks_rejected += 1


# Take a collection of IncompleteTesseract objects and yield only
# the ones that are "properly" 4-dimensional. A tesseract is properly 4d
# iff it has at least one W-pointing edge in all transformations.
# I.e. it should not be possible to rotate it out of the fourth dimension.
def remove_non_4d(tesseracts, stats):
    for t in tesseracts:
        log(f'Checking for proper 4-dimensionality of #{t.packed}.')
        stats.dimensionality_checks += 1

        if all(frozenset(tt.edges(unit=True)).intersection(W_EDGES)
                for tt in frozenset(t.transformations())):
            log(f'#{t.packed} is properly 4d!')
            stats.dimensionality_checks_accepted += 1
            yield t

        else:
            log(f'#{t.packed} is not properly 4d.')
            stats.dimensionality_checks_rejected += 1


# Print a log message
def log(msg, force=False):
    if log_messages_enabled or force:
        print(msg)


# Print a log message with a diagram of a tessaract
def log_diagram(t, force=False):
    if log_diagrams_enabled or force:
        print(f'(#{t.packed}):')
        print(t.ascii_drawing(
            indent_spaces=8,
            gap_width_spaces=4,
            horizontal_edge_length=6,
            vertical_edge_length=2,
            diagonal_edge_length=1))
        print('\n')


# Print pictures of all the tesseracts found
def print_final_report(results, stats, n):
    print("\n\n====== DIAGRAMS ======\n")
    for t in results:
        log_diagram(t, force=True)
    print('\n')
    print("\n\n====== SUMMARY OF RESULTS ======\n")
    print(f'N = {n}\n')
    print(f'Found {len(results)} results:')
    print(', '.join([f'#{t.packed}' for t in results]))
    print('\nStats:')
    for k, v in stats.__dict__.items():
        print(f'  {(k+":").ljust(35)}{str(v)}')
    print()


# Get an object to collect statistics in
def init_stats():
    class _Stats(object):
        def __getattr__(self, attr):
            setattr(self, attr, 0)
            return 0
    stats = _Stats()
    stats.start_time = datetime.now()
    return stats


# Crunch some final numbers to analyze runtime.
def finalize_stats(stats):
    stats.end_time = datetime.now()
    stats.elapsed_time = stats.end_time - stats.start_time
    stats.average_time_per_input = stats.elapsed_time / stats.inputs_processed
    stats.estimated_time_for_next_run = stats.average_time_per_input * stats.results_written_to_file


if __name__ == "__main__":
    try:
        n = int(sys.argv[1])
    except Exception as e:
        print("ERROR: Please give N as argument. (2 <= N <= 32)")
        exit(1)

    main(n)


