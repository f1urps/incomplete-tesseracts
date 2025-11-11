#! /usr/bin/python3

from incomplete_tesseract import IncompleteTesseract
from incomplete_tesseract import CACHE_MAXSIZE
import sys
from datetime import datetime
import psutil
import humanize


log_messages_enabled = False
log_diagrams_enabled = False


def main(starting_n):

    for n in range(starting_n,32):
        print(f'\nCalculating for n={n}...')

        stats = init_stats(n)

        # Load previously calculated results with n-1 edges
        previous = load_results(n-1)

        # Get all possible extensions with n edges
        t = generate_extensions(previous, stats)

        # Filter for uniqueness by symmetry
        t = remove_symmetrical_copies(t, stats)

        # Filter out non-4d shapes, once we get past small n
        if n >= 5:
            t = remove_non_4d(t, stats)

        # Convert each to their minimum representation
        t = convert_to_minimum(t, stats)

        # Write out the results for n
        write_results(t, n, stats)

        # Sort the results in the file
        sort_results(n, stats)

        # Print summary of findings
        finalize_stats(stats, n)
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
        return sorted([IncompleteTesseract(int(line.strip()))
            for line in f.readlines()])


# Write a results file with a list of IncompleteTesseract objects
# that all have n edges.
def write_results(tesseracts, n:int, stats):
    filename = f'results/results_with_{n}_edges'
    with open(filename, 'w') as f:
        for t in tesseracts:
            log(f'Writing result #{t.packed} to "{filename}".')
            stats.results_written_to_file += 1
            collect_memory_usage_stats(stats)

            f.write(f'{t.packed}\n')
            f.flush()


# Read the results file for n edges and write it back sorted.
def sort_results(n:int, stats):
    results = load_results(n)
    stats.results_written_to_file = 0
    write_results(sorted(results), n, stats)


# Take a collection of IncompleteTesseract objects with n edges and generate
# all the ways of adding one more connected edge to each one.
# Yields IncompleteTesseract objects with n+1 edges each.
def generate_extensions(tesseracts, stats):
    for t in tesseracts:
        log(f'Generating extensions of input tesseract #{t.packed}.')
        log_diagram(t)
        stats.inputs_processed += 1
        collect_memory_usage_stats(stats)

        v, e = t.graph(unit=True)

        for new_edge in ALL_EDGES:
            if ((new_edge not in e) and any(new_vertex in v for new_vertex in new_edge)):

                extension = t.with_edge(new_edge, unit=True)

                log(f'Extended #{t.packed} with edge ' +
                    f'{list(new_edge)} to create #{extension.packed}.')
                log_diagram(extension)
                stats.extensions_created += 1
                collect_memory_usage_stats(stats)

                yield extension


# Take a collection of IncompleteTesseract objects and yield only
# the ones that are unique up to symmetrical transformations.
def remove_symmetrical_copies(tesseracts, stats):
    results_seen = set()
    for t in tesseracts:
        log(f'Checking #{t.packed} for symmetrical uniqueness.')
        stats.uniqueness_checks += 1
        collect_memory_usage_stats(stats)

        if all(tt not in results_seen for tt in t.transformations()):
            results_seen.add(t)

            log(f'#{t.packed} is new!')
            stats.uniqueness_checks_accepted += 1

            yield t

        else:
            log(f'#{t.packed} has been seen before.')
            stats.uniqueness_checks_rejected += 1


# Take a collection of IncompleteTesseract objects and yield only
# the ones that are "properly" 4-dimensional. A tesseract is properly 4d
# iff it has at least one W-pointing edge in all transformations.
# I.e. it should not be possible to rotate it out of the fourth dimension.
def remove_non_4d(tesseracts, stats):
    for t in tesseracts:
        log(f'Checking for proper 4-dimensionality of #{t.packed}.')
        stats.dimensionality_checks += 1
        collect_memory_usage_stats(stats)

        if all(frozenset(tt.edges(unit=True)).intersection(W_EDGES)
                for tt in frozenset(t.transformations())):
            log(f'#{t.packed} is properly 4d!')
            stats.dimensionality_checks_accepted += 1
            yield t

        else:
            log(f'#{t.packed} is not properly 4d.')
            stats.dimensionality_checks_rejected += 1


# Convert each tesseract to its minimum transformation number.
def convert_to_minimum(tesseracts, stats):
    for t in tesseracts:
        log(f'Calculating minimum transformation for #{t.packed}.')
        stats.minimum_transformation_conversions += 1
        collect_memory_usage_stats(stats)

        yield min(t.transformations())


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
    print(f'Found {len(results)} results:')
    print(', '.join([f'#{t.packed}' for t in results]))
    print('\nStats:')
    for k, v in stats.__dict__.items():
        print(f'  {(k+":").ljust(40)}{str(v)}')
    print()


# Get an object to collect statistics in
def init_stats(n:int):
    class _Stats(object):
        def __getattr__(self, attr):
            setattr(self, attr, 0)
            return 0
    stats = _Stats()
    stats.n = n
    stats.log_messages_enabled = log_messages_enabled
    stats.log_diagrams_enabled = log_diagrams_enabled
    stats.cache_maxsize = CACHE_MAXSIZE
    collect_memory_usage_stats(stats)
    stats.start_time = datetime.now()
    stats.end_time = 0
    return stats


# Crunch some final numbers to analyze runtime.
def finalize_stats(stats, n:int):
    stats.end_time = datetime.now()
    stats.elapsed_time = stats.end_time - stats.start_time
    stats.average_time_per_input = stats.elapsed_time / stats.inputs_processed
    stats.estimated_time_for_next_run = (
            stats.average_time_per_input * stats.results_written_to_file)
    stats.max_memory_usage = humanize.naturalsize(
            stats.max_memory_usage)
    write_stats(stats, n)


# Write statistics out to a file
def write_stats(stats, n:int):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'statistics/{n}.{timestamp}'
    with open(filename, 'w') as f:
        for k, v in stats.__dict__.items():
            f.write(f'{(k+":").ljust(40)}{str(v)}\n')


# Keep track of maximum memory usage stats
def collect_memory_usage_stats(stats):
    p = psutil.Process()
    current_memory_usage_bytes = p.memory_info().rss
    current_memory_usage_percent = p.memory_percent()
    stats.max_memory_usage = max(
            stats.max_memory_usage, current_memory_usage_bytes)
    stats.max_memory_usage_percent = max(
            stats.max_memory_usage_percent, current_memory_usage_percent)


if __name__ == "__main__":
    try:
        starting_n = int(sys.argv[1])
    except Exception as e:
        print("ERROR: Please give starting value N as argument. (2 <= N <= 32)")
        exit(1)

    main(starting_n)



