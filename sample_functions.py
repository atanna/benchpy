from django.conf import settings
import html5lib

settings.configure()
from django.template import Template

def fib(n):
    return n if n <= 1 else fib(n-1) + fib(n-2)

def test_spam_bayes(messages, ham_classifier):
    for msg in messages:
            ham_classifier.score(msg)

def test_html5libs(spec_data):
    html5lib.parse(spec_data)


def count(iterations=1000000):
    """Count down from a given starting point."""
    while iterations > 0:
        iterations -= 1

def test_iterative_count():
   count()

def test_threaded_count(threads):
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


DJANGO_TMPL = Template("""<table>
{% for row in table %}
<tr>{% for col in row %}<td>{{ col|escape }}</td>{% endfor %}</tr>
{% endfor %}
</table>
""")

def test_django(context):
    data = DJANGO_TMPL.render(context)



# Pure-Python implementation of itertools.permutations().
def permutations(iterable, r=None):
    """permutations(range(3), 2) --> (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)"""
    pool = tuple(iterable)
    n = len(pool)
    if r is None:
        r = n
    indices = range(n)
    cycles = range(n-r+1, n+1)[::-1]
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return


# From http://code.activestate.com/recipes/576647/
def n_queens(queen_count):
    """N-Queens solver.

    Args:
        queen_count: the number of queens to solve for. This is also the
            board size.

    Yields:
        Solutions to the problem. Each yielded value is looks like
        (3, 8, 2, 1, 4, ..., 6) where each number is the column position for the
        queen, and the index into the tuple indicates the row.
    """
    cols = range(queen_count)
    for vec in permutations(cols):
        if (queen_count == len(set(vec[i]+i for i in cols))
                        == len(set(vec[i]-i for i in cols))):
            yield vec


def foo(a, b, c, d):
    # 20 calls
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)
    bar(a, b, c)


def bar(a, b, c):
    # 20 calls
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)
    baz(a, b)


def baz(a, b):
    # 20 calls
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)
    quux(a)


def quux(a):
    # 20 calls
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()
    qux()


def qux():
    pass


def bm_call():
    # 40 calls
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)
        foo(1, 2, 3, 4)


def test_list(n):
    arr = []
    for i in range(n):
        arr.append(i)


def test_circle_list():
    arr = []
    arr.append(arr)

