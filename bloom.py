"""bloom - a cheesy Bloom filter implementation

This module implements a lame version of a Bloom filter, a simple
set-like data structure. You don't need to worry about the details
of it if you don't want to, but it's a nice example of a cute
data structure.

A Bloom filter has the following properties:

- You can "add" known items to the filter
- You can then query the filter as to whether it contains a given
  item. There may be false positives (the filter says it contains the
  item when it doesn't really) but there are never false negatives
  (the filter never says that it doesn't contain the item when it
  does).

Bloom filters provide these characters cheaply in terms of both memory
and computation time. This makes them useful as filters to large
backing data stores; for instance, if you have a giant disk database
with many records that you're going to query for items it will
frequently not contain, a Bloom filter can allow you to avoid many
of the queries to the full disk.

The filter is defined by an array of "m" bits and a set of "k"
functions mapping a filter item to a number "n", 0 <= n < m. The bits
all start out as 0. When you add an item to the filter, you evaluate
each of the k functions and use each function's result as an index to
the bit array, setting the corresponding bits to 1. When you check for
the presence of an item, you evaluate the functions and test each
corresponding bit. If any are 0, the item has definitely not been
added to the filter.

For more information, see

http://en.wikipedia.org/wiki/Bloom_filter

Here are some equations describing the behavior of Bloom filters.
Let 

m = the number of bits in the filter
k = the number of functions
n = the number of items in the filter
p = the false positive rate

Given k, n, m, then

p = (1 - exp (-k * n / m)) ** k

Given n, m, then the optimal k is

k = (ln 2) * m / n

Given n, a desired p, and the desire to use the optimal k,

m = - n * ln p / (ln 2)**2
k = -ln p / (ln 2)

"""

# Activate "true" division: 1 / 2 = 0.5, not 0. See PEP238.
from __future__ import division
import numpy as N
import hashlib


def makeHashFunc (m, salt):
    """Return a *function* that hashes an input value. The function that
    we return takes a string argument and returns an integer between 0 and
    m, noninclusive. Our arguments are m, and a "salt" value that we mix
    into the hash. The allows us to create different hash functions from
    the single SHA1 function.

    These functions will be called many times, so speed is important. We
    take the simple route; a real implementation would worry a lot more
    about optimizing how this is done."""

    salt = str (salt)
    
    def f (value):
        # Get a 20-byte string representing the SHA1 cryptographic
        # hash of the input plus the "salt" string
        h = hashlib.sha1 ()
        h.update (salt)
        h.update (value)
        digest = h.digest ()

        # Convert the digest from a string to an integer modulo
        # m. Python handles bigints automagically so we don't have
        # to worry about the high bits causing problems.
        v = 0
        scale = 1

        for b in digest:
            v = (v + ord (b) * scale) % m
            scale *= 8

        return v

    return f


class BloomFilter (object):
    def __init__ (self, m, k):
        assert m > 0
        assert k > 0

        if m % 32 != 0:
            raise ValueError ("This lame function can only accept m's that"
                              " are multuples of 32; I got %d" % m)
        self.m = m
        self.k = k

        self.bits = N.zeros (m // 32, dtype=N.uint32)
        self.n = 0
        self.funcs = [makeHashFunc (m, i) for i in xrange (k)]


    def fprate (self):
        r = (1. - N.exp (-self.k * self.n / self.m))
        r **= self.k
        return r


    def add (self, item):
        for func in self.funcs:
            n = func (item)
            dword = n // 32
            bit = n % 32
            self.bits[dword] |= (1 << bit)

        self.n += 1


    def maycontain (self, item):
        for func in self.funcs:
            n = func (item)
            dword = n // 32
            bit = n % 32

            if self.bits[dword] & (1 << bit) == 0:
                return False

        return True


    def clear (self):
        self.bits.fill (0)


    # These functions allow us to save and load object state
    # through Python's standard "pickle" system. Populating the filter
    # is slow, so we'll speed things up by saving and restoring
    # the filter state.

    def __getstate__ (self):
        return self.k, self.n, self.bits


    def __setstate__ (self, state):
        self.k, self.n, self.bits = state
        self.m = self.bits.size * 32
        self.funcs = [makeHashFunc (self.m, i) for i in xrange (self.k)]


def optimalBloom (fprate, nexpected):
    """Create and return a well-optimized Bloom filter given a desired
    false-positive rate and an expected number of items to be added to
    the filter. This involves figuring out the optimal number of bits
    of data and the number of functions we need using the equations
    given in the module docstring."""

    ln2 = N.log (2)

    m = -nexpected * N.log (fprate) / ln2**2
    m = int (N.ceil (m))
    # round to nearest larger multiple of 32
    m = (m + 31) & ~0x1F

    k = ln2 * m / nexpected
    k = max (int (k), 1)

    return BloomFilter (m, k)
