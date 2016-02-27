#!/usr/bin/env python

for i in xrange(1, 256):
    print """        case %d:
                return _mm256_blend_epi32(a, b, %d);""" % (i, i)
