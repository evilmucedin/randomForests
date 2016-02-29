#!/usr/bin/env python

for i in xrange(0, 16):
    print """        case %d:
                return _mm128_blend_epi32(a, b, %d);""" % (i, i)
