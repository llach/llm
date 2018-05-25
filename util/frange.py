# -*- coding: utf-8 -*-

'''
Copyright 2016 Akseli Palén.
Created 2016-04-02.
Licensed under the MIT license.
<license>
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</lisence>
'''


def frange(x, y, jump=1.0):
    '''
    Range for floats.

    Parameters:
      x: range starting value, will be included.
      y: range ending value, will be excluded
      jump: the step value. Only positive steps are supported.

    Return:
      a generator that yields floats

    Usage:
    >>> list(frange(0, 1, 0.2))
    [0.0, 0.2, 0.4, 0.6000000000000001, 0.8]
    >>> list(frange(1, 0, 0.2))
    [1.0]
    >>> list(frange(0.0, 0.05, 0.1))
    [0.0]
    >>> list(frange(0.0, 0.15, 0.1))
    [0.0, 0.1]

    '''
    i = 0.0
    x = float(x)  # Prevent yielding integers.
    y = float(y)  # Comparison converts y to float every time otherwise.
    x0 = x
    epsilon = jump / 2.0
    yield x  # yield always first value
    while x + epsilon < y:
        i += 1.0
        x = x0 + i * jump
        yield x