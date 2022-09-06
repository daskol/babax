# BABAX

*bencharking TT-tensors in Python*

## Overview

At the table below one can see results of bencharking of a procedure for
calculating of a random single TT-tensor element with or witout JIT (i.e.
`True` for JITted functions). There are four implemention which were
benchmarked. They are original [TTPy][4], [Teneva][1] with Numba, [TTAX][2] in
JAX, and naive implemention in JAX. Unfortunately, TTAX is [a bit buggy][3] by
the moment.

```
----------------------------------- benchmark: 5 tests ------------------------
Name (time in us)                              Mean             StdDev
-------------------------------------------------------------------------------
TestGetItemTeneva::test_getitem[True]        1.4175 (1.0)       0.2381 (1.0)
TestGetItemTeneva::test_getitem[False]       1.9407 (1.37)      0.5820 (2.44)
TestGetItemJAX::test_getitem[True]           2.1229 (1.50)      1.1670 (4.90)
TestGetItemTTPy::test_getitem               57.3524 (40.46)    10.0930 (42.38)
TestGetItemJAX::test_getitem[False]        396.1317 (279.46)   57.9950 (243.55)
-------------------------------------------------------------------------------
```

One can see that a use of Numba in [Teneva][1] gives excelent performance for
an estimation of a random signle element of tensor in TT-format. Nevertheless,
naive TT-tensor implemention in JAX is worse by 50% due to heavy dispatching in
JAX internals.

```
------------------------------------ benchmark: 5 tests ------------------------------
Name (time in us)                                  Mean                StdDev
--------------------------------------------------------------------------------------
TestGetItemJAX::test_getitems[True]              3.9626 (1.0)         10.0755 (1.0)
TestGetItemTeneva::test_getitems[True]         168.4863 (42.52)       15.2934 (1.52)
TestGetItemTeneva::test_getitems[False]        215.9346 (54.49)       25.5915 (2.54)
TestGetItemTTPy::test_getitems               7,054.7623 (>1000.0)    299.3015 (29.71)
TestGetItemJAX::test_getitems[False]        50,191.7854 (>1000.0)  1,201.3886 (119.24)
--------------------------------------------------------------------------------------
```

So, calculation of a several elements of a TT-tensor should mitigate overhead
in JAX and the table above is just about it. Now, JAX implemention has better
performance than [Teneva][1].

In all experiments we use random 10 dimentional TT-tensor of rank 2 and
preliminary generated sets of random tensor indexes to access.

[1]: https://github.com/AndreiChertkov/teneva
[2]: https://github.com/fasghq/ttax
[3]: https://github.com/fasghq/ttax/issues/40
[4]: https://github.com/oseledets/ttpy
