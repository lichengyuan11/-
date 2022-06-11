# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 14:49:54 2022

@author: 李程远
"""



from numpy import r_, c_, ix_, zeros, pi, ones, exp, argmax, union1d

from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.makeBdc import makeBdc
from pypower.makeSbus import makeSbus
from pypower.dcpf import dcpf
from pypower.makeYbus import makeYbus


from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS
# from case33bw_mobi import case33bw as case
from case69 import case

def Ybus():

    ppc = case()

    ## add zero columns to branch for flows if needed
    if ppc["branch"].shape[1] < QT:
        ppc["branch"] = c_[ppc["branch"],
                           zeros((ppc["branch"].shape[0],
                                  QT - ppc["branch"].shape[1] + 1))]

    ## convert to internal indexing
    ppc = ext2int(ppc)
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]

    ref, pv, pq = bustypes(bus, gen)

    Va0 = bus[:, VA] * (pi / 180)

        ## build B matrices and phase shift injections
    B, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)

        ## compute complex bus power injections [generation - load]
        ## adjusted for phase shifters and real shunts
    Pbus = makeSbus(baseMVA, bus, gen).real - Pbusinj - bus[:, GS] / baseMVA

        ## "run" the power flow
    Va = dcpf(B, Pbus, Va0, ref, pv, pq)

        ## update data matrices with solution
    branch[:, [QF, QT]] = zeros((branch.shape[0], 2))
    branch[:, PF] = (Bf * Va + Pfinj) * baseMVA
    branch[:, PT] = -branch[:, PF]
    bus[:, VM] = ones(bus.shape[0])
    bus[:, VA] = Va * (180 / pi)
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    return Ybus.toarray()