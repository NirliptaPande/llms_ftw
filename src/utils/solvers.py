from utils.dsl import *
from utils.constants import *


def solve_67a3c6ac(I):
    O = vmirror(I)
    return O


def solve_68b16354(I):
    O = hmirror(I)
    return O


def solve_74dd1130(I):
    O = dmirror(I)
    return O


def solve_3c9b0459(I):
    O = rot180(I)
    return O


def solve_6150a2bd(I):
    O = rot180(I)
    return O


def solve_9172f3a0(I):
    O = upscale(I, THREE)
    return O


def solve_9dfd6313(I):
    O = dmirror(I)
    return O


def solve_a416b8f3(I):
    O = hconcat(I, I)
    return O


def solve_b1948b0a(I):
    O = replace(I, SIX, TWO)
    return O


def solve_c59eb873(I):
    O = upscale(I, TWO)
    return O


def solve_c8f0f002(I):
    O = replace(I, SEVEN, FIVE)
    return O


def solve_d10ecb37(I):
    O = crop(I, ORIGIN, TWO_BY_TWO)
    return O


def solve_d511f180(I):
    O = switch(I, FIVE, EIGHT)
    return O


def solve_ed36ccf7(I):
    O = rot270(I)
    return O


def solve_4c4377d9(I):
    x1 = hmirror(I)
    O = vconcat(x1, I)
    return O


def solve_6d0aefbc(I):
    x1 = vmirror(I)
    O = hconcat(I, x1)
    return O


def solve_6fa7a44f(I):
    x1 = hmirror(I)
    O = vconcat(I, x1)
    return O


def solve_5614dbcf(I):
    x1 = replace(I, FIVE, ZERO)
    O = downscale(x1, THREE)
    return O


def solve_5bd6f4ac(I):
    x1 = tojvec(SIX)
    O = crop(I, x1, THREE_BY_THREE)
    return O


def solve_5582e5ca(I):
    x1 = mostcolor(I)
    O = canvas(x1, THREE_BY_THREE)
    return O


def solve_8be77c9e(I):
    x1 = hmirror(I)
    O = vconcat(I, x1)
    return O


def solve_c9e6f938(I):
    x1 = vmirror(I)
    O = hconcat(I, x1)
    return O


def solve_2dee498d(I):
    x1 = hsplit(I, THREE)
    O = first(x1)
    return O


def solve_1cf80156(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = subgrid(x2, I)
    return O


def solve_32597951(I):
    x1 = ofcolor(I, EIGHT)
    x2 = delta(x1)
    O = fill(I, THREE, x2)
    return O


def solve_25ff71a9(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = move(I, x2, DOWN)
    return O


def solve_0b148d64(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O


def solve_1f85a75f(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    O = subgrid(x2, I)
    return O


def solve_23b5c85d(I):
    x1 = objects(I, T, T, T)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O


def solve_9ecd008a(I):
    x1 = vmirror(I)
    x2 = ofcolor(I, ZERO)
    O = subgrid(x2, x1)
    return O


def solve_ac0a08a4(I):
    x1 = colorcount(I, ZERO)
    x2 = subtract(NINE, x1)
    O = upscale(I, x2)
    return O


def solve_be94b721(I):
    x1 = objects(I, T, F, T)
    x2 = argmax(x1, size)
    O = subgrid(x2, I)
    return O


def solve_c909285e(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    O = subgrid(x2, I)
    return O


def solve_f25ffba3(I):
    x1 = bottomhalf(I)
    x2 = hmirror(x1)
    O = vconcat(x2, x1)
    return O


def solve_c1d99e64(I):
    x1 = frontiers(I)
    x2 = merge(x1)
    O = fill(I, TWO, x2)
    return O


def solve_b91ae062(I):
    x1 = numcolors(I)
    x2 = decrement(x1)
    O = upscale(I, x2)
    return O


def solve_3aa6fb7a(I):
    x1 = objects(I, T, F, T)
    x2 = mapply(corners, x1)
    O = underfill(I, ONE, x2)
    return O


def solve_7b7f7511(I):
    x1 = portrait(I)
    x2 = branch(x1, tophalf, lefthalf)
    O = x2(I)
    return O


def solve_4258a5f9(I):
    x1 = ofcolor(I, FIVE)
    x2 = mapply(neighbors, x1)
    O = fill(I, ONE, x2)
    return O


def solve_2dc579da(I):
    x1 = vsplit(I, TWO)
    x2 = rbind(hsplit, TWO)
    x3 = mapply(x2, x1)
    O = argmax(x3, numcolors)
    return O


def solve_28bf18c6(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = hconcat(x3, x3)
    return O


def solve_3af2c5a8(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_44f52bb0(I):
    x1 = vmirror(I)
    x2 = equality(x1, I)
    x3 = branch(x2, ONE, SEVEN)
    O = canvas(x3, UNITY)
    return O


def solve_62c24649(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_67e8384a(I):
    x1 = vmirror(I)
    x2 = hconcat(I, x1)
    x3 = hmirror(x2)
    O = vconcat(x2, x3)
    return O


def solve_7468f01a(I):
    x1 = objects(I, F, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = vmirror(x3)
    return O


def solve_662c240a(I):
    x1 = vsplit(I, THREE)
    x2 = fork(equality, dmirror, identity)
    x3 = compose(flip, x2)
    O = extract(x1, x3)
    return O


def solve_42a50994(I):
    x1 = objects(I, T, T, T)
    x2 = sizefilter(x1, ONE)
    x3 = merge(x2)
    O = cover(I, x3)
    return O


def solve_56ff96f3(I):
    x1 = fgpartition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = mapply(x2, x1)
    O = paint(I, x3)
    return O


def solve_50cb2852(I):
    x1 = objects(I, T, F, T)
    x2 = compose(backdrop, inbox)
    x3 = mapply(x2, x1)
    O = fill(I, EIGHT, x3)
    return O


def solve_4347f46a(I):
    x1 = objects(I, T, F, T)
    x2 = fork(difference, toindices, box)
    x3 = mapply(x2, x1)
    O = fill(I, ZERO, x3)
    return O


def solve_46f33fce(I):
    x1 = rot180(I)
    x2 = downscale(x1, TWO)
    x3 = rot180(x2)
    O = upscale(x3, FOUR)
    return O


def solve_a740d043(I):
    x1 = objects(I, T, T, T)
    x2 = merge(x1)
    x3 = subgrid(x2, I)
    O = replace(x3, ONE, ZERO)
    return O


def solve_a79310a0(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = move(I, x2, DOWN)
    O = replace(x3, EIGHT, TWO)
    return O


def solve_aabf363d(I):
    x1 = leastcolor(I)
    x2 = replace(I, x1, ZERO)
    x3 = leastcolor(x2)
    O = replace(x2, x3, x1)
    return O


def solve_ae4f1146(I):
    x1 = objects(I, F, F, T)
    x2 = rbind(colorcount, ONE)
    x3 = argmax(x1, x2)
    O = subgrid(x3, I)
    return O


def solve_b27ca6d3(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, TWO)
    x3 = mapply(outbox, x2)
    O = fill(I, THREE, x3)
    return O


def solve_ce22a75a(I):
    x1 = objects(I, T, F, T)
    x2 = apply(outbox, x1)
    x3 = mapply(backdrop, x2)
    O = fill(I, ONE, x3)
    return O


def solve_dc1df850(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, TWO)
    x3 = mapply(outbox, x2)
    O = fill(I, ONE, x3)
    return O


def solve_f25fbde4(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = upscale(x3, TWO)
    return O


def solve_44d8ac46(I):
    x1 = objects(I, T, F, T)
    x2 = apply(delta, x1)
    x3 = mfilter(x2, square)
    O = fill(I, TWO, x3)
    return O


def solve_1e0a9b12(I):
    x1 = rot270(I)
    x2 = rbind(order, identity)
    x3 = apply(x2, x1)
    O = rot90(x3)
    return O


def solve_0d3d703e(I):
    x1 = switch(I, THREE, FOUR)
    x2 = switch(x1, EIGHT, NINE)
    x3 = switch(x2, TWO, SIX)
    O = switch(x3, ONE, FIVE)
    return O


def solve_3618c87e(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = merge(x2)
    O = move(I, x3, TWO_BY_ZERO)
    return O


def solve_1c786137(I):
    x1 = objects(I, T, F, F)
    x2 = argmax(x1, height)
    x3 = subgrid(x2, I)
    O = trim(x3)
    return O


def solve_8efcae92(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ONE)
    x3 = compose(size, delta)
    x4 = argmax(x2, x3)
    O = subgrid(x4, I)
    return O


def solve_445eab21(I):
    x1 = objects(I, T, F, T)
    x2 = fork(multiply, height, width)
    x3 = argmax(x1, x2)
    x4 = color(x3)
    O = canvas(x4, TWO_BY_TWO)
    return O


def solve_6f8cd79b(I):
    x1 = asindices(I)
    x2 = apply(initset, x1)
    x3 = rbind(bordering, I)
    x4 = mfilter(x2, x3)
    O = fill(I, EIGHT, x4)
    return O


def solve_2013d3e2(I):
    x1 = objects(I, F, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = lefthalf(x3)
    O = tophalf(x4)
    return O


def solve_41e4d17e(I):
    x1 = objects(I, T, F, T)
    x2 = fork(combine, vfrontier, hfrontier)
    x3 = compose(x2, center)
    x4 = mapply(x3, x1)
    O = underfill(I, SIX, x4)
    return O


def solve_9565186b(I):
    x1 = shape(I)
    x2 = objects(I, T, F, F)
    x3 = argmax(x2, size)
    x4 = canvas(FIVE, x1)
    O = paint(x4, x3)
    return O


def solve_aedd82e4(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, TWO)
    x3 = sizefilter(x2, ONE)
    x4 = merge(x3)
    O = fill(I, ONE, x4)
    return O


def solve_bb43febb(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, FIVE)
    x3 = compose(backdrop, inbox)
    x4 = mapply(x3, x2)
    O = fill(I, TWO, x4)
    return O


def solve_e98196ab(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = objects(x1, T, F, T)
    x4 = merge(x3)
    O = paint(x2, x4)
    return O


def solve_f76d97a5(I):
    x1 = palette(I)
    x2 = first(x1)
    x3 = last(x1)
    x4 = switch(I, x2, x3)
    O = replace(x4, FIVE, ZERO)
    return O


def solve_ce9e57f2(I):
    x1 = objects(I, T, F, T)
    x2 = fork(connect, ulcorner, centerofmass)
    x3 = mapply(x2, x1)
    x4 = fill(I, EIGHT, x3)
    O = switch(x4, EIGHT, TWO)
    return O


def solve_22eb0ac0(I):
    x1 = fgpartition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = apply(x2, x1)
    x4 = mfilter(x3, hline)
    O = paint(I, x4)
    return O


def solve_9f236235(I):
    x1 = compress(I)
    x2 = objects(I, T, F, F)
    x3 = vmirror(x1)
    x4 = valmin(x2, width)
    O = downscale(x3, x4)
    return O


def solve_a699fb00(I):
    x1 = ofcolor(I, ONE)
    x2 = shift(x1, RIGHT)
    x3 = shift(x1, LEFT)
    x4 = intersection(x2, x3)
    O = fill(I, TWO, x4)
    return O


def solve_46442a0e(I):
    x1 = rot90(I)
    x2 = rot180(I)
    x3 = rot270(I)
    x4 = hconcat(I, x1)
    x5 = hconcat(x3, x2)
    O = vconcat(x4, x5)
    return O


def solve_7fe24cdd(I):
    x1 = rot90(I)
    x2 = rot180(I)
    x3 = rot270(I)
    x4 = hconcat(I, x1)
    x5 = hconcat(x3, x2)
    O = vconcat(x4, x5)
    return O


def solve_0ca9ddb6(I):
    x1 = ofcolor(I, ONE)
    x2 = ofcolor(I, TWO)
    x3 = mapply(dneighbors, x1)
    x4 = mapply(ineighbors, x2)
    x5 = fill(I, SEVEN, x3)
    O = fill(x5, FOUR, x4)
    return O


def solve_543a7ed5(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, SIX)
    x3 = mapply(outbox, x2)
    x4 = fill(I, THREE, x3)
    x5 = mapply(delta, x2)
    O = fill(x4, FOUR, x5)
    return O


def solve_0520fde7(I):
    x1 = vmirror(I)
    x2 = lefthalf(x1)
    x3 = righthalf(x1)
    x4 = vmirror(x3)
    x5 = cellwise(x2, x4, ZERO)
    O = replace(x5, ONE, TWO)
    return O


def solve_dae9d2b5(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, FOUR)
    x4 = ofcolor(x2, THREE)
    x5 = combine(x3, x4)
    O = fill(x1, SIX, x5)
    return O


def solve_8d5021e8(I):
    x1 = vmirror(I)
    x2 = hconcat(x1, I)
    x3 = hmirror(x2)
    x4 = vconcat(x2, x3)
    x5 = vconcat(x4, x2)
    O = hmirror(x5)
    return O


def solve_928ad970(I):
    x1 = ofcolor(I, FIVE)
    x2 = subgrid(x1, I)
    x3 = trim(x2)
    x4 = leastcolor(x3)
    x5 = inbox(x1)
    O = fill(I, x4, x5)
    return O


def solve_b60334d2(I):
    x1 = ofcolor(I, FIVE)
    x2 = replace(I, FIVE, ZERO)
    x3 = mapply(dneighbors, x1)
    x4 = mapply(ineighbors, x1)
    x5 = fill(x2, ONE, x3)
    O = fill(x5, FIVE, x4)
    return O


def solve_b94a9452(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = leastcolor(x3)
    x5 = mostcolor(x3)
    O = switch(x3, x4, x5)
    return O


def solve_d037b0a7(I):
    x1 = objects(I, T, F, T)
    x2 = rbind(shoot, DOWN)
    x3 = compose(x2, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x1)
    O = paint(I, x5)
    return O


def solve_d0f5fe59(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = astuple(x2, x2)
    x4 = canvas(ZERO, x3)
    x5 = shoot(ORIGIN, UNITY)
    O = fill(x4, EIGHT, x5)
    return O


def solve_e3497940(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = vmirror(x2)
    x4 = objects(x3, T, F, T)
    x5 = merge(x4)
    O = paint(x1, x5)
    return O


def solve_e9afcf9a(I):
    x1 = astuple(TWO, ONE)
    x2 = crop(I, ORIGIN, x1)
    x3 = hmirror(x2)
    x4 = hconcat(x2, x3)
    x5 = hconcat(x4, x4)
    O = hconcat(x5, x4)
    return O


def solve_48d8fb45(I):
    x1 = objects(I, T, T, T)
    x2 = matcher(size, ONE)
    x3 = extract(x1, x2)
    x4 = lbind(adjacent, x3)
    x5 = extract(x1, x4)
    O = subgrid(x5, I)
    return O


def solve_d406998b(I):
    x1 = vmirror(I)
    x2 = ofcolor(x1, FIVE)
    x3 = compose(even, last)
    x4 = sfilter(x2, x3)
    x5 = fill(x1, THREE, x4)
    O = vmirror(x5)
    return O


def solve_5117e062(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(numcolors, TWO)
    x3 = extract(x1, x2)
    x4 = subgrid(x3, I)
    x5 = mostcolor(x3)
    O = replace(x4, EIGHT, x5)
    return O


def solve_3906de3d(I):
    x1 = rot270(I)
    x2 = rbind(order, identity)
    x3 = switch(x1, ONE, TWO)
    x4 = apply(x2, x3)
    x5 = switch(x4, ONE, TWO)
    O = cmirror(x5)
    return O


def solve_00d62c1b(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    O = fill(I, FOUR, x5)
    return O


def solve_7b6016b9(I):
    x1 = objects(I, T, F, F)
    x2 = rbind(bordering, I)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x3)
    x5 = fill(I, TWO, x4)
    O = replace(x5, ZERO, THREE)
    return O


def solve_67385a82(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, THREE)
    x3 = sizefilter(x2, ONE)
    x4 = difference(x2, x3)
    x5 = merge(x4)
    O = fill(I, EIGHT, x5)
    return O


def solve_a5313dff(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = mfilter(x2, x4)
    O = fill(I, ONE, x5)
    return O


def solve_ea32f347(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, FIVE, FOUR)
    x3 = argmin(x1, size)
    x4 = argmax(x1, size)
    x5 = fill(x2, ONE, x4)
    O = fill(x5, TWO, x3)
    return O


def solve_d631b094(I):
    x1 = palette(I)
    x2 = other(x1, ZERO)
    x3 = ofcolor(I, x2)
    x4 = size(x3)
    x5 = astuple(ONE, x4)
    O = canvas(x2, x5)
    return O


def solve_10fcaaa3(I):
    x1 = leastcolor(I)
    x2 = hconcat(I, I)
    x3 = vconcat(x2, x2)
    x4 = ofcolor(x3, x1)
    x5 = mapply(ineighbors, x4)
    O = underfill(x3, EIGHT, x5)
    return O


def solve_007bbfb7(I):
    x1 = hupscale(I, THREE)
    x2 = vupscale(x1, THREE)
    x3 = hconcat(I, I)
    x4 = hconcat(x3, I)
    x5 = vconcat(x4, x4)
    x6 = vconcat(x5, x4)
    O = cellwise(x2, x6, ZERO)
    return O


def solve_496994bd(I):
    x1 = width(I)
    x2 = height(I)
    x3 = halve(x2)
    x4 = astuple(x3, x1)
    x5 = crop(I, ORIGIN, x4)
    x6 = hmirror(x5)
    O = vconcat(x5, x6)
    return O


def solve_1f876c06(I):
    x1 = fgpartition(I)
    x2 = compose(last, first)
    x3 = power(last, TWO)
    x4 = fork(connect, x2, x3)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x1)
    O = paint(I, x6)
    return O


def solve_05f2a901(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, TWO)
    x3 = first(x2)
    x4 = colorfilter(x1, EIGHT)
    x5 = first(x4)
    x6 = gravitate(x3, x5)
    O = move(I, x3, x6)
    return O


def solve_39a8645d(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = apply(color, x2)
    x4 = mostcommon(x3)
    x5 = matcher(color, x4)
    x6 = extract(x1, x5)
    O = subgrid(x6, I)
    return O


def solve_1b2d62fb(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = intersection(x3, x4)
    x6 = replace(x1, NINE, ZERO)
    O = fill(x6, EIGHT, x5)
    return O


def solve_90c28cc7(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = dedupe(x3)
    x5 = rot90(x4)
    x6 = dedupe(x5)
    O = rot270(x6)
    return O


def solve_b6afb2da(I):
    x1 = objects(I, T, F, F)
    x2 = replace(I, FIVE, TWO)
    x3 = colorfilter(x1, FIVE)
    x4 = mapply(box, x3)
    x5 = fill(x2, FOUR, x4)
    x6 = mapply(corners, x3)
    O = fill(x5, ONE, x6)
    return O


def solve_b9b7f026(I):
    x1 = objects(I, T, F, F)
    x2 = argmin(x1, size)
    x3 = rbind(adjacent, x2)
    x4 = remove(x2, x1)
    x5 = extract(x4, x3)
    x6 = color(x5)
    O = canvas(x6, UNITY)
    return O


def solve_ba97ae07(I):
    x1 = objects(I, T, F, T)
    x2 = totuple(x1)
    x3 = apply(color, x2)
    x4 = mostcommon(x3)
    x5 = ofcolor(I, x4)
    x6 = backdrop(x5)
    O = fill(I, x4, x6)
    return O


def solve_c9f8e694(I):
    x1 = height(I)
    x2 = width(I)
    x3 = ofcolor(I, ZERO)
    x4 = astuple(x1, ONE)
    x5 = crop(I, ORIGIN, x4)
    x6 = hupscale(x5, x2)
    O = fill(x6, ZERO, x3)
    return O


def solve_d23f8c26(I):
    x1 = asindices(I)
    x2 = width(I)
    x3 = halve(x2)
    x4 = matcher(last, x3)
    x5 = compose(flip, x4)
    x6 = sfilter(x1, x5)
    O = fill(I, ZERO, x6)
    return O


def solve_d5d6de2d(I):
    x1 = objects(I, T, F, T)
    x2 = sfilter(x1, square)
    x3 = difference(x1, x2)
    x4 = compose(backdrop, inbox)
    x5 = mapply(x4, x3)
    x6 = replace(I, TWO, ZERO)
    O = fill(x6, THREE, x5)
    return O


def solve_dbc1a6ce(I):
    x1 = ofcolor(I, ONE)
    x2 = product(x1, x1)
    x3 = fork(connect, first, last)
    x4 = apply(x3, x2)
    x5 = fork(either, vline, hline)
    x6 = mfilter(x4, x5)
    O = underfill(I, EIGHT, x6)
    return O


def solve_ded97339(I):
    x1 = ofcolor(I, EIGHT)
    x2 = product(x1, x1)
    x3 = fork(connect, first, last)
    x4 = apply(x3, x2)
    x5 = fork(either, vline, hline)
    x6 = mfilter(x4, x5)
    O = underfill(I, EIGHT, x6)
    return O


def solve_ea786f4a(I):
    x1 = width(I)
    x2 = shoot(ORIGIN, UNITY)
    x3 = decrement(x1)
    x4 = tojvec(x3)
    x5 = shoot(x4, DOWN_LEFT)
    x6 = combine(x2, x5)
    O = fill(I, ZERO, x6)
    return O


def solve_08ed6ac7(I):
    x1 = objects(I, T, F, T)
    x2 = totuple(x1)
    x3 = order(x1, height)
    x4 = size(x2)
    x5 = interval(x4, ZERO, NEG_ONE)
    x6 = mpapply(recolor, x5, x3)
    O = paint(I, x6)
    return O


def solve_40853293(I):
    x1 = partition(I)
    x2 = fork(recolor, color, backdrop)
    x3 = apply(x2, x1)
    x4 = mfilter(x3, hline)
    x5 = mfilter(x3, vline)
    x6 = paint(I, x4)
    O = paint(x6, x5)
    return O


def solve_5521c0d9(I):
    x1 = objects(I, T, F, T)
    x2 = merge(x1)
    x3 = cover(I, x2)
    x4 = chain(toivec, invert, height)
    x5 = fork(shift, identity, x4)
    x6 = mapply(x5, x1)
    O = paint(x3, x6)
    return O


def solve_f8ff0b80(I):
    x1 = objects(I, T, T, T)
    x2 = order(x1, size)
    x3 = apply(color, x2)
    x4 = rbind(canvas, UNITY)
    x5 = apply(x4, x3)
    x6 = merge(x5)
    O = hmirror(x6)
    return O


def solve_85c4e7cd(I):
    x1 = objects(I, T, F, F)
    x2 = compose(invert, size)
    x3 = order(x1, size)
    x4 = order(x1, x2)
    x5 = apply(color, x4)
    x6 = mpapply(recolor, x5, x3)
    O = paint(I, x6)
    return O


def solve_d2abd087(I):
    x1 = objects(I, T, F, T)
    x2 = matcher(size, SIX)
    x3 = compose(flip, x2)
    x4 = mfilter(x1, x2)
    x5 = mfilter(x1, x3)
    x6 = fill(I, TWO, x4)
    O = fill(x6, ONE, x5)
    return O


def solve_017c7c7b(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = equality(x1, x2)
    x4 = crop(I, TWO_BY_ZERO, THREE_BY_THREE)
    x5 = branch(x3, x2, x4)
    x6 = vconcat(I, x5)
    O = replace(x6, ONE, TWO)
    return O


def solve_363442ee(I):
    x1 = ofcolor(I, ONE)
    x2 = crop(I, ORIGIN, THREE_BY_THREE)
    x3 = asobject(x2)
    x4 = lbind(shift, x3)
    x5 = compose(x4, decrement)
    x6 = mapply(x5, x1)
    O = paint(I, x6)
    return O


def solve_5168d44c(I):
    x1 = ofcolor(I, THREE)
    x2 = height(x1)
    x3 = equality(x2, ONE)
    x4 = branch(x3, ZERO_BY_TWO, TWO_BY_ZERO)
    x5 = ofcolor(I, TWO)
    x6 = recolor(TWO, x5)
    O = move(I, x6, x4)
    return O


def solve_e9614598(I):
    x1 = ofcolor(I, ONE)
    x2 = fork(add, first, last)
    x3 = x2(x1)
    x4 = halve(x3)
    x5 = dneighbors(x4)
    x6 = insert(x4, x5)
    O = fill(I, THREE, x6)
    return O


def solve_d9fac9be(I):
    x1 = palette(I)
    x2 = objects(I, T, F, T)
    x3 = argmax(x2, size)
    x4 = color(x3)
    x5 = remove(ZERO, x1)
    x6 = other(x5, x4)
    O = canvas(x6, UNITY)
    return O


def solve_e50d258f(I):
    x1 = width(I)
    x2 = astuple(NINE, x1)
    x3 = canvas(ZERO, x2)
    x4 = vconcat(I, x3)
    x5 = objects(x4, F, F, T)
    x6 = rbind(colorcount, TWO)
    x7 = argmax(x5, x6)
    O = subgrid(x7, I)
    return O


def solve_810b9b61(I):
    x1 = objects(I, T, T, T)
    x2 = apply(toindices, x1)
    x3 = fork(either, vline, hline)
    x4 = sfilter(x2, x3)
    x5 = difference(x2, x4)
    x6 = fork(equality, identity, box)
    x7 = mfilter(x5, x6)
    O = fill(I, THREE, x7)
    return O


def solve_54d82841(I):
    x1 = height(I)
    x2 = objects(I, T, F, T)
    x3 = compose(last, center)
    x4 = apply(x3, x2)
    x5 = decrement(x1)
    x6 = lbind(astuple, x5)
    x7 = apply(x6, x4)
    O = fill(I, FOUR, x7)
    return O


def solve_60b61512(I):
    x1 = objects(I, T, T, T)
    x2 = mapply(delta, x1)
    O = fill(I, SEVEN, x2)
    return O


def solve_25d8a9c8(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, F)
    x3 = sizefilter(x2, THREE)
    x4 = mfilter(x3, hline)
    x5 = toindices(x4)
    x6 = difference(x1, x5)
    x7 = fill(I, FIVE, x5)
    O = fill(x7, ZERO, x6)
    return O


def solve_239be575(I):
    x1 = objects(I, F, T, T)
    x2 = lbind(contained, TWO)
    x3 = compose(x2, palette)
    x4 = sfilter(x1, x3)
    x5 = size(x4)
    x6 = greater(x5, ONE)
    x7 = branch(x6, ZERO, EIGHT)
    O = canvas(x7, UNITY)
    return O


def solve_67a423a3(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = colorfilter(x2, x1)
    x4 = merge(x3)
    x5 = delta(x4)
    x6 = first(x5)
    x7 = neighbors(x6)
    O = fill(I, FOUR, x7)
    return O


def solve_5c0a986e(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, ONE)
    x3 = lrcorner(x1)
    x4 = ulcorner(x2)
    x5 = shoot(x3, UNITY)
    x6 = shoot(x4, NEG_UNITY)
    x7 = fill(I, TWO, x5)
    O = fill(x7, ONE, x6)
    return O


def solve_6430c8c4(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = astuple(FOUR, FOUR)
    x4 = ofcolor(x1, ZERO)
    x5 = ofcolor(x2, ZERO)
    x6 = intersection(x4, x5)
    x7 = canvas(ZERO, x3)
    O = fill(x7, THREE, x6)
    return O


def solve_94f9d214(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = astuple(FOUR, FOUR)
    x6 = canvas(ZERO, x5)
    x7 = intersection(x3, x4)
    O = fill(x6, TWO, x7)
    return O


def solve_a1570a43(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, THREE)
    x3 = recolor(TWO, x1)
    x4 = ulcorner(x2)
    x5 = ulcorner(x1)
    x6 = subtract(x4, x5)
    x7 = increment(x6)
    O = move(I, x3, x7)
    return O


def solve_ce4f8723(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = intersection(x3, x4)
    x6 = astuple(FOUR, FOUR)
    x7 = canvas(THREE, x6)
    O = fill(x7, ZERO, x5)
    return O


def solve_d13f3404(I):
    x1 = objects(I, T, F, T)
    x2 = rbind(shoot, UNITY)
    x3 = compose(x2, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x1)
    x6 = astuple(SIX, SIX)
    x7 = canvas(ZERO, x6)
    O = paint(x7, x5)
    return O


def solve_dc433765(I):
    x1 = ofcolor(I, THREE)
    x2 = ofcolor(I, FOUR)
    x3 = first(x1)
    x4 = first(x2)
    x5 = subtract(x4, x3)
    x6 = sign(x5)
    x7 = recolor(THREE, x1)
    O = move(I, x7, x6)
    return O


def solve_f2829549(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = intersection(x3, x4)
    x6 = shape(x1)
    x7 = canvas(ZERO, x6)
    O = fill(x7, THREE, x5)
    return O


def solve_fafffa47(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = shape(x2)
    x4 = ofcolor(x1, ZERO)
    x5 = ofcolor(x2, ZERO)
    x6 = intersection(x4, x5)
    x7 = canvas(ZERO, x3)
    O = fill(x7, TWO, x6)
    return O


def solve_fcb5c309(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = colorfilter(x2, x1)
    x4 = difference(x2, x3)
    x5 = argmax(x4, size)
    x6 = color(x5)
    x7 = subgrid(x5, I)
    O = replace(x7, x6, x1)
    return O


def solve_ff805c23(I):
    x1 = hmirror(I)
    x2 = vmirror(I)
    x3 = ofcolor(I, ONE)
    x4 = subgrid(x3, x1)
    x5 = subgrid(x3, x2)
    x6 = palette(x4)
    x7 = contained(ONE, x6)
    O = branch(x7, x5, x4)
    return O


def solve_e76a88a6(I):
    x1 = objects(I, F, F, T)
    x2 = argmax(x1, numcolors)
    x3 = normalize(x2)
    x4 = remove(x2, x1)
    x5 = apply(ulcorner, x4)
    x6 = lbind(shift, x3)
    x7 = mapply(x6, x5)
    O = paint(I, x7)
    return O


def solve_7c008303(I):
    x1 = ofcolor(I, THREE)
    x2 = subgrid(x1, I)
    x3 = ofcolor(x2, ZERO)
    x4 = replace(I, THREE, ZERO)
    x5 = replace(x4, EIGHT, ZERO)
    x6 = compress(x5)
    x7 = upscale(x6, THREE)
    O = fill(x7, ZERO, x3)
    return O


def solve_7f4411dc(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = rbind(difference, x2)
    x4 = rbind(greater, TWO)
    x5 = chain(x4, size, x3)
    x6 = compose(x5, dneighbors)
    x7 = sfilter(x2, x6)
    O = fill(I, ZERO, x7)
    return O


def solve_b230c067(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = apply(normalize, x2)
    x4 = leastcommon(x3)
    x5 = matcher(normalize, x4)
    x6 = extract(x1, x5)
    x7 = replace(I, EIGHT, ONE)
    O = fill(x7, TWO, x6)
    return O


def solve_e8593010(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = sizefilter(x1, TWO)
    x4 = merge(x2)
    x5 = fill(I, THREE, x4)
    x6 = merge(x3)
    x7 = fill(x5, TWO, x6)
    O = replace(x7, ZERO, ONE)
    return O


def solve_6d75e8bb(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = ulcorner(x2)
    x4 = subgrid(x2, I)
    x5 = replace(x4, ZERO, TWO)
    x6 = asobject(x5)
    x7 = shift(x6, x3)
    O = paint(I, x7)
    return O


def solve_3f7978a0(I):
    x1 = fgpartition(I)
    x2 = matcher(color, FIVE)
    x3 = extract(x1, x2)
    x4 = ulcorner(x3)
    x5 = subtract(x4, DOWN)
    x6 = shape(x3)
    x7 = add(x6, TWO_BY_ZERO)
    O = crop(I, x5, x7)
    return O


def solve_1190e5a7(I):
    x1 = mostcolor(I)
    x2 = frontiers(I)
    x3 = sfilter(x2, vline)
    x4 = difference(x2, x3)
    x5 = astuple(x4, x3)
    x6 = apply(size, x5)
    x7 = increment(x6)
    O = canvas(x1, x7)
    return O


def solve_6e02f1e3(I):
    x1 = numcolors(I)
    x2 = canvas(ZERO, THREE_BY_THREE)
    x3 = equality(x1, THREE)
    x4 = equality(x1, TWO)
    x5 = branch(x3, TWO_BY_ZERO, ORIGIN)
    x6 = branch(x4, TWO_BY_TWO, ZERO_BY_TWO)
    x7 = connect(x5, x6)
    O = fill(x2, FIVE, x7)
    return O


def solve_a61f2674(I):
    x1 = objects(I, T, F, T)
    x2 = argmax(x1, size)
    x3 = argmin(x1, size)
    x4 = replace(I, FIVE, ZERO)
    x5 = recolor(ONE, x2)
    x6 = recolor(TWO, x3)
    x7 = combine(x5, x6)
    O = paint(x4, x7)
    return O


def solve_fcc82909(I):
    x1 = objects(I, F, T, T)
    x2 = rbind(add, DOWN)
    x3 = compose(x2, llcorner)
    x4 = compose(toivec, numcolors)
    x5 = fork(add, lrcorner, x4)
    x6 = fork(astuple, x3, x5)
    x7 = compose(box, x6)
    x8 = mapply(x7, x1)
    O = fill(I, THREE, x8)
    return O


def solve_72ca375d(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = rbind(subgrid, I)
    x4 = apply(x3, x2)
    x5 = apply(vmirror, x4)
    x6 = papply(equality, x4, x5)
    x7 = pair(x4, x6)
    x8 = extract(x7, last)
    O = first(x8)
    return O


def solve_253bf280(I):
    x1 = ofcolor(I, EIGHT)
    x2 = prapply(connect, x1, x1)
    x3 = rbind(greater, ONE)
    x4 = compose(x3, size)
    x5 = sfilter(x2, x4)
    x6 = fork(either, vline, hline)
    x7 = mfilter(x5, x6)
    x8 = fill(I, THREE, x7)
    O = fill(x8, EIGHT, x1)
    return O


def solve_694f12f3(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, FOUR)
    x3 = compose(backdrop, inbox)
    x4 = argmin(x2, size)
    x5 = argmax(x2, size)
    x6 = x3(x4)
    x7 = x3(x5)
    x8 = fill(I, ONE, x6)
    O = fill(x8, TWO, x7)
    return O


def solve_1f642eb9(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = difference(x1, x2)
    x4 = first(x3)
    x5 = rbind(gravitate, x4)
    x6 = compose(crement, x5)
    x7 = fork(shift, identity, x6)
    x8 = mapply(x7, x2)
    O = paint(I, x8)
    return O


def solve_31aa019c(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = first(x2)
    x4 = neighbors(x3)
    x5 = astuple(TEN, TEN)
    x6 = canvas(ZERO, x5)
    x7 = initset(x3)
    x8 = fill(x6, x1, x7)
    O = fill(x8, TWO, x4)
    return O


def solve_27a28665(I):
    x1 = objects(I, T, F, F)
    x2 = valmax(x1, size)
    x3 = equality(x2, ONE)
    x4 = equality(x2, FOUR)
    x5 = equality(x2, FIVE)
    x6 = branch(x3, TWO, ONE)
    x7 = branch(x4, THREE, x6)
    x8 = branch(x5, SIX, x7)
    O = canvas(x8, UNITY)
    return O


def solve_7ddcd7ec(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = difference(x1, x2)
    x4 = first(x3)
    x5 = color(x4)
    x6 = lbind(position, x4)
    x7 = fork(shoot, center, x6)
    x8 = mapply(x7, x2)
    O = fill(I, x5, x8)
    return O


def solve_3bd67248(I):
    x1 = height(I)
    x2 = decrement(x1)
    x3 = decrement(x2)
    x4 = astuple(x3, ONE)
    x5 = astuple(x2, ONE)
    x6 = shoot(x4, UP_RIGHT)
    x7 = shoot(x5, RIGHT)
    x8 = fill(I, TWO, x6)
    O = fill(x8, FOUR, x7)
    return O


def solve_73251a56(I):
    x1 = dmirror(I)
    x2 = papply(pair, I, x1)
    x3 = lbind(apply, maximum)
    x4 = apply(x3, x2)
    x5 = mostcolor(x4)
    x6 = replace(x4, ZERO, x5)
    x7 = index(x6, ORIGIN)
    x8 = shoot(ORIGIN, UNITY)
    O = fill(x6, x7, x8)
    return O


def solve_25d487eb(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = ofcolor(I, x1)
    x4 = center(x3)
    x5 = merge(x2)
    x6 = center(x5)
    x7 = subtract(x6, x4)
    x8 = shoot(x4, x7)
    O = underfill(I, x1, x8)
    return O


def solve_8f2ea7aa(I):
    x1 = objects(I, T, F, T)
    x2 = merge(x1)
    x3 = subgrid(x2, I)
    x4 = upscale(x3, THREE)
    x5 = hconcat(x3, x3)
    x6 = hconcat(x5, x3)
    x7 = vconcat(x6, x6)
    x8 = vconcat(x7, x6)
    O = cellwise(x4, x8, ZERO)
    return O


def solve_b8825c91(I):
    x1 = replace(I, FOUR, ZERO)
    x2 = dmirror(x1)
    x3 = papply(pair, x1, x2)
    x4 = lbind(apply, maximum)
    x5 = apply(x4, x3)
    x6 = cmirror(x5)
    x7 = papply(pair, x5, x6)
    x8 = apply(x4, x7)
    O = cmirror(x8)
    return O


def solve_cce03e0d(I):
    x1 = upscale(I, THREE)
    x2 = hconcat(I, I)
    x3 = hconcat(x2, I)
    x4 = vconcat(x3, x3)
    x5 = vconcat(x4, x3)
    x6 = ofcolor(x1, ZERO)
    x7 = ofcolor(x1, ONE)
    x8 = combine(x6, x7)
    O = fill(x5, ZERO, x8)
    return O


def solve_d364b489(I):
    x1 = ofcolor(I, ONE)
    x2 = shift(x1, DOWN)
    x3 = fill(I, EIGHT, x2)
    x4 = shift(x1, UP)
    x5 = fill(x3, TWO, x4)
    x6 = shift(x1, RIGHT)
    x7 = fill(x5, SIX, x6)
    x8 = shift(x1, LEFT)
    O = fill(x7, SEVEN, x8)
    return O


def solve_a5f85a15(I):
    x1 = objects(I, T, T, T)
    x2 = interval(ONE, NINE, ONE)
    x3 = apply(double, x2)
    x4 = apply(decrement, x3)
    x5 = papply(astuple, x4, x4)
    x6 = apply(ulcorner, x1)
    x7 = lbind(shift, x5)
    x8 = mapply(x7, x6)
    O = fill(I, FOUR, x8)
    return O


def solve_3ac3eb23(I):
    x1 = objects(I, T, F, T)
    x2 = chain(ineighbors, last, first)
    x3 = fork(recolor, color, x2)
    x4 = mapply(x3, x1)
    x5 = paint(I, x4)
    x6 = vsplit(x5, THREE)
    x7 = first(x6)
    x8 = vconcat(x7, x7)
    O = vconcat(x7, x8)
    return O


def solve_444801d8(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, ONE)
    x3 = rbind(toobject, I)
    x4 = chain(leastcolor, x3, delta)
    x5 = rbind(shift, UP)
    x6 = compose(x5, backdrop)
    x7 = fork(recolor, x4, x6)
    x8 = mapply(x7, x2)
    O = underpaint(I, x8)
    return O


def solve_22168020(I):
    x1 = palette(I)
    x2 = remove(ZERO, x1)
    x3 = lbind(ofcolor, I)
    x4 = lbind(prapply, connect)
    x5 = fork(x4, x3, x3)
    x6 = compose(merge, x5)
    x7 = fork(recolor, identity, x6)
    x8 = mapply(x7, x2)
    O = paint(I, x8)
    return O


def solve_6e82a1ae(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(sizefilter, x1)
    x3 = compose(merge, x2)
    x4 = x3(TWO)
    x5 = x3(THREE)
    x6 = x3(FOUR)
    x7 = fill(I, THREE, x4)
    x8 = fill(x7, TWO, x5)
    O = fill(x8, ONE, x6)
    return O


def solve_b2862040(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, NINE)
    x3 = colorfilter(x1, ONE)
    x4 = rbind(bordering, I)
    x5 = compose(flip, x4)
    x6 = mfilter(x2, x5)
    x7 = rbind(adjacent, x6)
    x8 = mfilter(x3, x7)
    O = fill(I, EIGHT, x8)
    return O


def solve_868de0fa(I):
    x1 = objects(I, T, F, F)
    x2 = sfilter(x1, square)
    x3 = compose(even, height)
    x4 = sfilter(x2, x3)
    x5 = difference(x2, x4)
    x6 = merge(x4)
    x7 = merge(x5)
    x8 = fill(I, TWO, x6)
    O = fill(x8, SEVEN, x7)
    return O


def solve_681b3aeb(I):
    x1 = rot270(I)
    x2 = objects(x1, T, F, T)
    x3 = argmax(x2, size)
    x4 = argmin(x2, size)
    x5 = color(x4)
    x6 = canvas(x5, THREE_BY_THREE)
    x7 = normalize(x3)
    x8 = paint(x6, x7)
    O = rot90(x8)
    return O


def solve_8e5a5113(I):
    x1 = crop(I, ORIGIN, THREE_BY_THREE)
    x2 = rot90(x1)
    x3 = rot180(x1)
    x4 = astuple(x2, x3)
    x5 = astuple(FOUR, EIGHT)
    x6 = apply(tojvec, x5)
    x7 = apply(asobject, x4)
    x8 = mpapply(shift, x7, x6)
    O = paint(I, x8)
    return O


def solve_025d127b(I):
    x1 = objects(I, T, F, T)
    x2 = apply(color, x1)
    x3 = merge(x1)
    x4 = lbind(colorfilter, x1)
    x5 = rbind(argmax, rightmost)
    x6 = compose(x5, x4)
    x7 = mapply(x6, x2)
    x8 = difference(x3, x7)
    O = move(I, x8, RIGHT)
    return O


def solve_2281f1f4(I):
    x1 = ofcolor(I, FIVE)
    x2 = product(x1, x1)
    x3 = power(first, TWO)
    x4 = power(last, TWO)
    x5 = fork(astuple, x3, x4)
    x6 = apply(x5, x2)
    x7 = urcorner(x1)
    x8 = remove(x7, x6)
    O = underfill(I, TWO, x8)
    return O


def solve_cf98881b(I):
    x1 = hsplit(I, THREE)
    x2 = first(x1)
    x3 = remove(x2, x1)
    x4 = first(x3)
    x5 = last(x3)
    x6 = ofcolor(x4, NINE)
    x7 = ofcolor(x2, FOUR)
    x8 = fill(x5, NINE, x6)
    O = fill(x8, FOUR, x7)
    return O


def solve_d4f3cd78(I):
    x1 = ofcolor(I, FIVE)
    x2 = delta(x1)
    x3 = fill(I, EIGHT, x2)
    x4 = box(x1)
    x5 = difference(x4, x1)
    x6 = position(x4, x5)
    x7 = first(x5)
    x8 = shoot(x7, x6)
    O = fill(x3, EIGHT, x8)
    return O


def solve_bda2d7a6(I):
    x1 = partition(I)
    x2 = order(x1, size)
    x3 = apply(color, x2)
    x4 = last(x2)
    x5 = remove(x4, x2)
    x6 = repeat(x4, ONE)
    x7 = combine(x6, x5)
    x8 = mpapply(recolor, x3, x7)
    O = paint(I, x8)
    return O


def solve_137eaa0f(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(first, FIVE)
    x3 = rbind(sfilter, x2)
    x4 = chain(invert, center, x3)
    x5 = fork(shift, identity, x4)
    x6 = canvas(ZERO, THREE_BY_THREE)
    x7 = mapply(x5, x1)
    x8 = shift(x7, UNITY)
    O = paint(x6, x8)
    return O


def solve_6455b5f5(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = argmax(x1, size)
    x4 = valmin(x1, size)
    x5 = sizefilter(x2, x4)
    x6 = recolor(ONE, x3)
    x7 = merge(x5)
    x8 = paint(I, x6)
    O = fill(x8, EIGHT, x7)
    return O


def solve_b8cdaf2b(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = shift(x2, UP)
    x4 = ulcorner(x3)
    x5 = urcorner(x3)
    x6 = shoot(x4, NEG_UNITY)
    x7 = shoot(x5, UP_RIGHT)
    x8 = combine(x6, x7)
    O = underfill(I, x1, x8)
    return O


def solve_bd4472b8(I):
    x1 = width(I)
    x2 = astuple(TWO, x1)
    x3 = crop(I, ORIGIN, x2)
    x4 = tophalf(x3)
    x5 = dmirror(x4)
    x6 = hupscale(x5, x1)
    x7 = repeat(x6, TWO)
    x8 = merge(x7)
    O = vconcat(x3, x8)
    return O


def solve_4be741c5(I):
    x1 = portrait(I)
    x2 = branch(x1, dmirror, identity)
    x3 = branch(x1, height, width)
    x4 = x3(I)
    x5 = astuple(ONE, x4)
    x6 = x2(I)
    x7 = crop(x6, ORIGIN, x5)
    x8 = apply(dedupe, x7)
    O = x2(x8)
    return O


def solve_bbc9ae5d(I):
    x1 = width(I)
    x2 = palette(I)
    x3 = halve(x1)
    x4 = vupscale(I, x3)
    x5 = rbind(shoot, UNITY)
    x6 = other(x2, ZERO)
    x7 = ofcolor(x4, x6)
    x8 = mapply(x5, x7)
    O = fill(x4, x6, x8)
    return O


def solve_d90796e8(I):
    x1 = objects(I, F, F, T)
    x2 = sizefilter(x1, TWO)
    x3 = lbind(contained, TWO)
    x4 = compose(x3, palette)
    x5 = mfilter(x2, x4)
    x6 = cover(I, x5)
    x7 = matcher(first, THREE)
    x8 = sfilter(x5, x7)
    O = fill(x6, EIGHT, x8)
    return O


def solve_2c608aff(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = argmax(x2, size)
    x4 = toindices(x3)
    x5 = ofcolor(I, x1)
    x6 = prapply(connect, x4, x5)
    x7 = fork(either, vline, hline)
    x8 = mfilter(x6, x7)
    O = underfill(I, x1, x8)
    return O


def solve_f8b3ba0a(I):
    x1 = compress(I)
    x2 = astuple(THREE, ONE)
    x3 = palette(x1)
    x4 = lbind(colorcount, x1)
    x5 = compose(invert, x4)
    x6 = order(x3, x5)
    x7 = rbind(canvas, UNITY)
    x8 = apply(x7, x6)
    x9 = merge(x8)
    O = crop(x9, DOWN, x2)
    return O


def solve_80af3007(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    x4 = upscale(x3, THREE)
    x5 = hconcat(x3, x3)
    x6 = hconcat(x5, x3)
    x7 = vconcat(x6, x6)
    x8 = vconcat(x7, x6)
    x9 = cellwise(x4, x8, ZERO)
    O = downscale(x9, THREE)
    return O


def solve_83302e8f(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = sfilter(x2, square)
    x4 = difference(x2, x3)
    x5 = merge(x3)
    x6 = recolor(THREE, x5)
    x7 = merge(x4)
    x8 = recolor(FOUR, x7)
    x9 = paint(I, x6)
    O = paint(x9, x8)
    return O


def solve_1fad071e(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, ONE)
    x3 = sizefilter(x2, FOUR)
    x4 = size(x3)
    x5 = subtract(FIVE, x4)
    x6 = astuple(ONE, x4)
    x7 = canvas(ONE, x6)
    x8 = astuple(ONE, x5)
    x9 = canvas(ZERO, x8)
    O = hconcat(x7, x9)
    return O


def solve_11852cab(I):
    x1 = objects(I, T, T, T)
    x2 = merge(x1)
    x3 = hmirror(x2)
    x4 = vmirror(x2)
    x5 = dmirror(x2)
    x6 = cmirror(x2)
    x7 = paint(I, x3)
    x8 = paint(x7, x4)
    x9 = paint(x8, x5)
    O = paint(x9, x6)
    return O


def solve_3428a4f5(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = astuple(SIX, FIVE)
    x4 = ofcolor(x1, TWO)
    x5 = ofcolor(x2, TWO)
    x6 = combine(x4, x5)
    x7 = intersection(x4, x5)
    x8 = difference(x6, x7)
    x9 = canvas(ZERO, x3)
    O = fill(x9, THREE, x8)
    return O


def solve_178fcbfb(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, TWO)
    x3 = mapply(vfrontier, x2)
    x4 = fill(I, TWO, x3)
    x5 = colorfilter(x1, TWO)
    x6 = difference(x1, x5)
    x7 = compose(hfrontier, center)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x6)
    O = paint(x4, x9)
    return O


def solve_3de23699(I):
    x1 = fgpartition(I)
    x2 = sizefilter(x1, FOUR)
    x3 = first(x2)
    x4 = difference(x1, x2)
    x5 = first(x4)
    x6 = color(x3)
    x7 = color(x5)
    x8 = subgrid(x3, I)
    x9 = trim(x8)
    O = replace(x9, x7, x6)
    return O


def solve_54d9e175(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = compose(neighbors, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x2)
    x6 = paint(I, x5)
    x7 = replace(x6, ONE, SIX)
    x8 = replace(x7, TWO, SEVEN)
    x9 = replace(x8, THREE, EIGHT)
    O = replace(x9, FOUR, NINE)
    return O


def solve_5ad4f10b(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = color(x2)
    x4 = subgrid(x2, I)
    x5 = leastcolor(x4)
    x6 = replace(x4, x5, ZERO)
    x7 = replace(x6, x3, x5)
    x8 = height(x7)
    x9 = divide(x8, THREE)
    O = downscale(x7, x9)
    return O


def solve_623ea044(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = center(x2)
    x4 = color(x2)
    x5 = astuple(UNITY, NEG_UNITY)
    x6 = astuple(UP_RIGHT, DOWN_LEFT)
    x7 = combine(x5, x6)
    x8 = lbind(shoot, x3)
    x9 = mapply(x8, x7)
    O = fill(I, x4, x9)
    return O


def solve_6b9890af(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, TWO)
    x3 = argmin(x1, size)
    x4 = subgrid(x2, I)
    x5 = width(x4)
    x6 = divide(x5, THREE)
    x7 = upscale(x3, x6)
    x8 = normalize(x7)
    x9 = shift(x8, UNITY)
    O = paint(x4, x9)
    return O


def solve_794b24be(I):
    x1 = ofcolor(I, ONE)
    x2 = size(x1)
    x3 = decrement(x2)
    x4 = canvas(ZERO, THREE_BY_THREE)
    x5 = tojvec(x3)
    x6 = connect(ORIGIN, x5)
    x7 = equality(x2, FOUR)
    x8 = insert(UNITY, x6)
    x9 = branch(x7, x8, x6)
    O = fill(x4, TWO, x9)
    return O


def solve_88a10436(I):
    x1 = objects(I, F, F, T)
    x2 = colorfilter(x1, FIVE)
    x3 = first(x2)
    x4 = center(x3)
    x5 = difference(x1, x2)
    x6 = first(x5)
    x7 = normalize(x6)
    x8 = shift(x7, x4)
    x9 = shift(x8, NEG_UNITY)
    O = paint(I, x9)
    return O


def solve_88a62173(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = tophalf(x1)
    x4 = tophalf(x2)
    x5 = bottomhalf(x1)
    x6 = bottomhalf(x2)
    x7 = astuple(x3, x4)
    x8 = astuple(x5, x6)
    x9 = combine(x7, x8)
    O = leastcommon(x9)
    return O


def solve_890034e9(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = inbox(x2)
    x4 = recolor(ZERO, x3)
    x5 = occurrences(I, x4)
    x6 = normalize(x2)
    x7 = shift(x6, NEG_UNITY)
    x8 = lbind(shift, x7)
    x9 = mapply(x8, x5)
    O = fill(I, x1, x9)
    return O


def solve_99b1bc43(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = ofcolor(x1, ZERO)
    x4 = ofcolor(x2, ZERO)
    x5 = combine(x3, x4)
    x6 = intersection(x3, x4)
    x7 = difference(x5, x6)
    x8 = shape(x1)
    x9 = canvas(ZERO, x8)
    O = fill(x9, THREE, x7)
    return O


def solve_a9f96cdd(I):
    x1 = ofcolor(I, TWO)
    x2 = replace(I, TWO, ZERO)
    x3 = shift(x1, NEG_UNITY)
    x4 = fill(x2, THREE, x3)
    x5 = shift(x1, UP_RIGHT)
    x6 = fill(x4, SIX, x5)
    x7 = shift(x1, DOWN_LEFT)
    x8 = fill(x6, EIGHT, x7)
    x9 = shift(x1, UNITY)
    O = fill(x8, SEVEN, x9)
    return O


def solve_af902bf9(I):
    x1 = ofcolor(I, FOUR)
    x2 = prapply(connect, x1, x1)
    x3 = fork(either, vline, hline)
    x4 = mfilter(x2, x3)
    x5 = underfill(I, NEG_ONE, x4)
    x6 = objects(x5, F, F, T)
    x7 = compose(backdrop, inbox)
    x8 = mapply(x7, x6)
    x9 = fill(x5, TWO, x8)
    O = replace(x9, NEG_ONE, ZERO)
    return O


def solve_b548a754(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, EIGHT, ZERO)
    x3 = leastcolor(x2)
    x4 = replace(x2, x3, ZERO)
    x5 = leastcolor(x4)
    x6 = merge(x1)
    x7 = backdrop(x6)
    x8 = box(x6)
    x9 = fill(I, x3, x7)
    O = fill(x9, x5, x8)
    return O


def solve_bdad9b1f(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, EIGHT)
    x3 = center(x1)
    x4 = center(x2)
    x5 = hfrontier(x3)
    x6 = vfrontier(x4)
    x7 = intersection(x5, x6)
    x8 = fill(I, TWO, x5)
    x9 = fill(x8, EIGHT, x6)
    O = fill(x9, FOUR, x7)
    return O


def solve_c3e719e8(I):
    x1 = mostcolor(I)
    x2 = hconcat(I, I)
    x3 = upscale(I, THREE)
    x4 = ofcolor(x3, x1)
    x5 = asindices(x3)
    x6 = difference(x5, x4)
    x7 = hconcat(x2, I)
    x8 = vconcat(x7, x7)
    x9 = vconcat(x8, x7)
    O = fill(x9, ZERO, x6)
    return O


def solve_de1cd16c(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, F)
    x3 = sizefilter(x2, ONE)
    x4 = difference(x2, x3)
    x5 = rbind(subgrid, I)
    x6 = apply(x5, x4)
    x7 = rbind(colorcount, x1)
    x8 = argmax(x6, x7)
    x9 = mostcolor(x8)
    O = canvas(x9, UNITY)
    return O


def solve_d8c310e9(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = hperiod(x2)
    x4 = multiply(x3, THREE)
    x5 = tojvec(x3)
    x6 = tojvec(x4)
    x7 = shift(x2, x5)
    x8 = shift(x2, x6)
    x9 = paint(I, x7)
    O = paint(x9, x8)
    return O


def solve_a3325580(I):
    x1 = objects(I, T, F, T)
    x2 = valmax(x1, size)
    x3 = sizefilter(x1, x2)
    x4 = order(x3, leftmost)
    x5 = apply(color, x4)
    x6 = astuple(ONE, x2)
    x7 = rbind(canvas, x6)
    x8 = apply(x7, x5)
    x9 = merge(x8)
    O = dmirror(x9)
    return O


def solve_8eb1be9a(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = interval(NEG_TWO, FOUR, ONE)
    x4 = lbind(shift, x2)
    x5 = height(x2)
    x6 = rbind(multiply, x5)
    x7 = apply(x6, x3)
    x8 = apply(toivec, x7)
    x9 = mapply(x4, x8)
    O = paint(I, x9)
    return O


def solve_321b1fc6(I):
    x1 = objects(I, F, F, T)
    x2 = colorfilter(x1, EIGHT)
    x3 = difference(x1, x2)
    x4 = first(x3)
    x5 = cover(I, x4)
    x6 = normalize(x4)
    x7 = lbind(shift, x6)
    x8 = apply(ulcorner, x2)
    x9 = mapply(x7, x8)
    O = paint(x5, x9)
    return O


def solve_1caeab9d(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, ONE)
    x3 = lowermost(x2)
    x4 = lbind(subtract, x3)
    x5 = chain(toivec, x4, lowermost)
    x6 = fork(shift, identity, x5)
    x7 = merge(x1)
    x8 = cover(I, x7)
    x9 = mapply(x6, x1)
    O = paint(x8, x9)
    return O


def solve_77fdfe62(I):
    x1 = ofcolor(I, EIGHT)
    x2 = subgrid(x1, I)
    x3 = replace(I, EIGHT, ZERO)
    x4 = replace(x3, ONE, ZERO)
    x5 = compress(x4)
    x6 = width(x2)
    x7 = halve(x6)
    x8 = upscale(x5, x7)
    x9 = ofcolor(x2, ZERO)
    O = fill(x8, ZERO, x9)
    return O


def solve_c0f76784(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = sfilter(x2, square)
    x4 = sizefilter(x3, ONE)
    x5 = merge(x4)
    x6 = argmax(x3, size)
    x7 = merge(x3)
    x8 = fill(I, SEVEN, x7)
    x9 = fill(x8, EIGHT, x6)
    O = fill(x9, SIX, x5)
    return O


def solve_1b60fb0c(I):
    x1 = rot90(I)
    x2 = ofcolor(I, ONE)
    x3 = ofcolor(x1, ONE)
    x4 = neighbors(ORIGIN)
    x5 = mapply(neighbors, x4)
    x6 = lbind(shift, x3)
    x7 = apply(x6, x5)
    x8 = lbind(intersection, x2)
    x9 = compose(size, x8)
    x10 = argmax(x7, x9)
    O = underfill(I, TWO, x10)
    return O


def solve_ddf7fa4f(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = colorfilter(x1, FIVE)
    x4 = product(x2, x3)
    x5 = fork(vmatching, first, last)
    x6 = sfilter(x4, x5)
    x7 = compose(color, first)
    x8 = fork(recolor, x7, last)
    x9 = mapply(x8, x6)
    O = paint(I, x9)
    return O


def solve_47c1f68c(I):
    x1 = leastcolor(I)
    x2 = vmirror(I)
    x3 = objects(I, T, T, T)
    x4 = merge(x3)
    x5 = mostcolor(x4)
    x6 = cellwise(I, x2, x1)
    x7 = hmirror(x6)
    x8 = cellwise(x6, x7, x1)
    x9 = compress(x8)
    O = replace(x9, x1, x5)
    return O


def solve_6c434453(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, EIGHT)
    x3 = dneighbors(UNITY)
    x4 = insert(UNITY, x3)
    x5 = merge(x2)
    x6 = cover(I, x5)
    x7 = apply(ulcorner, x2)
    x8 = lbind(shift, x4)
    x9 = mapply(x8, x7)
    O = fill(x6, TWO, x9)
    return O


def solve_23581191(I):
    x1 = objects(I, T, T, T)
    x2 = fork(combine, vfrontier, hfrontier)
    x3 = compose(x2, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x1)
    x6 = paint(I, x5)
    x7 = fork(intersection, first, last)
    x8 = apply(x3, x1)
    x9 = x7(x8)
    O = fill(x6, TWO, x9)
    return O


def solve_c8cbb738(I):
    x1 = mostcolor(I)
    x2 = fgpartition(I)
    x3 = valmax(x2, shape)
    x4 = canvas(x1, x3)
    x5 = apply(normalize, x2)
    x6 = lbind(subtract, x3)
    x7 = chain(halve, x6, shape)
    x8 = fork(shift, identity, x7)
    x9 = mapply(x8, x5)
    O = paint(x4, x9)
    return O


def solve_3eda0437(I):
    x1 = interval(TWO, TEN, ONE)
    x2 = prapply(astuple, x1, x1)
    x3 = lbind(canvas, ZERO)
    x4 = lbind(occurrences, I)
    x5 = lbind(lbind, shift)
    x6 = fork(apply, x5, x4)
    x7 = chain(x6, asobject, x3)
    x8 = mapply(x7, x2)
    x9 = argmax(x8, size)
    O = fill(I, SIX, x9)
    return O


def solve_dc0a314f(I):
    x1 = ofcolor(I, THREE)
    x2 = replace(I, THREE, ZERO)
    x3 = dmirror(x2)
    x4 = papply(pair, x2, x3)
    x5 = lbind(apply, maximum)
    x6 = apply(x5, x4)
    x7 = cmirror(x6)
    x8 = papply(pair, x6, x7)
    x9 = apply(x5, x8)
    O = subgrid(x1, x9)
    return O


def solve_d4469b4b(I):
    x1 = palette(I)
    x2 = other(x1, ZERO)
    x3 = equality(x2, ONE)
    x4 = equality(x2, TWO)
    x5 = branch(x3, UNITY, TWO_BY_TWO)
    x6 = branch(x4, RIGHT, x5)
    x7 = fork(combine, vfrontier, hfrontier)
    x8 = x7(x6)
    x9 = canvas(ZERO, THREE_BY_THREE)
    O = fill(x9, FIVE, x8)
    return O


def solve_6ecd11f4(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, size)
    x3 = argmin(x1, size)
    x4 = subgrid(x2, I)
    x5 = subgrid(x3, I)
    x6 = width(x4)
    x7 = width(x5)
    x8 = divide(x6, x7)
    x9 = downscale(x4, x8)
    x10 = ofcolor(x9, ZERO)
    O = fill(x5, ZERO, x10)
    return O


def solve_760b3cac(I):
    x1 = ofcolor(I, FOUR)
    x2 = ofcolor(I, EIGHT)
    x3 = ulcorner(x1)
    x4 = index(I, x3)
    x5 = equality(x4, FOUR)
    x6 = branch(x5, NEG_ONE, ONE)
    x7 = multiply(x6, THREE)
    x8 = tojvec(x7)
    x9 = vmirror(x2)
    x10 = shift(x9, x8)
    O = fill(I, EIGHT, x10)
    return O


def solve_c444b776(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = argmin(x2, size)
    x4 = backdrop(x3)
    x5 = toobject(x4, I)
    x6 = normalize(x5)
    x7 = lbind(shift, x6)
    x8 = compose(x7, ulcorner)
    x9 = mapply(x8, x2)
    O = paint(I, x9)
    return O


def solve_d4a91cb9(I):
    x1 = ofcolor(I, EIGHT)
    x2 = ofcolor(I, TWO)
    x3 = first(x1)
    x4 = first(x2)
    x5 = last(x3)
    x6 = first(x4)
    x7 = astuple(x6, x5)
    x8 = connect(x7, x3)
    x9 = connect(x7, x4)
    x10 = combine(x8, x9)
    O = underfill(I, FOUR, x10)
    return O


def solve_eb281b96(I):
    x1 = height(I)
    x2 = width(I)
    x3 = decrement(x1)
    x4 = astuple(x3, x2)
    x5 = crop(I, ORIGIN, x4)
    x6 = hmirror(x5)
    x7 = vconcat(I, x6)
    x8 = double(x3)
    x9 = astuple(x8, x2)
    x10 = crop(x7, DOWN, x9)
    O = vconcat(x7, x10)
    return O


def solve_ff28f65a(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = double(x2)
    x4 = interval(ZERO, x3, TWO)
    x5 = apply(tojvec, x4)
    x6 = astuple(ONE, NINE)
    x7 = canvas(ZERO, x6)
    x8 = fill(x7, ONE, x5)
    x9 = hsplit(x8, THREE)
    O = merge(x9)
    return O


def solve_7e0986d6(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = replace(I, x1, ZERO)
    x4 = leastcolor(x3)
    x5 = rbind(colorcount, x4)
    x6 = chain(positive, decrement, x5)
    x7 = rbind(toobject, x3)
    x8 = chain(x6, x7, dneighbors)
    x9 = sfilter(x2, x8)
    O = fill(x3, x4, x9)
    return O


def solve_09629e4f(I):
    x1 = objects(I, F, T, T)
    x2 = argmin(x1, numcolors)
    x3 = normalize(x2)
    x4 = upscale(x3, FOUR)
    x5 = paint(I, x4)
    x6 = ofcolor(I, FIVE)
    O = fill(x5, FIVE, x6)
    return O


def solve_a85d4709(I):
    x1 = ofcolor(I, FIVE)
    x2 = lbind(matcher, last)
    x3 = lbind(sfilter, x1)
    x4 = lbind(mapply, hfrontier)
    x5 = chain(x4, x3, x2)
    x6 = x5(ZERO)
    x7 = x5(TWO)
    x8 = x5(ONE)
    x9 = fill(I, TWO, x6)
    x10 = fill(x9, THREE, x7)
    O = fill(x10, FOUR, x8)
    return O


def solve_feca6190(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = multiply(x2, FIVE)
    x4 = astuple(x3, x3)
    x5 = canvas(ZERO, x4)
    x6 = rbind(shoot, UNITY)
    x7 = compose(x6, center)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x1)
    x10 = paint(x5, x9)
    O = hmirror(x10)
    return O


def solve_a68b268e(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = lefthalf(x1)
    x4 = righthalf(x1)
    x5 = lefthalf(x2)
    x6 = righthalf(x2)
    x7 = ofcolor(x4, FOUR)
    x8 = ofcolor(x3, SEVEN)
    x9 = ofcolor(x5, EIGHT)
    x10 = fill(x6, EIGHT, x9)
    x11 = fill(x10, FOUR, x7)
    O = fill(x11, SEVEN, x8)
    return O


def solve_beb8660c(I):
    x1 = shape(I)
    x2 = objects(I, T, F, T)
    x3 = compose(invert, size)
    x4 = order(x2, x3)
    x5 = apply(normalize, x4)
    x6 = size(x5)
    x7 = interval(ZERO, x6, ONE)
    x8 = apply(toivec, x7)
    x9 = mpapply(shift, x5, x8)
    x10 = canvas(ZERO, x1)
    x11 = paint(x10, x9)
    O = rot180(x11)
    return O


def solve_913fb3ed(I):
    x1 = ofcolor(I, THREE)
    x2 = ofcolor(I, EIGHT)
    x3 = ofcolor(I, TWO)
    x4 = mapply(neighbors, x1)
    x5 = mapply(neighbors, x2)
    x6 = mapply(neighbors, x3)
    x7 = fill(I, SIX, x4)
    x8 = fill(x7, FOUR, x5)
    O = fill(x8, ONE, x6)
    return O


def solve_0962bcdd(I):
    x1 = leastcolor(I)
    x2 = replace(I, ZERO, x1)
    x3 = leastcolor(x2)
    x4 = ofcolor(I, x3)
    x5 = mapply(dneighbors, x4)
    x6 = fill(I, x3, x5)
    x7 = objects(x6, F, T, T)
    x8 = fork(connect, ulcorner, lrcorner)
    x9 = fork(connect, llcorner, urcorner)
    x10 = fork(combine, x8, x9)
    x11 = mapply(x10, x7)
    O = fill(x6, x1, x11)
    return O


def solve_3631a71a(I):
    x1 = shape(I)
    x2 = replace(I, NINE, ZERO)
    x3 = lbind(apply, maximum)
    x4 = dmirror(x2)
    x5 = papply(pair, x2, x4)
    x6 = apply(x3, x5)
    x7 = subtract(x1, TWO_BY_TWO)
    x8 = crop(x6, TWO_BY_TWO, x7)
    x9 = vmirror(x8)
    x10 = objects(x9, T, F, T)
    x11 = merge(x10)
    x12 = shift(x11, TWO_BY_TWO)
    O = paint(x6, x12)
    return O


def solve_05269061(I):
    x1 = objects(I, T, T, T)
    x2 = neighbors(ORIGIN)
    x3 = mapply(neighbors, x2)
    x4 = rbind(multiply, THREE)
    x5 = apply(x4, x3)
    x6 = merge(x1)
    x7 = lbind(shift, x6)
    x8 = mapply(x7, x5)
    x9 = shift(x8, UP_RIGHT)
    x10 = shift(x8, DOWN_LEFT)
    x11 = paint(I, x8)
    x12 = paint(x11, x9)
    O = paint(x12, x10)
    return O


def solve_95990924(I):
    x1 = objects(I, T, F, T)
    x2 = apply(outbox, x1)
    x3 = apply(ulcorner, x2)
    x4 = apply(urcorner, x2)
    x5 = apply(llcorner, x2)
    x6 = apply(lrcorner, x2)
    x7 = fill(I, ONE, x3)
    x8 = fill(x7, TWO, x4)
    x9 = fill(x8, THREE, x5)
    O = fill(x9, FOUR, x6)
    return O


def solve_e509e548(I):
    x1 = objects(I, T, F, T)
    x2 = rbind(subgrid, I)
    x3 = chain(palette, trim, x2)
    x4 = lbind(contained, THREE)
    x5 = compose(x4, x3)
    x6 = fork(add, height, width)
    x7 = compose(decrement, x6)
    x8 = fork(equality, size, x7)
    x9 = mfilter(x1, x5)
    x10 = mfilter(x1, x8)
    x11 = replace(I, THREE, SIX)
    x12 = fill(x11, TWO, x9)
    O = fill(x12, ONE, x10)
    return O


def solve_d43fd935(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, THREE)
    x3 = sizefilter(x1, ONE)
    x4 = rbind(vmatching, x2)
    x5 = rbind(hmatching, x2)
    x6 = fork(either, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = rbind(gravitate, x2)
    x9 = fork(add, center, x8)
    x10 = fork(connect, center, x9)
    x11 = fork(recolor, color, x10)
    x12 = mapply(x11, x7)
    O = paint(I, x12)
    return O


def solve_db3e9e38(I):
    x1 = ofcolor(I, SEVEN)
    x2 = lrcorner(x1)
    x3 = shoot(x2, UP_RIGHT)
    x4 = shoot(x2, NEG_UNITY)
    x5 = combine(x3, x4)
    x6 = rbind(shoot, UP)
    x7 = mapply(x6, x5)
    x8 = last(x2)
    x9 = rbind(subtract, x8)
    x10 = chain(even, x9, last)
    x11 = fill(I, EIGHT, x7)
    x12 = sfilter(x7, x10)
    O = fill(x11, SEVEN, x12)
    return O


def solve_e73095fd(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = fork(equality, toindices, backdrop)
    x4 = sfilter(x2, x3)
    x5 = lbind(mapply, dneighbors)
    x6 = chain(x5, corners, outbox)
    x7 = fork(difference, x6, outbox)
    x8 = ofcolor(I, FIVE)
    x9 = rbind(intersection, x8)
    x10 = matcher(size, ZERO)
    x11 = chain(x10, x9, x7)
    x12 = mfilter(x4, x11)
    O = fill(I, FOUR, x12)
    return O


def solve_1bfc4729(I):
    x1 = asindices(I)
    x2 = tophalf(I)
    x3 = bottomhalf(I)
    x4 = leastcolor(x2)
    x5 = leastcolor(x3)
    x6 = hfrontier(TWO_BY_ZERO)
    x7 = box(x1)
    x8 = combine(x6, x7)
    x9 = fill(x2, x4, x8)
    x10 = hmirror(x9)
    x11 = replace(x10, x4, x5)
    O = vconcat(x9, x11)
    return O


def solve_93b581b8(I):
    x1 = fgpartition(I)
    x2 = chain(cmirror, dmirror, merge)
    x3 = x2(x1)
    x4 = upscale(x3, THREE)
    x5 = astuple(NEG_TWO, NEG_TWO)
    x6 = shift(x4, x5)
    x7 = underpaint(I, x6)
    x8 = toindices(x3)
    x9 = fork(combine, hfrontier, vfrontier)
    x10 = mapply(x9, x8)
    x11 = difference(x10, x8)
    O = fill(x7, ZERO, x11)
    return O


def solve_9edfc990(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = ofcolor(I, ONE)
    x4 = rbind(adjacent, x3)
    x5 = mfilter(x2, x4)
    x6 = recolor(ONE, x5)
    O = paint(I, x6)
    return O


def solve_a65b410d(I):
    x1 = ofcolor(I, TWO)
    x2 = urcorner(x1)
    x3 = shoot(x2, UP_RIGHT)
    x4 = shoot(x2, DOWN_LEFT)
    x5 = underfill(I, THREE, x3)
    x6 = underfill(x5, ONE, x4)
    x7 = rbind(shoot, LEFT)
    x8 = mapply(x7, x3)
    x9 = mapply(x7, x4)
    x10 = underfill(x6, ONE, x9)
    O = underfill(x10, THREE, x8)
    return O


def solve_7447852a(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = compose(last, center)
    x4 = order(x2, x3)
    x5 = size(x4)
    x6 = interval(ZERO, x5, THREE)
    x7 = rbind(contained, x6)
    x8 = compose(x7, last)
    x9 = interval(ZERO, x5, ONE)
    x10 = pair(x4, x9)
    x11 = sfilter(x10, x8)
    x12 = mapply(first, x11)
    O = fill(I, FOUR, x12)
    return O


def solve_97999447(I):
    x1 = objects(I, T, F, T)
    x2 = apply(toindices, x1)
    x3 = rbind(shoot, RIGHT)
    x4 = compose(x3, center)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x1)
    x7 = paint(I, x6)
    x8 = interval(ZERO, FIVE, ONE)
    x9 = apply(double, x8)
    x10 = apply(increment, x9)
    x11 = apply(tojvec, x10)
    x12 = prapply(shift, x2, x11)
    x13 = merge(x12)
    O = fill(x7, FIVE, x13)
    return O


def solve_91714a58(I):
    x1 = shape(I)
    x2 = asindices(I)
    x3 = objects(I, T, F, T)
    x4 = argmax(x3, size)
    x5 = mostcolor(x4)
    x6 = canvas(ZERO, x1)
    x7 = paint(x6, x4)
    x8 = rbind(toobject, x7)
    x9 = rbind(colorcount, x5)
    x10 = chain(x9, x8, neighbors)
    x11 = lbind(greater, THREE)
    x12 = compose(x11, x10)
    x13 = sfilter(x2, x12)
    O = fill(x7, ZERO, x13)
    return O


def solve_a61ba2ce(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(index, I)
    x3 = matcher(x2, ZERO)
    x4 = lbind(extract, x1)
    x5 = rbind(subgrid, I)
    x6 = lbind(compose, x3)
    x7 = chain(x5, x4, x6)
    x8 = x7(ulcorner)
    x9 = x7(urcorner)
    x10 = x7(llcorner)
    x11 = x7(lrcorner)
    x12 = hconcat(x11, x10)
    x13 = hconcat(x9, x8)
    O = vconcat(x12, x13)
    return O


def solve_8e1813be(I):
    x1 = replace(I, FIVE, ZERO)
    x2 = objects(x1, T, T, T)
    x3 = first(x2)
    x4 = vline(x3)
    x5 = branch(x4, dmirror, identity)
    x6 = x5(x1)
    x7 = objects(x6, T, T, T)
    x8 = order(x7, uppermost)
    x9 = apply(color, x8)
    x10 = dedupe(x9)
    x11 = size(x10)
    x12 = rbind(repeat, x11)
    x13 = apply(x12, x10)
    O = x5(x13)
    return O


def solve_bc1d5164(I):
    x1 = leastcolor(I)
    x2 = crop(I, ORIGIN, THREE_BY_THREE)
    x3 = crop(I, TWO_BY_ZERO, THREE_BY_THREE)
    x4 = tojvec(FOUR)
    x5 = crop(I, x4, THREE_BY_THREE)
    x6 = astuple(TWO, FOUR)
    x7 = crop(I, x6, THREE_BY_THREE)
    x8 = canvas(ZERO, THREE_BY_THREE)
    x9 = rbind(ofcolor, x1)
    x10 = astuple(x2, x3)
    x11 = astuple(x5, x7)
    x12 = combine(x10, x11)
    x13 = mapply(x9, x12)
    O = fill(x8, x1, x13)
    return O


def solve_ce602527(I):
    x1 = vmirror(I)
    x2 = fgpartition(x1)
    x3 = order(x2, size)
    x4 = last(x3)
    x5 = remove(x4, x3)
    x6 = compose(toindices, normalize)
    x7 = rbind(upscale, TWO)
    x8 = chain(toindices, x7, normalize)
    x9 = x6(x4)
    x10 = rbind(intersection, x9)
    x11 = chain(size, x10, x8)
    x12 = argmax(x5, x11)
    x13 = subgrid(x12, x1)
    O = vmirror(x13)
    return O


def solve_5c2c9af4(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = center(x2)
    x4 = ulcorner(x2)
    x5 = subtract(x3, x4)
    x6 = multiply(NEG_ONE, NINE)
    x7 = interval(ZERO, NINE, ONE)
    x8 = interval(ZERO, x6, NEG_ONE)
    x9 = lbind(multiply, x5)
    x10 = apply(x9, x7)
    x11 = apply(x9, x8)
    x12 = pair(x10, x11)
    x13 = mapply(box, x12)
    x14 = shift(x13, x3)
    O = fill(I, x1, x14)
    return O


def solve_75b8110e(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = tophalf(x1)
    x4 = bottomhalf(x1)
    x5 = tophalf(x2)
    x6 = bottomhalf(x2)
    x7 = rbind(ofcolor, ZERO)
    x8 = fork(difference, asindices, x7)
    x9 = fork(toobject, x8, identity)
    x10 = x9(x5)
    x11 = x9(x4)
    x12 = x9(x6)
    x13 = paint(x3, x12)
    x14 = paint(x13, x11)
    O = paint(x14, x10)
    return O


def solve_941d9a10(I):
    x1 = shape(I)
    x2 = objects(I, T, F, F)
    x3 = colorfilter(x2, ZERO)
    x4 = apply(toindices, x3)
    x5 = lbind(lbind, contained)
    x6 = lbind(extract, x4)
    x7 = compose(x6, x5)
    x8 = decrement(x1)
    x9 = astuple(FIVE, FIVE)
    x10 = x7(ORIGIN)
    x11 = x7(x8)
    x12 = x7(x9)
    x13 = fill(I, ONE, x10)
    x14 = fill(x13, THREE, x11)
    O = fill(x14, TWO, x12)
    return O


def solve_c3f564a4(I):
    x1 = asindices(I)
    x2 = dmirror(I)
    x3 = invert(NINE)
    x4 = papply(pair, I, x2)
    x5 = lbind(apply, maximum)
    x6 = apply(x5, x4)
    x7 = ofcolor(x6, ZERO)
    x8 = difference(x1, x7)
    x9 = toobject(x8, x6)
    x10 = interval(x3, NINE, ONE)
    x11 = interval(NINE, x3, NEG_ONE)
    x12 = pair(x10, x11)
    x13 = lbind(shift, x9)
    x14 = mapply(x13, x12)
    O = paint(x6, x14)
    return O


def solve_1a07d186(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = difference(x1, x2)
    x4 = apply(color, x3)
    x5 = rbind(contained, x4)
    x6 = compose(x5, color)
    x7 = sfilter(x2, x6)
    x8 = lbind(colorfilter, x3)
    x9 = chain(first, x8, color)
    x10 = fork(gravitate, identity, x9)
    x11 = fork(shift, identity, x10)
    x12 = mapply(x11, x7)
    x13 = merge(x2)
    x14 = cover(I, x13)
    O = paint(x14, x12)
    return O


def solve_d687bc17(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = difference(x1, x2)
    x4 = apply(color, x3)
    x5 = rbind(contained, x4)
    x6 = compose(x5, color)
    x7 = sfilter(x2, x6)
    x8 = lbind(colorfilter, x3)
    x9 = chain(first, x8, color)
    x10 = fork(gravitate, identity, x9)
    x11 = fork(shift, identity, x10)
    x12 = merge(x2)
    x13 = mapply(x11, x7)
    x14 = cover(I, x12)
    O = paint(x14, x13)
    return O


def solve_9af7a82c(I):
    x1 = objects(I, T, F, F)
    x2 = order(x1, size)
    x3 = valmax(x1, size)
    x4 = rbind(astuple, ONE)
    x5 = lbind(subtract, x3)
    x6 = compose(x4, size)
    x7 = chain(x4, x5, size)
    x8 = fork(canvas, color, x6)
    x9 = lbind(canvas, ZERO)
    x10 = compose(x9, x7)
    x11 = fork(vconcat, x8, x10)
    x12 = compose(cmirror, x11)
    x13 = apply(x12, x2)
    x14 = merge(x13)
    O = cmirror(x14)
    return O


def solve_6e19193c(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = rbind(toobject, I)
    x4 = compose(first, delta)
    x5 = rbind(colorcount, x1)
    x6 = matcher(x5, TWO)
    x7 = chain(x6, x3, dneighbors)
    x8 = rbind(sfilter, x7)
    x9 = chain(first, x8, toindices)
    x10 = fork(subtract, x4, x9)
    x11 = fork(shoot, x4, x10)
    x12 = mapply(x11, x2)
    x13 = fill(I, x1, x12)
    x14 = mapply(delta, x2)
    O = fill(x13, ZERO, x14)
    return O


def solve_ef135b50(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, ZERO)
    x3 = product(x1, x1)
    x4 = power(first, TWO)
    x5 = compose(first, last)
    x6 = fork(equality, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = fork(connect, first, last)
    x9 = mapply(x8, x7)
    x10 = intersection(x9, x2)
    x11 = fill(I, NINE, x10)
    x12 = trim(x11)
    x13 = asobject(x12)
    x14 = shift(x13, UNITY)
    O = paint(I, x14)
    return O


def solve_cbded52d(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = product(x2, x2)
    x4 = fork(vmatching, first, last)
    x5 = fork(hmatching, first, last)
    x6 = fork(either, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = compose(center, first)
    x9 = compose(center, last)
    x10 = fork(connect, x8, x9)
    x11 = chain(initset, center, x10)
    x12 = compose(color, first)
    x13 = fork(recolor, x12, x11)
    x14 = mapply(x13, x7)
    O = paint(I, x14)
    return O


def solve_8a004b2b(I):
    x1 = objects(I, F, T, T)
    x2 = ofcolor(I, FOUR)
    x3 = subgrid(x2, I)
    x4 = argmax(x1, lowermost)
    x5 = normalize(x4)
    x6 = replace(x3, FOUR, ZERO)
    x7 = objects(x6, T, F, T)
    x8 = merge(x7)
    x9 = width(x8)
    x10 = ulcorner(x8)
    x11 = width(x4)
    x12 = divide(x9, x11)
    x13 = upscale(x5, x12)
    x14 = shift(x13, x10)
    O = paint(x3, x14)
    return O


def solve_e26a3af2(I):
    x1 = rot90(I)
    x2 = apply(mostcommon, I)
    x3 = apply(mostcommon, x1)
    x4 = repeat(x2, ONE)
    x5 = repeat(x3, ONE)
    x6 = compose(size, dedupe)
    x7 = x6(x2)
    x8 = x6(x3)
    x9 = greater(x8, x7)
    x10 = branch(x9, height, width)
    x11 = x10(I)
    x12 = rot90(x4)
    x13 = branch(x9, x5, x12)
    x14 = branch(x9, vupscale, hupscale)
    O = x14(x13, x11)
    return O


def solve_6cf79266(I):
    x1 = ofcolor(I, ZERO)
    x2 = astuple(ZERO, ORIGIN)
    x3 = initset(x2)
    x4 = upscale(x3, THREE)
    x5 = toindices(x4)
    x6 = lbind(shift, x5)
    x7 = rbind(difference, x1)
    x8 = chain(size, x7, x6)
    x9 = matcher(x8, ZERO)
    x10 = lbind(add, NEG_UNITY)
    x11 = chain(flip, x9, x10)
    x12 = fork(both, x9, x11)
    x13 = sfilter(x1, x12)
    x14 = mapply(x6, x13)
    O = fill(I, ONE, x14)
    return O


def solve_a87f7484(I):
    x1 = numcolors(I)
    x2 = dmirror(I)
    x3 = portrait(I)
    x4 = branch(x3, dmirror, identity)
    x5 = x4(I)
    x6 = decrement(x1)
    x7 = hsplit(x5, x6)
    x8 = rbind(ofcolor, ZERO)
    x9 = apply(x8, x7)
    x10 = leastcommon(x9)
    x11 = matcher(x8, x10)
    x12 = extract(x7, x11)
    O = x4(x12)
    return O


def solve_4093f84a(I):
    x1 = leastcolor(I)
    x2 = replace(I, x1, FIVE)
    x3 = ofcolor(I, FIVE)
    x4 = portrait(x3)
    x5 = branch(x4, identity, dmirror)
    x6 = x5(x2)
    x7 = lefthalf(x6)
    x8 = righthalf(x6)
    x9 = rbind(order, identity)
    x10 = rbind(order, invert)
    x11 = apply(x9, x7)
    x12 = apply(x10, x8)
    x13 = hconcat(x11, x12)
    O = x5(x13)
    return O


def solve_ba26e723(I):
    x1 = rbind(divide, THREE)
    x2 = rbind(multiply, THREE)
    x3 = compose(x2, x1)
    x4 = fork(equality, identity, x3)
    x5 = compose(x4, last)
    x6 = ofcolor(I, FOUR)
    x7 = sfilter(x6, x5)
    O = fill(I, SIX, x7)
    return O


def solve_4612dd53(I):
    x1 = ofcolor(I, ONE)
    x2 = box(x1)
    x3 = fill(I, TWO, x2)
    x4 = subgrid(x1, x3)
    x5 = ofcolor(x4, ONE)
    x6 = mapply(vfrontier, x5)
    x7 = mapply(hfrontier, x5)
    x8 = size(x6)
    x9 = size(x7)
    x10 = greater(x8, x9)
    x11 = branch(x10, x7, x6)
    x12 = fill(x4, TWO, x11)
    x13 = ofcolor(x12, TWO)
    x14 = ulcorner(x1)
    x15 = shift(x13, x14)
    O = underfill(I, TWO, x15)
    return O


def solve_29c11459(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = objects(x2, T, F, T)
    x4 = objects(x1, T, F, T)
    x5 = compose(hfrontier, center)
    x6 = fork(recolor, color, x5)
    x7 = mapply(x6, x4)
    x8 = paint(x1, x7)
    x9 = mapply(x6, x3)
    x10 = paint(I, x9)
    x11 = objects(x8, T, F, T)
    x12 = apply(urcorner, x11)
    x13 = shift(x12, RIGHT)
    x14 = merge(x11)
    x15 = paint(x10, x14)
    O = fill(x15, FIVE, x13)
    return O


def solve_963e52fc(I):
    x1 = width(I)
    x2 = asobject(I)
    x3 = hperiod(x2)
    x4 = height(x2)
    x5 = astuple(x4, x3)
    x6 = ulcorner(x2)
    x7 = crop(I, x6, x5)
    x8 = rot90(x7)
    x9 = double(x1)
    x10 = divide(x9, x3)
    x11 = increment(x10)
    x12 = repeat(x8, x11)
    x13 = merge(x12)
    x14 = rot270(x13)
    x15 = astuple(x4, x9)
    O = crop(x14, ORIGIN, x15)
    return O


def solve_ae3edfdc(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, THREE, ZERO)
    x3 = replace(x2, SEVEN, ZERO)
    x4 = lbind(colorfilter, x1)
    x5 = lbind(rbind, gravitate)
    x6 = chain(x5, first, x4)
    x7 = x6(TWO)
    x8 = x6(ONE)
    x9 = x4(THREE)
    x10 = x4(SEVEN)
    x11 = fork(shift, identity, x7)
    x12 = fork(shift, identity, x8)
    x13 = mapply(x11, x9)
    x14 = mapply(x12, x10)
    x15 = paint(x3, x13)
    O = paint(x15, x14)
    return O


def solve_1f0c79e5(I):
    x1 = ofcolor(I, TWO)
    x2 = replace(I, TWO, ZERO)
    x3 = leastcolor(x2)
    x4 = ofcolor(x2, x3)
    x5 = combine(x1, x4)
    x6 = recolor(x3, x5)
    x7 = compose(decrement, double)
    x8 = ulcorner(x5)
    x9 = invert(x8)
    x10 = shift(x1, x9)
    x11 = apply(x7, x10)
    x12 = interval(ZERO, NINE, ONE)
    x13 = prapply(multiply, x11, x12)
    x14 = lbind(shift, x6)
    x15 = mapply(x14, x13)
    O = paint(I, x15)
    return O


def solve_56dc2b01(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, THREE)
    x3 = first(x2)
    x4 = ofcolor(I, TWO)
    x5 = gravitate(x3, x4)
    x6 = first(x5)
    x7 = equality(x6, ZERO)
    x8 = branch(x7, width, height)
    x9 = x8(x3)
    x10 = gravitate(x4, x3)
    x11 = sign(x10)
    x12 = multiply(x11, x9)
    x13 = crement(x12)
    x14 = recolor(EIGHT, x4)
    x15 = shift(x14, x13)
    x16 = paint(I, x15)
    O = move(x16, x3, x5)
    return O


def solve_e48d4e1a(I):
    x1 = shape(I)
    x2 = ofcolor(I, FIVE)
    x3 = fill(I, ZERO, x2)
    x4 = leastcolor(x3)
    x5 = size(x2)
    x6 = ofcolor(I, x4)
    x7 = rbind(toobject, I)
    x8 = rbind(colorcount, x4)
    x9 = chain(x8, x7, dneighbors)
    x10 = matcher(x9, FOUR)
    x11 = extract(x6, x10)
    x12 = multiply(DOWN_LEFT, x5)
    x13 = add(x12, x11)
    x14 = canvas(ZERO, x1)
    x15 = fork(combine, vfrontier, hfrontier)
    x16 = x15(x13)
    O = fill(x14, x4, x16)
    return O


def solve_6773b310(I):
    x1 = compress(I)
    x2 = neighbors(ORIGIN)
    x3 = insert(ORIGIN, x2)
    x4 = rbind(multiply, THREE)
    x5 = apply(x4, x3)
    x6 = astuple(FOUR, FOUR)
    x7 = shift(x5, x6)
    x8 = fork(insert, identity, neighbors)
    x9 = apply(x8, x7)
    x10 = rbind(toobject, x1)
    x11 = apply(x10, x9)
    x12 = rbind(colorcount, SIX)
    x13 = matcher(x12, TWO)
    x14 = mfilter(x11, x13)
    x15 = fill(x1, ONE, x14)
    x16 = replace(x15, SIX, ZERO)
    O = downscale(x16, THREE)
    return O


def solve_780d0b14(I):
    x1 = asindices(I)
    x2 = objects(I, T, T, T)
    x3 = rbind(greater, TWO)
    x4 = compose(x3, size)
    x5 = sfilter(x2, x4)
    x6 = totuple(x5)
    x7 = apply(color, x6)
    x8 = apply(center, x6)
    x9 = pair(x7, x8)
    x10 = fill(I, ZERO, x1)
    x11 = paint(x10, x9)
    x12 = rbind(greater, ONE)
    x13 = compose(dedupe, totuple)
    x14 = chain(x12, size, x13)
    x15 = sfilter(x11, x14)
    x16 = rot90(x15)
    x17 = sfilter(x16, x14)
    O = rot270(x17)
    return O


def solve_2204b7a8(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(sfilter, x1)
    x3 = compose(size, x2)
    x4 = x3(vline)
    x5 = x3(hline)
    x6 = greater(x4, x5)
    x7 = branch(x6, lefthalf, tophalf)
    x8 = branch(x6, righthalf, bottomhalf)
    x9 = branch(x6, hconcat, vconcat)
    x10 = x7(I)
    x11 = x8(I)
    x12 = index(x10, ORIGIN)
    x13 = shape(x11)
    x14 = decrement(x13)
    x15 = index(x11, x14)
    x16 = replace(x10, THREE, x12)
    x17 = replace(x11, THREE, x15)
    O = x9(x16, x17)
    return O


def solve_d9f24cd1(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, FIVE)
    x3 = prapply(connect, x1, x2)
    x4 = mfilter(x3, vline)
    x5 = underfill(I, TWO, x4)
    x6 = matcher(numcolors, TWO)
    x7 = objects(x5, F, F, T)
    x8 = sfilter(x7, x6)
    x9 = difference(x7, x8)
    x10 = colorfilter(x9, TWO)
    x11 = mapply(toindices, x10)
    x12 = apply(urcorner, x8)
    x13 = shift(x12, UNITY)
    x14 = rbind(shoot, UP)
    x15 = mapply(x14, x13)
    x16 = fill(x5, TWO, x15)
    x17 = mapply(vfrontier, x11)
    O = fill(x16, TWO, x17)
    return O


def solve_b782dc8a(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, F)
    x3 = ofcolor(I, x1)
    x4 = first(x3)
    x5 = dneighbors(x4)
    x6 = toobject(x5, I)
    x7 = mostcolor(x6)
    x8 = ofcolor(I, x7)
    x9 = colorfilter(x2, ZERO)
    x10 = rbind(adjacent, x8)
    x11 = mfilter(x9, x10)
    x12 = toindices(x11)
    x13 = rbind(manhattan, x3)
    x14 = chain(even, x13, initset)
    x15 = sfilter(x12, x14)
    x16 = difference(x12, x15)
    x17 = fill(I, x1, x15)
    O = fill(x17, x7, x16)
    return O


def solve_673ef223(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, EIGHT)
    x3 = replace(I, EIGHT, FOUR)
    x4 = colorfilter(x1, TWO)
    x5 = argmin(x1, uppermost)
    x6 = apply(uppermost, x4)
    x7 = fork(subtract, maximum, minimum)
    x8 = x7(x6)
    x9 = toivec(x8)
    x10 = leftmost(x5)
    x11 = equality(x10, ZERO)
    x12 = branch(x11, LEFT, RIGHT)
    x13 = rbind(shoot, x12)
    x14 = mapply(x13, x2)
    x15 = underfill(x3, EIGHT, x14)
    x16 = shift(x2, x9)
    x17 = mapply(hfrontier, x16)
    O = underfill(x15, EIGHT, x17)
    return O


def solve_f5b8619d(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = mapply(vfrontier, x2)
    x4 = underfill(I, EIGHT, x3)
    x5 = hconcat(x4, x4)
    O = vconcat(x5, x5)
    return O


def solve_f8c80d96(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, F)
    x3 = colorfilter(x2, x1)
    x4 = argmax(x3, size)
    x5 = argmin(x2, width)
    x6 = size(x5)
    x7 = equality(x6, ONE)
    x8 = branch(x7, identity, outbox)
    x9 = chain(outbox, outbox, x8)
    x10 = power(x9, TWO)
    x11 = power(x9, THREE)
    x12 = x9(x4)
    x13 = x10(x4)
    x14 = x11(x4)
    x15 = fill(I, x1, x12)
    x16 = fill(x15, x1, x13)
    x17 = fill(x16, x1, x14)
    O = replace(x17, ZERO, FIVE)
    return O


def solve_ecdecbb3(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, TWO)
    x3 = colorfilter(x1, EIGHT)
    x4 = product(x2, x3)
    x5 = fork(gravitate, first, last)
    x6 = compose(crement, x5)
    x7 = compose(center, first)
    x8 = fork(add, x7, x6)
    x9 = fork(connect, x7, x8)
    x10 = apply(x9, x4)
    x11 = lbind(greater, EIGHT)
    x12 = compose(x11, size)
    x13 = mfilter(x10, x12)
    x14 = fill(I, TWO, x13)
    x15 = apply(x8, x4)
    x16 = intersection(x13, x15)
    x17 = mapply(neighbors, x16)
    O = fill(x14, EIGHT, x17)
    return O


def solve_e5062a87(I):
    x1 = ofcolor(I, TWO)
    x2 = recolor(ZERO, x1)
    x3 = normalize(x2)
    x4 = occurrences(I, x2)
    x5 = lbind(shift, x3)
    x6 = apply(x5, x4)
    x7 = astuple(ONE, THREE)
    x8 = astuple(FIVE, ONE)
    x9 = astuple(TWO, SIX)
    x10 = initset(x7)
    x11 = insert(x8, x10)
    x12 = insert(x9, x11)
    x13 = rbind(contained, x12)
    x14 = chain(flip, x13, ulcorner)
    x15 = sfilter(x6, x14)
    x16 = merge(x15)
    x17 = recolor(TWO, x16)
    O = paint(I, x17)
    return O


def solve_a8d7556c(I):
    x1 = initset(ORIGIN)
    x2 = recolor(ZERO, x1)
    x3 = upscale(x2, TWO)
    x4 = occurrences(I, x3)
    x5 = lbind(shift, x3)
    x6 = mapply(x5, x4)
    x7 = fill(I, TWO, x6)
    x8 = add(SIX, SIX)
    x9 = astuple(EIGHT, x8)
    x10 = index(x7, x9)
    x11 = equality(x10, TWO)
    x12 = initset(x9)
    x13 = add(x9, DOWN)
    x14 = insert(x13, x12)
    x15 = toobject(x14, x7)
    x16 = toobject(x14, I)
    x17 = branch(x11, x16, x15)
    O = paint(x7, x17)
    return O


def solve_4938f0c2(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, TWO)
    x3 = vmirror(x2)
    x4 = height(x2)
    x5 = width(x2)
    x6 = toivec(x4)
    x7 = tojvec(x5)
    x8 = add(x7, ZERO_BY_TWO)
    x9 = add(x6, TWO_BY_ZERO)
    x10 = shift(x3, x8)
    x11 = fill(I, TWO, x10)
    x12 = ofcolor(x11, TWO)
    x13 = hmirror(x12)
    x14 = shift(x13, x9)
    x15 = fill(x11, TWO, x14)
    x16 = size(x1)
    x17 = greater(x16, FOUR)
    O = branch(x17, I, x15)
    return O


def solve_834ec97d(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = first(x2)
    x4 = shift(x3, DOWN)
    x5 = fill(I, ZERO, x3)
    x6 = paint(x5, x4)
    x7 = uppermost(x4)
    x8 = leftmost(x4)
    x9 = subtract(x8, TEN)
    x10 = add(x8, TEN)
    x11 = interval(x9, x10, TWO)
    x12 = lbind(greater, x7)
    x13 = compose(x12, first)
    x14 = rbind(contained, x11)
    x15 = compose(x14, last)
    x16 = sfilter(x1, x13)
    x17 = sfilter(x16, x15)
    O = fill(x6, FOUR, x17)
    return O


def solve_846bdb03(I):
    x1 = objects(I, F, F, T)
    x2 = rbind(colorcount, FOUR)
    x3 = matcher(x2, ZERO)
    x4 = extract(x1, x3)
    x5 = remove(x4, x1)
    x6 = merge(x5)
    x7 = subgrid(x6, I)
    x8 = index(x7, DOWN)
    x9 = subgrid(x4, I)
    x10 = lefthalf(x9)
    x11 = palette(x10)
    x12 = other(x11, ZERO)
    x13 = equality(x8, x12)
    x14 = branch(x13, identity, vmirror)
    x15 = x14(x4)
    x16 = normalize(x15)
    x17 = shift(x16, UNITY)
    O = paint(x7, x17)
    return O


def solve_90f3ed37(I):
    x1 = objects(I, T, T, T)
    x2 = order(x1, uppermost)
    x3 = first(x2)
    x4 = remove(x3, x2)
    x5 = normalize(x3)
    x6 = lbind(shift, x5)
    x7 = compose(x6, ulcorner)
    x8 = interval(TWO, NEG_ONE, NEG_ONE)
    x9 = apply(tojvec, x8)
    x10 = rbind(apply, x9)
    x11 = lbind(compose, size)
    x12 = lbind(lbind, intersection)
    x13 = compose(x11, x12)
    x14 = lbind(lbind, shift)
    x15 = chain(x10, x14, x7)
    x16 = fork(argmax, x15, x13)
    x17 = mapply(x16, x4)
    O = underfill(I, ONE, x17)
    return O


def solve_8403a5d5(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = first(x2)
    x4 = color(x3)
    x5 = leftmost(x3)
    x6 = interval(x5, TEN, TWO)
    x7 = rbind(contained, x6)
    x8 = compose(x7, last)
    x9 = sfilter(x1, x8)
    x10 = increment(x5)
    x11 = add(x5, THREE)
    x12 = interval(x10, TEN, FOUR)
    x13 = interval(x11, TEN, FOUR)
    x14 = lbind(astuple, NINE)
    x15 = apply(tojvec, x12)
    x16 = apply(x14, x13)
    x17 = fill(I, x4, x9)
    x18 = fill(x17, FIVE, x15)
    O = fill(x18, FIVE, x16)
    return O


def solve_91413438(I):
    x1 = colorcount(I, ZERO)
    x2 = subtract(NINE, x1)
    x3 = multiply(x1, THREE)
    x4 = multiply(x3, x1)
    x5 = subtract(x4, THREE)
    x6 = astuple(THREE, x5)
    x7 = canvas(ZERO, x6)
    x8 = hconcat(I, x7)
    x9 = objects(x8, T, T, T)
    x10 = first(x9)
    x11 = lbind(shift, x10)
    x12 = compose(x11, tojvec)
    x13 = interval(ZERO, x2, ONE)
    x14 = rbind(multiply, THREE)
    x15 = apply(x14, x13)
    x16 = mapply(x12, x15)
    x17 = paint(x8, x16)
    x18 = hsplit(x17, x1)
    O = merge(x18)
    return O


def solve_539a4f51(I):
    x1 = shape(I)
    x2 = index(I, ORIGIN)
    x3 = colorcount(I, ZERO)
    x4 = decrement(x1)
    x5 = positive(x3)
    x6 = branch(x5, x4, x1)
    x7 = crop(I, ORIGIN, x6)
    x8 = width(x7)
    x9 = astuple(ONE, x8)
    x10 = crop(x7, ORIGIN, x9)
    x11 = vupscale(x10, x8)
    x12 = dmirror(x11)
    x13 = hconcat(x7, x11)
    x14 = hconcat(x12, x7)
    x15 = vconcat(x13, x14)
    x16 = asobject(x15)
    x17 = multiply(UNITY, TEN)
    x18 = canvas(x2, x17)
    O = paint(x18, x16)
    return O


def solve_5daaa586(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = extract(x2, x4)
    x6 = outbox(x5)
    x7 = subgrid(x6, I)
    x8 = fgpartition(x7)
    x9 = argmax(x8, size)
    x10 = color(x9)
    x11 = toindices(x9)
    x12 = prapply(connect, x11, x11)
    x13 = mfilter(x12, vline)
    x14 = mfilter(x12, hline)
    x15 = size(x13)
    x16 = size(x14)
    x17 = greater(x15, x16)
    x18 = branch(x17, x13, x14)
    O = fill(x7, x10, x18)
    return O


def solve_3bdb4ada(I):
    x1 = objects(I, T, F, T)
    x2 = totuple(x1)
    x3 = compose(increment, ulcorner)
    x4 = compose(decrement, lrcorner)
    x5 = apply(x3, x2)
    x6 = apply(x4, x2)
    x7 = papply(connect, x5, x6)
    x8 = apply(last, x5)
    x9 = compose(last, first)
    x10 = power(last, TWO)
    x11 = fork(subtract, x9, x10)
    x12 = compose(even, x11)
    x13 = lbind(rbind, astuple)
    x14 = lbind(compose, x12)
    x15 = compose(x14, x13)
    x16 = fork(sfilter, first, x15)
    x17 = pair(x7, x8)
    x18 = mapply(x16, x17)
    O = fill(I, ZERO, x18)
    return O


def solve_ec883f72(I):
    x1 = palette(I)
    x2 = objects(I, T, T, T)
    x3 = fork(multiply, height, width)
    x4 = argmax(x2, x3)
    x5 = color(x4)
    x6 = remove(ZERO, x1)
    x7 = other(x6, x5)
    x8 = lrcorner(x4)
    x9 = llcorner(x4)
    x10 = urcorner(x4)
    x11 = ulcorner(x4)
    x12 = shoot(x8, UNITY)
    x13 = shoot(x9, DOWN_LEFT)
    x14 = shoot(x10, UP_RIGHT)
    x15 = shoot(x11, NEG_UNITY)
    x16 = combine(x12, x13)
    x17 = combine(x14, x15)
    x18 = combine(x16, x17)
    O = underfill(I, x7, x18)
    return O


def solve_2bee17df(I):
    x1 = height(I)
    x2 = rot90(I)
    x3 = subtract(x1, TWO)
    x4 = interval(ZERO, x1, ONE)
    x5 = rbind(colorcount, ZERO)
    x6 = matcher(x5, x3)
    x7 = rbind(vsplit, x1)
    x8 = lbind(apply, x6)
    x9 = compose(x8, x7)
    x10 = x9(I)
    x11 = pair(x4, x10)
    x12 = sfilter(x11, last)
    x13 = mapply(hfrontier, x12)
    x14 = x9(x2)
    x15 = pair(x14, x4)
    x16 = sfilter(x15, first)
    x17 = mapply(vfrontier, x16)
    x18 = astuple(x13, x17)
    x19 = merge(x18)
    O = underfill(I, THREE, x19)
    return O


def solve_e8dc4411(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, ZERO)
    x3 = ofcolor(I, x1)
    x4 = position(x2, x3)
    x5 = fork(connect, ulcorner, lrcorner)
    x6 = x5(x2)
    x7 = intersection(x2, x6)
    x8 = equality(x6, x7)
    x9 = fork(subtract, identity, crement)
    x10 = fork(add, identity, x9)
    x11 = branch(x8, identity, x10)
    x12 = shape(x2)
    x13 = multiply(x12, x4)
    x14 = apply(x11, x13)
    x15 = interval(ONE, FIVE, ONE)
    x16 = lbind(multiply, x14)
    x17 = apply(x16, x15)
    x18 = lbind(shift, x2)
    x19 = mapply(x18, x17)
    O = fill(I, x1, x19)
    return O


def solve_e40b9e2f(I):
    x1 = objects(I, F, T, T)
    x2 = neighbors(ORIGIN)
    x3 = mapply(neighbors, x2)
    x4 = first(x1)
    x5 = lbind(intersection, x4)
    x6 = compose(hmirror, vmirror)
    x7 = x6(x4)
    x8 = lbind(shift, x7)
    x9 = apply(x8, x3)
    x10 = argmax(x9, x5)
    x11 = paint(I, x10)
    x12 = objects(x11, F, T, T)
    x13 = first(x12)
    x14 = compose(size, x5)
    x15 = compose(vmirror, dmirror)
    x16 = x15(x13)
    x17 = lbind(shift, x16)
    x18 = apply(x17, x3)
    x19 = argmax(x18, x14)
    O = paint(x11, x19)
    return O


def solve_29623171(I):
    x1 = leastcolor(I)
    x2 = interval(ZERO, NINE, FOUR)
    x3 = product(x2, x2)
    x4 = rbind(add, THREE)
    x5 = rbind(interval, ONE)
    x6 = fork(x5, identity, x4)
    x7 = compose(x6, first)
    x8 = compose(x6, last)
    x9 = fork(product, x7, x8)
    x10 = rbind(colorcount, x1)
    x11 = rbind(toobject, I)
    x12 = compose(x10, x11)
    x13 = apply(x9, x3)
    x14 = valmax(x13, x12)
    x15 = matcher(x12, x14)
    x16 = compose(flip, x15)
    x17 = mfilter(x13, x15)
    x18 = mfilter(x13, x16)
    x19 = fill(I, x1, x17)
    O = fill(x19, ZERO, x18)
    return O


def solve_a2fd1cf0(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, THREE)
    x3 = uppermost(x1)
    x4 = leftmost(x1)
    x5 = uppermost(x2)
    x6 = leftmost(x2)
    x7 = astuple(x3, x5)
    x8 = minimum(x7)
    x9 = maximum(x7)
    x10 = astuple(x8, x6)
    x11 = astuple(x9, x6)
    x12 = connect(x10, x11)
    x13 = astuple(x4, x6)
    x14 = minimum(x13)
    x15 = maximum(x13)
    x16 = astuple(x3, x14)
    x17 = astuple(x3, x15)
    x18 = connect(x16, x17)
    x19 = combine(x12, x18)
    O = underfill(I, EIGHT, x19)
    return O


def solve_b0c4d837(I):
    x1 = ofcolor(I, FIVE)
    x2 = ofcolor(I, EIGHT)
    x3 = height(x1)
    x4 = decrement(x3)
    x5 = height(x2)
    x6 = subtract(x4, x5)
    x7 = astuple(ONE, x6)
    x8 = canvas(EIGHT, x7)
    x9 = subtract(SIX, x6)
    x10 = astuple(ONE, x9)
    x11 = canvas(ZERO, x10)
    x12 = hconcat(x8, x11)
    x13 = hsplit(x12, TWO)
    x14 = first(x13)
    x15 = last(x13)
    x16 = vmirror(x15)
    x17 = vconcat(x14, x16)
    x18 = astuple(ONE, THREE)
    x19 = canvas(ZERO, x18)
    O = vconcat(x17, x19)
    return O


def solve_8731374e(I):
    x1 = objects(I, T, F, F)
    x2 = argmax(x1, size)
    x3 = subgrid(x2, I)
    x4 = height(x3)
    x5 = width(x3)
    x6 = vsplit(x3, x4)
    x7 = lbind(greater, FOUR)
    x8 = compose(x7, numcolors)
    x9 = sfilter(x6, x8)
    x10 = merge(x9)
    x11 = rot90(x10)
    x12 = vsplit(x11, x5)
    x13 = sfilter(x12, x8)
    x14 = merge(x13)
    x15 = rot270(x14)
    x16 = leastcolor(x15)
    x17 = ofcolor(x15, x16)
    x18 = fork(combine, vfrontier, hfrontier)
    x19 = mapply(x18, x17)
    O = fill(x15, x16, x19)
    return O


def solve_272f95fa(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = apply(toindices, x2)
    x4 = rbind(bordering, I)
    x5 = compose(flip, x4)
    x6 = extract(x3, x5)
    x7 = remove(x6, x3)
    x8 = lbind(vmatching, x6)
    x9 = lbind(hmatching, x6)
    x10 = sfilter(x7, x8)
    x11 = sfilter(x7, x9)
    x12 = argmin(x10, uppermost)
    x13 = argmax(x10, uppermost)
    x14 = argmin(x11, leftmost)
    x15 = argmax(x11, leftmost)
    x16 = fill(I, SIX, x6)
    x17 = fill(x16, TWO, x12)
    x18 = fill(x17, ONE, x13)
    x19 = fill(x18, FOUR, x14)
    O = fill(x19, THREE, x15)
    return O


def solve_db93a21d(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, NINE)
    x3 = colorfilter(x1, NINE)
    x4 = rbind(shoot, DOWN)
    x5 = mapply(x4, x2)
    x6 = underfill(I, ONE, x5)
    x7 = compose(halve, width)
    x8 = rbind(greater, ONE)
    x9 = compose(x8, x7)
    x10 = matcher(x7, THREE)
    x11 = power(outbox, TWO)
    x12 = power(outbox, THREE)
    x13 = mapply(outbox, x3)
    x14 = sfilter(x3, x9)
    x15 = sfilter(x3, x10)
    x16 = mapply(x11, x14)
    x17 = mapply(x12, x15)
    x18 = fill(x6, THREE, x13)
    x19 = fill(x18, THREE, x16)
    O = fill(x19, THREE, x17)
    return O


def solve_53b68214(I):
    x1 = width(I)
    x2 = objects(I, T, T, T)
    x3 = first(x2)
    x4 = vperiod(x3)
    x5 = toivec(x4)
    x6 = interval(ZERO, NINE, ONE)
    x7 = lbind(multiply, x5)
    x8 = apply(x7, x6)
    x9 = lbind(shift, x3)
    x10 = mapply(x9, x8)
    x11 = astuple(x1, x1)
    x12 = portrait(x3)
    x13 = shape(x3)
    x14 = add(DOWN, x13)
    x15 = decrement(x14)
    x16 = shift(x3, x15)
    x17 = branch(x12, x10, x16)
    x18 = canvas(ZERO, x11)
    x19 = paint(x18, x3)
    O = paint(x19, x17)
    return O


def solve_d6ad076f(I):
    x1 = objects(I, T, F, T)
    x2 = argmin(x1, size)
    x3 = argmax(x1, size)
    x4 = vmatching(x2, x3)
    x5 = branch(x4, DOWN, RIGHT)
    x6 = branch(x4, uppermost, leftmost)
    x7 = valmax(x1, x6)
    x8 = x6(x2)
    x9 = equality(x7, x8)
    x10 = branch(x9, NEG_ONE, ONE)
    x11 = multiply(x5, x10)
    x12 = inbox(x2)
    x13 = rbind(shoot, x11)
    x14 = mapply(x13, x12)
    x15 = underfill(I, EIGHT, x14)
    x16 = objects(x15, T, F, T)
    x17 = colorfilter(x16, EIGHT)
    x18 = rbind(bordering, I)
    x19 = mfilter(x17, x18)
    O = cover(x15, x19)
    return O


def solve_6cdd2623(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = prapply(connect, x2, x2)
    x4 = fgpartition(I)
    x5 = merge(x4)
    x6 = cover(I, x5)
    x7 = fork(either, hline, vline)
    x8 = box(x5)
    x9 = rbind(difference, x8)
    x10 = chain(positive, size, x9)
    x11 = fork(both, x7, x10)
    x12 = mfilter(x3, x11)
    O = fill(x6, x1, x12)
    return O


def solve_a3df8b1e(I):
    x1 = shape(I)
    x2 = ofcolor(I, ONE)
    x3 = first(x2)
    x4 = shoot(x3, UP_RIGHT)
    x5 = fill(I, ONE, x4)
    x6 = ofcolor(x5, ONE)
    x7 = urcorner(x6)
    x8 = shoot(x7, NEG_UNITY)
    x9 = fill(x5, ONE, x8)
    x10 = objects(x9, T, T, T)
    x11 = first(x10)
    x12 = subgrid(x11, x9)
    x13 = shape(x12)
    x14 = subtract(x13, DOWN)
    x15 = crop(x12, DOWN, x14)
    x16 = vconcat(x15, x15)
    x17 = vconcat(x16, x16)
    x18 = vconcat(x17, x17)
    x19 = hmirror(x18)
    x20 = crop(x19, ORIGIN, x1)
    O = hmirror(x20)
    return O


def solve_8d510a79(I):
    x1 = ofcolor(I, ONE)
    x2 = ofcolor(I, TWO)
    x3 = ofcolor(I, FIVE)
    x4 = uppermost(x3)
    x5 = chain(toivec, decrement, double)
    x6 = lbind(greater, x4)
    x7 = compose(x6, first)
    x8 = chain(invert, x5, x7)
    x9 = fork(shoot, identity, x8)
    x10 = compose(x5, x7)
    x11 = fork(shoot, identity, x10)
    x12 = lbind(matcher, x7)
    x13 = compose(x12, x7)
    x14 = fork(sfilter, x11, x13)
    x15 = mapply(x9, x1)
    x16 = mapply(x14, x2)
    x17 = underfill(I, TWO, x16)
    O = fill(x17, ONE, x15)
    return O


def solve_cdecee7f(I):
    x1 = objects(I, T, F, T)
    x2 = astuple(ONE, THREE)
    x3 = size(x1)
    x4 = order(x1, leftmost)
    x5 = apply(color, x4)
    x6 = rbind(canvas, UNITY)
    x7 = apply(x6, x5)
    x8 = merge(x7)
    x9 = dmirror(x8)
    x10 = subtract(NINE, x3)
    x11 = astuple(ONE, x10)
    x12 = canvas(ZERO, x11)
    x13 = hconcat(x9, x12)
    x14 = hsplit(x13, THREE)
    x15 = merge(x14)
    x16 = crop(x15, ORIGIN, x2)
    x17 = crop(x15, DOWN, x2)
    x18 = crop(x15, TWO_BY_ZERO, x2)
    x19 = vmirror(x17)
    x20 = vconcat(x16, x19)
    O = vconcat(x20, x18)
    return O


def solve_3345333e(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = cover(I, x2)
    x4 = leastcolor(x3)
    x5 = ofcolor(x3, x4)
    x6 = neighbors(ORIGIN)
    x7 = mapply(neighbors, x6)
    x8 = vmirror(x5)
    x9 = lbind(shift, x8)
    x10 = apply(x9, x7)
    x11 = rbind(intersection, x5)
    x12 = compose(size, x11)
    x13 = argmax(x10, x12)
    O = fill(x3, x4, x13)
    return O


def solve_b190f7f5(I):
    x1 = portrait(I)
    x2 = branch(x1, vsplit, hsplit)
    x3 = x2(I, TWO)
    x4 = argmin(x3, numcolors)
    x5 = argmax(x3, numcolors)
    x6 = width(x5)
    x7 = rbind(repeat, x6)
    x8 = chain(dmirror, merge, x7)
    x9 = upscale(x5, x6)
    x10 = x8(x4)
    x11 = x8(x10)
    x12 = ofcolor(x11, ZERO)
    O = fill(x9, ZERO, x12)
    return O


def solve_caa06a1f(I):
    x1 = asobject(I)
    x2 = shape(I)
    x3 = decrement(x2)
    x4 = index(I, x3)
    x5 = double(x2)
    x6 = canvas(x4, x5)
    x7 = paint(x6, x1)
    x8 = objects(x7, F, F, T)
    x9 = first(x8)
    x10 = shift(x9, LEFT)
    x11 = vperiod(x10)
    x12 = hperiod(x10)
    x13 = neighbors(ORIGIN)
    x14 = lbind(mapply, neighbors)
    x15 = power(x14, TWO)
    x16 = x15(x13)
    x17 = astuple(x11, x12)
    x18 = lbind(multiply, x17)
    x19 = apply(x18, x16)
    x20 = lbind(shift, x10)
    x21 = mapply(x20, x19)
    O = paint(I, x21)
    return O


def solve_e21d9049(I):
    x1 = asindices(I)
    x2 = leastcolor(I)
    x3 = objects(I, T, F, T)
    x4 = ofcolor(I, x2)
    x5 = merge(x3)
    x6 = shape(x5)
    x7 = neighbors(ORIGIN)
    x8 = lbind(mapply, neighbors)
    x9 = power(x8, TWO)
    x10 = x9(x7)
    x11 = lbind(multiply, x6)
    x12 = lbind(shift, x5)
    x13 = apply(x11, x10)
    x14 = mapply(x12, x13)
    x15 = lbind(hmatching, x4)
    x16 = lbind(vmatching, x4)
    x17 = fork(either, x15, x16)
    x18 = compose(x17, initset)
    x19 = paint(I, x14)
    x20 = sfilter(x1, x18)
    x21 = difference(x1, x20)
    O = cover(x19, x21)
    return O


def solve_d89b689b(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, EIGHT)
    x3 = sizefilter(x1, ONE)
    x4 = apply(initset, x2)
    x5 = lbind(argmin, x4)
    x6 = lbind(rbind, manhattan)
    x7 = compose(x5, x6)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x3)
    x10 = merge(x3)
    x11 = cover(I, x10)
    O = paint(x11, x9)
    return O


def solve_746b3537(I):
    x1 = chain(size, dedupe, first)
    x2 = x1(I)
    x3 = equality(x2, ONE)
    x4 = branch(x3, dmirror, identity)
    x5 = x4(I)
    x6 = objects(x5, T, F, F)
    x7 = order(x6, leftmost)
    x8 = apply(color, x7)
    x9 = repeat(x8, ONE)
    O = x4(x9)
    return O


def solve_63613498(I):
    x1 = crop(I, ORIGIN, THREE_BY_THREE)
    x2 = ofcolor(x1, ZERO)
    x3 = asindices(x1)
    x4 = difference(x3, x2)
    x5 = normalize(x4)
    x6 = objects(I, T, F, T)
    x7 = compose(toindices, normalize)
    x8 = matcher(x7, x5)
    x9 = mfilter(x6, x8)
    x10 = fill(I, FIVE, x9)
    x11 = asobject(x1)
    O = paint(x10, x11)
    return O


def solve_06df4c85(I):
    x1 = partition(I)
    x2 = mostcolor(I)
    x3 = ofcolor(I, x2)
    x4 = colorfilter(x1, ZERO)
    x5 = argmax(x1, size)
    x6 = difference(x1, x4)
    x7 = remove(x5, x6)
    x8 = merge(x7)
    x9 = product(x8, x8)
    x10 = power(first, TWO)
    x11 = compose(first, last)
    x12 = fork(equality, x10, x11)
    x13 = sfilter(x9, x12)
    x14 = compose(last, first)
    x15 = power(last, TWO)
    x16 = fork(connect, x14, x15)
    x17 = fork(recolor, color, x16)
    x18 = apply(x17, x13)
    x19 = fork(either, vline, hline)
    x20 = mfilter(x18, x19)
    x21 = paint(I, x20)
    O = fill(x21, x2, x3)
    return O


def solve_f9012d9b(I):
    x1 = objects(I, T, F, F)
    x2 = ofcolor(I, ZERO)
    x3 = lbind(contained, ZERO)
    x4 = chain(flip, x3, palette)
    x5 = mfilter(x1, x4)
    x6 = vsplit(I, TWO)
    x7 = hsplit(I, TWO)
    x8 = extract(x6, x4)
    x9 = extract(x7, x4)
    x10 = asobject(x8)
    x11 = asobject(x9)
    x12 = vperiod(x10)
    x13 = hperiod(x11)
    x14 = neighbors(ORIGIN)
    x15 = mapply(neighbors, x14)
    x16 = astuple(x12, x13)
    x17 = rbind(multiply, x16)
    x18 = apply(x17, x15)
    x19 = lbind(shift, x5)
    x20 = mapply(x19, x18)
    x21 = paint(I, x20)
    O = subgrid(x2, x21)
    return O


def solve_4522001f(I):
    x1 = objects(I, F, F, T)
    x2 = first(x1)
    x3 = toindices(x2)
    x4 = contained(ZERO_BY_TWO, x3)
    x5 = contained(TWO_BY_TWO, x3)
    x6 = contained(TWO_BY_ZERO, x3)
    x7 = astuple(NINE, NINE)
    x8 = canvas(ZERO, x7)
    x9 = astuple(THREE, ORIGIN)
    x10 = initset(x9)
    x11 = upscale(x10, TWO)
    x12 = upscale(x11, TWO)
    x13 = shape(x12)
    x14 = shift(x12, x13)
    x15 = combine(x12, x14)
    x16 = paint(x8, x15)
    x17 = rot90(x16)
    x18 = rot180(x16)
    x19 = rot270(x16)
    x20 = branch(x4, x17, x16)
    x21 = branch(x5, x18, x20)
    O = branch(x6, x19, x21)
    return O


def solve_a48eeaf7(I):
    x1 = ofcolor(I, TWO)
    x2 = outbox(x1)
    x3 = apply(initset, x2)
    x4 = ofcolor(I, FIVE)
    x5 = lbind(argmin, x3)
    x6 = lbind(lbind, manhattan)
    x7 = compose(x6, initset)
    x8 = compose(x5, x7)
    x9 = mapply(x8, x4)
    x10 = cover(I, x4)
    O = fill(x10, FIVE, x9)
    return O


def solve_eb5a1d5d(I):
    x1 = compose(dmirror, dedupe)
    x2 = x1(I)
    x3 = x1(x2)
    x4 = fork(remove, last, identity)
    x5 = compose(hmirror, x4)
    x6 = fork(vconcat, identity, x5)
    x7 = x6(x3)
    x8 = dmirror(x7)
    O = x6(x8)
    return O


def solve_e179c5f4(I):
    x1 = height(I)
    x2 = ofcolor(I, ONE)
    x3 = first(x2)
    x4 = shoot(x3, UP_RIGHT)
    x5 = fill(I, ONE, x4)
    x6 = ofcolor(x5, ONE)
    x7 = urcorner(x6)
    x8 = shoot(x7, NEG_UNITY)
    x9 = fill(x5, ONE, x8)
    x10 = ofcolor(x9, ONE)
    x11 = subgrid(x10, x9)
    x12 = height(x11)
    x13 = width(x11)
    x14 = decrement(x12)
    x15 = astuple(x14, x13)
    x16 = ulcorner(x10)
    x17 = crop(x9, x16, x15)
    x18 = repeat(x17, NINE)
    x19 = merge(x18)
    x20 = astuple(x1, x13)
    x21 = crop(x19, ORIGIN, x20)
    x22 = hmirror(x21)
    O = replace(x22, ZERO, EIGHT)
    return O


def solve_228f6490(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, ZERO)
    x3 = rbind(bordering, I)
    x4 = compose(flip, x3)
    x5 = sfilter(x2, x4)
    x6 = first(x5)
    x7 = last(x5)
    x8 = difference(x1, x2)
    x9 = compose(normalize, toindices)
    x10 = x9(x6)
    x11 = x9(x7)
    x12 = matcher(x9, x10)
    x13 = matcher(x9, x11)
    x14 = extract(x8, x12)
    x15 = extract(x8, x13)
    x16 = ulcorner(x6)
    x17 = ulcorner(x7)
    x18 = ulcorner(x14)
    x19 = ulcorner(x15)
    x20 = subtract(x16, x18)
    x21 = subtract(x17, x19)
    x22 = move(I, x14, x20)
    O = move(x22, x15, x21)
    return O


def solve_995c5fa3(I):
    x1 = hsplit(I, THREE)
    x2 = astuple(TWO, ONE)
    x3 = rbind(ofcolor, ZERO)
    x4 = compose(ulcorner, x3)
    x5 = compose(size, x3)
    x6 = matcher(x5, ZERO)
    x7 = matcher(x4, UNITY)
    x8 = matcher(x4, DOWN)
    x9 = matcher(x4, x2)
    x10 = rbind(multiply, THREE)
    x11 = power(double, TWO)
    x12 = compose(double, x6)
    x13 = chain(x11, double, x7)
    x14 = compose(x10, x8)
    x15 = compose(x11, x9)
    x16 = fork(add, x12, x13)
    x17 = fork(add, x14, x15)
    x18 = fork(add, x16, x17)
    x19 = rbind(canvas, UNITY)
    x20 = compose(x19, x18)
    x21 = apply(x20, x1)
    x22 = merge(x21)
    O = hupscale(x22, THREE)
    return O


def solve_d06dbe63(I):
    x1 = ofcolor(I, EIGHT)
    x2 = center(x1)
    x3 = connect(ORIGIN, DOWN)
    x4 = connect(ORIGIN, ZERO_BY_TWO)
    x5 = combine(x3, x4)
    x6 = subtract(x2, TWO_BY_ZERO)
    x7 = shift(x5, x6)
    x8 = astuple(NEG_TWO, TWO)
    x9 = interval(ZERO, FIVE, ONE)
    x10 = lbind(multiply, x8)
    x11 = apply(x10, x9)
    x12 = lbind(shift, x7)
    x13 = mapply(x12, x11)
    x14 = fill(I, FIVE, x13)
    x15 = rot180(x14)
    x16 = ofcolor(x15, EIGHT)
    x17 = center(x16)
    x18 = subtract(x17, x6)
    x19 = shift(x13, x18)
    x20 = toivec(NEG_TWO)
    x21 = shift(x19, x20)
    x22 = fill(x15, FIVE, x21)
    O = rot180(x22)
    return O


def solve_36fdfd69(I):
    x1 = upscale(I, TWO)
    x2 = objects(x1, T, T, T)
    x3 = colorfilter(x2, TWO)
    x4 = fork(manhattan, first, last)
    x5 = lbind(greater, FIVE)
    x6 = compose(x5, x4)
    x7 = product(x3, x3)
    x8 = sfilter(x7, x6)
    x9 = apply(merge, x8)
    x10 = mapply(delta, x9)
    x11 = fill(x1, FOUR, x10)
    x12 = merge(x3)
    x13 = paint(x11, x12)
    O = downscale(x13, TWO)
    return O


def solve_0a938d79(I):
    x1 = portrait(I)
    x2 = branch(x1, dmirror, identity)
    x3 = x2(I)
    x4 = fgpartition(x3)
    x5 = merge(x4)
    x6 = chain(double, decrement, width)
    x7 = x6(x5)
    x8 = compose(vfrontier, tojvec)
    x9 = lbind(mapply, x8)
    x10 = rbind(interval, x7)
    x11 = width(x3)
    x12 = rbind(x10, x11)
    x13 = chain(x9, x12, leftmost)
    x14 = fork(recolor, color, x13)
    x15 = mapply(x14, x4)
    x16 = paint(x3, x15)
    O = x2(x16)
    return O


def solve_045e512c(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = lbind(shift, x2)
    x5 = lbind(mapply, x4)
    x6 = double(TEN)
    x7 = interval(FOUR, x6, FOUR)
    x8 = rbind(apply, x7)
    x9 = lbind(position, x2)
    x10 = lbind(rbind, multiply)
    x11 = chain(x8, x10, x9)
    x12 = compose(x5, x11)
    x13 = fork(recolor, color, x12)
    x14 = mapply(x13, x3)
    O = paint(I, x14)
    return O


def solve_82819916(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = normalize(x2)
    x5 = compose(last, last)
    x6 = rbind(argmin, x5)
    x7 = compose(first, x6)
    x8 = fork(other, palette, x7)
    x9 = x7(x4)
    x10 = matcher(first, x9)
    x11 = sfilter(x4, x10)
    x12 = difference(x4, x11)
    x13 = compose(toivec, uppermost)
    x14 = lbind(shift, x11)
    x15 = lbind(shift, x12)
    x16 = compose(x14, x13)
    x17 = compose(x15, x13)
    x18 = fork(recolor, x7, x16)
    x19 = fork(recolor, x8, x17)
    x20 = fork(combine, x18, x19)
    x21 = mapply(x20, x3)
    O = paint(I, x21)
    return O


def solve_99fa7670(I):
    x1 = shape(I)
    x2 = objects(I, T, F, T)
    x3 = rbind(shoot, RIGHT)
    x4 = compose(x3, center)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x2)
    x7 = paint(I, x6)
    x8 = add(x1, DOWN_LEFT)
    x9 = initset(x8)
    x10 = recolor(ZERO, x9)
    x11 = objects(x7, T, F, T)
    x12 = insert(x10, x11)
    x13 = order(x12, uppermost)
    x14 = first(x13)
    x15 = remove(x10, x13)
    x16 = remove(x14, x13)
    x17 = compose(lrcorner, first)
    x18 = compose(lrcorner, last)
    x19 = fork(connect, x17, x18)
    x20 = compose(color, first)
    x21 = fork(recolor, x20, x19)
    x22 = pair(x15, x16)
    x23 = mapply(x21, x22)
    O = underpaint(x7, x23)
    return O


def solve_72322fa7(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(numcolors, ONE)
    x3 = sfilter(x1, x2)
    x4 = difference(x1, x3)
    x5 = lbind(matcher, first)
    x6 = compose(x5, mostcolor)
    x7 = fork(sfilter, identity, x6)
    x8 = fork(difference, identity, x7)
    x9 = lbind(occurrences, I)
    x10 = compose(x9, x7)
    x11 = compose(x9, x8)
    x12 = compose(ulcorner, x8)
    x13 = fork(subtract, ulcorner, x12)
    x14 = lbind(rbind, add)
    x15 = compose(x14, x13)
    x16 = fork(apply, x15, x11)
    x17 = lbind(lbind, shift)
    x18 = compose(x17, normalize)
    x19 = fork(mapply, x18, x10)
    x20 = fork(mapply, x18, x16)
    x21 = mapply(x19, x4)
    x22 = mapply(x20, x4)
    x23 = paint(I, x21)
    O = paint(x23, x22)
    return O


def solve_855e0971(I):
    x1 = rot90(I)
    x2 = frontiers(I)
    x3 = sfilter(x2, hline)
    x4 = size(x3)
    x6 = positive(x4)
    x7 = branch(x6, identity, dmirror)
    x8 = x7(I)
    x9 = rbind(subgrid, x8)
    x10 = matcher(color, ZERO)
    x11 = compose(flip, x10)
    x12 = partition(x8)
    x13 = sfilter(x12, x11)
    x14 = rbind(ofcolor, ZERO)
    x15 = lbind(mapply, vfrontier)
    x16 = chain(x15, x14, x9)
    x17 = fork(shift, x16, ulcorner)
    x18 = fork(intersection, toindices, x17)
    x19 = mapply(x18, x13)
    x20 = fill(x8, ZERO, x19)
    O = x7(x20)
    return O


def solve_a78176bb(I):
    x1 = palette(I)
    x2 = objects(I, T, F, T)
    x3 = remove(ZERO, x1)
    x4 = other(x3, FIVE)
    x5 = colorfilter(x2, FIVE)
    x6 = lbind(index, I)
    x7 = compose(x6, urcorner)
    x8 = matcher(x7, FIVE)
    x9 = sfilter(x5, x8)
    x10 = difference(x5, x9)
    x11 = apply(urcorner, x9)
    x12 = apply(llcorner, x10)
    x13 = rbind(add, UP_RIGHT)
    x14 = rbind(add, DOWN_LEFT)
    x15 = apply(x13, x11)
    x16 = apply(x14, x12)
    x17 = rbind(shoot, UNITY)
    x18 = rbind(shoot, NEG_UNITY)
    x19 = fork(combine, x17, x18)
    x20 = mapply(x19, x15)
    x21 = mapply(x19, x16)
    x22 = combine(x20, x21)
    x23 = fill(I, x4, x22)
    O = replace(x23, FIVE, ZERO)
    return O


def solve_952a094c(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = argmax(x1, size)
    x4 = outbox(x3)
    x5 = corners(x4)
    x6 = lbind(rbind, manhattan)
    x7 = lbind(argmax, x2)
    x8 = chain(x7, x6, initset)
    x9 = compose(color, x8)
    x10 = fork(astuple, x9, identity)
    x11 = apply(x10, x5)
    x12 = merge(x2)
    x13 = cover(I, x12)
    O = paint(x13, x11)
    return O


def solve_6d58a25d(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = merge(x3)
    x5 = color(x4)
    x6 = uppermost(x2)
    x7 = rbind(greater, x6)
    x8 = compose(x7, uppermost)
    x9 = rbind(vmatching, x2)
    x10 = fork(both, x9, x8)
    x11 = sfilter(x3, x10)
    x12 = increment(x6)
    x13 = rbind(greater, x12)
    x14 = compose(x13, first)
    x15 = rbind(sfilter, x14)
    x16 = chain(x15, vfrontier, center)
    x17 = mapply(x16, x11)
    O = underfill(I, x5, x17)
    return O


def solve_6aa20dc0(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, numcolors)
    x3 = normalize(x2)
    x4 = lbind(matcher, first)
    x5 = compose(x4, mostcolor)
    x6 = fork(sfilter, identity, x5)
    x7 = fork(difference, identity, x6)
    x8 = lbind(rbind, upscale)
    x9 = interval(ONE, FOUR, ONE)
    x10 = apply(x8, x9)
    x11 = initset(identity)
    x12 = insert(vmirror, x11)
    x13 = insert(hmirror, x12)
    x14 = insert(cmirror, x13)
    x15 = insert(dmirror, x14)
    x16 = fork(compose, first, last)
    x17 = lbind(occurrences, I)
    x18 = lbind(lbind, shift)
    x19 = compose(x17, x7)
    x20 = product(x15, x10)
    x21 = apply(x16, x20)
    x22 = rapply(x21, x3)
    x23 = fork(mapply, x18, x19)
    x24 = mapply(x23, x22)
    O = paint(I, x24)
    return O


def solve_e6721834(I):
    x1 = portrait(I)
    x2 = branch(x1, vsplit, hsplit)
    x3 = x2(I, TWO)
    x4 = order(x3, numcolors)
    x5 = first(x4)
    x6 = last(x4)
    x7 = objects(x6, F, F, T)
    x8 = merge(x7)
    x9 = mostcolor(x8)
    x10 = matcher(first, x9)
    x11 = compose(flip, x10)
    x12 = rbind(sfilter, x11)
    x13 = lbind(occurrences, x5)
    x14 = compose(x13, x12)
    x15 = chain(positive, size, x14)
    x16 = sfilter(x7, x15)
    x17 = chain(first, x13, x12)
    x18 = compose(ulcorner, x12)
    x19 = fork(subtract, x17, x18)
    x20 = fork(shift, identity, x19)
    x21 = apply(x20, x16)
    x22 = compose(decrement, width)
    x23 = chain(positive, decrement, x22)
    x24 = mfilter(x21, x23)
    O = paint(x5, x24)
    return O


def solve_447fd412(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, numcolors)
    x3 = normalize(x2)
    x4 = lbind(matcher, first)
    x5 = compose(x4, mostcolor)
    x6 = fork(sfilter, identity, x5)
    x7 = fork(difference, identity, x6)
    x8 = lbind(rbind, upscale)
    x9 = interval(ONE, FOUR, ONE)
    x10 = apply(x8, x9)
    x11 = lbind(recolor, ZERO)
    x12 = compose(x11, outbox)
    x13 = fork(combine, identity, x12)
    x14 = lbind(occurrences, I)
    x15 = lbind(rbind, subtract)
    x16 = lbind(apply, increment)
    x17 = lbind(lbind, shift)
    x18 = chain(x15, ulcorner, x7)
    x19 = chain(x14, x13, x7)
    x20 = fork(apply, x18, x19)
    x21 = compose(x16, x20)
    x22 = fork(mapply, x17, x21)
    x23 = rapply(x10, x3)
    x24 = mapply(x22, x23)
    O = paint(I, x24)
    return O


def solve_2bcee788(I):
    x1 = mostcolor(I)
    x2 = objects(I, T, F, T)
    x3 = replace(I, x1, THREE)
    x4 = argmax(x2, size)
    x5 = argmin(x2, size)
    x6 = position(x4, x5)
    x7 = first(x6)
    x8 = last(x6)
    x9 = subgrid(x4, x3)
    x10 = hline(x5)
    x11 = hmirror(x9)
    x12 = vmirror(x9)
    x13 = branch(x10, x11, x12)
    x14 = branch(x10, x7, ZERO)
    x15 = branch(x10, ZERO, x8)
    x16 = asobject(x13)
    x17 = matcher(first, THREE)
    x18 = compose(flip, x17)
    x19 = sfilter(x16, x18)
    x20 = ulcorner(x4)
    x21 = shape(x4)
    x22 = astuple(x14, x15)
    x23 = multiply(x21, x22)
    x24 = add(x20, x23)
    x25 = shift(x19, x24)
    O = paint(x3, x25)
    return O


def solve_776ffc46(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, FIVE)
    x3 = fork(equality, toindices, box)
    x4 = extract(x2, x3)
    x5 = inbox(x4)
    x6 = subgrid(x5, I)
    x7 = asobject(x6)
    x8 = matcher(first, ZERO)
    x9 = compose(flip, x8)
    x10 = sfilter(x7, x9)
    x11 = normalize(x10)
    x12 = toindices(x11)
    x13 = compose(toindices, normalize)
    x14 = matcher(x13, x12)
    x15 = mfilter(x1, x14)
    x16 = color(x11)
    O = fill(I, x16, x15)
    return O


def solve_f35d900a(I):
    x1 = objects(I, T, F, T)
    x2 = palette(I)
    x3 = remove(ZERO, x2)
    x4 = lbind(other, x3)
    x5 = compose(x4, color)
    x6 = fork(recolor, x5, outbox)
    x7 = mapply(x6, x1)
    x8 = mapply(toindices, x1)
    x9 = box(x8)
    x10 = difference(x9, x8)
    x11 = lbind(argmin, x8)
    x12 = rbind(compose, initset)
    x13 = lbind(rbind, manhattan)
    x14 = chain(x12, x13, initset)
    x15 = chain(initset, x11, x14)
    x16 = fork(manhattan, initset, x15)
    x17 = compose(even, x16)
    x18 = sfilter(x10, x17)
    x19 = paint(I, x7)
    O = fill(x19, FIVE, x18)
    return O


def solve_0dfd9992(I):
    x1 = height(I)
    x2 = width(I)
    x3 = partition(I)
    x4 = colorfilter(x3, ZERO)
    x5 = difference(x3, x4)
    x6 = merge(x5)
    x7 = astuple(x1, ONE)
    x8 = astuple(ONE, x2)
    x9 = decrement(x1)
    x10 = decrement(x2)
    x11 = toivec(x10)
    x12 = tojvec(x9)
    x13 = crop(I, x11, x8)
    x14 = crop(I, x12, x7)
    x15 = asobject(x14)
    x16 = asobject(x13)
    x17 = vperiod(x15)
    x18 = hperiod(x16)
    x19 = astuple(x17, x18)
    x20 = lbind(multiply, x19)
    x21 = neighbors(ORIGIN)
    x22 = mapply(neighbors, x21)
    x23 = apply(x20, x22)
    x24 = lbind(shift, x6)
    x25 = mapply(x24, x23)
    O = paint(I, x25)
    return O


def solve_29ec7d0e(I):
    x1 = height(I)
    x2 = width(I)
    x3 = partition(I)
    x4 = colorfilter(x3, ZERO)
    x5 = difference(x3, x4)
    x6 = merge(x5)
    x7 = astuple(x1, ONE)
    x8 = astuple(ONE, x2)
    x9 = decrement(x1)
    x10 = decrement(x2)
    x11 = toivec(x10)
    x12 = tojvec(x9)
    x13 = crop(I, x11, x8)
    x14 = crop(I, x12, x7)
    x15 = asobject(x14)
    x16 = asobject(x13)
    x17 = vperiod(x15)
    x18 = hperiod(x16)
    x19 = astuple(x17, x18)
    x20 = lbind(multiply, x19)
    x21 = neighbors(ORIGIN)
    x22 = mapply(neighbors, x21)
    x23 = apply(x20, x22)
    x24 = lbind(shift, x6)
    x25 = mapply(x24, x23)
    O = paint(I, x25)
    return O


def solve_36d67576(I):
    x1 = objects(I, F, F, T)
    x2 = argmax(x1, numcolors)
    x3 = astuple(TWO, FOUR)
    x4 = rbind(contained, x3)
    x5 = compose(x4, first)
    x6 = rbind(sfilter, x5)
    x7 = lbind(rbind, subtract)
    x8 = lbind(occurrences, I)
    x9 = lbind(lbind, shift)
    x10 = compose(x7, ulcorner)
    x11 = chain(x10, x6, normalize)
    x12 = chain(x8, x6, normalize)
    x13 = fork(apply, x11, x12)
    x14 = compose(x9, normalize)
    x15 = fork(mapply, x14, x13)
    x16 = astuple(cmirror, dmirror)
    x17 = astuple(hmirror, vmirror)
    x18 = combine(x16, x17)
    x19 = product(x18, x18)
    x20 = fork(compose, first, last)
    x21 = apply(x20, x19)
    x22 = totuple(x21)
    x23 = combine(x18, x22)
    x24 = rapply(x23, x2)
    x25 = mapply(x15, x24)
    O = paint(I, x25)
    return O


def solve_98cf29f8(I):
    x1 = fgpartition(I)
    x2 = fork(multiply, height, width)
    x3 = fork(equality, size, x2)
    x4 = extract(x1, x3)
    x5 = other(x1, x4)
    x6 = color(x5)
    x7 = rbind(greater, THREE)
    x8 = rbind(toobject, I)
    x9 = rbind(colorcount, x6)
    x10 = chain(x8, ineighbors, last)
    x11 = chain(x7, x9, x10)
    x12 = sfilter(x5, x11)
    x13 = outbox(x12)
    x14 = backdrop(x13)
    x15 = cover(I, x5)
    x16 = gravitate(x14, x4)
    x17 = shift(x14, x16)
    O = fill(x15, x6, x17)
    return O


def solve_469497ad(I):
    x1 = numcolors(I)
    x2 = decrement(x1)
    x3 = upscale(I, x2)
    x4 = objects(x3, F, F, T)
    x5 = argmin(x4, size)
    x6 = ulcorner(x5)
    x7 = llcorner(x5)
    x8 = shoot(x6, NEG_UNITY)
    x9 = shoot(x6, UNITY)
    x10 = shoot(x7, DOWN_LEFT)
    x11 = shoot(x7, UP_RIGHT)
    x12 = combine(x8, x9)
    x13 = combine(x10, x11)
    x14 = combine(x12, x13)
    x15 = underfill(x3, TWO, x14)
    x16 = objects(x15, T, F, T)
    x17 = argmax(x16, lrcorner)
    O = paint(x15, x17)
    return O


def solve_39e1d7f9(I):
    x1 = fgpartition(I)
    x2 = objects(I, T, F, T)
    x3 = order(x1, height)
    x4 = last(x3)
    x5 = remove(x4, x3)
    x6 = last(x5)
    x7 = color(x6)
    x8 = colorfilter(x2, x7)
    x9 = power(outbox, TWO)
    x10 = rbind(toobject, I)
    x11 = mostcolor(I)
    x12 = lbind(remove, x11)
    x13 = chain(size, x12, palette)
    x14 = chain(x13, x10, x9)
    x15 = argmax(x8, x14)
    x16 = ulcorner(x15)
    x17 = shape(x15)
    x18 = subtract(x16, x17)
    x19 = decrement(x18)
    x20 = multiply(x17, THREE)
    x21 = add(x20, TWO_BY_TWO)
    x22 = crop(I, x19, x21)
    x23 = asobject(x22)
    x24 = apply(ulcorner, x8)
    x25 = increment(x17)
    x26 = rbind(subtract, x25)
    x27 = apply(x26, x24)
    x28 = lbind(shift, x23)
    x29 = mapply(x28, x27)
    O = paint(I, x29)
    return O


def solve_484b58aa(I):
    x1 = height(I)
    x2 = width(I)
    x3 = partition(I)
    x4 = colorfilter(x3, ZERO)
    x5 = difference(x3, x4)
    x6 = merge(x5)
    x7 = astuple(x1, TWO)
    x8 = astuple(TWO, x2)
    x9 = power(decrement, TWO)
    x10 = x9(x1)
    x11 = x9(x2)
    x12 = toivec(x11)
    x13 = tojvec(x10)
    x14 = crop(I, x12, x8)
    x15 = crop(I, x13, x7)
    x16 = asobject(x15)
    x17 = asobject(x14)
    x18 = vperiod(x16)
    x19 = hperiod(x17)
    x20 = astuple(x18, x19)
    x21 = lbind(multiply, x20)
    x22 = neighbors(ORIGIN)
    x23 = mapply(neighbors, x22)
    x24 = apply(x21, x23)
    x25 = lbind(shift, x6)
    x26 = mapply(x25, x24)
    O = paint(I, x26)
    return O


def solve_3befdf3e(I):
    x1 = objects(I, F, F, T)
    x2 = leastcolor(I)
    x3 = palette(I)
    x4 = remove(ZERO, x3)
    x5 = other(x4, x2)
    x6 = switch(I, x2, x5)
    x7 = compose(width, inbox)
    x8 = lbind(power, outbox)
    x9 = compose(x8, x7)
    x10 = initset(x9)
    x11 = lbind(rapply, x10)
    x12 = chain(initset, first, x11)
    x13 = fork(rapply, x12, identity)
    x14 = compose(first, x13)
    x15 = compose(backdrop, x14)
    x16 = lbind(chain, backdrop)
    x17 = lbind(x16, inbox)
    x18 = compose(x17, x9)
    x19 = lbind(apply, initset)
    x20 = chain(x19, corners, x15)
    x21 = fork(mapply, x18, x20)
    x22 = fork(intersection, x15, x21)
    x23 = mapply(x15, x1)
    x24 = mapply(x22, x1)
    x25 = underfill(x6, x5, x23)
    O = fill(x25, ZERO, x24)
    return O


def solve_9aec4887(I):
    x1 = objects(I, F, T, T)
    x2 = argmin(x1, numcolors)
    x3 = other(x1, x2)
    x4 = subgrid(x3, I)
    x5 = normalize(x2)
    x6 = shift(x5, UNITY)
    x7 = toindices(x6)
    x8 = normalize(x3)
    x9 = lbind(argmin, x8)
    x11 = lbind(rbind, manhattan)
    x12 = rbind(compose, initset)
    x13 = chain(x12, x11, initset)
    x14 = chain(first, x9, x13)
    x15 = fork(astuple, x14, identity)
    x16 = apply(x15, x7)
    x17 = paint(x4, x16)
    x18 = fork(connect, ulcorner, lrcorner)
    x19 = x18(x7)
    x20 = fork(combine, identity, vmirror)
    x21 = x20(x19)
    x22 = intersection(x7, x21)
    O = fill(x17, EIGHT, x22)
    return O


def solve_49d1d64f(I):
    x1 = shape(I)
    x2 = add(x1, TWO)
    x3 = canvas(ZERO, x2)
    x4 = asobject(I)
    x5 = shift(x4, UNITY)
    x6 = paint(x3, x5)
    x7 = asindices(x3)
    x8 = fork(difference, box, corners)
    x9 = x8(x7)
    x10 = lbind(lbind, manhattan)
    x11 = rbind(compose, initset)
    x12 = chain(x11, x10, initset)
    x13 = lbind(argmin, x5)
    x14 = chain(first, x13, x12)
    x15 = fork(astuple, x14, identity)
    x16 = apply(x15, x9)
    O = paint(x6, x16)
    return O


def solve_57aa92db(I):
    x1 = objects(I, F, T, T)
    x2 = objects(I, T, F, T)
    x3 = lbind(lbind, colorcount)
    x4 = fork(apply, x3, palette)
    x5 = compose(maximum, x4)
    x6 = compose(minimum, x4)
    x7 = fork(subtract, x5, x6)
    x8 = argmax(x1, x7)
    x9 = leastcolor(x8)
    x10 = normalize(x8)
    x11 = matcher(first, x9)
    x12 = sfilter(x10, x11)
    x13 = ulcorner(x12)
    x14 = colorfilter(x2, x9)
    x15 = rbind(toobject, I)
    x16 = lbind(remove, ZERO)
    x17 = chain(first, x16, palette)
    x18 = chain(x17, x15, outbox)
    x19 = lbind(multiply, x13)
    x20 = compose(x19, width)
    x21 = fork(subtract, ulcorner, x20)
    x22 = lbind(shift, x10)
    x23 = compose(x22, x21)
    x24 = fork(upscale, x23, width)
    x25 = fork(recolor, x18, x24)
    x26 = mapply(x25, x14)
    x27 = paint(I, x26)
    x28 = merge(x2)
    O = paint(x27, x28)
    return O


def solve_aba27056(I):
    x1 = objects(I, T, F, T)
    x2 = mapply(toindices, x1)
    x3 = box(x2)
    x4 = difference(x3, x2)
    x5 = delta(x2)
    x6 = position(x5, x4)
    x7 = interval(ZERO, NINE, ONE)
    x8 = lbind(multiply, x6)
    x9 = apply(x8, x7)
    x10 = lbind(shift, x4)
    x11 = mapply(x10, x9)
    x12 = fill(I, FOUR, x5)
    x13 = fill(x12, FOUR, x11)
    x14 = corners(x4)
    x15 = ofcolor(x13, ZERO)
    x16 = rbind(toobject, x13)
    x17 = rbind(colorcount, ZERO)
    x18 = chain(x17, x16, dneighbors)
    x19 = matcher(x18, TWO)
    x20 = rbind(adjacent, x2)
    x21 = rbind(adjacent, x11)
    x22 = fork(both, x20, x21)
    x23 = compose(x22, initset)
    x24 = sfilter(x15, x19)
    x25 = sfilter(x24, x23)
    x26 = product(x14, x25)
    x27 = fork(subtract, last, first)
    x28 = fork(shoot, first, x27)
    x29 = mapply(x28, x26)
    O = fill(x13, FOUR, x29)
    return O


def solve_f1cefba8(I):
    x1 = palette(I)
    x2 = objects(I, F, F, T)
    x3 = ofcolor(I, ZERO)
    x4 = first(x2)
    x5 = ulcorner(x4)
    x6 = subgrid(x4, I)
    x7 = power(trim, TWO)
    x8 = x7(x6)
    x9 = asindices(x8)
    x10 = shift(x9, TWO_BY_TWO)
    x11 = fill(x6, ZERO, x10)
    x12 = leastcolor(x11)
    x13 = remove(ZERO, x1)
    x14 = other(x13, x12)
    x15 = ofcolor(x11, x12)
    x16 = shift(x15, x5)
    x17 = ofcolor(I, x12)
    x18 = uppermost(x17)
    x19 = lowermost(x17)
    x20 = matcher(first, x18)
    x21 = matcher(first, x19)
    x22 = fork(either, x20, x21)
    x23 = sfilter(x16, x22)
    x24 = difference(x16, x23)
    x25 = mapply(vfrontier, x23)
    x26 = mapply(hfrontier, x24)
    x27 = combine(x25, x26)
    x28 = intersection(x3, x27)
    x29 = fill(I, x14, x27)
    O = fill(x29, x12, x28)
    return O


def solve_1e32b0e9(I):
    x1 = height(I)
    x2 = mostcolor(I)
    x3 = asobject(I)
    x4 = subtract(x1, TWO)
    x5 = divide(x4, THREE)
    x6 = astuple(x5, x5)
    x7 = crop(I, ORIGIN, x6)
    x8 = partition(x7)
    x9 = matcher(color, ZERO)
    x10 = compose(flip, x9)
    x11 = extract(x8, x10)
    x12 = initset(x2)
    x13 = palette(x3)
    x14 = palette(x11)
    x15 = difference(x13, x14)
    x16 = difference(x15, x12)
    x17 = first(x16)
    x18 = interval(ZERO, THREE, ONE)
    x19 = product(x18, x18)
    x20 = totuple(x19)
    x21 = apply(first, x20)
    x22 = apply(last, x20)
    x23 = lbind(multiply, x5)
    x24 = apply(x23, x21)
    x25 = apply(x23, x22)
    x26 = papply(add, x24, x21)
    x27 = papply(add, x25, x22)
    x28 = papply(astuple, x26, x27)
    x29 = lbind(shift, x11)
    x30 = mapply(x29, x28)
    O = underfill(I, x17, x30)
    return O


def solve_28e73c20(I):
    x1 = width(I)
    x2 = astuple(ONE, TWO)
    x3 = astuple(TWO, TWO)
    x4 = astuple(TWO, ONE)
    x5 = astuple(THREE, ONE)
    x6 = canvas(THREE, UNITY)
    x7 = upscale(x6, FOUR)
    x8 = initset(DOWN)
    x9 = insert(UNITY, x8)
    x10 = insert(x2, x9)
    x11 = insert(x3, x10)
    x12 = fill(x7, ZERO, x11)
    x13 = vupscale(x6, FIVE)
    x14 = hupscale(x13, THREE)
    x15 = insert(x4, x9)
    x16 = insert(x5, x15)
    x17 = fill(x14, ZERO, x16)
    x18 = even(x1)
    x19 = branch(x18, x12, x17)
    x20 = canvas(ZERO, UNITY)
    x21 = lbind(hupscale, x20)
    x22 = chain(x21, decrement, height)
    x23 = rbind(hconcat, x6)
    x24 = compose(x23, x22)
    x25 = lbind(hupscale, x6)
    x26 = compose(x25, height)
    x27 = fork(vconcat, x24, rot90)
    x28 = fork(vconcat, x26, x27)
    x29 = subtract(x1, FOUR)
    x30 = power(x28, x29)
    O = x30(x19)
    return O


def solve_4c5c2cf0(I):
    x1 = objects(I, T, T, T)
    x2 = objects(I, F, T, T)
    x3 = first(x2)
    x4 = rbind(subgrid, I)
    x5 = fork(equality, identity, rot90)
    x6 = compose(x5, x4)
    x7 = extract(x1, x6)
    x8 = center(x7)
    x9 = subgrid(x3, I)
    x10 = hmirror(x9)
    x11 = objects(x10, F, T, T)
    x12 = first(x11)
    x13 = objects(x10, T, T, T)
    x14 = rbind(subgrid, x10)
    x15 = compose(x5, x14)
    x16 = extract(x13, x15)
    x17 = center(x16)
    x18 = subtract(x8, x17)
    x19 = shift(x12, x18)
    x20 = paint(I, x19)
    x21 = objects(x20, F, T, T)
    x22 = first(x21)
    x23 = subgrid(x22, x20)
    x24 = vmirror(x23)
    x25 = objects(x24, F, T, T)
    x26 = first(x25)
    x27 = objects(x24, T, T, T)
    x28 = color(x7)
    x29 = matcher(color, x28)
    x30 = extract(x27, x29)
    x31 = center(x30)
    x32 = subtract(x8, x31)
    x33 = shift(x26, x32)
    O = paint(x20, x33)
    return O


def solve_508bd3b6(I):
    x1 = width(I)
    x2 = objects(I, T, T, T)
    x3 = argmin(x2, size)
    x4 = argmax(x2, size)
    x5 = ulcorner(x3)
    x6 = urcorner(x3)
    x7 = index(I, x5)
    x8 = equality(x7, EIGHT)
    x9 = branch(x8, x5, x6)
    x10 = branch(x8, UNITY, DOWN_LEFT)
    x11 = multiply(x10, x1)
    x12 = double(x11)
    x13 = add(x9, x12)
    x14 = subtract(x9, x12)
    x15 = connect(x13, x14)
    x16 = fill(I, THREE, x15)
    x17 = paint(x16, x4)
    x18 = objects(x17, T, F, T)
    x19 = rbind(adjacent, x4)
    x20 = extract(x18, x19)
    x21 = first(x20)
    x22 = last(x21)
    x23 = flip(x8)
    x24 = branch(x23, UNITY, DOWN_LEFT)
    x25 = multiply(x24, x1)
    x26 = double(x25)
    x27 = add(x22, x26)
    x28 = subtract(x22, x26)
    x29 = connect(x27, x28)
    x30 = fill(x17, THREE, x29)
    x31 = paint(x30, x3)
    O = paint(x31, x4)
    return O


def solve_6d0160f0(I):
    x1 = ofcolor(I, FOUR)
    x2 = first(x1)
    x3 = first(x2)
    x4 = last(x2)
    x5 = greater(x3, THREE)
    x6 = greater(x3, SEVEN)
    x7 = greater(x4, THREE)
    x8 = greater(x4, SEVEN)
    x9 = branch(x5, FOUR, ZERO)
    x10 = branch(x6, EIGHT, x9)
    x11 = branch(x7, FOUR, ZERO)
    x12 = branch(x8, EIGHT, x11)
    x13 = astuple(x10, x12)
    x14 = initset(ZERO)
    x15 = insert(FOUR, x14)
    x16 = insert(EIGHT, x15)
    x17 = product(x16, x16)
    x18 = crop(I, ORIGIN, THREE_BY_THREE)
    x19 = asindices(x18)
    x20 = recolor(ZERO, x19)
    x21 = lbind(shift, x20)
    x22 = mapply(x21, x17)
    x23 = paint(I, x22)
    x24 = crop(I, x13, THREE_BY_THREE)
    x25 = replace(x24, FIVE, ZERO)
    x26 = ofcolor(x25, FOUR)
    x27 = first(x26)
    x28 = asindices(x25)
    x29 = toobject(x28, x25)
    x30 = multiply(x27, FOUR)
    x31 = shift(x29, x30)
    O = paint(x23, x31)
    return O


def solve_f8a8fe49(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, FIVE, ZERO)
    x3 = colorfilter(x1, TWO)
    x4 = first(x3)
    x5 = portrait(x4)
    x6 = branch(x5, hsplit, vsplit)
    x7 = branch(x5, vmirror, hmirror)
    x8 = ofcolor(I, TWO)
    x9 = subgrid(x8, I)
    x10 = trim(x9)
    x11 = x7(x10)
    x12 = x6(x11, TWO)
    x13 = compose(normalize, asobject)
    x14 = apply(x13, x12)
    x15 = last(x14)
    x16 = first(x14)
    x17 = ulcorner(x8)
    x18 = increment(x17)
    x19 = shift(x15, x18)
    x20 = shift(x16, x18)
    x21 = branch(x5, width, height)
    x22 = branch(x5, tojvec, toivec)
    x23 = x21(x15)
    x24 = double(x23)
    x25 = compose(x22, increment)
    x26 = x25(x23)
    x27 = invert(x26)
    x28 = x25(x24)
    x29 = shift(x19, x27)
    x30 = shift(x20, x28)
    x31 = paint(x2, x29)
    O = paint(x31, x30)
    return O


def solve_d07ae81c(I):
    x1 = objects(I, T, F, F)
    x2 = sizefilter(x1, ONE)
    x3 = apply(color, x2)
    x4 = difference(x1, x2)
    x5 = apply(color, x4)
    x6 = first(x5)
    x7 = last(x5)
    x8 = ofcolor(I, x6)
    x9 = ofcolor(I, x7)
    x10 = rbind(shoot, UNITY)
    x11 = rbind(shoot, NEG_UNITY)
    x12 = rbind(shoot, DOWN_LEFT)
    x13 = rbind(shoot, UP_RIGHT)
    x14 = fork(combine, x10, x11)
    x15 = fork(combine, x12, x13)
    x16 = fork(combine, x14, x15)
    x17 = compose(x16, center)
    x18 = mapply(x17, x2)
    x19 = intersection(x8, x18)
    x20 = intersection(x9, x18)
    x21 = first(x2)
    x22 = color(x21)
    x23 = center(x21)
    x24 = neighbors(x23)
    x25 = toobject(x24, I)
    x26 = mostcolor(x25)
    x27 = other(x3, x22)
    x28 = equality(x26, x6)
    x29 = branch(x28, x22, x27)
    x30 = branch(x28, x27, x22)
    x31 = fill(I, x29, x19)
    O = fill(x31, x30, x20)
    return O


def solve_6a1e5592(I):
    x1 = width(I)
    x2 = objects(I, T, F, T)
    x3 = astuple(FIVE, x1)
    x4 = crop(I, ORIGIN, x3)
    x5 = colorfilter(x2, FIVE)
    x6 = merge(x5)
    x7 = cover(I, x6)
    x8 = compose(toindices, normalize)
    x9 = apply(x8, x5)
    x10 = asindices(x4)
    x11 = ofcolor(x4, ZERO)
    x12 = ofcolor(x4, TWO)
    x13 = rbind(multiply, TEN)
    x14 = rbind(multiply, FIVE)
    x15 = rbind(intersection, x12)
    x16 = rbind(intersection, x11)
    x17 = rbind(intersection, x10)
    x18 = chain(x13, size, x15)
    x19 = chain(size, x16, delta)
    x20 = compose(x14, uppermost)
    x21 = chain(size, x16, outbox)
    x22 = chain(x13, size, x17)
    x23 = compose(invert, x18)
    x24 = fork(add, x22, x23)
    x25 = fork(subtract, x24, x21)
    x26 = fork(subtract, x25, x20)
    x27 = fork(subtract, x26, x19)
    x28 = rbind(apply, x10)
    x29 = lbind(lbind, shift)
    x30 = rbind(argmax, x27)
    x31 = chain(x30, x28, x29)
    x32 = mapply(x31, x9)
    O = fill(x7, ONE, x32)
    return O


def solve_0e206a2e(I):
    x1 = palette(I)
    x2 = objects(I, F, F, T)
    x3 = rbind(greater, ONE)
    x4 = compose(x3, numcolors)
    x5 = sfilter(x2, x4)
    x6 = remove(ZERO, x1)
    x7 = lbind(colorcount, I)
    x8 = argmax(x6, x7)
    x9 = remove(x8, x6)
    x10 = rbind(contained, x9)
    x11 = compose(x10, first)
    x12 = rbind(sfilter, x11)
    x13 = lbind(rbind, subtract)
    x14 = lbind(occurrences, I)
    x15 = lbind(lbind, shift)
    x16 = compose(x13, ulcorner)
    x17 = chain(x16, x12, normalize)
    x18 = chain(x14, x12, normalize)
    x19 = fork(apply, x17, x18)
    x20 = compose(x15, normalize)
    x21 = fork(mapply, x20, x19)
    x22 = astuple(cmirror, dmirror)
    x23 = astuple(hmirror, vmirror)
    x24 = combine(x22, x23)
    x25 = product(x24, x24)
    x26 = fork(compose, first, last)
    x27 = apply(x26, x25)
    x28 = totuple(x27)
    x29 = combine(x24, x28)
    x30 = lbind(rapply, x29)
    x31 = mapply(x30, x5)
    x32 = mapply(x21, x31)
    x33 = paint(I, x32)
    x34 = merge(x5)
    O = cover(x33, x34)
    return O


def solve_d22278a0(I):
    x1 = asindices(I)
    x2 = objects(I, T, F, T)
    x3 = fork(multiply, sign, identity)
    x4 = lbind(apply, x3)
    x5 = chain(even, maximum, x4)
    x6 = lbind(sfilter, x1)
    x7 = fork(add, first, last)
    x8 = rbind(remove, x2)
    x9 = compose(center, last)
    x10 = fork(subtract, first, x9)
    x11 = compose(x5, x10)
    x12 = lbind(rbind, equality)
    x13 = lbind(argmin, x2)
    x14 = chain(x7, x4, x10)
    x15 = lbind(lbind, astuple)
    x16 = lbind(rbind, astuple)
    x17 = lbind(compose, x11)
    x18 = lbind(compose, x14)
    x19 = compose(x18, x15)
    x20 = compose(x18, x16)
    x21 = compose(x13, x19)
    x22 = rbind(compose, x21)
    x23 = lbind(lbind, valmin)
    x24 = rbind(compose, x19)
    x25 = chain(x24, x23, x8)
    x26 = lbind(fork, greater)
    x27 = fork(x26, x25, x20)
    x28 = chain(x6, x17, x16)
    x29 = chain(x6, x22, x12)
    x30 = fork(intersection, x28, x29)
    x31 = compose(x6, x27)
    x32 = fork(intersection, x30, x31)
    x33 = fork(recolor, color, x32)
    x34 = mapply(x33, x2)
    O = paint(I, x34)
    return O


def solve_4290ef0e(I):
    x1 = mostcolor(I)
    x2 = fgpartition(I)
    x3 = objects(I, T, F, T)
    x4 = rbind(valmax, width)
    x5 = lbind(colorfilter, x3)
    x6 = compose(x5, color)
    x7 = compose(double, x4)
    x8 = lbind(prapply, manhattan)
    x9 = fork(x8, identity, identity)
    x10 = lbind(remove, ZERO)
    x11 = compose(x10, x9)
    x12 = rbind(branch, NEG_TWO)
    x13 = fork(x12, positive, decrement)
    x14 = chain(x13, minimum, x11)
    x15 = fork(add, x14, x7)
    x16 = compose(x15, x6)
    x17 = compose(invert, x16)
    x18 = order(x2, x17)
    x19 = rbind(argmin, centerofmass)
    x20 = compose(initset, vmirror)
    x21 = fork(insert, dmirror, x20)
    x22 = fork(insert, cmirror, x21)
    x23 = fork(insert, hmirror, x22)
    x24 = compose(x19, x23)
    x25 = apply(x24, x18)
    x26 = size(x2)
    x27 = apply(size, x2)
    x28 = contained(ONE, x27)
    x29 = increment(x26)
    x30 = branch(x28, x26, x29)
    x31 = double(x30)
    x32 = decrement(x31)
    x33 = apply(normalize, x25)
    x34 = interval(ZERO, x30, ONE)
    x35 = pair(x34, x34)
    x36 = mpapply(shift, x33, x35)
    x37 = astuple(x32, x32)
    x38 = canvas(x1, x37)
    x39 = paint(x38, x36)
    x40 = rot90(x39)
    x41 = paint(x40, x36)
    x42 = rot90(x41)
    x43 = paint(x42, x36)
    x44 = rot90(x43)
    O = paint(x44, x36)
    return O


def solve_50846271(I):
    x1 = ofcolor(I, TWO)
    x2 = prapply(connect, x1, x1)
    x3 = lbind(greater, SIX)
    x4 = compose(x3, size)
    x5 = fork(either, vline, hline)
    x6 = fork(both, x4, x5)
    x7 = mfilter(x2, x6)
    x8 = fill(I, TWO, x7)
    x9 = objects(x8, T, F, F)
    x10 = colorfilter(x9, TWO)
    x11 = valmax(x10, width)
    x12 = halve(x11)
    x13 = toivec(x12)
    x14 = tojvec(x12)
    x15 = rbind(add, ZERO_BY_TWO)
    x16 = rbind(add, TWO_BY_ZERO)
    x17 = rbind(subtract, ZERO_BY_TWO)
    x18 = rbind(subtract, TWO_BY_ZERO)
    x19 = rbind(colorcount, TWO)
    x20 = rbind(toobject, x8)
    x21 = compose(initset, x15)
    x22 = fork(insert, x16, x21)
    x23 = fork(insert, x17, x22)
    x24 = fork(insert, x18, x23)
    x25 = fork(combine, dneighbors, x24)
    x26 = chain(x19, x20, x25)
    x27 = rbind(argmax, x26)
    x28 = compose(x27, toindices)
    x29 = apply(x28, x10)
    x30 = rbind(add, x13)
    x31 = rbind(subtract, x13)
    x32 = rbind(add, x14)
    x33 = rbind(subtract, x14)
    x34 = fork(connect, x30, x31)
    x35 = fork(connect, x32, x33)
    x36 = fork(combine, x34, x35)
    x37 = mapply(x36, x29)
    x38 = fill(x8, EIGHT, x37)
    O = fill(x38, TWO, x1)
    return O


def solve_b527c5c6(I):
    x1 = objects(I, F, F, T)
    x2 = matcher(first, TWO)
    x3 = rbind(sfilter, x2)
    x4 = compose(lowermost, x3)
    x5 = compose(rightmost, x3)
    x6 = compose(uppermost, x3)
    x7 = compose(leftmost, x3)
    x8 = fork(equality, x4, lowermost)
    x9 = fork(equality, x5, rightmost)
    x10 = fork(equality, x6, uppermost)
    x11 = fork(equality, x7, leftmost)
    x12 = compose(invert, x10)
    x13 = compose(invert, x11)
    x14 = fork(add, x12, x8)
    x15 = fork(add, x13, x9)
    x16 = fork(astuple, x14, x15)
    x17 = compose(center, x3)
    x18 = fork(shoot, x17, x16)
    x19 = mapply(x18, x1)
    x20 = fill(I, TWO, x19)
    x21 = compose(vline, x18)
    x22 = sfilter(x1, x21)
    x23 = difference(x1, x22)
    x24 = chain(decrement, minimum, shape)
    x25 = compose(increment, x24)
    x26 = compose(invert, x24)
    x27 = rbind(interval, ONE)
    x28 = fork(x27, x26, x25)
    x29 = lbind(apply, toivec)
    x30 = lbind(apply, tojvec)
    x31 = lbind(lbind, shift)
    x32 = compose(x31, x18)
    x33 = compose(x29, x28)
    x34 = compose(x30, x28)
    x35 = fork(mapply, x32, x33)
    x36 = fork(mapply, x32, x34)
    x37 = mapply(x35, x23)
    x38 = mapply(x36, x22)
    x39 = combine(x37, x38)
    O = underfill(x20, THREE, x39)
    return O


def solve_150deff5(I):
    x1 = canvas(FIVE, TWO_BY_TWO)
    x2 = asobject(x1)
    x3 = occurrences(I, x2)
    x4 = lbind(shift, x2)
    x5 = mapply(x4, x3)
    x6 = fill(I, EIGHT, x5)
    x7 = canvas(FIVE, UNITY)
    x8 = astuple(TWO, ONE)
    x9 = canvas(EIGHT, x8)
    x10 = vconcat(x9, x7)
    x11 = asobject(x10)
    x12 = occurrences(x6, x11)
    x13 = lbind(shift, x11)
    x14 = mapply(x13, x12)
    x15 = fill(x6, TWO, x14)
    x16 = astuple(ONE, THREE)
    x17 = canvas(FIVE, x16)
    x18 = asobject(x17)
    x19 = occurrences(x15, x18)
    x20 = lbind(shift, x18)
    x21 = mapply(x20, x19)
    x22 = fill(x15, TWO, x21)
    x23 = hmirror(x10)
    x24 = asobject(x23)
    x25 = occurrences(x22, x24)
    x26 = lbind(shift, x24)
    x27 = mapply(x26, x25)
    x28 = fill(x22, TWO, x27)
    x29 = dmirror(x10)
    x30 = asobject(x29)
    x31 = occurrences(x28, x30)
    x32 = lbind(shift, x30)
    x33 = mapply(x32, x31)
    x34 = fill(x28, TWO, x33)
    x35 = vmirror(x29)
    x36 = asobject(x35)
    x37 = occurrences(x34, x36)
    x38 = lbind(shift, x36)
    x39 = mapply(x38, x37)
    O = fill(x34, TWO, x39)
    return O


def solve_b7249182(I):
    x1 = objects(I, T, F, T)
    x2 = merge(x1)
    x3 = portrait(x2)
    x4 = branch(x3, identity, dmirror)
    x5 = x4(I)
    x6 = objects(x5, T, F, T)
    x7 = order(x6, uppermost)
    x8 = first(x7)
    x9 = last(x7)
    x10 = color(x8)
    x11 = color(x9)
    x12 = compose(first, toindices)
    x13 = x12(x8)
    x14 = x12(x9)
    x15 = connect(x13, x14)
    x16 = centerofmass(x15)
    x17 = connect(x13, x16)
    x18 = fill(x5, x11, x15)
    x19 = fill(x18, x10, x17)
    x20 = add(x16, DOWN)
    x21 = initset(x16)
    x22 = insert(x20, x21)
    x23 = toobject(x22, x19)
    x24 = astuple(ZERO, NEG_TWO)
    x25 = shift(x23, ZERO_BY_TWO)
    x26 = shift(x23, x24)
    x27 = combine(x25, x26)
    x28 = ulcorner(x27)
    x29 = urcorner(x27)
    x30 = connect(x28, x29)
    x31 = shift(x30, UP)
    x32 = llcorner(x27)
    x33 = lrcorner(x27)
    x34 = connect(x32, x33)
    x35 = shift(x34, DOWN)
    x36 = paint(x19, x27)
    x37 = fill(x36, x10, x31)
    x38 = fill(x37, x11, x35)
    x39 = cover(x38, x22)
    O = x4(x39)
    return O


def solve_9d9215db(I):
    x1 = rot90(I)
    x2 = rot180(I)
    x3 = rot270(I)
    x4 = initset(I)
    x5 = chain(numcolors, lefthalf, tophalf)
    x6 = insert(x1, x4)
    x7 = insert(x2, x6)
    x8 = insert(x3, x7)
    x9 = argmax(x8, x5)
    x10 = vmirror(x9)
    x11 = papply(pair, x9, x10)
    x12 = lbind(apply, maximum)
    x13 = apply(x12, x11)
    x14 = partition(x13)
    x15 = sizefilter(x14, FOUR)
    x16 = apply(llcorner, x15)
    x17 = apply(lrcorner, x15)
    x18 = combine(x16, x17)
    x19 = cover(x13, x18)
    x20 = tojvec(NEG_TWO)
    x21 = rbind(add, ZERO_BY_TWO)
    x22 = rbind(add, x20)
    x23 = compose(x21, ulcorner)
    x24 = compose(x22, urcorner)
    x25 = fork(connect, x23, x24)
    x26 = compose(even, last)
    x27 = rbind(sfilter, x26)
    x28 = chain(normalize, x27, x25)
    x29 = fork(shift, x28, x23)
    x30 = fork(recolor, color, x29)
    x31 = mapply(x30, x15)
    x32 = paint(x19, x31)
    x33 = rot90(x32)
    x34 = rot180(x32)
    x35 = rot270(x32)
    x36 = papply(pair, x32, x33)
    x37 = apply(x12, x36)
    x38 = papply(pair, x37, x34)
    x39 = apply(x12, x38)
    x40 = papply(pair, x39, x35)
    O = apply(x12, x40)
    return O


def solve_6855a6e4(I):
    x1 = fgpartition(I)
    x2 = rot90(I)
    x3 = colorfilter(x1, TWO)
    x4 = first(x3)
    x5 = portrait(x4)
    x6 = branch(x5, I, x2)
    x7 = objects(x6, T, F, T)
    x8 = colorfilter(x7, FIVE)
    x9 = apply(center, x8)
    x10 = valmin(x9, first)
    x11 = compose(first, center)
    x12 = matcher(x11, x10)
    x13 = compose(flip, x12)
    x14 = extract(x8, x12)
    x15 = extract(x8, x13)
    x16 = ulcorner(x14)
    x17 = ulcorner(x15)
    x18 = subgrid(x14, x6)
    x19 = subgrid(x15, x6)
    x20 = hmirror(x18)
    x21 = hmirror(x19)
    x22 = ofcolor(x20, FIVE)
    x23 = recolor(FIVE, x22)
    x24 = ofcolor(x21, FIVE)
    x25 = recolor(FIVE, x24)
    x26 = height(x23)
    x27 = height(x25)
    x28 = add(THREE, x26)
    x29 = add(THREE, x27)
    x30 = toivec(x28)
    x31 = toivec(x29)
    x32 = add(x16, x30)
    x33 = subtract(x17, x31)
    x34 = shift(x23, x32)
    x35 = shift(x25, x33)
    x36 = merge(x8)
    x37 = cover(x6, x36)
    x38 = paint(x37, x34)
    x39 = paint(x38, x35)
    x40 = rot270(x39)
    O = branch(x5, x39, x40)
    return O


def solve_264363fd(I):
    x1 = objects(I, F, F, T)
    x2 = argmin(x1, size)
    x3 = normalize(x2)
    x4 = height(x2)
    x5 = width(x2)
    x6 = equality(x4, FIVE)
    x7 = equality(x5, FIVE)
    x8 = astuple(x6, x7)
    x9 = add(UNITY, x8)
    x10 = invert(x9)
    x11 = center(x2)
    x12 = index(I, x11)
    x13 = branch(x6, UP, RIGHT)
    x14 = add(x13, x11)
    x15 = index(I, x14)
    x16 = astuple(x12, ORIGIN)
    x17 = initset(x16)
    x18 = cover(I, x2)
    x19 = mostcolor(x18)
    x20 = ofcolor(x18, x19)
    x21 = occurrences(x18, x17)
    x22 = objects(x18, F, F, T)
    x23 = rbind(occurrences, x17)
    x24 = rbind(subgrid, x18)
    x25 = compose(x23, x24)
    x26 = lbind(mapply, vfrontier)
    x27 = lbind(mapply, hfrontier)
    x28 = compose(x26, x25)
    x29 = compose(x27, x25)
    x30 = branch(x6, x28, x29)
    x31 = branch(x7, x29, x28)
    x32 = fork(combine, x30, x31)
    x33 = lbind(recolor, x15)
    x34 = compose(x33, x32)
    x35 = fork(paint, x24, x34)
    x36 = compose(asobject, x35)
    x37 = fork(shift, x36, ulcorner)
    x38 = mapply(x37, x22)
    x39 = paint(x18, x38)
    x40 = shift(x3, x10)
    x41 = lbind(shift, x40)
    x42 = mapply(x41, x21)
    x43 = paint(x39, x42)
    O = fill(x43, x19, x20)
    return O


def solve_7df24a62(I):
    x1 = height(I)
    x2 = width(I)
    x3 = ofcolor(I, ONE)
    x4 = ofcolor(I, FOUR)
    x5 = ulcorner(x3)
    x6 = subgrid(x3, I)
    x7 = rot90(x6)
    x8 = rot180(x6)
    x9 = rot270(x6)
    x10 = matcher(size, ZERO)
    x11 = rbind(ofcolor, ONE)
    x12 = compose(normalize, x11)
    x13 = rbind(ofcolor, FOUR)
    x14 = rbind(shift, x5)
    x15 = compose(x14, x13)
    x16 = lbind(subtract, x1)
    x17 = chain(increment, x16, height)
    x18 = lbind(subtract, x2)
    x19 = chain(increment, x18, width)
    x20 = rbind(interval, ONE)
    x21 = lbind(x20, ZERO)
    x22 = compose(x21, x17)
    x23 = compose(x21, x19)
    x24 = fork(product, x22, x23)
    x25 = rbind(shift, NEG_UNITY)
    x26 = lbind(lbind, shift)
    x27 = chain(x26, x25, x12)
    x28 = astuple(x6, x7)
    x29 = astuple(x8, x9)
    x30 = combine(x28, x29)
    x31 = apply(x15, x30)
    x32 = lbind(difference, x4)
    x33 = apply(x32, x31)
    x34 = apply(normalize, x31)
    x35 = apply(x24, x34)
    x36 = lbind(rbind, difference)
    x37 = apply(x26, x34)
    x38 = apply(x36, x33)
    x39 = papply(compose, x38, x37)
    x40 = lbind(compose, x10)
    x41 = apply(x40, x39)
    x42 = papply(sfilter, x35, x41)
    x43 = apply(x27, x30)
    x44 = mpapply(mapply, x43, x42)
    O = fill(I, ONE, x44)
    return O


def solve_f15e1fac(I):
    x1 = ofcolor(I, TWO)
    x2 = portrait(x1)
    x3 = branch(x2, identity, dmirror)
    x4 = x3(I)
    x5 = leftmost(x1)
    x6 = equality(x5, ZERO)
    x7 = branch(x6, identity, vmirror)
    x8 = x7(x4)
    x9 = ofcolor(x8, EIGHT)
    x10 = uppermost(x9)
    x11 = equality(x10, ZERO)
    x12 = branch(x11, identity, hmirror)
    x13 = x12(x8)
    x14 = ofcolor(x13, EIGHT)
    x15 = ofcolor(x13, TWO)
    x16 = rbind(shoot, DOWN)
    x17 = mapply(x16, x14)
    x18 = height(x13)
    x19 = apply(first, x15)
    x20 = insert(ZERO, x19)
    x21 = insert(x18, x19)
    x22 = apply(decrement, x21)
    x23 = order(x20, identity)
    x24 = order(x22, identity)
    x25 = size(x15)
    x26 = increment(x25)
    x27 = interval(ZERO, x26, ONE)
    x28 = apply(tojvec, x27)
    x29 = pair(x23, x24)
    x30 = lbind(sfilter, x17)
    x31 = compose(first, last)
    x32 = chain(decrement, first, first)
    x33 = fork(greater, x31, x32)
    x34 = chain(increment, last, first)
    x35 = fork(greater, x34, x31)
    x36 = fork(both, x33, x35)
    x37 = lbind(lbind, astuple)
    x38 = lbind(compose, x36)
    x39 = chain(x30, x38, x37)
    x40 = apply(x39, x29)
    x41 = papply(shift, x40, x28)
    x42 = merge(x41)
    x43 = fill(x13, EIGHT, x42)
    x44 = chain(x3, x7, x12)
    O = x44(x43)
    return O


def solve_234bbc79(I):
    x1 = objects(I, F, F, T)
    x2 = rbind(other, FIVE)
    x3 = compose(x2, palette)
    x4 = fork(recolor, x3, identity)
    x5 = apply(x4, x1)
    x6 = order(x5, leftmost)
    x7 = compose(last, last)
    x8 = lbind(matcher, x7)
    x9 = compose(x8, leftmost)
    x10 = compose(x8, rightmost)
    x11 = fork(sfilter, identity, x9)
    x12 = fork(sfilter, identity, x10)
    x13 = compose(dneighbors, last)
    x14 = rbind(chain, x13)
    x15 = lbind(x14, size)
    x16 = lbind(rbind, intersection)
    x17 = chain(x15, x16, toindices)
    x18 = fork(argmin, x11, x17)
    x19 = fork(argmin, x12, x17)
    x20 = compose(last, x18)
    x21 = compose(last, x19)
    x22 = astuple(ZERO, DOWN_LEFT)
    x23 = initset(x22)
    x24 = lbind(add, RIGHT)
    x25 = chain(x20, first, last)
    x26 = compose(x21, first)
    x27 = fork(subtract, x26, x25)
    x28 = compose(first, last)
    x29 = compose(x24, x27)
    x30 = fork(shift, x28, x29)
    x31 = fork(combine, first, x30)
    x32 = fork(remove, x28, last)
    x33 = fork(astuple, x31, x32)
    x34 = size(x1)
    x35 = power(x33, x34)
    x36 = astuple(x23, x6)
    x37 = x35(x36)
    x38 = first(x37)
    x39 = width(x38)
    x40 = decrement(x39)
    x41 = astuple(THREE, x40)
    x42 = canvas(ZERO, x41)
    O = paint(x42, x38)
    return O


def solve_22233c11(I):
    x1 = objects(I, T, T, T)
    x2 = rbind(upscale, TWO)
    x3 = chain(invert, halve, shape)
    x4 = fork(combine, hfrontier, vfrontier)
    x5 = compose(x2, vmirror)
    x6 = fork(shift, x5, x3)
    x7 = compose(toindices, x6)
    x8 = lbind(mapply, x4)
    x9 = compose(x8, toindices)
    x10 = fork(difference, x7, x9)
    x11 = mapply(x10, x1)
    O = fill(I, EIGHT, x11)
    return O


def solve_2dd70a9a(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, THREE)
    x3 = vline(x1)
    x4 = vline(x2)
    x5 = center(x1)
    x6 = branch(x4, uppermost, rightmost)
    x7 = x6(x1)
    x8 = x6(x2)
    x9 = greater(x7, x8)
    x10 = both(x4, x9)
    x11 = branch(x10, lowermost, uppermost)
    x12 = x11(x2)
    x13 = branch(x4, leftmost, rightmost)
    x14 = x13(x2)
    x15 = astuple(x12, x14)
    x16 = other(x2, x15)
    x17 = subtract(x15, x16)
    x18 = shoot(x15, x17)
    x19 = underfill(I, ONE, x18)
    x20 = objects(x19, T, F, F)
    x21 = colorfilter(x20, ONE)
    x22 = rbind(adjacent, x2)
    x23 = sfilter(x21, x22)
    x24 = difference(x21, x23)
    x25 = merge(x24)
    x26 = cover(x19, x25)
    x27 = shoot(x5, DOWN)
    x28 = shoot(x5, UP)
    x29 = shoot(x5, LEFT)
    x30 = shoot(x5, RIGHT)
    x31 = combine(x27, x28)
    x32 = combine(x29, x30)
    x33 = branch(x3, x31, x32)
    x34 = ofcolor(x26, ONE)
    x35 = initset(x15)
    x36 = rbind(manhattan, x35)
    x37 = compose(x36, initset)
    x38 = argmax(x34, x37)
    x39 = initset(x38)
    x40 = gravitate(x39, x33)
    x41 = crement(x40)
    x42 = add(x38, x41)
    x43 = connect(x38, x42)
    x44 = fill(x26, ONE, x43)
    x45 = connect(x42, x5)
    x46 = underfill(x44, ONE, x45)
    O = replace(x46, ONE, THREE)
    return O


def solve_a64e4611(I):
    x1 = asindices(I)
    x2 = fork(product, identity, identity)
    x3 = lbind(canvas, ZERO)
    x4 = compose(asobject, x3)
    x5 = fork(multiply, first, last)
    x6 = compose(positive, size)
    x7 = lbind(lbind, shift)
    x8 = rbind(fork, x5)
    x9 = lbind(x8, multiply)
    x10 = lbind(chain, x6)
    x11 = rbind(x10, x4)
    x12 = lbind(lbind, occurrences)
    x13 = chain(x9, x11, x12)
    x14 = compose(x2, first)
    x15 = compose(x13, last)
    x16 = fork(argmax, x14, x15)
    x17 = chain(x7, x4, x16)
    x18 = compose(x4, x16)
    x19 = fork(occurrences, last, x18)
    x20 = fork(mapply, x17, x19)
    x21 = multiply(TWO, SIX)
    x22 = interval(THREE, x21, ONE)
    x23 = astuple(x22, I)
    x24 = x20(x23)
    x25 = fill(I, THREE, x24)
    x26 = interval(THREE, TEN, ONE)
    x27 = astuple(x26, x25)
    x28 = x20(x27)
    x29 = fill(x25, THREE, x28)
    x30 = astuple(x26, x29)
    x31 = x20(x30)
    x32 = fill(x29, THREE, x31)
    x33 = rbind(toobject, x32)
    x34 = rbind(colorcount, THREE)
    x35 = chain(x34, x33, neighbors)
    x36 = matcher(x35, EIGHT)
    x37 = sfilter(x1, x36)
    x38 = fill(I, THREE, x37)
    x39 = ofcolor(x38, ZERO)
    x40 = rbind(bordering, x38)
    x41 = compose(x40, initset)
    x42 = lbind(contained, THREE)
    x43 = rbind(toobject, x38)
    x44 = chain(x42, palette, x43)
    x45 = compose(x44, dneighbors)
    x46 = fork(both, x45, x41)
    x47 = sfilter(x39, x46)
    O = fill(x38, THREE, x47)
    return O


def solve_7837ac64(I):
    x1 = fgpartition(I)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = merge(x3)
    x5 = subgrid(x4, I)
    x6 = chain(color, merge, frontiers)
    x7 = x6(I)
    x8 = objects(x5, T, F, F)
    x9 = colorfilter(x8, ZERO)
    x10 = rbind(toobject, x5)
    x11 = chain(x10, corners, outbox)
    x12 = lbind(contained, x7)
    x13 = chain(x12, palette, x11)
    x14 = compose(numcolors, x11)
    x15 = compose(flip, x13)
    x16 = matcher(x14, ONE)
    x17 = fork(both, x15, x16)
    x18 = sfilter(x9, x17)
    x19 = compose(color, x11)
    x20 = fork(recolor, x19, identity)
    x21 = mapply(x20, x18)
    x22 = paint(x5, x21)
    x23 = first(x9)
    x24 = height(x23)
    x25 = height(x5)
    x26 = increment(x24)
    x27 = interval(ZERO, x25, x26)
    x28 = interval(ZERO, x25, ONE)
    x29 = rbind(contained, x27)
    x30 = chain(flip, x29, last)
    x31 = lbind(apply, first)
    x32 = rbind(sfilter, x30)
    x33 = rbind(pair, x28)
    x34 = chain(x31, x32, x33)
    x35 = compose(dmirror, x34)
    x36 = power(x35, TWO)
    x37 = x36(x22)
    O = downscale(x37, x24)
    return O


def solve_a8c38be5(I):
    x1 = replace(I, FIVE, ZERO)
    x2 = objects(x1, T, F, T)
    x3 = apply(normalize, x2)
    x4 = astuple(NINE, NINE)
    x5 = canvas(FIVE, x4)
    x6 = asindices(x5)
    x7 = box(x6)
    x8 = center(x6)
    x9 = lbind(contained, ZERO)
    x10 = rbind(subtract, x8)
    x11 = compose(x9, x10)
    x12 = chain(outbox, outbox, initset)
    x13 = corners(x6)
    x14 = mapply(x12, x13)
    x15 = difference(x7, x14)
    x16 = inbox(x7)
    x17 = sfilter(x16, x11)
    x18 = combine(x15, x17)
    x19 = fill(x5, ONE, x18)
    x20 = objects(x19, T, F, T)
    x21 = apply(toindices, x20)
    x22 = lbind(matcher, normalize)
    x23 = lbind(extract, x21)
    x24 = chain(ulcorner, x23, x22)
    x25 = compose(x24, toindices)
    x26 = fork(shift, identity, x25)
    x27 = mapply(x26, x3)
    O = paint(x5, x27)
    return O


def solve_b775ac94(I):
    x1 = objects(I, F, T, T)
    x2 = lbind(rbind, equality)
    x3 = rbind(compose, first)
    x4 = chain(x3, x2, mostcolor)
    x5 = fork(sfilter, identity, x4)
    x6 = fork(difference, identity, x5)
    x7 = lbind(rbind, adjacent)
    x8 = rbind(compose, initset)
    x9 = chain(x8, x7, x6)
    x10 = fork(extract, x5, x9)
    x11 = fork(insert, x10, x6)
    x12 = lbind(recolor, ZERO)
    x13 = chain(x12, delta, x11)
    x14 = fork(combine, x11, x13)
    x15 = fork(position, x5, x6)
    x16 = chain(toivec, first, x15)
    x17 = chain(tojvec, last, x15)
    x18 = fork(multiply, shape, x16)
    x19 = fork(multiply, shape, x17)
    x20 = fork(multiply, shape, x15)
    x21 = fork(shift, hmirror, x18)
    x22 = fork(shift, vmirror, x19)
    x23 = compose(hmirror, vmirror)
    x24 = fork(shift, x23, x20)
    x25 = lbind(compose, x5)
    x26 = x25(x21)
    x27 = x25(x22)
    x28 = x25(x24)
    x29 = compose(crement, invert)
    x30 = lbind(compose, x29)
    x31 = x30(x16)
    x32 = x30(x17)
    x33 = x30(x15)
    x34 = fork(shift, x26, x31)
    x35 = fork(shift, x27, x32)
    x36 = fork(shift, x28, x33)
    x37 = lbind(index, I)
    x38 = lbind(compose, toindices)
    x39 = x38(x14)
    x40 = x38(x34)
    x41 = x38(x35)
    x42 = x38(x36)
    x43 = fork(intersection, x39, x40)
    x44 = fork(intersection, x39, x41)
    x45 = fork(intersection, x39, x42)
    x46 = chain(x37, first, x43)
    x47 = chain(x37, first, x44)
    x48 = chain(x37, first, x45)
    x49 = fork(recolor, x46, x34)
    x50 = fork(recolor, x47, x35)
    x51 = fork(recolor, x48, x36)
    x52 = mapply(x49, x1)
    x53 = mapply(x50, x1)
    x54 = mapply(x51, x1)
    x55 = paint(I, x52)
    x56 = paint(x55, x53)
    O = paint(x56, x54)
    return O


def solve_97a05b5b(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, size)
    x3 = subgrid(x2, I)
    x4 = rbind(greater, ONE)
    x5 = compose(x4, numcolors)
    x6 = sfilter(x1, x5)
    x7 = lbind(rbind, subtract)
    x8 = switch(x3, TWO, ZERO)
    x9 = lbind(occurrences, x8)
    x10 = lbind(lbind, shift)
    x11 = compose(x7, ulcorner)
    x12 = matcher(first, TWO)
    x13 = compose(flip, x12)
    x14 = rbind(sfilter, x12)
    x15 = rbind(sfilter, x13)
    x16 = lbind(recolor, ZERO)
    x17 = compose(x16, x15)
    x18 = fork(combine, x17, x14)
    x19 = chain(x11, x18, normalize)
    x20 = objects(x8, T, T, T)
    x21 = apply(toindices, x20)
    x22 = chain(x9, x18, normalize)
    x23 = rbind(colorcount, TWO)
    x24 = lbind(sfilter, x21)
    x25 = chain(size, first, x24)
    x26 = compose(positive, size)
    x27 = lbind(lbind, contained)
    x28 = chain(x26, x24, x27)
    x29 = compose(x25, x27)
    x30 = rbind(sfilter, x28)
    x31 = compose(x30, x22)
    x32 = lbind(rbind, equality)
    x33 = rbind(compose, x29)
    x34 = chain(x33, x32, x23)
    x35 = fork(sfilter, x31, x34)
    x36 = fork(apply, x19, x35)
    x37 = compose(x10, normalize)
    x38 = fork(mapply, x37, x36)
    x39 = astuple(cmirror, dmirror)
    x40 = astuple(hmirror, vmirror)
    x41 = combine(x39, x40)
    x42 = product(x41, x41)
    x43 = fork(compose, first, last)
    x44 = apply(x43, x42)
    x45 = lbind(rapply, x44)
    x46 = mapply(x45, x6)
    x47 = mapply(x38, x46)
    x48 = paint(x3, x47)
    x49 = palette(x47)
    x50 = lbind(remove, TWO)
    x51 = x50(x49)
    x52 = chain(first, x50, palette)
    x53 = rbind(contained, x51)
    x54 = chain(flip, x53, x52)
    x55 = sfilter(x6, x54)
    x56 = fork(apply, x19, x22)
    x57 = fork(mapply, x37, x56)
    x58 = mapply(x45, x55)
    x59 = mapply(x57, x58)
    O = paint(x48, x59)
    return O


def solve_3e980e27(I):
    x1 = objects(I, F, T, T)
    x2 = astuple(TEN, TEN)
    x3 = invert(x2)
    x4 = astuple(TWO, x3)
    x5 = astuple(THREE, x3)
    x6 = initset(x4)
    x7 = insert(x5, x6)
    x8 = insert(x7, x1)
    x9 = lbind(contained, TWO)
    x10 = lbind(contained, THREE)
    x11 = compose(invert, ulcorner)
    x12 = lbind(compose, x11)
    x13 = lbind(rbind, sfilter)
    x14 = compose(x12, x13)
    x15 = rbind(compose, center)
    x16 = lbind(lbind, shift)
    x17 = x14(x9)
    x18 = x14(x10)
    x19 = fork(shift, identity, x17)
    x20 = fork(shift, identity, x18)
    x21 = compose(x9, palette)
    x22 = compose(x10, palette)
    x23 = sfilter(x8, x21)
    x24 = argmax(x23, size)
    x25 = remove(x24, x23)
    x26 = vmirror(x24)
    x27 = chain(x15, x16, x19)
    x28 = x27(x26)
    x29 = mapply(x28, x25)
    x30 = sfilter(x8, x22)
    x31 = argmax(x30, size)
    x32 = remove(x31, x30)
    x33 = chain(x15, x16, x20)
    x34 = x33(x31)
    x35 = mapply(x34, x32)
    x36 = combine(x29, x35)
    O = paint(I, x36)
    return O

description_solvers={
    "67a3c6ac": "Reflect the entire grid left–right (vertical mirror).",
    "68b16354": "Reflect the entire grid top–bottom (horizontal mirror).",
    "74dd1130": "Reflect the grid across the main diagonal (transpose-like diagonal mirror).",
    "3c9b0459": "Rotate the grid by 180 degrees.",
    "6150a2bd": "Rotate the grid by 180 degrees.",
    "9172f3a0": "Uniformly upscale the input by a factor of 3 (pixel replication).",
    "9dfd6313": "Reflect the grid across the main diagonal.",
    "a416b8f3": "Duplicate the grid side-by-side (horizontal concatenation of the grid with itself).",
    "b1948b0a": "Replace all color 6 cells with color 2.",
    "c59eb873": "Uniformly upscale the input by a factor of 2.",
    "c8f0f002": "Replace all color 7 cells with color 5.",
    "d10ecb37": "Crop the top-left 2×2 patch of the grid.",
    "d511f180": "Swap colors 5 and 8 everywhere.",
    "ed36ccf7": "Rotate the grid 270° clockwise (or 90° counter-clockwise).",
    "4c4377d9": "Mirror the grid top–bottom and stack it above the original (mirror on top of original).",
    "6d0aefbc": "Mirror the grid left–right and place it to the right of the original (original then its mirror).",
    "6fa7a44f": "Mirror the grid top–bottom and place it below the original (original then its mirror).",
    "5614dbcf": "Turn 5s into background (0), then downscale by factor 3 (take every 3rd row/col).",
    "5bd6f4ac": "Crop a centered 3×3 window starting at row 0, col 6 (i.e., a 3×3 cut beginning at column 6).",
    "5582e5ca": "Create a solid 3×3 canvas filled with the input’s most common color.",
    "8be77c9e": "Mirror the grid top–bottom and stack below the original (original on top, its mirror beneath).",
    "c9e6f938": "Mirror the grid left–right and place to the right of the original (original then mirror).",
    "2dee498d": "Split the grid into three vertical slices and return the left slice.",
    "1cf80156": "Find the first (foreground) object (univalued, diagonal allowed) and return its tight subgrid.",
    "32597951": "Take all background-adjacent cells around color 8 and fill that surrounding frame with color 3.",
    "25ff71a9": "Move the first object (univalued, diagonal allowed) one cell down on the grid.",
    "0b148d64": "Partition by color; return the tight subgrid of the smallest color-region.",
    "1f85a75f": "Return the tight subgrid of the largest foreground object.",
    "23b5c85d": "Return the tight subgrid of the smallest foreground object.",
    "9ecd008a": "Mirror the grid left–right; then take the tight subgrid that bounds all 0-colored cells within that mirror.",
    "ac0a08a4": "Upscale the grid by (9 − count_of_zeroes) (replicating pixels).",
    "be94b721": "Return the tight subgrid of the largest 4-connected object (no diagonal).",
    "c909285e": "Return the tight subgrid of all cells colored with the least frequent color.",
    "f25ffba3": "Take the bottom half, mirror it top–bottom, and stack above to form a symmetric grid.",
    "c1d99e64": "Detect uniform (all-one-color) row/column frontiers and fill them with color 2.",
    "b91ae062": "Upscale the grid by (number_of_colors − 1).",
    "3aa6fb7a": "For every non-diagonal foreground object, draw 1-colored corners of each object (underfill only background).",
    "7b7f7511": "If taller than wide, return top half; otherwise return left half.",
    "4258a5f9": "Fill all neighbors of color-5 cells with color 1 (including adjacency and diagonals via two passes of neighbors merge).",
    "2dc579da": "Split grid into two horizontal halves, then split each half into two vertical parts; return the subgrid with the most colors.",
    "28bf18c6": "Extract the first foreground object and duplicate its tight subgrid side-by-side.",
    "3af2c5a8": "Tile the grid into a 2×2 kaleidoscopic symmetry (original|vmirror horizontally, then mirrored vertically).",
    "44f52bb0": "Output a 1×1 cell: 1 if the grid is left–right symmetric, else 7.",
    "62c24649": "Create a 2×2 symmetric tiling (same as 3af2c5a8/67e8384a).",
    "67e8384a": "Create a 2×2 symmetric tiling of the grid (original+vmirror, then mirror top–bottom).",
    "7468f01a": "Take the first multi-colored object (diagonal allowed) and mirror its subgrid left–right.",
    "662c240a": "Split into three horizontal bands and return the band that is NOT diagonal-symmetric.",
    "42a50994": "Remove all singleton objects (size 1) by painting them as background.",
    "56ff96f3": "For each foreground color-region, recolor its bounding box border with that region’s color.",
    "50cb2852": "For each (non-diagonal) object, draw the border of its ‘inbox’ backdrop in color 8.",
    "4347f46a": "For each object, zero-fill the outline (box) minus the object itself (hollowing outlines).",
    "46f33fce": "Rotate 180°, downscale by 2, rotate back, then upscale by 4 (normalize/thicken shapes).",
    "a740d043": "Paint all foreground onto a tight subgrid and turn color 1 into 0 there.",
    "a79310a0": "Move the first (non-diagonal) object one cell down, then change color 8 cells to color 2.",
    "aabf363d": "Swap the two rarest colors: make least-common into 0, then next least-common into the previous least-common.",
    "ae4f1146": "Return the tight subgrid of the object containing the most cells of color 1.",
    "b27ca6d3": "For every size-2 object, draw its surrounding outbox with color 3.",
    "ce22a75a": "For every object, fill the backdrop of its outbox with color 1.",
    "dc1df850": "For every color-2 object, draw its outbox in color 1.",
    "f25fbde4": "Extract the first (foreground) object’s subgrid and upscale it by 2.",
    "44d8ac46": "Find deltas (bounding box minus object) of objects that are squares and fill those cells with color 2.",
    "1e0a9b12": "Rotate right, sort rows lexicographically, then rotate back (row ordering).",
    "0d3d703e": "Apply a chain of color swaps: (3↔4), (8↔9), (2↔6), (1↔5).",
    "3618c87e": "Shift all singleton objects two rows down (move size-1 foreground cells).",
    "1c786137": "Take the tallest object’s tight subgrid and trim its 1-cell border.",
    "8efcae92": "Among non-diagonal objects of color 1, pick the one with largest delta (background in its bounding box) and return its subgrid.",
    "445eab21": "Make a solid 2×2 block colored with the color of the largest-area object.",
    "6f8cd79b": "Fill with color 8 all cells lying on the outer border of the grid.",
    "2013d3e2": "From the first multi-colored object, take its subgrid, then return the top-left quadrant (left half then top half).",
    "41e4d17e": "For each (non-diagonal) object center, draw full row+column ‘frontiers’ and underfill them with color 6 only on background.",
    "9565186b": "Create a canvas of 5s sized like input and paint the largest (non-diagonal) object onto it.",
    "aedd82e4": "Fill all singleton color-2 objects with color 1 (on the original grid).",
    "bb43febb": "For every color-5 object, fill the backdrop of its inbox with color 2.",
    "e98196ab": "Copy all (non-diagonal) objects from the top half and paint them in the bottom half at same positions.",
    "f76d97a5": "Swap the lexicographically first and last palette colors, then convert 5s to 0s.",
    "ce9e57f2": "For each (non-diagonal) object, draw a line from its upper-left corner to its center of mass, finally swap colors 8↔2.",
    "22eb0ac0": "Recolor only horizontal lines corresponding to fg partitions: draw each partition’s backdrop border in its color if it forms a horizontal line.",
    "9f236235": "Compress out uniform rows/cols, mirror left–right, then downscale by the minimum object width.",
    "a699fb00": "Where a color-1 cell has both left and right neighbors also 1, fill that intersection position with color 2.",
    "46442a0e": "Create a 2×2 rotation collage: [I | rot90; rot270 | rot180].",
    "7fe24cdd": "Create the same 2×2 rotation collage as 46442a0e.",
    "0ca9ddb6": "Color neighbors of 1s with 7 (orthogonal) and neighbors of 2s with 4 (diagonal).",
    "543a7ed5": "For color-6 objects: draw their outboxes in 3 and fill the gaps (delta) around them with 4.",
    "0520fde7": "Mirror the grid left–right, compare left vs mirrored-right cellwise (match→keep, else 0), then replace 1s by 2.",
    "dae9d2b5": "Take left/right halves; mark positions of 4s (left) and 3s (right), fill those in the left half with 6.",
    "8d5021e8": "Build a multi-fold mirror tiling (left–right mirror, then top–bottom mirror, then repeat and mirror again).",
    "928ad970": "Find color-5 region, crop and trim it to learn its least color, then draw an inbox around 5 using that least color.",
    "b60334d2": "Around color-5 cells: set orthogonal neighbors to 1 on a 5→0 cleared grid, then set diagonal neighbors to 5.",
    "b94a9452": "Take the first multi-colored object’s subgrid and swap its least frequent and most frequent colors.",
    "d037b0a7": "For each (non-diagonal) object, shoot a vertical line down from its center, recolored with the object’s color.",
    "d0f5fe59": "Create an N×N canvas of 0s where N=#objects; draw the main diagonal in color 8.",
    "e3497940": "Copy objects from the right half mirrored into the left half.",
    "e9afcf9a": "Tile mirrored 2×1 blocks to create a wider repeating horizontal pattern from the top-left 2×1 crop.",
    "48d8fb45": "Among objects, find a size-1 object and then another object adjacent to it; return the latter’s subgrid.",
    "d406998b": "Mirror grid left–right; among color-5 cells on even columns, fill them with 3; mirror back.",
    "5117e062": "Extract the first multi-colored object; in its tight subgrid, replace 8s with that object’s most common color.",
    "3906de3d": "Rotate right, swap 1↔2, sort rows, swap back, then counter-diagonal mirror.",
    "00d62c1b": "Find background-colored (0) objects not touching the border and fill them with color 4.",
    "7b6016b9": "Fill all interior (non-border-touching) objects with color 2, then turn resulting 0s into 3s.",
    "67385a82": "Among color-3 objects, fill all non-singleton ones with color 8.",
    "a5313dff": "Find background-colored (0) objects not touching border and fill them with color 1.",
    "ea32f347": "Replace 5 with 4; then color the largest object’s cells with 1 and the smallest object’s cells with 2 (overlays).",
    "d631b094": "Make a 1×K canvas (K=#cells of the non-zero color) filled with the non-zero color.",
    "10fcaaa3": "Tile the grid 2×2; expand least-frequent color’s diagonal neighbors across the big tile with color 8 but only on background.",
    "007bbfb7": "Compare a 3× upscaled grid to a 3×3 tiled original and keep matching cells, else 0 (cellwise equality mask).",
    "496994bd": "Take the top half of the grid, mirror it top–bottom, and stack to form vertical symmetry.",
    "1f876c06": "For each foreground partition, connect first and last cells (by index order) and paint those connections in the partition’s color.",
    "05f2a901": "Move the first color-2 object toward the first color-8 object until adjacent (gravity), updating its position.",
    "39a8645d": "Select the most common object color and return the tight subgrid of one object of that color.",
    "1b2d62fb": "Compare left and right halves of the grid; on the left half, paint 8 wherever both halves are background at the same relative positions (also turn 9s into background).",
    "90c28cc7": "Crop the first foreground object, then remove duplicate rows and columns from it (compressing repeated stripes) while keeping its original orientation.",
    "b6afb2da": "For every color-5 object, draw its bounding box in color 4 and mark the box’s four corners with color 1; also convert any 5s in the grid to 2.",
    "b9b7f026": "Find the smallest object and output a 1×1 tile colored like one of the objects that touches it.",
    "ba97ae07": "Find the most common foreground color and draw the outline of its overall bounding box onto the image using that same color.",
    "c9f8e694": "Replicate the first column across the whole width, then force all locations that were background in the input to remain background.",
    "d23f8c26": "Erase every column except the central column, leaving only the middle column of the image.",
    "d5d6de2d": "For all non-square objects, paint their inner rectangular frame (the ‘inbox’ outline) with color 3; also replace color 2 everywhere with background.",
    "dbc1a6ce": "For every pair of color-1 cells, draw the straight horizontal/vertical connection between them and paint those background cells with color 8.",
    "ded97339": "For every pair of color-8 cells, draw the straight horizontal/vertical connection between them and paint those background cells with color 8.",
    "ea786f4a": "Draw both main diagonals of the grid (an ‘X’) in background color.",
    "08ed6ac7": "Rank foreground objects by height and recolor them with descending labels (n, n−1, …, 1) according to increasing height, overlaying on the input.",
    "40853293": "For each color, frame its extent with a colored rectangle and paint only the straight sides (horizontal/vertical edges) back onto the image.",
    "5521c0d9": "Move every foreground object straight up by its own height (on a cleared background) and paint them in those shifted positions.",
    "f8ff0b80": "List object colors in order of increasing size as a vertical 1-cell-wide strip and vertically mirror that strip.",
    "85c4e7cd": "Permute object colors by size rank: the largest takes the smallest’s color, the second-largest takes the second-smallest’s, etc., without moving shapes.",
    "d2abd087": "Recolor objects of size 6 to color 2 and all other foreground objects to color 1.",
    "017c7c7b": "Append content beneath the image: if top and bottom halves match, append the bottom half; otherwise append a specific 3×3 patch. Finally change all 1s to 2s.",
    "363442ee": "Take the top-left 3×3 patch and stamp that pattern (offset so its center aligns) around every cell colored 1.",
    "5168d44c": "If the color-3 region is a single row, shift the color-2 object two cells right; otherwise shift it two cells down.",
    "e9614598": "Find the midpoint between the first and last color-1 cells and mark a plus-shaped cross (center and its 4-neighbors) with color 3.",
    "d9fac9be": "Output a 1×1 tile of the non-background color that is present in the image but not used by the largest object.",
    "e50d258f": "Identify the object that contains the most color-2 cells and output its tight crop from the original image.",
    "810b9b61": "Detect objects that are perfect rectangular frames (not solid rectangles or simple lines) and paint their cells with color 3.",
    "54d82841": "For each foreground object, mark on the bottom row the column under its center with color 4.",
    "60b61512": "For each object, fill the cells inside its bounding box that are not part of the object (the box’s interior gaps) with color 7.",
    "25d8a9c8": "Keep only horizontal length-3 objects and color them 5; set all other cells to background.",
    "239be575": "If at least two distinct (non-background) components contain color 2, output 0; otherwise output 8 (as a 1×1 tile).",
    "67a423a3": "Around the rarest-color region, pick one adjacent background cell and paint its 4-neighborhood (a plus) with color 4.",
    "5c0a986e": "From the lower-right corner of the color-2 region draw a (1,1) diagonal of 2s; from the upper-left corner of the color-1 region draw a (−1,−1) diagonal of 1s.",
    "6430c8c4": "Create a 4×4 grid and mark with color 3 the positions that are background in both the top and bottom halves of the input at the same relative locations.",
    "94f9d214": "Create a 4×4 grid and mark with color 2 the positions that are background in both the top and bottom halves at the same relative locations.",
    "a1570a43": "Shift the color-2 object so its upper-left corner moves to one cell beyond the upper-left corner of the color-3 object (down-right by that corner difference plus one).",
    "ce4f8723": "Create a 4×4 grid filled with 3 and carve out zeros wherever the top and bottom halves of the input are both background at the same relative positions.",
    "d13f3404": "On a 6×6 canvas, from each object’s center draw a down-right 45° ray colored like that object.",
    "dc433765": "Move the color-3 object by one step toward the color-4 object (along row/column/diagonal direction) and repaint.",
    "f2829549": "Make a canvas shaped like the left half and mark with color 3 the positions that are background in both left and right halves at the same relative locations.",
    "fafffa47": "Make a canvas shaped like the bottom half and mark with color 2 the positions that are background in both top and bottom halves at the same relative locations.",
    "fcb5c309": "Find the largest object that is not of the rarest color and recolor it to that rare color within its bounding crop.",
    "ff805c23": "Take the region shaped like the color-1 footprint from both the horizontally and vertically mirrored images; return whichever subgrid still contains a 1.",
    "e76a88a6": "Copy the object with the most distinct colors, normalize it to the origin, and stamp that copy at the upper-left corner of every other object.",
    "7c008303": "Remove colors 3 and 8, compress the grid (removing uniform rows/cols), upscale by 3, then restore zero ‘holes’ where the zeros lay within the original crop around the 3-colored area.",
    "7f4411dc": "Erode the rarest-color cells: any such cell with more than two non-same-color 4-neighbors is turned to background.",
    "b230c067": "Turn all 8s into 1s, then identify the rarest (by normalized shape) foreground object and recolor precisely that object to color 2.",
    "e8593010": "Classify tiny components: recolor all single-cell objects to 3 and all size-2 objects to 2, then turn the background (0) to 1.",
    "6d75e8bb": "Take the first object and fill its entire bounding rectangle: keep the object as is and paint every background cell inside its bbox with color 2.",
    "3f7978a0": "Find the color-5 object and crop a window around it that is vertically expanded by one cell above and below (same width).",
    "1190e5a7": "Count uniform rows and uniform columns (frontiers). Output a solid canvas of the most common color whose size is (#uniform rows+1) by (#uniform columns+1).",
    "6e02f1e3": "Produce a 3×3 template whose line depends on the number of colors in the input: two colors → bottom horizontal line of 5s; three colors → counter-diagonal of 5s; otherwise → rightmost vertical from (2,0) to (0,2) fallback (per the chosen endpoints).",
    "a61f2674": "Erase color 5 to background, then highlight extremes: recolor the largest object to 1 and the smallest object to 2.",
    "fcc82909": "For each (foreground) object, draw a rectangular frame of color 3 whose top-left is one cell below the object’s lower-left corner and whose bottom extends downward by the grid’s color count from the object’s lower-right corner.",
    "72ca375d": "Among the objects, return the tight crop of the one that is vertically symmetric (unchanged under a vertical mirror).",
    "253bf280": "Connect every pair of color-8 cells with straight horizontal/vertical segments and paint those segments 3, then restore the original 8 cells.",
    "694f12f3": "Among color-4 objects, draw the inner border (inbox/backdrop outline): paint the smallest one’s outline with 1 and the largest one’s outline with 2.",
    "1f642eb9": "Move every single-cell object straight toward the first multi-cell object until just adjacent, then paint them at their new positions.",
    "31aa019c": "Create a 10×10 canvas. Place the rarest color at the location of its first occurrence, and paint all 8-neighbors around that cell with color 2.",
    "27a28665": "Classify the scene by the largest object size and output a 1×1 pixel: size==1→2, size==4→3, size==5→6, otherwise→1.",
    "7ddcd7ec": "Take the first non-singleton object’s color. From each singleton’s center, shoot a ray in the direction given by its relative position to that object and paint those rays with the non-singleton’s color.",
    "3bd67248": "Draw two guide lines anchored near the bottom-left: an up-right diagonal in color 2 and a rightward horizontal in color 4.",
    "73251a56": "Blend the grid with its main diagonal mirror by taking per-cell maxima (and replacing 0s by the global mode), then color the main diagonal with the (new) top-left value.",
    "25d487eb": "Find the least frequent color. Shoot a line from its center toward the global center of all foreground cells and paint that line with the least-frequent color only where the background is present.",
    "8f2ea7aa": "Compare a 3× upscaled foreground crop to a 3×3 tiling of the original crop and keep only cells where they agree; elsewhere write 0. (A periodicity/consistency mask.)",
    "b8825c91": "Enforce both diagonal and counter-diagonal symmetry by repeatedly taking per-cell maxima with mirrored versions (after zeroing color 4), then mirror back.",
    "cce03e0d": "Tile the original grid into a 3×3 mosaic, then project the 3× upscaled grid’s zeros and ones as a mask onto the mosaic and set those positions to 0.",
    "d364b489": "From every color-1 cell, color its immediate neighbors: below→8, above→2, right→6, left→7.",
    "a5f85a15": "From each object’s upper-left corner, stamp a sequence of marks along the main diagonal using offsets (1,1), (3,3), …, (15,15), painting those locations with color 4.",
    "3ac3eb23": "First, for each object paint its diagonal neighbors with the object’s own color; then output three stacked copies of the left third of the resulting grid.",
    "444801d8": "For each color-1 object, take the backdrop just above it and recolor that outline with the least frequent color found there, underpainting onto background only.",
    "22168020": "For each non-background color, connect all its pixels pairwise with straight lines and paint those lines in that same color.",
    "6e82a1ae": "Recolor by size tier: paint all size-2 objects with 3, size-3 with 2, and size-4 with 1 (later writes override earlier).",
    "b2862040": "Find color-9 objects not touching the border; any color-1 object adjacent to such a 9 is recolored to 8.",
    "868de0fa": "Among square objects, paint even-sized squares with 2 and odd-sized squares with 7.",
    "681b3aeb": "Rotate the scene, take the largest object’s shape, render it onto a 3×3 canvas colored by the smallest object’s color, then rotate back.",
    "8e5a5113": "Copy the top-left 3×3 patch: place its 90° and 180° rotations to the right (shifted by 4 and 8 columns) onto the original grid.",
    "025d127b": "Within each color, keep the rightmost extreme fixed and shift all other pixels one step to the right (so objects advance without moving their leading edge).",
    "2281f1f4": "For color-5 pixels, form all “cross” coordinates by combining each pair’s row and column; fill those crossing points with 2 except the upper-right original, painting only on background.",
    "cf98881b": "Split the grid into three vertical panels. Copy the pattern of 9s from the middle panel and 4s from the left panel into the rightmost panel at the same coordinates.",
    "d4f3cd78": "Around the color-5 object, first fill the ring just inside its bounding box with 8, then extend and draw an aligned 8-colored line along the frame direction.",
    "bda2d7a6": "Repaint only the largest color-region: give it the color of the smallest region, leave all other regions unchanged.",
    "137eaa0f": "Extract the (diagonally connected) foreground cluster that involves color 5, recenter its shape, and place it in the middle of a 3×3 canvas.",
    "6455b5f5": "Color the single largest object to 1, and recolor the smallest zero-colored fragments to 8; leave everything else as is.",
    "b8cdaf2b": "Take the least-frequent color and extend two diagonals from the shifted top corners of its area; draw those diagonals in that color only through background.",
    "bd4472b8": "Take the top two-row band, mirror/transpose its top half into a column, stretch it horizontally, and stack it under the original band to form a tall motif.",
    "4be741c5": "Depending on portrait/landscape, read the first row or first column, compress it to its unique color sequence (order-preserving), and write that sequence back in the same orientation.",
    "bbc9ae5d": "Vertically stretch the image, then from each cell of the single non-background color cast 45° rays; paint those rays in that color.",
    "d90796e8": "Remove two-cell foreground pieces that include color 2; among those whose leading cell is color 3, recolor their cells to 8.",
    "2c608aff": "Find the largest object and the least-frequent color; draw axis-aligned lines connecting the large object to every cell of that least color, painting only the background with that color.",
    "f8b3ba0a": "Summarize the compressed image by color frequency and output a 3×1 strip encoding the most frequent colors in order.",
    "80af3007": "Tile the object and compare to a 3× upscaled copy; keep only pixels that agree across the tiling, then downscale back—producing the motif invariant under the 3×3 repetition.",
    "83302e8f": "Classify zero-colored regions: paint square holes with 3 and non-square zero regions with 4.",
    "1fad071e": "Count how many color-1 objects have size 4 and output a 1×5 bar: that many 1s followed by 0s.",
    "11852cab": "Mirror the foreground across horizontal, vertical, main diagonal, and counterdiagonal axes and paint all reflections.",
    "3428a4f5": "Mark where color 2 appears in exactly one of the top/bottom halves; output a fixed 6×5 canvas with those XOR positions in color 3.",
    "178fcbfb": "Fill entire columns passing through color-2 cells with 2; for every non-2 object, draw a horizontal line through its center in its own color.",
    "3de23699": "Within the trimmed subgrid of a chosen 4-cell object, replace occurrences of the other object’s color with the 4-cell object’s color.",
    "54d9e175": "Grow every single-pixel object into a plus-shape around its center, then remap colors 1→6, 2→7, 3→8, 4→9.",
    "5ad4f10b": "On the largest object’s subgrid, swap its main color with the least-used color (least→0, main→least), then downscale so the height becomes 3.",
    "623ea044": "From the first object’s center, draw the two diagonals (an “X”) in its own color.",
    "6b9890af": "Inside the bounding box of color 2, place a normalized, scaled-up copy of the smallest object and paint it one cell down-right from the top-left.",
    "794b24be": "In a 3×3, draw a horizontal bar of length (count(color-1)−1) in color 2; if there are exactly four 1s, also fill the center cell.",
    "88a10436": "Move a chosen non-5 object so that it sits just up-left of the center of a color-5 object, then paint it there.",
    "88a62173": "Return the quadrant (top-left, top-right, bottom-left, bottom-right) that is least common among the four (breaking ties by content).",
    "890034e9": "Replicate the least-color’s shape at every place where its bounding-box outline occurs, shifted one up and one left; paint those copies in the least color.",
    "99b1bc43": "XOR the zero-colored cells of the top and bottom halves and output a canvas the size of the top half with those XOR cells colored 3.",
    "a9f96cdd": "Around every color-2 cell, draw a four-direction diagonal halo: (-1,-1)→3, (-1,+1)→6, (+1,-1)→8, (+1,+1)→7 (leaving the original 2s intact).",
    "af902bf9": "Connect every pair of color-4 cells with straight lines, then draw bounding-box frames around the resulting non-background segments in color 2; restore background elsewhere.",
    "b548a754": "Compute a global foreground box; fill its interior with the least-frequent non-8 color and outline the box with the next least-frequent non-8 color.",
    "bdad9b1f": "Draw the row through the center of the 2s in color 2 and the column through the center of the 8s in color 8; color their intersection 4.",
    "c3e719e8": "Create a 3×3 tiling and punch out (set to 0) wherever the 3× upscaled original is non-background—producing a tiled silhouette mask.",
    "de1cd16c": "Among multi-cell objects, find the one containing the most of the least-frequent color; output a 1×1 pixel of that object’s dominant color.",
    "d8c310e9": "Detect a horizontally periodic mixed-color object and stamp copies shifted by one and three periods to the right.",
    "a3325580": "For the largest objects (ties), left-to-right, output a vertical stack of equal-length color bars whose length equals that maximum size.",
    "8eb1be9a": "Tile the first object vertically at offsets of −2h, −h, 0, h, 2h (h = its height) and paint all copies.",
    "321b1fc6": "Erase one non-8 shape and copy its normalized form to the upper-left corners of all color-8 objects.",
    "1caeab9d": "Vertically align all objects so their bottoms sit on the lowest row occupied by color 1, then repaint them in place.",
    "77fdfe62": "Remove 8s and 1s, compress the remainder, scale it to match half the width of the 8-region, and preserve the zero-mask from the 8-region onto this scaled canvas.",
    "c0f76784": "Within zero-colored regions: paint all square zero blocks with 7, highlight the largest such square with 8, and mark single-cell zeros with 6.",
    "1b60fb0c": "Find the shift that best aligns the pattern of 1s in the 90°-rotated image with the original; draw that best-overlap footprint in color 2 on background only.",
    "ddf7fa4f": "For every color-5 object that shares a column with a 1-pixel object, recolor that color-5 object to the singleton’s color and paint it back onto the grid.",
    "47c1f68c": "Enforce bilateral symmetry by intersecting the grid with its vertical and horizontal mirrors using the least color as fallback, compress away uniform borders, then replace the fallback with the most frequent object color.",
    "6c434453": "Erase all size-8 objects and stamp a plus-shaped marker (center and four von-Neumann neighbors) colored 2 at each such object’s upper-left corner.",
    "23581191": "Through each object’s center, draw its full row and column in the object’s color; wherever these cross (centers intersect), fill those crossing cells with color 2.",
    "c8cbb738": "Normalize all foreground objects and place them centered inside a canvas sized to the largest object’s bounding box, using the grid’s most common color as background.",
    "3eda0437": "Find the largest axis-aligned rectangle that occurs at least twice in the grid and highlight all its cells with color 6.",
    "dc0a314f": "Build a symmetry-max composite from the grid and its diagonal/counter-diagonal mirrors after zeroing color 3, then crop the result to the region spanned by the original color-3 cells.",
    "d4469b4b": "On a 3×3 canvas, draw the row and column frontiers through a chosen pivot (depending on whether the only non-zero color is 1 or 2) and color that plus sign with 5.",
    "6ecd11f4": "Extract the largest and smallest connected objects; downscale the larger by the ratio of their widths and copy its zeros as a mask to set corresponding cells to 0 within the smaller object’s subgrid.",
    "760b3cac": "Mirror the color-8 pattern vertically and shift it up or down by three rows (direction chosen from the color at the color-4 region’s upper-left); paint 8 at the shifted positions.",
    "c444b776": "For each zero-colored object, place a normalized copy of its bounding backdrop aligned at its own upper-left corner, effectively outlining/backfilling frames of zeros over those regions.",
    "d4a91cb9": "Connect the first pixel of the color-8 set and the first pixel of the color-2 set to a L-shaped junction and underfill those connecting segments with color 4.",
    "eb281b96": "Mirror all but the last row vertically, stack beneath the original, crop to align heights, then stack again—producing a vertically palindromic tiling.",
    "ff28f65a": "Count connected objects, build a 1×(2·count) barcode of ones, split into three equal vertical bands, and merge them into a single output strip.",
    "7e0986d6": "Replace the background with zero, find the new least color, and for each original least-color pixel that has a neighbor of that new least color, recolor that pixel to the new least color.",
    "09629e4f": "Take the multicolor object with the fewest distinct colors, normalize and upscale it by 4, paint it onto the grid, and fill the positions of color 5 with color 5 (reinforcing fives).",
    "a85d4709": "For color-5 pixels, project horizontal frontiers and color the ones lying on rows 0, 1, or 2 with colors 2, 3, and 4 respectively (layered horizon bands).",
    "feca6190": "Place each object’s color along a long ray shot from its center in direction (1,1) on a large zero canvas and finally mirror horizontally, yielding diagonally radiating colored rays.",
    "a68b268e": "Partition the grid into quadrants; copy color-8 from bottom-left onto bottom-right, copy color-4 from top-right onto bottom-right, and copy color-7 from top-left onto bottom-left—filling those colors into the paired quadrants.",
    "beb8660c": "Normalize objects, order by decreasing size, vertically stack them with incremental downward shifts on a blank canvas, then rotate 180°, producing a size-sorted inverted montage.",
    "913fb3ed": "Expand the neighborhoods (8-adjacency) of colors 3, 8, and 2 and fill those neighborhoods with colors 6, 4, and 1 respectively.",
    "0962bcdd": "Promote the least color into background, recolor its neighbors with that least color to frame each object, then draw the rectangle outline (box) around each connected component, finally refilling the outlines with the original background color.",
    "3631a71a": "Mirror-pair the grid, take cellwise maxima, crop off a 2×2 frame, mirror vertically, and paint the mirrored non-zero objects shifted back by (2,2) onto the max composite.",
    "05269061": "From the union of all objects, stamp shifted copies of that union at offsets around origin (up/right diagonals and their opposites) and paint all copies back to create a 3-copy star of the scene.",
    "95990924": "For each object, draw the four corners of its outbox on the original grid and color them 1 (UL), 2 (UR), 3 (LL), and 4 (LR).",
    "e509e548": "Among objects, keep those whose trimmed subgrid contains color 3 and/or whose size equals (height+width−1); recolor 3→6 globally, then fill qualifying cells with color 2 and color 1 respectively.",
    "d43fd935": "Find 1-pixel objects that share a row or column with any color-3 structure and draw a line from each such pixel to the nearest approach point on the 3-object, colored with the pixel’s color.",
    "db3e9e38": "From the LR corner of the color-7 region, shoot diagonals and build vertical rays; paint these rays 8 and then recolor every other step along them (even offsets) back to 7, forming striped beacons.",
    "e73095fd": "Locate zero-colored rectangles whose backdrop equals their indices; around each, take the outbox corners, exclude the outbox itself, intersect with color-5 cells, and fill the empty remainder with color 4.",
    "1bfc4729": "Compute a box outline plus a middle horizontal frontier, fill the top half with its least color along those lines, mirror it for the bottom, and then replace that least color with the bottom half’s least color to form a mirrored double-stripe.",
    "93b581b8": "Mirror the merged foreground across both diagonals, upscale and shift it slightly, underpaint onto the original, then draw a zero-colored frame around the mirrored shape’s indices except at the indices themselves.",
    "9edfc990": "Among zero-colored objects, keep those adjacent to any color-1 pixel and recolor those adjacent zeros to 1.",
    "a65b410d": "From the UR corner of color-2, draw both diagonals and underfill one diagonal with 3 and the other with 1; then project horizontal shoots from both and underfill further with the same colors to thicken the diagonals.",
    "7447852a": "Order zero-colored objects by the column of their center, select those whose order index is a multiple of three, and fill those selected objects with color 4.",
    "97999447": "Extend each object rightward from its center with a color-matched ray; then add a staggered set of vertical dotted guides at odd columns and fill them with color 5 where they intersect objects’ spans.",
    "91714a58": "Isolate the largest object; on a blank canvas with just that object, mark those background cells whose neighborhood contains more than three pixels of the object’s predominant color, and set those cells to 0 (carving near-object voids).",
    "a61ba2ce": "For each corner function (UL, UR, LL, LR), extract the subgrid of the first object whose corner cell is background (color 0), and tile those four subgrids into a 2×2 mosaic.",
    "8e1813be": "If the single non-five object is a vertical line, first rotate by diagonal mirror; then list object colors top-to-bottom, deduplicate, repeat each to the count of unique colors, and output that color list as a row/column (mirrored back if needed).",
    "bc1d5164": "Across the four 3×3 quadrants, detect the least color and mark its indices in each quadrant; output a 3×3 grid colored with that least color at the union of those indices.",
    "ce602527": "Mirror left–right, keep the largest foreground object and the next best match whose 2× upscaled normalized indices overlap most with the largest; output that runner-up object mirrored back.",
    "5c2c9af4": "From the least-color region, compute the vector from its UL corner to its center; generate a family of concentric rectangular frames stepping in both positive and negative multiples of that vector and stamp those frames with the least color.",
    "75b8110e": "Split the grid into four quadrants and move the non-zero shapes from bottom-right into the top-left and top-right empties (preserving background elsewhere), effectively redistributing shapes across quadrants.",
    "941d9a10": "For each of three reference points (origin, size−1, and (5,5)), pick the unique zero-object containing that point and fill its cells with colors 1, 3, and 2 respectively.",
    "c3f564a4": "Create a cellwise max between the grid and its diagonal mirror, then move a complement-shaped object around in steps from −9 to 9 and paint all its shifted copies onto that max composite.",
    "1a07d186": "Find singleton pixels whose color appears among larger objects; for each such singleton, gravitate it toward the nearest larger object of the same color and paint it at that offset over a cleared background of the singletons.",
    "d687bc17": "Same as 1a07d186 but defined via an equivalent ordering of merging and painting: move singletons toward matching larger objects and paint them back after removing all singletons from the grid.",
    "9af7a82c": "For each object, build a two-row banner: its color as a solid bar above and a zero bar sized from the size difference to the largest object below; mirror those banners across the counter-diagonal and merge to form a symmetric collage.",
    "6e19193c": "Let x be the least color; for each object whose immediate neighborhood contains exactly two x’s, shoot a line from its first delta cell toward that neighbor and fill the line with x; finally fill all objects’ deltas with 0.",
    "ef135b50": "Connect pairs of color-2 pixels that share the same row or column, intersect those connections with zero-colored cells, color the intersections 9, trim borders, shift by (1,1), and paint that onto the original.",
    "cbded52d": "For all pairs of singletons that share a row or column, connect their centers, recolor those lines with the first pixel’s color, and paint the resulting connectors onto the grid.",
    "8a004b2b": "Within the color-4 bounding box, normalize the lowest object in the full grid, compute how many times it tiles across the internal zero-structure, upscale accordingly, shift into place, and paint it back inside the color-4 window.",
    "e26a3af2": "Rebuilds the grid by making every row or every column uniformly its majority color, choosing the orientation (rows vs. columns) that has the greater variety of majority colors, and expanding that 1-pixel strip to the full grid size along the chosen axis.",
    "6cf79266": "Detects solid 3×3 background (color 0) squares and fills each such square (once per square) with color 1.",
    "a87f7484": "Orients the grid to be wider than tall, splits it into (num_colors−1) equal vertical slices, finds the slice whose zero-pattern is the rarest among the slices, and returns that slice (restored to the original orientation).",
    "4093f84a": "Highlights the least frequent color as 5, orients the grid to a tall view if needed, sorts the left half and right half in opposite ordering (lexicographic vs. reverse by row content), concatenates them, then restores the original orientation.",
    "ba26e723": "From the pixels of color 4, selects those whose coordinates lie on the 3-cell grid (both i and j multiples of 3) and recolors those selected positions to 6.",
    "4612dd53": "Finds the color-1 object, works inside its bounding box, and draws background-only guide lines through the object’s rows or columns—whichever direction yields fewer lines—using color 2; paints these lines back into the original grid only over background.",
    "29c11459": "Processes left and right halves independently: for each object, draws a same-colored horizontal line through its center; then repaints objects and finally marks a 5 one step to the right of each object’s upper-right corner before merging the halves back together.",
    "963e52fc": "Detects the grid’s fundamental horizontal period and horizontally tiles that periodic slice to produce a result that is twice the original width (cropped to exact size), preserving height.",
    "ae3edfdc": "Removes colors 3 and 7 temporarily, then moves all 3-colored objects next to the nearest 2-colored mass and all 7-colored objects next to the nearest 1-colored mass (full gravitation to adjacency), and paints the moved objects back in.",
    "1f0c79e5": "Combines the color-2 pixels with the grid’s least frequent color, recenters the 2-shape, generates a sequence of scaled/stepped offsets from that shape, and repeatedly stamps the combined shape along those offsets across the grid.",
    "56dc2b01": "Finds a color-3 object and the color-2 mass, computes the gravitation between them, shifts the 2s toward the 3 by a sized step (with a small correction), paints that shifted 2-trail, and then moves the 3 to become adjacent to the 2s.",
    "e48d4e1a": "Counts the number of 5s, locates a special pixel of the least-color with exactly four same-color cardinal neighbors, offsets that location diagonally by the 5-count, and draws an orthogonal cross (row+column) through that point in the least color.",
    "6773b310": "Compresses out solid borders, examines a 3×3 sampling lattice, finds the cells whose local 3×3 neighborhood contains exactly two 6s, marks those centers with 1, removes 6s to background, and finally downscales by 3.",
    "780d0b14": "Marks each sizable object at its center with the object’s color on a blank background and then retains only those centers that share a row and also a column with at least one other differently colored center; outputs the filtered center-dot grid.",
    "2204b7a8": "Chooses to split the grid horizontally or vertically based on whether vertical or horizontal 1-pixel objects dominate; replaces all 3s in each half by that half’s nearest border corner color, then recombines the halves along the chosen split axis.",
    "d9f24cd1": "Connects every 2 to every 5 and keeps vertical connections, paints those lines with 2 on background, identifies multi-color components to place upward shoots from their upper-right corners, and reinforces vertical frontiers through 2-only components with color 2.",
    "b782dc8a": "Finds the least frequent color and the most frequent color adjacent to it; on the region bordering that dominant color, assigns the least color to positions at even Manhattan distance from the least-color seed and the dominant color to the rest (checker-like split).",
    "673ef223": "Treats 8s as movable bands: determines a vertical offset from the spread of 2-colored objects, shifts the 8-cells by that amount, draws horizontal rays from the 8s toward the nearer side (left or right decided by the topmost object’s position), and underfills those rays with 8.",
    "f5b8619d": "Through every least-color pixel, draws an infinite vertical line (on background only) in color 8, then tiles the result into a 2×2 repetition (duplicated in both directions).",
    "f8c80d96": "Takes the largest object of the least frequent color, builds concentric rectangular outlines (outboxes) around it at one, two, and three layers, fills those outlines with the least color, and finally converts background 0s to 5.",
    "ecdecbb3": "For each pair of a color-2 object and a color-8 object, computes a step toward adjacency, draws a long ray from the 2-object’s center in that direction (only sufficiently long rays are kept) in color 2, and highlights the ray’s immediate neighborhood endpoints with 8.",
    "e5062a87": "Normalizes the color-2 shape, finds all occurrences of that shape, filters out occurrences anchored at a small forbidden set of upper-left corners, and repaints the remaining occurrences back onto the grid in color 2.",
    "a8d7556c": "Finds all 2×2 zero blocks and fills them with color 2; then inspects a particular reference coordinate and, depending on whether it is already 2, places a tiny 2-mark (two adjacent cells) derived from either the original grid or the updated one.",
    "4938f0c2": "Mirrors the color-2 mass horizontally and vertically with offsets derived from its width and height to form a 2×2 tiling of mirror copies; if the scene already has many objects (>4), leaves the grid unchanged.",
    "834ec97d": "Moves the first object one cell downward, swaps it into place, and then paints a vertical dashed stripe of color 4 above it near the object’s left edge using every other row and a limited column window.",
    "846bdb03": "Within the trimmed scene, identifies the single object that lacks color 4, optionally flips it horizontally to match a local neighbor color test, normalizes it, shifts it to start at (1,1), and paints it onto the cropped background.",
    "90f3ed37": "Orders objects top-to-bottom, takes the top one as a reference, and for each remaining object chooses the rightward shift (from a small set) that maximizes overlap with shifted copies of the reference around its upper-left; stamps 1s under those best overlaps on background.",
    "8403a5d5": "Finds the first object’s leftmost column and color, paints that color on every cell whose column index lies in an arithmetic progression derived from that column, and places two interleaved series of color-5 markers along the bottom row at related step-4 column positions.",
    "91413438": "Replicates the leftmost detected object in 3-column steps across an appended blank strip, then collapses back to the original width—effectively stamping a periodic series of that object and merging the stamps into the final grid.",
    "539a4f51": "Takes a (possibly trimmed) version of the input’s top-left block and builds a four-quadrant, mirror-symmetric mosaic from a single row slice, then paints it onto a 10×10 canvas using the color of the original top-left cell.",
    "5daaa586": "Finds a zero-colored interior region, boxes it, identifies the largest foreground piece inside, and draws a dominant axis skeleton (longest straight lines through its pixels) in that piece’s color within the boxed subgrid.",
    "3bdb4ada": "For each foreground object (ignoring background), links a shrunken upper-left to a shrunken lower-right corner and selectively erases (paints with 0) those internal connectors based on a parity test of object extents.",
    "ec883f72": "Locates the largest object, takes the non-background, non-object color, and underfills background cells along rays shot outward from the object’s four bounding-box corners (cardinal/diagonal directions) using that other color.",
    "2bee17df": "Detects rows and columns that are almost entirely background and underfills those entire frontiers (full rows/columns) with a fixed color, effectively drawing a grid/frame at those near-empty lines.",
    "e8dc4411": "Computes the relative offset between the background (0) mask and the rarest color, then repeatedly shifts the background’s footprint by that offset and recolors those hits with the rare color—tiling a motif at a fixed step.",
    "e40b9e2f": "Takes the first object and places mirrored variants (composed reflections) at offsets that maximize overlap with it under neighborhood moves, painting those best-overlapping placements to complete a symmetric layout.",
    "29623171": "Enumerates a fixed set of axis-aligned blocks on a coarse grid, chooses the block containing the most cells of the rarest color, fills that chosen block with the rare color, and erases the others.",
    "a2fd1cf0": "From the extreme positions of colors 2 and 3, draws one horizontal and one vertical segment that form an orthogonal corner connecting those extremes, underfilling those segments with color 8.",
    "b0c4d837": "Synthesizes a vertically symmetric columnar emblem from the relative heights of colors 5 and 8: constructs an 8/0 strip sized by their height difference, mirrors it, stacks it, and appends a small zero footer.",
    "8731374e": "Within the largest object’s subgrid, isolates the most colorful lines, finds the rarest color there, and draws crosshair frontiers (full row and column lines) through its occurrences using that rare color.",
    "272f95fa": "Finds a zero-colored interior region and the zero shapes sharing its rows/columns; colors the extreme ones by role—topmost, bottommost, leftmost, and rightmost—using fixed target colors.",
    "db93a21d": "For color-9 elements, shoots downward traces (underfilling with 1), then draws one-, two-, and three-step outbox outlines around sufficiently wide 9-objects—coloring all such outlines with color 3.",
    "53b68214": "Detects the object’s vertical periodicity and either tiles it by that period (if portrait) or shifts it to a target corner; paints the original object and the periodic or shifted copy onto a square canvas.",
    "d6ad076f": "Chooses a principal axis from the smallest and largest objects, then shoots rays from the small one toward the extremum along that axis (marking with 8); finally removes any 8-markings that end up touching the border.",
    "6cdd2623": "Connects rare-color points pairwise but keeps only pure straight segments that lie strictly inside the overall bounding box of the figure; paints that interior line skeleton with the rare color onto a cleared background.",
    "a3df8b1e": "Grows a diagonal ‘1’ trace from a seed, isolates the resulting motif, then vertically doubles it repeatedly and mirrors it, finally cropping and mirroring back to fit—producing a repeated diagonal band pattern of 1s.",
    "8d510a79": "Uses the top of color-5 as a threshold to decide, for each color-2, whether to shoot an up or down vertical ray of a computed length; underfills along those selected rays with 2 and also extends 1s along their own rays.",
    "cdecee7f": "Reads object colors in left-to-right order, assembles them into a compact palette strip via unit tiles and diagonal mirroring, pads to three panels, then crops and stacks to output a 3×3 color logo encoding that order.",
    "3345333e": "Removes the least color, mirrors its positions, explores neighbor-based shifts of the mirror, and fills using the shift that best overlaps the original set—reconstructing a missing symmetric counterpart of that color.",
    "b190f7f5": "Splits the grid along the long side, upscales the more colorful half by its width, and overlays a doubly mirrored/repeated version of the less colorful half as a zero-mask—punching structured holes into the upscaled half.",
    "caa06a1f": "Finds an object’s 2D periodicity (vertical and horizontal) and stamps periodic copies of that motif across a doubled canvas, then paints the resulting tiled pattern back onto the original.",
    "e21d9049": "Replicates the merged figure at lattice offsets derived from its bounding shape, paints all copies, then erases any cells not horizontally or vertically aligned with the least-color template—leaving a lattice-aligned residue.",
    "d89b689b": "For each singleton object, finds the nearest location among the eights and paints a single pixel there with the singleton’s color, after removing larger objects—assigning each 1-cell piece to its nearest 8.",
    "746b3537": "If the first row is uniform, diagonally mirror the input; then list object colors in left-to-right order and return that list, mirrored or not depending on the initial uniformity test.",
    "63613498": "Takes the top-left 3×3 pattern of non-zeros, finds every object in the image whose normalized shape matches that motif, paints those matches with color 5, and then restores the original 3×3 block.",
    "06df4c85": "For all colors except the largest region, connects pairs of same-color components with straight segments recolored to that color, keeps only true lines, paints those connectors, and finally restores the dominant background color over its original area.",
    "f9012d9b": "Infers a 2D lattice from the blank halves of the grid (vertical period from the non-zero vertical half, horizontal period from the non-zero horizontal half), then replicates all non-background objects at the neighbor offsets scaled by those periods, and finally crops the result to the original background region.",
    "4522001f": "Builds a fixed 9×9 pattern made of two aligned upscaled blocks and chooses its orientation by checking which reference offsets are present inside the single foreground object; returns the appropriately rotated version of that pattern.",
    "a48eeaf7": "For every cell of color 5, finds the nearest cell on the perimeter surrounding the color-2 region and moves the 5 onto that perimeter, erasing the original 5s.",
    "eb5a1d5d": "Creates a kaleidoscopic symmetry: deduplicates, mirrors, and vertically concatenates the input with a trimmed horizontal mirror, then repeats the same process after a diagonal mirror to produce a symmetric composite.",
    "e179c5f4": "Grows a diagonal from the first 1 to complete a near-rectangle of 1s, extracts its interior strip (height−1), tiles that strip to fill the canvas, horizontally mirrors the tiling, and finally converts background to color 8.",
    "228f6490": "Finds zero-colored, border-touching shapes and their matching colored counterparts (same normalized shape). It translates each colored counterpart so it sits where its zero-shape sits.",
    "995c5fa3": "Splits the grid into three vertical thirds, computes a tiny score per third from zero-cell position tests, renders those scores as single-cell tiles, merges them, and outputs a horizontally upscaled bar (factor 3).",
    "d06dbe63": "Centers on an 8-colored target, stamps a small plus/‘corner’ motif of color 5 at regularly spaced diagonal offsets, mirrors the whole construction through 180°, aligns it to the opposed center, and merges both.",
    "36fdfd69": "Upscales to thicken, then for each pair of color-2 objects that are far apart, fills the space between their combined outlines with color 4; finally restores the objects and downscales.",
    "0a938d79": "Normalizes orientation (mirror if portrait), then for each foreground blob draws a vertical frontier line at a deterministic column derived from the blob’s width/left edge and colors that line with the blob’s color, then restores the original orientation.",
    "045e512c": "Takes the largest object as a template. For each smaller object, it places multiple shifted copies of the large object along the direction toward that smaller object at fixed step sizes, recoloring each copy to the smaller object’s color.",
    "82819916": "Chooses the most prominent multi-color object as a prototype, splits its normalized shape into ‘key’ vs ‘other’ color parts, recenters both to their own top edge, assigns two target colors from that prototype, and applies the same split-and-recolor to every other object before painting them.",
    "99fa7670": "Shoots rightward rays from the center of each object using the object’s color, then orders objects and connects successive ones with straight connector lines between their lower-right corners, underpainting those connectors onto the background.",
    "72322fa7": "Separates single-color from multi-color objects. For the multi-color ones, identifies the dominant-color component and its complement, finds all occurrences/placements of these patterns, computes needed shifts, and paints normalized copies to all matched locations.",
    "855e0971": "Detects whether horizontal frontiers exist (rotating to a convenient orientation if needed). Within zero-colored partitions it computes vertical frontiers intersecting their subgrids and clears those intersections to zero, then undoes the rotation.",
    "a78176bb": "From 5-colored objects, selects those whose upper-right corner sits on a 5. From near the chosen corners it shoots diagonals in both ‘X’ directions by one-cell-offset, drawing those diagonals in the ‘other’ non-zero color and finally removing remaining 5s.",
    "952a094c": "Finds the largest object and all singleton pixels. For each corner of the largest object’s bounding box, colors that corner with the color of the nearest singleton and removes the singletons from the scene.",
    "6d58a25d": "Uses the top of the largest object as a threshold. For each object that shares a column with it and lies above that threshold, fills the background beneath its vertical centerline with the global (merged) foreground color.",
    "6aa20dc0": "Takes the most colorful object, splits off its dominant vs. remaining colors, then searches the grid for occurrences of those remaining-color parts under all mirror symmetries and upscales (×1..×3); for each match it shifts and paints the full prototype accordingly.",
    "e6721834": "Splits the grid into two halves along the longer dimension, picks the simpler half, and copies into it the objects from the complex half whose dominant-color parts also occur in the simple half, shifting them by the displacement between those matches and keeping only sufficiently wide results.",
    "447fd412": "Like the previous symmetry-and-occurrence propagation, but first augments each candidate with a zero-colored outline (outbox), computes occurrence displacements with an incremented offset, and stamps mirrored/upscaled variants of the prototype accordingly.",
    "2bcee788": "Recolors background to 3, takes the largest and smallest objects, orients a copy of the large one toward the small (horizontal vs vertical mirror depending on the small being a horizontal line), and stamps only the 3-colored part at an offset proportional to the inter-object position.",
    "776ffc46": "Finds a 5-colored rectangular ‘frame’ (object equal to its box outline), extracts its inner box, locates all other objects whose normalized indices match that inner shape, and fills them with the frame’s inner color.",
    "f35d900a": "For every object, paints its immediate outbox with a secondary (non-zero) color, computes the box perimeter cells at even Manhattan distances from the nearest object cells, and marks those perimeter cells with color 5.",
    "0dfd9992": "Derives vertical and horizontal repetition periods from the last row/column strips, builds a neighborhood of offsets scaled by those periods, and stamps all non-zero cells at those lattice positions onto the canvas.",
    "29ec7d0e": "Same as 0dfd9992: infer v/h periodicity from edge strips, generate a local lattice of scaled neighbor offsets, and replicate the foreground over that lattice.",
    "36d67576": "Chooses the most colorful object, selects its parts whose colors are in {2,4}, for all combinations of basic mirrors (including compositions) builds symmetric variants, computes per-variant shifts from occurrences, and paints the shifted variants back.",
    "98cf29f8": "Among foreground pieces, finds a solid rectangle and the other pieces. It selects those other pieces whose diagonal neighborhood is rich in the shared color, takes the outbox/backdrop of that subset, slides (gravitates) this outline until adjacent to the rectangle, and fills it with the subset’s color.",
    "469497ad": "Upscales by (num_colors−1), takes the smallest object, shoots four diagonals from its upper-left and lower-left corners, underfills those diagonals with color 2, then identifies the bottom-rightmost resulting object and paints it on top.",
    "39e1d7f9": "Select the most contextually distinctive object of a certain color, crop an enlarged window around it, and stamp that crop (as an object) around all objects of that same color aligned by their upper-left corners.",
    "484b58aa": "Infer vertical and horizontal repeat periods from border crops, then replicate the interior foreground motif on a lattice of offsets determined by those periods.",
    "3befdf3e": "For each foreground object, derive a framed backdrop (inbox/outbox/corners), fill the interior backdrop with a chosen non-background color, and clear cells that fall outside the intersection of those backdrops.",
    "9aec4887": "Project the simpler object onto the complex one by mapping each shifted cell to its nearest site in the larger object, then overlay an X-shaped mask and mark its intersections with a highlight color.",
    "49d1d64f": "Pad the image with a one-cell border and, from every border cell except corners, draw a shortest Manhattan path to the nearest foreground cell, painting those paths onto the padded canvas.",
    "57aa92db": "Identify the foreground object with the greatest internal color imbalance, extract its rarest-color subshape, upscale/align copies of that subshape near matching-colored objects, and finally restore all objects.",
    "aba27056": "Compute the union box around all objects, thicken/extend that box along a direction, find special background targets adjacent to both the box and its shifts, and fire rays from the box’s corners to fill those targets.",
    "f1cefba8": "Within the first object’s window, detect the rarest color and place full row/column frontiers through its extreme occurrences using a chosen partner color, while preserving intersections on original background as the rarest color.",
    "1e32b0e9": "Sample a small top-left tile, find a color present globally but missing from that tile, and stamp the tile’s content at the nine grid-centered positions of a 3×3 layout, filling only background with that color.",
    "28e73c20": "Generate a fixed decorative tile made of two colors (choosing a variant by input parity) and repeatedly concatenate/rotate it to build a patterned strip that matches the image dimensions.",
    "4c5c2cf0": "Detect a rotationally symmetric template, mirror the relevant subgrids, and shift/paint the corresponding pieces so the assembled figure achieves the intended bilateral/rotational symmetry.",
    "508bd3b6": "Take the smallest and largest objects, draw long guide lines through the scene from a corner tied to the small object, paint those lines with a marker color, and then re-apply the two objects on top.",
    "6d0160f0": "Place a set of 3×3 reference markers across the grid from fixed offsets, choose one 3×3 window based on the position of a key color, extract its feature, and project that feature to a scaled location before repainting.",
    "f8a8fe49": "Isolate the main colored motif, mirror it according to its orientation, split it into two halves, and then shift/paint the halves to opposite sides around the motif’s anchor after clearing an interfering color.",
    "d07ae81c": "From each singleton marker, shoot rays in cardinal and diagonal directions, test which colored regions they hit, and fill the contacted cells with colors chosen by a local neighbor/mode rule.",
    "6a1e5592": "Use statistics from a top band (counts over zeros/twos and box features) to score candidate shifts for the ‘5’ clusters, choose the best shift per cluster, clear the fives, and place single markers at the chosen destinations.",
    "0e206a2e": "Find multicolor motifs and their symmetric variants (reflections and compositions), locate all occurrences via transformations and relative offsets, paint all aligned copies, and remove the originals.",
    "d22278a0": "For each object, build a center-relative parity stencil over all coordinates, intersect it with constraints derived from the object set, and recolor the surviving cells with the object’s color.",
    "4290ef0e": "Score foreground components by a width-and-distance objective, pick exemplars per color group, choose symmetry transforms that best align centers, normalize and place them on a lattice, then stamp the lattice under four 90° rotations to form a kaleidoscopic tiling.",
    "50846271": "Complete long straight connections among pixels of a given color, select a best center per run, and draw a plus-shaped marker of fixed half-extent at that center before restoring the original pixels.",
    "b527c5c6": "Extend each object along a direction inferred from its relation to a specific colored subset (drawing straight guides for vertical-line cases and stepped ribbons otherwise), paint those guides with one color, and underfill adjacent cells with a secondary color.",
    "150deff5": "Detects specific small prototypes and overwrites them: first turns every 2×2 block of color 5 into color 8, then finds any length-3 straight segment that contains exactly two 8s and one 5 (in any orientation or mirror) and recolors those segments to color 2.",
    "b7249182": "Normalizes orientation if needed, identifies the topmost and bottommost foreground objects, draws a straight connector between their first cells, colors the connector with the bottom object’s color and a branch toward the connector midpoint with the top object’s color, adds a small bracket/rail around that midpoint, and finally restores the original orientation.",
    "9d9215db": "Chooses the rotation of the input that maximizes a color-diversity score on the upper-left portion, then, within that chosen view, isolates four equal tiles/blocks, prunes their even/odd edge traces into corner-to-corner lines, recolors those lines onto the canvas, and finally replicates the result across all rotations to enforce rotational consistency.",
    "6855a6e4": "If a tall/portrait target piece is present, use the input as is; otherwise rotate 90°. Among the colored objects (focusing on color 5), pick the two that are opposite around a common center. Mirror their local patches horizontally, extract their color-5 silhouettes, shift them to symmetric destinations offset from the originals, erase the old pair, paint the mirrored copies in their new symmetric locations, then unrotate if needed.",
    "264363fd": "Finds the smallest non-background object, measures whether it is 5×5, samples its center color and a neighbor to infer the ‘fill’ color, then for each background object draws faint vertical/horizontal guide frontiers around where a 1×1 probe of the center color occurs, paints those guides with the sampled color, shifts the small object toward the inferred center by a vector derived from its size test, and finally floods remaining background with the scene’s background color.",
    "7df24a62": "Cuts out a 1-colored patch (color 1) and generates its three rotations. For every rotation, it computes all placements over a lattice of valid positions, filters those placements that don’t collide with the scene’s color-4 structure, and paints the union of all non-colliding placements of the rotated copies back into the original grid as color-1.",
    "f15e1fac": "Canonicalizes orientation so that a color-2 bar is left-aligned and color-8 sits at the top; shoots rays from every color-8 cell, gathers the vertical indices of the color-2 structure, then, using a set of index-pair predicates (relative comparisons on first/last endpoints), selects ray segments that lie strictly between successive row/column bounds. Those selected segments are shifted into parallel lanes and painted as color-8; finally the earlier canonicalization transforms are undone.",
    "234bbc79": "Colors every object by ‘the other palette color’ seen next to it, arranges objects left→right, and then—by repeatedly picking extreme members in subsets that share a leftmost/rightmost row—moves each selected object one step diagonally (down-left) while keeping the rest in place. After applying this iterative pairing/motion program to the whole list, it renders the resulting object set onto a blank canvas the size of the moved ensemble.",
    "22233c11": "For each connected object, upscales a mirrored copy, centers it by half the object’s shape (rounded toward the center), computes the union of infinite vertical and horizontal guide frontiers through the upscaled copy, subtracts those guides from the copy’s index set, and fills the remaining indices on the original grid with color 8—i.e., paint the object’s ‘thickened outline’ minus its axial guides.",
    "2dd70a9a": "Given a color-2 and color-3 pair of lines, determines whether they are vertical or horizontal and which has priority by extremal index; extends a line from the priority anchor toward the other with background-preserving drawing (underfill as color 1), segments the new color-1 components, keeps only those adjacent to the original pair, erases the rest, then ‘gravitates’ the nearest kept component toward the anchor with a one-step offset and completes a two-stage connection (anchor→midpoint→center) before turning all produced color-1 into color-3.",
    "a64e4611": "Builds a parametric ‘stamping’ function that chooses, for a given scalar n, the best-matching canvas pattern (by object occurrence scoring), stamps it repeatedly at indices tied to n, and paints the stamps as color-3. It sweeps n over two ranges to accumulate stamps, then uses local neighbor palette tests to add extra color-3 at cells that have exactly eight color-3 neighbors and sit on the border or are adjacent to a border cell containing a 3; finally it expands color-3 to the remaining eligible background cells that pass both tests.",
    "7837ac64": "Removes the largest foreground blob to get a ‘working window’, detects background-colored sub-objects inside it that lie in single-color outboxes and whose outbox corners carry the global frontier color; recolors those qualifying sub-objects with the outbox-corner color and paints them back. Then, based on the height of a reference object, it periodically samples rows, mirrors the sampling pattern across the diagonal twice, applies it to the enriched window, and finally downsamples by the reference height.",
    "a8c38be5": "Starts from a blank 9×9 canvas of color 5 framed by a complex ring (box, outbox-of-outbox corners, and an inner ‘inbox’ mask selected by parity relative to center). It paints that ring as color 1, partitions the ring into objects, and for each normalized input object finds the matching ring object by shape, aligns the input object to the matching ring object’s upper-left, and paints all normalized input objects onto the prepared ring canvas.",
    "b775ac94": "Splits each multi-part object into a ‘majority-color core’ and its complement; isolates the complement piece that sits adjacent to the majority core; grows three symmetry-shifted shadows of the core (horizontal, vertical, and both) positioned by the relative pose of core vs. complement; for each shadow, looks up the color from the original input at the first overlapping index with the assembled figure and recolors that shadow accordingly; finally paints all recolored shadows back onto the original grid.",
    "97a05b5b": "Focuses on the largest composite object then switches its local colors (2↔0) to make a working mask. It splits each sub-object into ‘is-2’ and ‘not-2’ parts, aligns them by upper-left, finds placements where these normalized parts co-occur within the mask, and filters those placements by counting how many training occurrences of color-2 each has. It then applies a library of symmetry transforms in pairs, evaluates each transformed placement by that count predicate, and paints all accepted placements back onto the crop; finally, from the colors that appear in the painted placements, it selects all except color 2 and applies an additional round on the remaining candidates before merging everything back.",
    "3e980e27": "Targets objects that contain colors 2 and 3. For the 2-colored subset, mirrors a reference object vertically and shifts all peers toward it by the negative of each peer’s upper-left; for the 3-colored subset, shifts all peers toward the center of a reference 3-object by the same rule. It combines both moved sets and paints them onto the original grid, effectively ‘aligning’ every 2-object to a mirrored anchor and every 3-object to a centered anchor, then overlaying the aligned copies.",
}
