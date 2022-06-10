hw.module @Test(%a: i1, %b: i1, %c: i1, %d: i1) -> (e: i1, f: i1) {
  %0, %1 = comb.minimize %a, %b, %c, %d {
    table = ["0011-", "0101-", "0110-", "1001-", "-1101", "-1110", "-00-0", "0---0"]
  } : (i1, i1, i1, i1) -> (i1, i1)
  hw.output %0, %1 : i1, i1
}
