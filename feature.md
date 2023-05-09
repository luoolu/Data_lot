# Dlt 5 qianqu
## 和值计算公式
=SUM(B2:F2)

## 最大间距
=MAX(ABS(B2 - C2), ABS(C2 - D2), ABS(D2 - E2), ABS(E2 - F2))

## 最小间距
=MIN(ABS(B2 - C2), ABS(C2 - D2), ABS(D2 - E2), ABS(E2 - F2))

## 极值
=MAX(B2, C2, D2, E2, F2) - MIN(B2, C2, D2, E2, F2)

## 质数个数
=IF(AND(B2<>1,OR(B2=2,AND(MOD(B2,ROW(INDIRECT("2:"&INT(SQRT(B2)))))<>0))),1,0)
+IF(AND(C2<>1,OR(C2=2,AND(MOD(C2,ROW(INDIRECT("2:"&INT(SQRT(C2)))))<>0))),1,0)
+IF(AND(D2<>1,OR(D2=2,AND(MOD(D2,ROW(INDIRECT("2:"&INT(SQRT(D2)))))<>0))),1,0)
+IF(AND(E2<>1,OR(E2=2,AND(MOD(E2,ROW(INDIRECT("2:"&INT(SQRT(E2)))))<>0))),1,0)
+IF(AND(F2<>1,OR(F2=2,AND(MOD(F2,ROW(INDIRECT("2:"&INT(SQRT(F2)))))<>0))),1,0)

## 合数个数
=5 - (
 IF(AND(B2<>1,OR(B2=2,AND(MOD(B2,ROW(INDIRECT("2:"&INT(SQRT(B2)))))<>0))),1,0)
+IF(AND(C2<>1,OR(C2=2,AND(MOD(C2,ROW(INDIRECT("2:"&INT(SQRT(C2)))))<>0))),1,0)
+IF(AND(D2<>1,OR(D2=2,AND(MOD(D2,ROW(INDIRECT("2:"&INT(SQRT(D2)))))<>0))),1,0)
+IF(AND(E2<>1,OR(E2=2,AND(MOD(E2,ROW(INDIRECT("2:"&INT(SQRT(E2)))))<>0))),1,0)
+IF(AND(F2<>1,OR(F2=2,AND(MOD(F2,ROW(INDIRECT("2:"&INT(SQRT(F2)))))<>0))),1,0)
)

## 与上一组相同数字个数
=SUM(--(ISNUMBER(MATCH(B3:F3, B2:F2, 0))))

## 奇数个数
=IF(MOD(B2, 2) = 1, 1, 0) + IF(MOD(C2, 2) = 1, 1, 0) + IF(MOD(D2, 2) = 1, 1, 0) + IF(MOD(E2, 2) = 1, 1, 0) + IF(MOD(F2, 2) = 1, 1, 0)

## 偶数个数
=IF(MOD(B2, 2) = 0, 1, 0) + IF(MOD(C2, 2) = 0, 1, 0) + IF(MOD(D2, 2) = 0, 1, 0) + IF(MOD(E2, 2) = 0, 1, 0) + IF(MOD(F2, 2) = 0, 1, 0)

## 均值
=AVERAGE(B2, C2, D2, E2, F2)

## 相邻个数
=(IF(ABS(B2-C2)=1, 1, 0) + IF(ABS(C2-D2)=1, 1, 0) + IF(ABS(D2-E2)=1, 1, 0) + IF(ABS(E2-F2)=1, 1, 0) + IF(ABS(F2-B2)=1, 1, 0))

## 2-35余数计算公式

=IF(MOD(B2, 2) = 0, 1, 0) + IF(MOD(C2, 2) = 0, 1, 0) + IF(MOD(D2, 2) = 0, 1, 0) + IF(MOD(E2, 2) = 0, 1, 0) + IF(MOD(F2, 2) = 0, 1, 0)
=IF(MOD(B2, 3) = 0, 1, 0) + IF(MOD(C2, 3) = 0, 1, 0) + IF(MOD(D2, 3) = 0, 1, 0) + IF(MOD(E2, 3) = 0, 1, 0) + IF(MOD(F2, 3) = 0, 1, 0)
=IF(MOD(B2, 4) = 0, 1, 0) + IF(MOD(C2, 4) = 0, 1, 0) + IF(MOD(D2, 4) = 0, 1, 0) + IF(MOD(E2, 4) = 0, 1, 0) + IF(MOD(F2, 4) = 0, 1, 0)
=IF(MOD(B2, 5) = 0, 1, 0) + IF(MOD(C2, 5) = 0, 1, 0) + IF(MOD(D2, 5) = 0, 1, 0) + IF(MOD(E2, 5) = 0, 1, 0) + IF(MOD(F2, 5) = 0, 1, 0)
=IF(MOD(B2, 6) = 0, 1, 0) + IF(MOD(C2, 6) = 0, 1, 0) + IF(MOD(D2, 6) = 0, 1, 0) + IF(MOD(E2, 6) = 0, 1, 0) + IF(MOD(F2, 6) = 0, 1, 0)
=IF(MOD(B2, 7) = 0, 1, 0) + IF(MOD(C2, 7) = 0, 1, 0) + IF(MOD(D2, 7) = 0, 1, 0) + IF(MOD(E2, 7) = 0, 1, 0) + IF(MOD(F2, 7) = 0, 1, 0)
=IF(MOD(B2, 8) = 0, 1, 0) + IF(MOD(C2, 8) = 0, 1, 0) + IF(MOD(D2, 8) = 0, 1, 0) + IF(MOD(E2, 8) = 0, 1, 0) + IF(MOD(F2, 8) = 0, 1, 0)
=IF(MOD(B2, 9) = 0, 1, 0) + IF(MOD(C2, 9) = 0, 1, 0) + IF(MOD(D2, 9) = 0, 1, 0) + IF(MOD(E2, 9) = 0, 1, 0) + IF(MOD(F2, 9) = 0, 1, 0)
=IF(MOD(B2, 10) = 0, 1, 0) + IF(MOD(C2, 10) = 0, 1, 0) + IF(MOD(D2, 10) = 0, 1, 0) + IF(MOD(E2, 10) = 0, 1, 0) + IF(MOD(F2, 10) = 0, 1, 0)
=IF(MOD(B2, 11) = 0, 1, 0) + IF(MOD(C2, 11) = 0, 1, 0) + IF(MOD(D2, 11) = 0, 1, 0) + IF(MOD(E2, 11) = 0, 1, 0) + IF(MOD(F2, 11) = 0, 1, 0)
=IF(MOD(B2, 12) = 0, 1, 0) + IF(MOD(C2, 12) = 0, 1, 0) + IF(MOD(D2, 12) = 0, 1, 0) + IF(MOD(E2, 12) = 0, 1, 0) + IF(MOD(F2, 12) = 0, 1, 0)
=IF(MOD(B2, 13) = 0, 1, 0) + IF(MOD(C2, 13) = 0, 1, 0) + IF(MOD(D2, 13) = 0, 1, 0) + IF(MOD(E2, 13) = 0, 1, 0) + IF(MOD(F2, 13) = 0, 1, 0)
=IF(MOD(B2, 14) = 0, 1, 0) + IF(MOD(C2, 14) = 0, 1, 0) + IF(MOD(D2, 14) = 0, 1, 0) + IF(MOD(E2, 14) = 0, 1, 0) + IF(MOD(F2, 14) = 0, 1, 0)
=IF(MOD(B2, 15) = 0, 1, 0) + IF(MOD(C2, 15) = 0, 1, 0) + IF(MOD(D2, 15) = 0, 1, 0) + IF(MOD(E2, 15) = 0, 1, 0) + IF(MOD(F2, 15) = 0, 1, 0)
=IF(MOD(B2, 16) = 0, 1, 0) + IF(MOD(C2, 16) = 0, 1, 0) + IF(MOD(D2, 16) = 0, 1, 0) + IF(MOD(E2, 16) = 0, 1, 0) + IF(MOD(F2, 16) = 0, 1, 0)
=IF(MOD(B2, 17) = 0, 1, 0) + IF(MOD(C2, 17) = 0, 1, 0) + IF(MOD(D2, 17) = 0, 1, 0) + IF(MOD(E2, 17) = 0, 1, 0) + IF(MOD(F2, 17) = 0, 1, 0)
=IF(MOD(B2, 18) = 0, 1, 0) + IF(MOD(C2, 18) = 0, 1, 0) + IF(MOD(D2, 18) = 0, 1, 0) + IF(MOD(E2, 18) = 0, 1, 0) + IF(MOD(F2, 18) = 0, 1, 0)
=IF(MOD(B2, 19) = 0, 1, 0) + IF(MOD(C2, 19) = 0, 1, 0) + IF(MOD(D2, 19) = 0, 1, 0) + IF(MOD(E2, 19) = 0, 1, 0) + IF(MOD(F2, 19) = 0, 1, 0)
=IF(MOD(B2, 20) = 0, 1, 0) + IF(MOD(C2, 20) = 0, 1, 0) + IF(MOD(D2, 20) = 0, 1, 0) + IF(MOD(E2, 20) = 0, 1, 0) + IF(MOD(F2, 20) = 0, 1, 0)
=IF(MOD(B2, 21) = 0, 1, 0) + IF(MOD(C2, 21) = 0, 1, 0) + IF(MOD(D2, 21) = 0, 1, 0) + IF(MOD(E2, 21) = 0, 1, 0) + IF(MOD(F2, 21) = 0, 1, 0)
=IF(MOD(B2, 22) = 0, 1, 0) + IF(MOD(C2, 22) = 0, 1, 0) + IF(MOD(D2, 22) = 0, 1, 0) + IF(MOD(E2, 22) = 0, 1, 0) + IF(MOD(F2, 22) = 0, 1, 0)
=IF(MOD(B2, 23) = 0, 1, 0) + IF(MOD(C2, 23) = 0, 1, 0) + IF(MOD(D2, 23) = 0, 1, 0) + IF(MOD(E2, 23) = 0, 1, 0) + IF(MOD(F2, 23) = 0, 1, 0)
=IF(MOD(B2, 24) = 0, 1, 0) + IF(MOD(C2, 24) = 0, 1, 0) + IF(MOD(D2, 24) = 0, 1, 0) + IF(MOD(E2, 24) = 0, 1, 0) + IF(MOD(F2, 24) = 0, 1, 0)
=IF(MOD(B2, 25) = 0, 1, 0) + IF(MOD(C2, 25) = 0, 1, 0) + IF(MOD(D2, 25) = 0, 1, 0) + IF(MOD(E2, 25) = 0, 1, 0) + IF(MOD(F2, 25) = 0, 1, 0)
=IF(MOD(B2, 26) = 0, 1, 0) + IF(MOD(C2, 26) = 0, 1, 0) + IF(MOD(D2, 26) = 0, 1, 0) + IF(MOD(E2, 26) = 0, 1, 0) + IF(MOD(F2, 26) = 0, 1, 0)
=IF(MOD(B2, 27) = 0, 1, 0) + IF(MOD(C2, 27) = 0, 1, 0) + IF(MOD(D2, 27) = 0, 1, 0) + IF(MOD(E2, 27) = 0, 1, 0) + IF(MOD(F2, 27) = 0, 1, 0)
=IF(MOD(B2, 28) = 0, 1, 0) + IF(MOD(C2, 28) = 0, 1, 0) + IF(MOD(D2, 28) = 0, 1, 0) + IF(MOD(E2, 28) = 0, 1, 0) + IF(MOD(F2, 28) = 0, 1, 0)
=IF(MOD(B2, 29) = 0, 1, 0) + IF(MOD(C2, 29) = 0, 1, 0) + IF(MOD(D2, 29) = 0, 1, 0) + IF(MOD(E2, 29) = 0, 1, 0) + IF(MOD(F2, 29) = 0, 1, 0)
=IF(MOD(B2, 30) = 0, 1, 0) + IF(MOD(C2, 30) = 0, 1, 0) + IF(MOD(D2, 30) = 0, 1, 0) + IF(MOD(E2, 30) = 0, 1, 0) + IF(MOD(F2, 30) = 0, 1, 0)
=IF(MOD(B2, 31) = 0, 1, 0) + IF(MOD(C2, 31) = 0, 1, 0) + IF(MOD(D2, 31) = 0, 1, 0) + IF(MOD(E2, 31) = 0, 1, 0) + IF(MOD(F2, 31) = 0, 1, 0)
=IF(MOD(B2, 32) = 0, 1, 0) + IF(MOD(C2, 32) = 0, 1, 0) + IF(MOD(D2, 32) = 0, 1, 0) + IF(MOD(E2, 32) = 0, 1, 0) + IF(MOD(F2, 32) = 0, 1, 0)
=IF(MOD(B2, 33) = 0, 1, 0) + IF(MOD(C2, 33) = 0, 1, 0) + IF(MOD(D2, 33) = 0, 1, 0) + IF(MOD(E2, 33) = 0, 1, 0) + IF(MOD(F2, 33) = 0, 1, 0)
=IF(MOD(B2, 34) = 0, 1, 0) + IF(MOD(C2, 34) = 0, 1, 0) + IF(MOD(D2, 34) = 0, 1, 0) + IF(MOD(E2, 34) = 0, 1, 0) + IF(MOD(F2, 34) = 0, 1, 0)
=IF(MOD(B2, 35) = 0, 1, 0) + IF(MOD(C2, 35) = 0, 1, 0) + IF(MOD(D2, 35) = 0, 1, 0) + IF(MOD(E2, 35) = 0, 1, 0) + IF(MOD(F2, 35) = 0, 1, 0)

**********************************************************************************************************************************
# Dlt 2 houqu
## 和值计算公式
=SUM(B2:C2)

## 两数相邻
=IF(ABS(B2-C2)=1, 1, 0)

## 亲和数（Amicable numbers）是指一对正整数，其中每个数都是另一个数的真因数之和
=IF(SUMPRODUCT((MOD(B2,ROW(INDIRECT("1:"&INT(B2/2))))=0)*ROW(INDIRECT("1:"&INT(B2/2))))=C2, IF(SUMPRODUCT((MOD(C2,ROW(INDIRECT("1:"&INT(C2/2))))=0)*ROW(INDIRECT("1:"&INT(C2/2))))=B2, 1, 0), 0)

## 间距
=MAX(ABS(B2 - C2))

## 质数个数
=IF(AND(B2<>1,OR(B2=2,AND(MOD(B2,ROW(INDIRECT("2:"&INT(SQRT(B2)))))<>0))),1,0)
+IF(AND(C2<>1,OR(C2=2,AND(MOD(C2,ROW(INDIRECT("2:"&INT(SQRT(C2)))))<>0))),1,0)

## 合数个数
=2 - (
 IF(AND(B2<>1,OR(B2=2,AND(MOD(B2,ROW(INDIRECT("2:"&INT(SQRT(B2)))))<>0))),1,0)
+IF(AND(C2<>1,OR(C2=2,AND(MOD(C2,ROW(INDIRECT("2:"&INT(SQRT(C2)))))<>0))),1,0)
)

## 与上一组相同数字个数
=SUM(--(ISNUMBER(MATCH(B3:C3, B2:C2, 0))))

## 奇数个数
=IF(MOD(B2, 2) = 1, 1, 0) + IF(MOD(C2, 2) = 1, 1, 0)

## 偶数个数
=IF(MOD(B2, 2) = 0, 1, 0) + IF(MOD(C2, 2) = 0, 1, 0)

## 均值
=AVERAGE(B2, C2)

## 2-16余数计算公式

=IF(MOD(B2, 2) = 0, 1, 0) + IF(MOD(C2, 2) = 0, 1, 0)
=IF(MOD(B2, 3) = 0, 1, 0) + IF(MOD(C2, 3) = 0, 1, 0)
=IF(MOD(B2, 4) = 0, 1, 0) + IF(MOD(C2, 4) = 0, 1, 0)
=IF(MOD(B2, 5) = 0, 1, 0) + IF(MOD(C2, 5) = 0, 1, 0)
=IF(MOD(B2, 6) = 0, 1, 0) + IF(MOD(C2, 6) = 0, 1, 0)
=IF(MOD(B2, 7) = 0, 1, 0) + IF(MOD(C2, 7) = 0, 1, 0)
=IF(MOD(B2, 8) = 0, 1, 0) + IF(MOD(C2, 8) = 0, 1, 0)
=IF(MOD(B2, 9) = 0, 1, 0) + IF(MOD(C2, 9) = 0, 1, 0)
=IF(MOD(B2, 10) = 0, 1, 0) + IF(MOD(C2, 10) = 0, 1, 0)
=IF(MOD(B2, 11) = 0, 1, 0) + IF(MOD(C2, 11) = 0, 1, 0)
=IF(MOD(B2, 12) = 0, 1, 0) + IF(MOD(C2, 12) = 0, 1, 0)
=IF(MOD(B2, 13) = 0, 1, 0) + IF(MOD(C2, 13) = 0, 1, 0)
=IF(MOD(B2, 14) = 0, 1, 0) + IF(MOD(C2, 14) = 0, 1, 0)
=IF(MOD(B2, 15) = 0, 1, 0) + IF(MOD(C2, 15) = 0, 1, 0)
=IF(MOD(B2, 16) = 0, 1, 0) + IF(MOD(C2, 16) = 0, 1, 0)

————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# SSQ 6 red
## 和值计算公式
=SUM(B2:G2)

## 最大间距
=MAX(ABS(B2 - C2), ABS(C2 - D2), ABS(D2 - E2), ABS(E2 - F2), ABS(F2 - G2))

## 最小间距
=MIN(ABS(B2 - C2), ABS(C2 - D2), ABS(D2 - E2), ABS(E2 - F2), ABS(F2 - G2))

## 极值
=MAX(B2, C2, D2, E2, F2, G2) - MIN(B2, C2, D2, E2, F2, G2)

## 质数个数
=IF(AND(B2<>1,OR(B2=2,AND(MOD(B2,ROW(INDIRECT("2:"&INT(SQRT(B2)))))<>0))),1,0)
+IF(AND(C2<>1,OR(C2=2,AND(MOD(C2,ROW(INDIRECT("2:"&INT(SQRT(C2)))))<>0))),1,0)
+IF(AND(D2<>1,OR(D2=2,AND(MOD(D2,ROW(INDIRECT("2:"&INT(SQRT(D2)))))<>0))),1,0)
+IF(AND(E2<>1,OR(E2=2,AND(MOD(E2,ROW(INDIRECT("2:"&INT(SQRT(E2)))))<>0))),1,0)
+IF(AND(F2<>1,OR(F2=2,AND(MOD(F2,ROW(INDIRECT("2:"&INT(SQRT(F2)))))<>0))),1,0)
+IF(AND(F2<>1,OR(F2=2,AND(MOD(F2,ROW(INDIRECT("2:"&INT(SQRT(G2)))))<>0))),1,0)

## 合数个数
=6 - (
 IF(AND(B2<>1,OR(B2=2,AND(MOD(B2,ROW(INDIRECT("2:"&INT(SQRT(B2)))))<>0))),1,0)
+IF(AND(C2<>1,OR(C2=2,AND(MOD(C2,ROW(INDIRECT("2:"&INT(SQRT(C2)))))<>0))),1,0)
+IF(AND(D2<>1,OR(D2=2,AND(MOD(D2,ROW(INDIRECT("2:"&INT(SQRT(D2)))))<>0))),1,0)
+IF(AND(E2<>1,OR(E2=2,AND(MOD(E2,ROW(INDIRECT("2:"&INT(SQRT(E2)))))<>0))),1,0)
+IF(AND(F2<>1,OR(F2=2,AND(MOD(F2,ROW(INDIRECT("2:"&INT(SQRT(F2)))))<>0))),1,0)
+IF(AND(F2<>1,OR(F2=2,AND(MOD(F2,ROW(INDIRECT("2:"&INT(SQRT(G2)))))<>0))),1,0)
)

## 与上一组相同数字个数
=SUM(--(ISNUMBER(MATCH(B3:G3, B2:G2, 0))))

## 奇数个数
=IF(MOD(B2, 2) = 1, 1, 0) + IF(MOD(C2, 2) = 1, 1, 0) + IF(MOD(D2, 2) = 1, 1, 0) + IF(MOD(E2, 2) = 1, 1, 0) + IF(MOD(F2, 2) = 1, 1, 0) + IF(MOD(G2, 2) = 1, 1, 0)

## 偶数个数
=IF(MOD(B2, 2) = 0, 1, 0) + IF(MOD(C2, 2) = 0, 1, 0) + IF(MOD(D2, 2) = 0, 1, 0) + IF(MOD(E2, 2) = 0, 1, 0) + IF(MOD(F2, 2) = 0, 1, 0) + IF(MOD(G2, 2) = 0, 1, 0)

## 均值
=AVERAGE(B2, C2, D2, E2, F2, G2)

## 相邻个数
=(IF(ABS(B2-C2)=1, 1, 0) + IF(ABS(C2-D2)=1, 1, 0) + IF(ABS(D2-E2)=1, 1, 0) + IF(ABS(E2-F2)=1, 1, 0) + IF(ABS(F2-G2)=1, 1, 0) + IF(ABS(G2-B2)=1, 1, 0))

## 2-17余数计算公式

=IF(MOD(B2, 2) = 0, 1, 0) + IF(MOD(C2, 2) = 0, 1, 0) + IF(MOD(D2, 2) = 0, 1, 0) + IF(MOD(E2, 2) = 0, 1, 0) + IF(MOD(F2, 2) = 0, 1, 0) + IF(MOD(G2, 2) = 0, 1, 0)
=IF(MOD(B2, 3) = 0, 1, 0) + IF(MOD(C2, 3) = 0, 1, 0) + IF(MOD(D2, 3) = 0, 1, 0) + IF(MOD(E2, 3) = 0, 1, 0) + IF(MOD(F2, 3) = 0, 1, 0) + IF(MOD(G2, 3) = 0, 1, 0)
=IF(MOD(B2, 4) = 0, 1, 0) + IF(MOD(C2, 4) = 0, 1, 0) + IF(MOD(D2, 4) = 0, 1, 0) + IF(MOD(E2, 4) = 0, 1, 0) + IF(MOD(F2, 4) = 0, 1, 0) + IF(MOD(G2, 4) = 0, 1, 0)
=IF(MOD(B2, 5) = 0, 1, 0) + IF(MOD(C2, 5) = 0, 1, 0) + IF(MOD(D2, 5) = 0, 1, 0) + IF(MOD(E2, 5) = 0, 1, 0) + IF(MOD(F2, 5) = 0, 1, 0) + IF(MOD(G2, 5) = 0, 1, 0)
=IF(MOD(B2, 6) = 0, 1, 0) + IF(MOD(C2, 6) = 0, 1, 0) + IF(MOD(D2, 6) = 0, 1, 0) + IF(MOD(E2, 6) = 0, 1, 0) + IF(MOD(F2, 6) = 0, 1, 0) + IF(MOD(G2, 6) = 0, 1, 0)
=IF(MOD(B2, 7) = 0, 1, 0) + IF(MOD(C2, 7) = 0, 1, 0) + IF(MOD(D2, 7) = 0, 1, 0) + IF(MOD(E2, 7) = 0, 1, 0) + IF(MOD(F2, 7) = 0, 1, 0) + IF(MOD(G2, 7) = 0, 1, 0)
=IF(MOD(B2, 8) = 0, 1, 0) + IF(MOD(C2, 8) = 0, 1, 0) + IF(MOD(D2, 8) = 0, 1, 0) + IF(MOD(E2, 8) = 0, 1, 0) + IF(MOD(F2, 8) = 0, 1, 0) + IF(MOD(G2, 8) = 0, 1, 0)
=IF(MOD(B2, 9) = 0, 1, 0) + IF(MOD(C2, 9) = 0, 1, 0) + IF(MOD(D2, 9) = 0, 1, 0) + IF(MOD(E2, 9) = 0, 1, 0) + IF(MOD(F2, 9) = 0, 1, 0) + IF(MOD(G2, 9) = 0, 1, 0)
=IF(MOD(B2, 10) = 0, 1, 0) + IF(MOD(C2, 10) = 0, 1, 0) + IF(MOD(D2, 10) = 0, 1, 0) + IF(MOD(E2, 10) = 0, 1, 0) + IF(MOD(F2, 10) = 0, 1, 0) + IF(MOD(G2, 10) = 0, 1, 0)
=IF(MOD(B2, 11) = 0, 1, 0) + IF(MOD(C2, 11) = 0, 1, 0) + IF(MOD(D2, 11) = 0, 1, 0) + IF(MOD(E2, 11) = 0, 1, 0) + IF(MOD(F2, 11) = 0, 1, 0) + IF(MOD(G2, 11) = 0, 1, 0)
=IF(MOD(B2, 12) = 0, 1, 0) + IF(MOD(C2, 12) = 0, 1, 0) + IF(MOD(D2, 12) = 0, 1, 0) + IF(MOD(E2, 12) = 0, 1, 0) + IF(MOD(F2, 12) = 0, 1, 0) + IF(MOD(G2, 12) = 0, 1, 0)
=IF(MOD(B2, 13) = 0, 1, 0) + IF(MOD(C2, 13) = 0, 1, 0) + IF(MOD(D2, 13) = 0, 1, 0) + IF(MOD(E2, 13) = 0, 1, 0) + IF(MOD(F2, 13) = 0, 1, 0) + IF(MOD(G2, 13) = 0, 1, 0)
=IF(MOD(B2, 14) = 0, 1, 0) + IF(MOD(C2, 14) = 0, 1, 0) + IF(MOD(D2, 14) = 0, 1, 0) + IF(MOD(E2, 14) = 0, 1, 0) + IF(MOD(F2, 14) = 0, 1, 0) + IF(MOD(G2, 14) = 0, 1, 0)
=IF(MOD(B2, 15) = 0, 1, 0) + IF(MOD(C2, 15) = 0, 1, 0) + IF(MOD(D2, 15) = 0, 1, 0) + IF(MOD(E2, 15) = 0, 1, 0) + IF(MOD(F2, 15) = 0, 1, 0) + IF(MOD(G2, 15) = 0, 1, 0)
=IF(MOD(B2, 16) = 0, 1, 0) + IF(MOD(C2, 16) = 0, 1, 0) + IF(MOD(D2, 16) = 0, 1, 0) + IF(MOD(E2, 16) = 0, 1, 0) + IF(MOD(F2, 16) = 0, 1, 0) + IF(MOD(G2, 16) = 0, 1, 0)
=IF(MOD(B2, 17) = 0, 1, 0) + IF(MOD(C2, 17) = 0, 1, 0) + IF(MOD(D2, 17) = 0, 1, 0) + IF(MOD(E2, 17) = 0, 1, 0) + IF(MOD(F2, 17) = 0, 1, 0) + IF(MOD(G2, 17) = 0, 1, 0)






