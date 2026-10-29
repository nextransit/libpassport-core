# MRZ GT vs PRED 对比（来源: photo_direct_bench_tess_conditional_v3.json）

照片总数 115  真值来源 photo_auto_gt_all.json

## 验收汇总

| 项目 | 数值 | 门槛 | 判定 |
|---|---|---|---|
| 官方 PASS 张数 | 105 | >= 105 | PASS |
| raw fuse decode 张数 | 84 | >= 84 | PASS |
| 官方路径速率（张/秒） | 4.432 | >= 2.0 | PASS |
| 总体结论 | PASS | | PASS |

- 速率【产品路径（全引擎并发）】：4.432 张/秒，平均 225.61 ms，中位 194.21 ms，P95 441.6 ms
- 速率【单引擎最快】：87.342 张/秒；批测墙钟吞吐 1.235 张/秒（参考）
- 通过率：官方 PASS 105/115 = 91.3%，REJECT 9，raw fuse decode 84
- 基线对比（photo_direct_bench_train3.json）：PASS +0，raw fuse +0，速率 +1.528 张/秒，退化 0 张


## 08806d29a6268e1f_0003.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=452
- 官方 line1: PMCHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
- 官方 line2: S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<`
- GT line2: `S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06`

### traditional  [decoded] conf1=74% conf2=73%
- error: decode failed: check digit mismatch
GT  : P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
PRED: AALLLAL<P<1LWGGLO<<P3GAAE<<<<<<<<<<<<<<<<<<<
      ^^^^^^^ ^^^^^^^ ^   ^^^^                       [19 mismatch]
GT  : S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
PRED: A24L9L23<LLLL74C8C4GP2E322G3<<<<<<<<<<<<<<<E
      ^ ^^^^   ^^^^ ^^ ^^^^ ^^  ^               ^^   [20 mismatch]

### cnn  [decoded] conf1=97% conf2=98%
- error: decode failed: check digit mismatch
GT  : P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
PRED: P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
PRED: 5210002356CHE7108019M2002293<5<55<<<<<<<<<<6
      ^       ^                    ^ ^^         ^    [6 mismatch]

### crnn  [decoded] conf1=98% conf2=95%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch
GT  : P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
PRED: PMCHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
PRED: S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<<6
                                                ^    [1 mismatch]

### fuse  [decoded] conf1=97% conf2=98%
- error: decode failed: check digit mismatch
GT  : P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
PRED: P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
PRED: 5210002356CHE7108019M2002293<<<<<<<<<<<<<<<6
      ^       ^                                 ^    [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (70 ms)
GT  : P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
PRED: PMCHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
PRED: S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=94%
- backend: paddle-onnxruntime  (91 ms)
GT  : P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
PRED: PMCHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
PRED: S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=98%
- backend: indra  (49 ms)
GT  : P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
PRED: PMCHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
PRED: 82100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
      ^                                              [1 mismatch]

### indra_fullpage  [DECODE-OK] conf1=94% conf2=100%
- backend: indra  (202 ms)
GT  : P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<
PRED: CISSADINANTANAZLUNALITADKATIONALITY<<<<<<<<<
      ^^^^^ ^^^^^^ ^^ ^^^^ ^^^^^^^^^^^^^^            [31 mismatch]
GT  : S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06
PRED: SCHWEIZSU2SSASVIZZE1ASVIZRA3WITZERLA<<<<<<80
       ^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^      ^^   [36 mismatch]

## 0f9780b8e1e3cc21_0008.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=166
- 官方 line1: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 3400072370USA6707046F1608078910000193<113538
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `3400072370USA6707046F1608078910000193<113538`

### traditional  [decoded] conf1=49% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: KT77<<<F726TFA<6<LTLL6<<LTJFFT<<<<<<T<T<<<<<
      ^^^^^^^^^^^^^^ ^^^^^^^^ ^^^^^^      ^ ^        [30 mismatch]
GT  : 3400072370USA6707046F1608078910000193<113538
PRED: K3ACCC7237CC9AK7C7C46P46C8C7894CCCC4V3<44353
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]

### cnn  [DECODE-OK] conf1=78% conf2=98%
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<GGASYNHWRCKKGGGAQGGGSNH<<<ELFGGG<<<S<UJ<<<
        ^    ^    ^ ^    ^^^ ^^   ^^^^^^   ^ ^^      [18 mismatch]
GT  : 3400072370USA6707046F1608078910000193<113538
PRED: 3400072370USA6707046F1608078910000193<113538
                                                     [0 mismatch]

### crnn  [decoded] conf1=96% conf2=90%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
        ^^ ^^^^^^^^^ ^^^^^ ^^                        [18 mismatch]
GT  : 3400072370USA6707046F1608078910000193<113538
PRED: 5400072370USA6707046F1608078910000193<113538
      ^                                              [1 mismatch]

### fuse  [DECODE-OK] conf1=89% conf2=98%
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<GGASYNHWRCKEGGGAQGGGS<H<<<ELFGGG<<<S<UJ<<<
        ^    ^    ^^^    ^^^  ^   ^^^^^^   ^ ^^      [18 mismatch]
GT  : 3400072370USA6707046F1608078910000193<113538
PRED: 3400072370USA6707046F1608078910000193<113538
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (76 ms)
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
        ^^ ^^^^^^^^^ ^^^^^ ^^                        [18 mismatch]
GT  : 3400072370USA6707046F1608078910000193<113538
PRED: 3400072370USA6707046F1608078910000193<113538
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=99%
- backend: paddle-onnxruntime  (82 ms)
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
        ^^ ^^^^^^^^^ ^^^^^ ^^                        [18 mismatch]
GT  : 3400072370USA6707046F1608078910000193<113538
PRED: 3400072370USA6707046F1608078910000193<113538
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=75% conf2=97%
- backend: indra  (53 ms)
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PRUSATRAVELERKHAPPYKKKKKKKKKKKKKKKKKKKKK<<<<
       ^^^ ^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^       [37 mismatch]
GT  : 3400072370USA6707046F1608078910000193<113538
PRED: 3400072370USA6707046F1608078910000193<113538
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=94% conf2=96%
- backend: indra  (217 ms)
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: ONONFSTERITYDOORDAINANDESTABLISHTHIS<<<<<<<<
      ^^^^^ ^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^           [34 mismatch]
GT  : 3400072370USA6707046F1608078910000193<113538
PRED: CONSTITUT2ONFORTBEU7ITEDSTA1ESOFAMERI<<<<<04
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^   [43 mismatch]

## 16c07a99a0fb81e4_0013.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=146
- 官方 line1: P<CZESPECIMEN<<VZOR<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 99009054<4CZE6906229F16072996956220612<<<<74
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `R900R054<4CZL6906229F16072996956220612<KKK74`

### traditional  [decoded] conf1=75% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<C2E9PEC1YEW<<V73M<<<<<<<<<<<<<<<<<<<<<<<<<
         ^ ^   ^^ ^  ^^^^                            [9 mismatch]
GT  : R900R054<4CZL6906229F16072996956220612<KKK74
PRED: VGCEG35A<AC2LK9E422GP3EE729G4V542234M2<<<<7A
      ^^^^^^ ^ ^ ^ ^ ^^  ^^^^^   ^^^ ^  ^^^  ^^^ ^   [28 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=95% conf2=85%
- backend: crnn  (11 ms)
- error: decode failed: bad MRZ length
GT  : P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CZESPECIMEN<<VZOR<<<<<<<<<<<<<<<<<<<<<<<<<
           ^         ^^ ^                            [4 mismatch]
GT  : R900R054<4CZL6906229F16072996956220612<KKK74
PRED: 99009054<4Q2E8906229F16072996950220612<<<<7 
      ^   ^     ^^^^                 ^       ^^^ ^   [11 mismatch]

### fuse  [decoded] conf1=93% conf2=93%
- error: decode failed: check digit mismatch
GT  : P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CZEWPECIMEN<<NGOH<<<<<<<<<A<<<<<<<<<<<<<<<
                                  ^                  [1 mismatch]
GT  : R900R054<4CZL6906229F16072996956220612<KKK74
PRED: 9900R054<4CZL6906229F16072996956220612<K<K64
      ^                                       ^ ^    [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (75 ms)
GT  : P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CZESPECIMEN<<VZOR<<<<<<<<<<<<<<<<<<<<<<<<<
           ^         ^^ ^                            [4 mismatch]
GT  : R900R054<4CZL6906229F16072996956220612<KKK74
PRED: 99009054<4CZE6906229F16072996956220612<<<<74
      ^   ^       ^                          ^^^     [6 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=99%
- backend: paddle-onnxruntime  (85 ms)
GT  : P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CZESPECIMEN<<VZOR<<<<<<<<<<<<<<<<<<<<<<<<<
           ^         ^^ ^                            [4 mismatch]
GT  : R900R054<4CZL6906229F16072996956220612<KKK74
PRED: 99009054<4CZE6906229F16072996956220612<<<<74
      ^   ^       ^                          ^^^     [6 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=98%
- backend: indra  (60 ms)
GT  : P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CZESPECIMEN<<VZOR<<<<<<<<<<<<<<<<<<<<<<<<<
           ^         ^^ ^                            [4 mismatch]
GT  : R900R054<4CZL6906229F16072996956220612<KKK74
PRED: 99009054<4CZE6906229F16072996956220612<<<<74
      ^   ^       ^                          ^^^     [6 mismatch]

### indra_fullpage  [decoded] conf1=85% conf2=92%
- backend: indra  (223 ms)
- error: lines=34 edits=4 composite_edits=1 country=SSO
GT  : P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: CESSOVNIPASPASSPORTPASEPORTESKREPUBLIKACZECH
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]
GT  : R900R054<4CZL6906229F16072996956220612<KKK74
PRED: ISIOPASUP2SSOORTNOP6SSEFORT1<<<<<<<<<<<<<<06
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^   [43 mismatch]

## 1792f7a2d79c7a8a_0017.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=151
- 官方 line1: P<CHNZHENGJIAN<<YANGBEN<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E900000821CHN8108038F2110189NGKELMPONBPJB978
- 理由: Tesseract 识别并通过校验

### traditional  [decoded] conf1=72% conf2=68%
- error: decode failed: check digit mismatch

### cnn  [decoded] conf1=74% conf2=76%
- error: decode failed: check digit mismatch

### crnn  [decoded] conf1=96% conf2=92%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch

### fuse  [decoded] conf1=78% conf2=80%
- error: decode failed: check digit mismatch

### tesseract  [DECODE-OK]
- backend: tesseract  (67 ms)

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (78 ms)

### indra  [DECODE-OK] conf1=95% conf2=98%
- backend: indra  (51 ms)

### indra_fullpage  [DECODE-OK] conf1=94% conf2=98%
- backend: indra  (291 ms)

## 1c6acd092dee7031_0222.png
- 官方判定: REJECT  verified=False  chosen=None  ms=13
- 理由: 未在照片中定位到 MRZ 双行

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [FAIL]
- backend: indra  (125 ms)
- error: indra 未找到 MRZ 候选行 (0 行)

## 1cce0f3fc387c893_0001.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=143
- 官方 line1: PSPOLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: ZC39917886POL8101028M2909185<<<<<<<<<<<<<<04
- 理由: IndraOCR 识别并通过校验

- GT line1: `P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `5C39917886POL8101028M2909185<<<<<<<<<<<<<<04`

### traditional  [decoded] conf1=71% conf2=71%
- error: decode failed: check digit mismatch
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A4ALALCWAL9NI<<LMAH<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^   ^^   ^^^^                            [13 mismatch]
GT  : 5C39917886POL8101028M2909185<<<<<<<<<<<<<<04
PRED: 2L34447884PCL84CJC28H24C4485<<<<<<<<<<<<<<C<
      ^^ ^^^   ^ ^  ^^^^  ^ ^^^^                ^^   [18 mismatch]

### cnn  [skip]

### crnn  [DECODE-OK] conf1=93% conf2=98%
- backend: crnn  (13 ms)
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSPOLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : 5C39917886POL8101028M2909185<<<<<<<<<<<<<<04
PRED: ZC39917886POL8101028M2909185<<<<<<<<<<<<<<04
      ^                                              [1 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=98%
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 5C39917886POL8101028M2909185<<<<<<<<<<<<<<04
PRED: 2009917886HOL8101028M2909185<<<<<<<<<<<<<<04
      ^^^       ^                                    [4 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (67 ms)
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSPOLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : 5C39917886POL8101028M2909185<<<<<<<<<<<<<<04
PRED: ZC39917886POL8101028M2909185<<<<<<<<<<<<<<04
      ^                                              [1 mismatch]

### paddle  [DECODE-OK] conf1=95% conf2=92%
- backend: paddle-onnxruntime  (82 ms)
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSPOLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : 5C39917886POL8101028M2909185<<<<<<<<<<<<<<04
PRED: ZC39917886POL8101028M2909185<<<<<<<<<04<<<28
      ^                                    ^^   ^^   [5 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=98%
- backend: indra  (53 ms)
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSPOLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : 5C39917886POL8101028M2909185<<<<<<<<<<<<<<04
PRED: ZC39917886POL8101028M2909185<<<<<<<<<<<<<<04
      ^                                              [1 mismatch]

### indra_fullpage  [DECODE-OK] conf1=94% conf2=98%
- backend: indra  (212 ms)
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSPOLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : 5C39917886POL8101028M2909185<<<<<<<<<<<<<<04
PRED: ZC39917886POL8101028M2909185<<<<<<<<<<<<<<04
      ^                                              [1 mismatch]

## 239e042aba0c3ab2_0102.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=166
- 官方 line1: PSCHNYANG<<ZHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: SE00000008CHN8509037F150501819200001<<<<<<92
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `5B00000008CHN8509037F1505018192000015555KK92`

### traditional  [decoded] conf1=72% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P9CHNYAWM<<2HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^     ^^  ^                                   [4 mismatch]
GT  : 5B00000008CHN8509037F1505018192000015555KK92
PRED: 9MOOCOCOO8CHH85C9C37A49C9C484V2MDCM4<<<<<<V2
      ^^^^^^^^^   ^  ^ ^  ^^^^^^^ ^^ ^^^^^^^^^^^^    [33 mismatch]

### cnn  [DECODE-OK] conf1=91% conf2=93%
GT  : P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 5B00000008CHN8509037F1505018192000015555KK92
PRED: 5B00000008CHN8509037F1505018192000015555KK92
                                                     [0 mismatch]

### crnn  [DECODE-OK] conf1=97% conf2=94%
- backend: crnn  (15 ms)
GT  : P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSCHNYANG<<ZHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^         ^                                   [2 mismatch]
GT  : 5B00000008CHN8509037F1505018192000015555KK92
PRED: SE00000008CHN8509037F150501819200001<<<<<<92
      ^^                                  ^^^^^^     [8 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=95%
GT  : P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 5B00000008CHN8509037F1505018192000015555KK92
PRED: 5B00000008CHN8509037F1505018192000015555KK92
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (90 ms)
GT  : P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSCHNYANG<<ZHAO0<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^         ^   ^                               [3 mismatch]
GT  : 5B00000008CHN8509037F1505018192000015555KK92
PRED: SE00000008CHN8509037F150501819200001<<<<<<92
      ^^                                  ^^^^^^     [8 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=98%
- backend: paddle-onnxruntime  (89 ms)
GT  : P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSCHNYANG<<ZHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^         ^                                   [2 mismatch]
GT  : 5B00000008CHN8509037F1505018192000015555KK92
PRED: SE00000008CHN8509037F150501819200001<<<<<<92
      ^^                                  ^^^^^^     [8 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=99%
- backend: indra  (57 ms)
GT  : P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSCHNYANG<<ZHA0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^         ^  ^                                [3 mismatch]
GT  : 5B00000008CHN8509037F1505018192000015555KK92
PRED: SE00000008CHN8509037F150501819200001<<<<<<92
      ^^                                  ^^^^^^     [8 mismatch]

### indra_fullpage  [DECODE-OK] conf1=95% conf2=99%
- backend: indra  (244 ms)
GT  : P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PSCHNYANG<<ZHAD<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^         ^  ^                                [3 mismatch]
GT  : 5B00000008CHN8509037F1505018192000015555KK92
PRED: SE00000008CHN8509037F150501819200001<<<<<<92
      ^^                                  ^^^^^^     [8 mismatch]

## 23f16a7d9b6ab9da_0062.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=178
- 官方 line1: P<URRGRYYSDNCO<<INAN<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 4400000147UKH8307255M19080333052125257S5S500
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `4400000147UKH8307255M19080333052125257S5S500`

### traditional  [decoded] conf1=73% conf2=70%
- error: decode failed: check digit mismatch
GT  : P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
PRED: D<1IWWGRYT5EN<C<<3UAW<<<<<<<<<<<<<<<<<<<<<<<
      ^ ^^^^^^^^^^^^^ ^^^^^                          [19 mismatch]
GT  : 4400000147UKH8307255M19080333052125257S5S500
PRED: E<CCCCC3<7O<803C7255PM9CBCS3SC52425257<<<<CC
      ^^^^^^^^^ ^^^^ ^    ^^ ^^^^ ^^  ^     ^^^^^^   [29 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=96% conf2=87%
- backend: crnn  (12 ms)
- error: decode failed: check digit mismatch
GT  : P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 4400000147UKH8307255M19080333052125257S5S500
PRED: E<000001<70<B8307258M19080333052125257<<<<00
      ^^      ^ ^^^      ^                  ^^^^     [11 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<URRGRYYSDNCO<<INAN<<<<<<<<<<<<<<<<<<<<<<<<
         ^    ^ ^ ^    ^                             [5 mismatch]
GT  : 4400000147UKH8307255M19080333052125257S5S500
PRED: 4400000147UKH8307255M19080333052125257S5S500
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (132 ms)
GT  : P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 4400000147UKH8307255M19080333052125257S5S500
PRED: EKO0O001<7UKR8307255M19080333052125257<<<<00
      ^^^ ^   ^   ^                         ^^^^     [10 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=98%
- backend: paddle-onnxruntime  (79 ms)
GT  : P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 4400000147UKH8307255M19080333052125257S5S500
PRED: EKO0O001<7UKR8307255M19080333052125257<<<<00
      ^^^ ^   ^   ^                         ^^^^     [10 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=97%
- backend: indra  (51 ms)
GT  : P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 4400000147UKH8307255M19080333052125257S5S500
PRED: EK000Q01<7UKR8307255M19080333052125257<<<<00
      ^^   ^  ^   ^                         ^^^^     [9 mismatch]

### indra_fullpage  [DECODE-OK] conf1=90% conf2=99%
- backend: indra  (187 ms)
GT  : P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 4400000147UKH8307255M19080333052125257S5S500
PRED: EK000Q01<7UKR8307255M19080333052125257<<<<00
      ^^   ^  ^   ^                         ^^^^     [9 mismatch]

## 28a42952ba25584a_0078.png
- 官方判定: PASS  verified=True  chosen=paddle  ms=153
- 官方 line1: Q<CHNNGKELMPONBPJLBLO<<<<<<<<<<<<<<<<<<<<SAC
- 官方 line2: Q100547988CHN8410124F190717319204415<<<<<<46
- 理由: IndraOCR 识别并通过校验

- GT line1: `P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `Q100547988CHN8410124F1907173192044155<<5<<46`

### traditional  [decoded] conf1=70% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<
PRED: Q<ELNKGXELNPOWDP3LGLLC<<<<<<<<<<<<<<<<<<<<4A
      ^ ^^ ^ ^     ^^ ^ ^ ^^                    ^^   [13 mismatch]
GT  : Q100547988CHN8410124F1907173192044155<<5<<46
PRED: Q3CCS47V28CHK244C424F3VC74734V2C<<45<<<<<<<K
       ^^^^  ^^   ^^ ^^^   ^^^ ^  ^^ ^^^^ ^  ^  ^^   [25 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=97% conf2=93%
- backend: crnn  (13 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<
PRED: Q<CHNNGKELMPONBPJLBLO<<<<<<<<<<<<<<<<<<<<SAC
      ^         ^                              ^^^   [5 mismatch]
GT  : Q100547988CHN8410124F1907173192044155<<5<<46
PRED: Q100547988CHN8410124F190717319204<15<<<<<<<6
                                       ^  ^  ^  ^    [4 mismatch]

### fuse  [DECODE-OK] conf1=95% conf2=98%
GT  : P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNNGCELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : Q100547988CHN8410124F1907173192044155<<5<<46
PRED: Q100547988CHN8410124F1907173192044155<<5<<46
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (66 ms)
GT  : P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<
PRED: Q<CHNNGKELMPONBPJLBL0O<<<<<<<<<<<<<<<<<<<<SA
      ^         ^         ^^                    ^^   [6 mismatch]
GT  : Q100547988CHN8410124F1907173192044155<<5<<46
PRED: Q100547988CHN8410124F190717319204415<<<<<<46
                                          ^  ^       [2 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (81 ms)
GT  : P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<
PRED: Q<CHNNGKELMPONBPJLBLO<<<<<<<<<<<<<<<<<<<<SAC
      ^         ^                              ^^^   [5 mismatch]
GT  : Q100547988CHN8410124F1907173192044155<<5<<46
PRED: Q100547988CHN8410124F190717319204415<<<<<<46
                                          ^  ^       [2 mismatch]

### indra  [DECODE-OK] conf1=97% conf2=99%
- backend: indra  (52 ms)
GT  : P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<
PRED: Q<CHNNGKELMPONBPJLBLO<<<<<<<<<<<<<<<<<<<<SAC
      ^         ^                              ^^^   [5 mismatch]
GT  : Q100547988CHN8410124F1907173192044155<<5<<46
PRED: Q100547988CHN8410124F190717319204415<<<<<<46
                                          ^  ^       [2 mismatch]

### indra_fullpage  [DECODE-OK] conf1=96% conf2=98%
- backend: indra  (173 ms)
GT  : P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<
PRED: Q<CHNNGKELMPONBPJLBLO<<<<<<<<<<<<<<<<<<<<SAC
      ^         ^                              ^^^   [5 mismatch]
GT  : Q100547988CHN8410124F1907173192044155<<5<<46
PRED: Q100547988CHN8410124F190717319204415<<<<<<46
                                          ^  ^       [2 mismatch]

## 28ccd101a752acef_0016.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=189
- 官方 line1: P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: PO00V000A0PHL8003166F2706269<<<<<<<<<<<<<<02
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02`

### traditional  [decoded] conf1=74% conf2=71%
- error: decode failed: check digit mismatch
GT  : P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
PRED: D<PHLDELA<CHLZ<<NAWJA<<<<<<<<<<<<<<<<<<<<<<<
      ^          ^^   ^ ^^                           [6 mismatch]
GT  : P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
PRED: PCCCCCCCACPHC8CC3466A24C6269<<<<<<<<<<<<<<C2
       ^^^^^^^ ^  ^ ^^ ^  ^ ^^                  ^    [16 mismatch]

### cnn  [decoded] conf1=85% conf2=86%
- error: decode failed: check digit mismatch
GT  : P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<YYUSCGA<S<<<<<SA<IA<<<<<<<<<<<<<<<<<<<<<<<
        ^^^^^^  ^^^^  ^ ^                            [12 mismatch]
GT  : P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
PRED: C000000010C<L8003166F21062692<222<1<<22CC102
      ^       ^ ^^                ^ ^^^ ^  ^^^^^     [14 mismatch]

### crnn  [decoded] conf1=95% conf2=95%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
PRED: P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
                                                     [0 mismatch]

### fuse  [decoded] conf1=90% conf2=89%
- error: decode failed: check digit mismatch
GT  : P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<YYUSGGA<S<<<<<SA<IA<<<<<<<<<<<<<<<<<<<<<<<
        ^^^^^^  ^^^^  ^ ^                            [12 mismatch]
GT  : P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
PRED: C000000010CEL8003166F2106269<<<<<<<<<22CC102
      ^       ^ ^^                         ^^^^^     [9 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (148 ms)
GT  : P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
PRED: O00000D0A9PHL8003166F2706269<<<<<<<<<<<<<<04
      ^     ^  ^            ^                    ^   [5 mismatch]

### paddle  [decoded] conf1=98% conf2=92%
- backend: paddle-onnxruntime  (190 ms)
- error: decode failed: check digit mismatch
GT  : P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
PRED: P0000000A0PHL8003166F2106269<<<<<<<<<<<<02<<
                                              ^^^^   [4 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=99%
- backend: indra  (53 ms)
GT  : P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
PRED: P0000D0OA0PHL8003166F2706269<<<<<<<<<<<<<<02
           ^ ^              ^                        [3 mismatch]

### indra_fullpage  [decoded] conf1=99% conf2=89%
- backend: indra  (204 ms)
- error: lines=39 edits=4 composite_edits=1 country=IEB
GT  : P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<
PRED: REIEBLIKANGPILIPINASREPUBLICOFTHEPHILIPPINES
      ^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [43 mismatch]
GT  : P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02
PRED: PASAPORTE2IEBPASSPO8T<<<<<<0<<<<<<<<<<<<<<08
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^               ^   [28 mismatch]

## 2c793efe12276f9b_0258.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=169
- 官方 line1: P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
- 官方 line2: LV90008223LVA8612010F2501121111111<11410<<70
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<`
- GT line2: `LV90008223LVA8612010F2501121111111<11410<<70`

### traditional  [decoded] conf1=71% conf2=59%
- error: decode failed: check digit mismatch
GT  : P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
PRED: P<LVACZCLXALNA4<LIENE<HAHA<<<<<<<<<<<<<<<<<<
           ^ ^ ^    ^       ^ ^                      [6 mismatch]
GT  : LV90008223LVA8612010F2501121111111<11410<<70
PRED: LV9CCC8223LVA8K42C4CP25C4424444444<4<4LL4<4L
         ^^^        ^^ ^^^^  ^^^ ^^^^^^^ ^^ ^^^ ^^   [26 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=98%
GT  : P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
PRED: P<LNAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
         ^                                           [1 mismatch]
GT  : LV90008223LVA8612010F2501121111111<11410<<70
PRED: LV90008223LVA8612010F2501121111111<114105570
                                              ^^     [2 mismatch]

### crnn  [decoded] conf1=98% conf2=95%
- backend: crnn  (12 ms)
- error: decode failed: bad MRZ length
GT  : P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
PRED: P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : LV90008223LVA8612010F2501121111111<11410<<70
PRED: V90008223LVA8612010F2501121111111<11410<<7G 
      ^^^  ^^ ^^^^^^^^^^^^^^^^ ^^      ^^ ^^^^ ^^^   [32 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
PRED: P<LNAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
         ^                                           [1 mismatch]
GT  : LV90008223LVA8612010F2501121111111<11410<<70
PRED: LV90008223LVA8612010F2501121111111<114105570
                                              ^^     [2 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (64 ms)
GT  : P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
PRED: P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : LV90008223LVA8612010F2501121111111<11410<<70
PRED: LV90008223LVA8612010F2501121111111<11410<<70
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (84 ms)
GT  : P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
PRED: P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : LV90008223LVA8612010F2501121111111<11410<<70
PRED: LV90008223LVA8612010F2501121111111<11410<<70
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=90% conf2=98%
- backend: indra  (51 ms)
GT  : P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
PRED: P<LVAOZOLKALNAK<LIENE<MARA<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : LV90008223LVA8612010F2501121111111<11410<<70
PRED: LV90008223LVA8612010F2501121111111<11410<<70
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=96% conf2=99%
- backend: indra  (231 ms)
GT  : P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
PRED: LACODJALATVIALETTONIE<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^  ^^^^                     [24 mismatch]
GT  : LV90008223LVA8612010F2501121111111<11410<<70
PRED: VALSTSKOD4CODEOFISS3INGSTAT9CODEDUPA<<<<<<16
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^   [42 mismatch]

## 31611582cbf14d7f_0018.jpg
- 官方判定: REJECT  verified=False  chosen=None  ms=13
- 理由: 未在照片中定位到 MRZ 双行

- GT line1: `P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `ZF40004998P0L8101028M3203031<<<<<<<<<<<<<<00`

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [DECODE-OK] conf1=98% conf2=95%
- backend: indra  (187 ms)
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: RZECTPOSPOLITAPOLSKAREPUBLICOFPOLAND<<<<<<<<
      ^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^           [35 mismatch]
GT  : ZF40004998P0L8101028M3203031<<<<<<<<<<<<<<00
PRED: NUMERPASZ6ECTUPASSP0RT<<<<<3<<<<<<<<<<<<<<02
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^               ^   [29 mismatch]

## 3f5bb54276c6b754_0110.png
- 官方判定: PASS  verified=True  chosen=paddle  ms=248
- 官方 line1: PPLIESPECIMEN<<SOPHIA<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: PP00000000LIE9001011F3602025<<<<<<<<<<<<<<08
- 理由: IndraOCR 识别并通过校验

### traditional  [decoded] conf1=75% conf2=68%
- error: decode failed: bad MRZ length

### cnn  [decoded] conf1=92% conf2=84%
- error: decode failed: check digit mismatch

### crnn  [decoded] conf1=95% conf2=98%
- backend: crnn  (13 ms)
- error: decode failed: bad MRZ length

### fuse  [decoded] conf1=96% conf2=86%
- error: decode failed: check digit mismatch

### tesseract  [FAIL]
- backend: tesseract  (176 ms)
- error: no MRZ candidate decoded

### paddle  [decoded] conf1=98% conf2=89%
- backend: paddle-onnxruntime  (168 ms)
- error: decode failed: check digit mismatch

### indra  [DECODE-OK] conf1=99% conf2=98%
- backend: indra  (56 ms)

### indra_fullpage  [DECODE-OK] conf1=98% conf2=98%
- backend: indra  (262 ms)

## 4023a0d69dbd6cf7_0120.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=194
- 官方 line1: P<CHNYANG<<IHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 1C00000003CHN8509037F2611158NB00HFNB<<<<A912
- 理由: 融合CNN+CRNN 识别并通过校验

### traditional  [decoded] conf1=74% conf2=65%
- error: decode failed: check digit mismatch

### cnn  [DECODE-OK] conf1=95% conf2=96%

### crnn  [decoded] conf1=95% conf2=83%
- backend: crnn  (16 ms)
- error: decode failed: check digit mismatch

### fuse  [DECODE-OK] conf1=96% conf2=97%

### tesseract  [DECODE-OK]
- backend: tesseract  (86 ms)

### paddle  [DECODE-OK] conf1=99% conf2=97%
- backend: paddle-onnxruntime  (91 ms)

### indra  [DECODE-OK] conf1=94% conf2=92%
- backend: indra  (53 ms)

### indra_fullpage  [DECODE-OK] conf1=95% conf2=92%
- backend: indra  (270 ms)

## 4086f51e94e9c5c7_0007.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=181
- 官方 line1: P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 1100000000HUS8008046F0902274555<55555<5K5K08
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<`
- GT line2: `1100000000HUS8008046F0902274555<55555<5K5K08`

### traditional  [decoded] conf1=73% conf2=70%
- error: decode failed: check digit mismatch
GT  : P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
PRED: MMMWL99NIWHOVA<<VALENT1HA<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^ ^^^^^^^^^^                      [24 mismatch]
GT  : 1100000000HUS8008046F0902274555<55555<5K5K08
PRED: M44CCCCCCCCAC98CC8CA4ACVC2274<<<<<<<<<<<<<<C
      ^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^ ^^^^^ ^^^^^^   [41 mismatch]

### cnn  [DECODE-OK] conf1=95% conf2=94%
GT  : P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
PRED: P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 1100000000HUS8008046F0902274555<55555<5K5K08
PRED: 1100000000HUS8008046F0902274555<55555<5K5K08
                                                     [0 mismatch]

### crnn  [DECODE-OK] conf1=98% conf2=92%
- backend: crnn  (14 ms)
GT  : P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
PRED: PDRVSSMIRNOVA<<VALENTINA<<<<<<<<<<<<<<<<<<<<
       ^^^  ^    ^   ^    ^                          [7 mismatch]
GT  : 1100000000HUS8008046F0902274555<55555<5K5K08
PRED: 1100000000R088008046F0902274<<<<<<<<<<<<<<08
                ^^^               ^^^ ^^^^^ ^^^^     [15 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=98%
GT  : P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
PRED: P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 1100000000HUS8008046F0902274555<55555<5K5K08
PRED: 1100000000HUS8008046F0902274555<55555<5K5K08
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (84 ms)
GT  : P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
PRED: PDRUSSMIRNOVA<<VALENTINA<<<<<<<<<<<<<<<<<<<<
       ^^   ^    ^   ^    ^                          [6 mismatch]
GT  : 1100000000HUS8008046F0902274555<55555<5K5K08
PRED: 1100000000RUS8008046F0902274<<<<<<<<<<<<<<08
                ^                 ^^^ ^^^^^ ^^^^     [13 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=89%
- backend: paddle-onnxruntime  (94 ms)
GT  : P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
PRED: PDRUSSMIRNOVA<<VALENTINA<<<<<<<<<<<<<<<<<<<<
       ^^   ^    ^   ^    ^                          [6 mismatch]
GT  : 1100000000HUS8008046F0902274555<55555<5K5K08
PRED: 1100000000RUS8008046F0902274<<<<<<<<<<<<0846
                ^                 ^^^ ^^^^^ ^^^^^^   [15 mismatch]

### indra  [DECODE-OK] conf1=95% conf2=98%
- backend: indra  (56 ms)
GT  : P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
PRED: PDRUSSMIRNOVA<<VALENTINA<<<<<<<<<<<<<<<<<<<<
       ^^   ^    ^   ^    ^                          [6 mismatch]
GT  : 1100000000HUS8008046F0902274555<55555<5K5K08
PRED: 1100000000RUS8008046F0902274<<<<<<<<<<<<<<08
                ^                 ^^^ ^^^^^ ^^^^     [13 mismatch]

### indra_fullpage  [DECODE-OK] conf1=92% conf2=98%
- backend: indra  (185 ms)
GT  : P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<
PRED: PDRUSSMIRNOVAK<VALENTINAK<<<<<<<<<<<<<<<<<<<
       ^^   ^    ^ ^ ^    ^   ^                      [8 mismatch]
GT  : 1100000000HUS8008046F0902274555<55555<5K5K08
PRED: 1100000000RUS8008046F0902274<<<<<<<<<<<<<<08
                ^                 ^^^ ^^^^^ ^^^^     [13 mismatch]

## 41ab45d4bde758dc_0306.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=169
- 官方 line1: PCAZEHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: X110003442AZE7503153M230801030LJV5Z<<<<<<<86
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `3110003442AZE7503153M230801030KJU52555<<<<86`

### traditional  [decoded] conf1=70% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
PRED: PCA2EFL5EYKLI4<CR<HAK4<4<4444<<<<<<<<<<<<<<4
       ^ ^^^^^  ^  ^ ^ ^  ^^ ^ ^^^^              ^   [18 mismatch]
GT  : 3110003442AZE7503153M230801030KJU52555<<<<86
PRED: W44CCC3K4242G75C3453A23C8C4C3CL3V52<<<<<<<86
      ^^^^^^ ^  ^^^  ^ ^  ^  ^ ^^^ ^^^^  ^^^         [24 mismatch]

### cnn  [DECODE-OK] conf1=94% conf2=93%
GT  : P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 3110003442AZE7503153M230801030KJU52555<<<<86
PRED: 3110003442AZE7503153M230801030KJU52555<<<<86
                                                     [0 mismatch]

### crnn  [decoded] conf1=95% conf2=88%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
PRED: PCAZEHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^^                                          [3 mismatch]
GT  : 3110003442AZE7503153M230801030KJU52555<<<<86
PRED: X110003442N2E7503153M230801030LJV52<<<<<<<8<
      ^         ^^                  ^ ^  ^^^     ^   [9 mismatch]

### fuse  [DECODE-OK] conf1=97% conf2=95%
GT  : P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 3110003442AZE7503153M230801030KJU52555<<<<86
PRED: 3110003442AZE7503153M230801030KJU52555<<<<86
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (85 ms)
GT  : P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
PRED: PCAZEHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^^                                          [3 mismatch]
GT  : 3110003442AZE7503153M230801030KJU52555<<<<86
PRED: X110003442AZE7503153M230801030LJV5Z<<<<<<<86
      ^                             ^ ^ ^^^^         [7 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=98%
- backend: paddle-onnxruntime  (89 ms)
GT  : P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
PRED: PCAZEHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^^                                          [3 mismatch]
GT  : 3110003442AZE7503153M230801030KJU52555<<<<86
PRED: X110003442AZE7503153M2308010230LJV5Z<<<<<<86
      ^                           ^^^^^^^^^^         [11 mismatch]

### indra  [DECODE-OK] conf1=92% conf2=98%
- backend: indra  (53 ms)
GT  : P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
PRED: PCAZEHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^^                                          [3 mismatch]
GT  : 3110003442AZE7503153M230801030KJU52555<<<<86
PRED: X110003442AZE7503153M230801030LJV5Z<<<<<<<86
      ^                             ^ ^ ^^^^         [7 mismatch]

### indra_fullpage  [DECODE-OK] conf1=97% conf2=97%
- backend: indra  (211 ms)
GT  : P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<
PRED: AZMBAYCANRESPUBLKASIREPUBLICOFAZERBAUJAN<<<<
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       [40 mismatch]
GT  : 3110003442AZE7503153M230801030KJU52555<<<<86
PRED: PASPORTUN3MBASIPASS7ORT<<<<6<<<<<<<<<<<<<<02
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^   [40 mismatch]

## 43ecb49033fb30c1_0073.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=180
- 官方 line1: P<USATRUNZO<<ONDINA<MICHELLE<<<<<<<<<<<<<<<<
- 官方 line2: 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
- 理由: IndraOCR 识别并通过校验

- GT line1: `P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<`
- GT line2: `0418276532USA3309209F9707014<<<<<<<<<<<<<<06`

### traditional  [decoded] conf1=76% conf2=72%
- error: decode failed: check digit mismatch
GT  : P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<
PRED: P<CG41MCH2M<<CHDINA<N1CHELLL<<<<<<<<<<<<<<<<
        ^^^^^^^^^  ^^     ^^^ ^  ^                   [16 mismatch]
GT  : 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
PRED: CA4827A932L9A33C92CDG47M7M4A<<<<<<<<<<<<<<<A
      ^^^   ^^  ^^   ^  ^^^^ ^ ^^^              ^^   [18 mismatch]

### cnn  [decoded] conf1=90% conf2=95%
- error: decode failed: check digit mismatch
GT  : P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<
PRED: P<UIKYAUNZO<<OND<NA<MA<YSCLE<<<<<<<<<<<<<<<<
                      ^      ^ ^                     [3 mismatch]
GT  : 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
PRED: 7478276532USA3309209F9707014<<<<<<<<<<<<<<<1
      ^ ^                                       ^^   [4 mismatch]

### crnn  [decoded] conf1=96% conf2=96%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch
GT  : P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<
PRED: P<USATRUNZO<<ONDINA<MICHELLE<<<<<<<<<<<<<<<<
         ^^^^              ^^ ^                      [7 mismatch]
GT  : 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
PRED: 0418276532USA3309209F9707014<<<<<<<<<<<<<<<6
                                                ^    [1 mismatch]

### fuse  [decoded] conf1=96% conf2=98%
- error: decode failed: check digit mismatch
GT  : P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<
PRED: P<UIKYAUNZO<<OND<NA<MA<HSLLE<<<<<<<<<<<<<<<<
                      ^                              [1 mismatch]
GT  : 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
PRED: 7418276532USA3309209F9707014<<<<<<<<<<<<<<<1
      ^                                         ^^   [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (76 ms)
GT  : P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<
PRED: P<USATRUNZO<<ONDINA<MICHELLE<<<<<<<<<<<<<<<<
         ^^^^              ^^ ^                      [7 mismatch]
GT  : 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
PRED: 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=97%
- backend: paddle-onnxruntime  (82 ms)
GT  : P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<
PRED: P<USATRUNZO<<ONDINA<MICHELLE<<<<<<<<<<<<<<<<
         ^^^^              ^^ ^                      [7 mismatch]
GT  : 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
PRED: 0418276532USA3309209F9707014<<<<<<<<<<<<<682
                                               ^^^   [3 mismatch]

### indra  [DECODE-OK] conf1=99% conf2=100%
- backend: indra  (54 ms)
GT  : P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<
PRED: P<USATRUNZO<<ONDINA<MICHELLE<<<<<<<<<<<<<<<<
         ^^^^              ^^ ^                      [7 mismatch]
GT  : 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
PRED: 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=87% conf2=95%
- backend: indra  (294 ms)
GT  : P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<
PRED: MICHALLEKKKKKKKKKKKKKKKK<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^                   [28 mismatch]
GT  : 0418276532USA3309209F9707014<<<<<<<<<<<<<<06
PRED: 0418276532CHA3309209F9707014<<<<<<<<<<<<<<06
                ^^                                   [2 mismatch]

## 4c5f5e1654002cd0_0050.png
- 官方判定: PASS  verified=True  chosen=tesseract  ms=157
- 官方 line1: P<USAOBAMA<<MICHELLE<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 9102392482USA6401171F1812051900781200<129676
- 理由: PaddleOCR 识别并通过校验

### traditional  [decoded] conf1=70% conf2=62%
- error: decode failed: check digit mismatch

### cnn  [skip]

### crnn  [decoded] conf1=93% conf2=90%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch

### fuse  [DECODE-OK] conf1=96% conf2=93%

### tesseract  [DECODE-OK]
- backend: tesseract  (72 ms)

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (84 ms)

### indra  [DECODE-OK] conf1=86% conf2=98%
- backend: indra  (54 ms)

### indra_fullpage  [DECODE-OK] conf1=91% conf2=96%
- backend: indra  (253 ms)

## 4dbd1b9e24b069f4_0023.png
- 官方判定: PASS  verified=True  chosen=indra  ms=203
- 官方 line1: P<DAAMUSTERMANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: C01X00T478DAA6408125F2702283<<<<<<<<<<<<<<04
- 理由: IndraOCR 识别并通过校验

- GT line1: `P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04`

### traditional  [decoded] conf1=66% conf2=57%
- error: decode failed: check digit mismatch
GT  : P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<
PRED: M<M<<MURTEMMANN<<MMXXA<<<<<<<<<<<<<<<<<<4<<<
      ^ ^^^^ ^^ ^^     ^^^^                   ^      [14 mismatch]
GT  : C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04
PRED: LA<ATAATX7AG<<XXHAT4TJAFL7TAEIAA<<<<<<<<T<<<
      ^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^        ^ ^^   [34 mismatch]

### cnn  [DECODE-OK] conf1=91% conf2=86%
GT  : P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<
PRED: P<DSSNUSYERNANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<
          ^             ^                            [2 mismatch]
GT  : C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04
PRED: C01AM0Y478BA<6408125F27022833554545434544454
         ^^ ^   ^^                ^^^^^^^^^^^^^^^    [20 mismatch]

### crnn  [decoded] conf1=94% conf2=94%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<
PRED: P<D<<MUSTERMANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<
         ^^^  ^  ^      ^                            [6 mismatch]
GT  : C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04
PRED: 001X00T4780<<6408125F2702283<<<<<<<<<<<<<<<4
      ^         ^                               ^    [3 mismatch]

### fuse  [DECODE-OK] conf1=93% conf2=90%
GT  : P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<
PRED: P<DSSNUSYERNANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<
          ^             ^                            [2 mismatch]
GT  : C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04
PRED: K01A00Y478BA<6408125F27022833554545454555454
      ^  ^  ^   ^^                ^^^^^^^^^^^^^^^    [20 mismatch]

### tesseract  [FAIL]
- backend: tesseract  (127 ms)
- error: no MRZ candidate decoded
- PRED line1: <空>
- PRED line2: <空>

### paddle  [DECODE-OK] conf1=97% conf2=98%
- backend: paddle-onnxruntime  (104 ms)
GT  : P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<
PRED: P<D<<MUSTERMANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<
         ^^^  ^  ^      ^                            [6 mismatch]
GT  : C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04
PRED: C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=97%
- backend: indra  (54 ms)
GT  : P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<
PRED: P<DAAMUSTERMANN<<ERIKA<<<<<<<<<<<<<<<<<<<<<<
         ^^^  ^  ^      ^                            [6 mismatch]
GT  : C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04
PRED: C01X00T478DAA6408125F2702283<<<<<<<<<<<<<<04
                 ^^                                  [2 mismatch]

### indra_fullpage  [DECODE-OK] conf1=97% conf2=94%
- backend: indra  (285 ms)
GT  : P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<
PRED: 10SNIERSCHVIFTDERINHABERIN<<<<<<<<<<<<<<<<<<
      ^^^^^^^ ^^^^^^^^^^ ^^^^^^^                     [24 mismatch]
GT  : C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04
PRED: DESINHABE0SNIGNATUR1OFBEAR<3<<<<<<<<<<<<<<06
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^                ^   [28 mismatch]

## 526d2e38b6b977bc_0014.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=150
- 官方 line1: PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
- 官方 line2: 0400210397ROU8001014F34090284800101017005<70
- 理由: PaddleOCR 识别并通过校验

- GT line1: `PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<`
- GT line2: `0400210397ROU8001014F34090284800101017005<70`

### traditional  [decoded] conf1=58% conf2=55%
- error: decode failed: check digit mismatch
GT  : PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
PRED: Y<VTTTVF<<<<<YMFJTN<<VL<4<PTTTF<FAN<<T4<<<<<
      ^^^^^^^^^^^^ ^ ^^^^^^^^^^^^^^^^ ^^^  ^^        [34 mismatch]
GT  : 0400210397ROU8001014F34090284800101017005<70
PRED: <<7<<<3<<<<J6<<<<5<<<<<F<<<<5<<<6<L<<<<<<L<L
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]

### cnn  [decoded] conf1=87% conf2=57%
- error: decode failed: check digit mismatch
GT  : PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
PRED: P<UPDPEWCU<<VDHACLA<GTDFANDA<<<<<<<<<<<<<<<<
       ^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^                 [28 mismatch]
GT  : 0400210397ROU8001014F34090284800101017005<70
PRED: 0D004XX9H0NVH9698040M8597043H80XUPH0040X0070
       ^  ^^^^^^^^^^^^^ ^^^^^^^ ^^^  ^^^^ ^^ ^^^     [33 mismatch]

### crnn  [DECODE-OK] conf1=95% conf2=98%
- backend: crnn  (15 ms)
GT  : PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
PRED: PER0UP0PESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
         ^  ^                                        [2 mismatch]
GT  : 0400210397ROU8001014F34090284800101017005<70
PRED: 0400210397ROU8001014F34090284800101017005<70
                                                     [0 mismatch]

### fuse  [decoded] conf1=89% conf2=61%
- error: decode failed: check digit mismatch
GT  : PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
PRED: P<UPDPEWCU<<MDNADLA<GTDFANDA<<<<<<<<<<<<<<<<
       ^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^                 [28 mismatch]
GT  : 0400210397ROU8001014F34090284800101017005<70
PRED: 00004XX9H0NVH9690040M0597843H80XUPH0040X0070
       ^  ^^^^^^^^^^^^^ ^^^^^^^^^^^  ^^^^ ^^ ^^^     [34 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (62 ms)
GT  : PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
PRED: PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 0400210397ROU8001014F34090284800101017005<70
PRED: 0400210397ROU8001014F34090284800101017005<70
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (83 ms)
GT  : PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
PRED: PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 0400210397ROU8001014F34090284800101017005<70
PRED: 0400210397ROU8001014F34090284800101017005<70
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (33 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=96% conf2=93%
- backend: indra  (270 ms)
GT  : PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<
PRED: ROCNDAROMANIAROUMANIE<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^ ^^^ ^^^^^^^^                 [28 mismatch]
GT  : 0400210397ROU8001014F34090284800101017005<70
PRED: TPTYPATYP0CNDULFRIC3UNTRYCO3ECODEDUPAYSNUM08
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]

## 52f421c92ccf6ba4_0136.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=169
- 官方 line1: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: RC00150586UTO9105045F2509013<99999000T<<<<82
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `KCG01505B6U<O9105045F2509013N999990001<<5<82`

### traditional  [decoded] conf1=75% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: JP<AL5CITI2EN<<3ANE<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^ ^^^^^                            [18 mismatch]
GT  : KCG01505B6U<O9105045F2509013N999990001<<5<82
PRED: MHCCC35C586L1C9MC5CA5P25CVC43<99V99CCC1<<<<8
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^ ^^^^^ ^ ^^   [39 mismatch]

### cnn  [DECODE-OK] conf1=98% conf2=93%
GT  : P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : KCG01505B6U<O9105045F2509013N999990001<<5<82
PRED: KCG01505B6U<O9105045F2509013N999990001<<5<82
                                                     [0 mismatch]

### crnn  [DECODE-OK] conf1=96% conf2=97%
- backend: crnn  (15 ms)
GT  : P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCITI2EN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^    ^       ^                             [3 mismatch]
GT  : KCG01505B6U<O9105045F2509013N999990001<<5<82
PRED: RC00150586UTO9105045F2509013<99999000T<<<<82
      ^ ^     ^  ^                ^        ^  ^      [7 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=92%
GT  : P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : KCG01505B6U<O9105045F2509013N999990001<<5<82
PRED: KCG01505B6UXO9105045F2509013N999990001<<5<82
                 ^                                   [1 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (90 ms)
GT  : P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^            ^                             [2 mismatch]
GT  : KCG01505B6U<O9105045F2509013N999990001<<5<82
PRED: RC00150586UTO9105045F2509013<99999000T<<<<82
      ^ ^     ^  ^                ^        ^  ^      [7 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (77 ms)
GT  : P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^            ^                             [2 mismatch]
GT  : KCG01505B6U<O9105045F2509013N999990001<<5<82
PRED: RC00150586UTO9105045F2509013<99999000T<<<<82
      ^ ^     ^  ^                ^        ^  ^      [7 mismatch]

### indra  [DECODE-OK] conf1=97% conf2=99%
- backend: indra  (53 ms)
GT  : P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<ATSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^^            ^                             [3 mismatch]
GT  : KCG01505B6U<O9105045F2509013N999990001<<5<82
PRED: RC00150586ATS9105045F2509013<99999000T<<<<82
      ^ ^     ^ ^^^               ^        ^  ^      [9 mismatch]

### indra_fullpage  [DECODE-OK] conf1=93% conf2=99%
- backend: indra  (159 ms)
GT  : P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<ATSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^^            ^                             [3 mismatch]
GT  : KCG01505B6U<O9105045F2509013N999990001<<5<82
PRED: RC00150586ATS9105045F2509013<99999000T<<<<82
      ^ ^     ^ ^^^               ^        ^  ^      [9 mismatch]

## 556644d1d8bf4672_0215.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=216
- 官方 line1: P<IWN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 98S0701807KNH7307052F1204089K201781154<<<<54
- 理由: 融合CNN+CRNN 识别并通过校验

### traditional  [decoded] conf1=75% conf2=68%
- error: decode failed: check digit mismatch

### cnn  [skip]

### crnn  [decoded] conf1=97% conf2=94%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch

### fuse  [DECODE-OK] conf1=95% conf2=96%

### tesseract  [decoded]
- backend: tesseract  (158 ms)
- error: no MRZ candidate decoded

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (81 ms)

### indra  [DECODE-OK] conf1=72% conf2=99%
- backend: indra  (47 ms)

### indra_fullpage  [DECODE-OK] conf1=92% conf2=84%
- backend: indra  (333 ms)

## 61d671ec161f22e3_0042.png
- 官方判定: REJECT  verified=False  chosen=None  ms=14
- 理由: 未在照片中定位到 MRZ 双行

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [DECODE-OK] conf1=99% conf2=98%
- backend: indra  (255 ms)

## 6b0e5919dd6f3ac8_0006.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=157
- 官方 line1: P<ESPESPANOLA<ESPANOLA<<CARMEN<<<<<<<<<<<<<<
- 官方 line2: ZAB0002549ESP8001014F2501017A9999999900<<<44
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<`
- GT line2: `ZAB0002549ESP8001014F2501017A9999999900<<<44`

### traditional  [decoded] conf1=69% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
PRED: M<L9ML4AAKCLA<L9MAHMLA<<LAWNLK<<<<<<<<<4<<44
      ^ ^^^^^^ ^^   ^^^ ^^ ^  ^^^ ^^         ^  ^^   [23 mismatch]
GT  : ZAB0002549ESP8001014F2501017A9999999900<<<44
PRED: 2AMCCC2944C9A8CC4C44A29M4M47AVV9VV9VVCC4<<A4
      ^ ^^^^ ^ ^^^^ ^^^^^ ^ ^^^^^  ^^ ^^ ^^^^^  ^    [31 mismatch]

### cnn  [decoded] conf1=91% conf2=83%
- error: decode failed: check digit mismatch
GT  : P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
PRED: P<SSPRSPANOLA<ESPANDEK<<EKNNEN<<<<<<<<<<<<<<
                          ^                          [1 mismatch]
GT  : ZAB0002549ESP8001014F2501017A9999999900<<<44
PRED: YMH0002549KSP8001016F25510178999999990038844
      ^^^       ^        ^   ^    ^          ^^^     [10 mismatch]

### crnn  [decoded] conf1=97% conf2=94%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
PRED: P<ESPESPANOLA<ESPANOLA<<CARMEN<<<<<<<<<<<<<<
        ^  ^             ^ ^  ^^^^                   [8 mismatch]
GT  : ZAB0002549ESP8001014F2501017A9999999900<<<44
PRED: ZAB0002549ESP8001014F2501017A9999999900<<<<4
                                                ^    [1 mismatch]

### fuse  [decoded] conf1=96% conf2=91%
- error: decode failed: check digit mismatch
GT  : P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
PRED: P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : ZAB0002549ESP8001014F2501017A9999999900<<<44
PRED: IMH0002549ESP8001014F2551017K999999990038844
      ^^^                    ^    ^          ^^^     [8 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (69 ms)
GT  : P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
PRED: P<ESPESPANOLA<ESPANOLA<<CARMEN<<<<<<<<<<<<<<
        ^  ^             ^ ^  ^^^^                   [8 mismatch]
GT  : ZAB0002549ESP8001014F2501017A9999999900<<<44
PRED: ZAB0002549ESP8001014F2501017A9999999900<<<44
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (86 ms)
GT  : P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
PRED: P<ESPESPANOLA<ESPANOLA<<CARMEN<<<<<<<<<<<<<<
        ^  ^             ^ ^  ^^^^                   [8 mismatch]
GT  : ZAB0002549ESP8001014F2501017A9999999900<<<44
PRED: ZAB0002549ESP8001014F25010171A999999900<<<44
                                  ^^                 [2 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=97%
- backend: indra  (48 ms)
GT  : P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
PRED: PKESPESPANOLAKESPANOLA<<CARMENK<<<<<<<<<<<<<
       ^^  ^       ^     ^ ^  ^^^^  ^                [11 mismatch]
GT  : ZAB0002549ESP8001014F2501017A9999999900<<<44
PRED: ZAB0002549ESP8001014F2501017A9999999900<<<44
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=95% conf2=98%
- backend: indra  (211 ms)
GT  : P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<
PRED: P<ESPESPANOLA<ESPANOLAK<CARMEN<<<<<<<<<<<<<<
        ^  ^             ^ ^^ ^^^^                   [9 mismatch]
GT  : ZAB0002549ESP8001014F2501017A9999999900<<<44
PRED: ZAB0002549ESP8001014F2501017A9999999900<<<44
                                                     [0 mismatch]

## 6b91117492dc24f3_0022.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=169
- 官方 line1: P<OMNAL<SIYABI<<ISSA<ABDULLAH<SALEH<<<<<<<<<
- 官方 line2: JG20705515OMN7803048M241118530003614<<<<<<50
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<`
- GT line2: `JG20705515OMN7803048M2411185300036145<<<5<50`

### traditional  [decoded] conf1=73% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
PRED: P<CHHAL<91YAM1<<139A<AMMLLLAH<9ALLH<<<<<<<<<
        ^ ^   ^^  ^^  ^^^   ^^^     ^ ^^             [15 mismatch]
GT  : JG20705515OMN7803048M2411185300036145<<<5<50
PRED: 3G2C7C553SDHH78C3C48H24444853CCC364<<<<<<<4C
      ^  ^ ^  ^^^^^  ^ ^  ^  ^^^   ^^^  ^^^   ^ ^^   [23 mismatch]

### cnn  [DECODE-OK] conf1=95% conf2=94%
GT  : P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
PRED: P<UHNAL<SIYARI<<ISSA<ARBUELAH<SBEEH<<<<<<<<<
                               ^     ^               [2 mismatch]
GT  : JG20705515OMN7803048M2411185300036145<<<5<50
PRED: JG20705515OMN7803048M2411185300036145<<<5<50
                                                     [0 mismatch]

### crnn  [decoded] conf1=98% conf2=96%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
PRED: P<OMNAL<SIYABI<<ISSA<ABDULLAH<SALEH<<<<<<<<<
        ^^        ^         ^^        ^              [6 mismatch]
GT  : JG20705515OMN7803048M2411185300036145<<<5<50
PRED: J020705515OMN7803048M241118530003614<<<<<<50
       ^                                  ^   ^      [3 mismatch]

### fuse  [DECODE-OK] conf1=98% conf2=97%
GT  : P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
PRED: P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
                                                     [0 mismatch]
GT  : JG20705515OMN7803048M2411185300036145<<<5<50
PRED: JG20705515OMN7803048M2411185300036145<<<5<50
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (79 ms)
GT  : P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
PRED: P<OMNAL<SIYABI<<ISSA<ABDULLAH<SALEH<<<<<<<<<
        ^^        ^         ^^        ^              [6 mismatch]
GT  : JG20705515OMN7803048M2411185300036145<<<5<50
PRED: JG20705515OMN7803048M241118530003614<<<<<<50
                                          ^   ^      [2 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=97%
- backend: paddle-onnxruntime  (87 ms)
GT  : P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
PRED: P<OMNAL<SIYABI<<ISSA<ABDULLAH<SALEH<<<<<<<<<
        ^^        ^         ^^        ^              [6 mismatch]
GT  : JG20705515OMN7803048M2411185300036145<<<5<50
PRED: JG20705515OMN7803048M241118530003614<<<<<<50
                                          ^   ^      [2 mismatch]

### indra  [DECODE-OK] conf1=95% conf2=99%
- backend: indra  (51 ms)
GT  : P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
PRED: P<OMNAL<SIYABIK<ISSAKABDULLAH<SALEH<<<<<<<<<
        ^^        ^ ^     ^ ^^        ^              [8 mismatch]
GT  : JG20705515OMN7803048M2411185300036145<<<5<50
PRED: JG20705515OMN7803048M241118530003614<<<<<<50
                                          ^   ^      [2 mismatch]

### indra_fullpage  [DECODE-OK] conf1=97% conf2=100%
- backend: indra  (198 ms)
GT  : P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<
PRED: P<OMNAL<SIYABI<<ISSAKABDULLAH<SALEH<<<<<<<<<
        ^^        ^       ^ ^^        ^              [7 mismatch]
GT  : JG20705515OMN7803048M2411185300036145<<<5<50
PRED: JG20705515OMN7803048M241118530003614<<<<<<50
                                          ^   ^      [2 mismatch]

## 6dafa3b35c792d06_0039.png
- 官方判定: PASS  verified=True  chosen=indra  ms=312
- 官方 line1: P<MMRABDELLAHI<MAHFOUDH<<MOHAMMED<<<<<<<<<<<
- 官方 line2: ABI2345678MMRT7511217M15111331234567B9<<<<14
- 理由: IndraOCR 识别并通过校验

### traditional  [decoded] conf1=67% conf2=67%
- error: decode failed: check digit mismatch

### cnn  [decoded] conf1=96% conf2=88%
- error: decode failed: check digit mismatch

### crnn  [decoded] conf1=98% conf2=94%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch

### fuse  [decoded] conf1=97% conf2=89%
- error: decode failed: check digit mismatch

### tesseract  [decoded]
- backend: tesseract  (146 ms)
- error: no MRZ candidate decoded

### paddle  [decoded] conf1=95% conf2=99%
- backend: paddle-onnxruntime  (200 ms)
- error: decode failed: check digit mismatch

### indra  [DECODE-OK] conf1=98% conf2=100%
- backend: indra  (52 ms)

### indra_fullpage  [DECODE-OK] conf1=97% conf2=100%
- backend: indra  (241 ms)

## 7294ad26ea8d95e0_0010.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=181
- 官方 line1: P<BRAFARIAS<DOS<SANTOS<<RODRIGO<<<<<<<<<<<<<
- 官方 line2: AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
- 理由: IndraOCR 识别并通过校验

- GT line1: `P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<`
- GT line2: `AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00`

### traditional  [decoded] conf1=72% conf2=70%
- error: decode failed: check digit mismatch
GT  : P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
PRED: P<BHAFARIA9<DCG<GAHTC9<<RCDRIGC<<<<<<<<<<<<<
        ^^      ^  ^^ ^ ^^^^  ^^    ^                [13 mismatch]
GT  : AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
PRED: AACCC264<7AAACAC34A2H29C7C53<<<<<<<<<<<<<<CC
        ^^^  ^  ^^ ^^^ ^^ ^ ^^ ^                ^^   [17 mismatch]

### cnn  [decoded] conf1=95% conf2=93%
- error: decode failed: check digit mismatch
GT  : P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
PRED: P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
PRED: AA00026157BHA0403162M25070535555555<55555500
              ^  ^                ^^^^^^^ ^^^^^^     [15 mismatch]

### crnn  [DECODE-OK] conf1=98% conf2=99%
- backend: crnn  (18 ms)
GT  : P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
PRED: P<BRAFARIAS<DOS<SANTOS<<RODRIGO<<<<<<<<<<<<<
        ^^               ^    ^                      [4 mismatch]
GT  : AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
PRED: AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
                                                     [0 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=94%
GT  : P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
PRED: P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
PRED: AA000261<7BHA0403162M2507053<5<<5<<<<<<<<<00
                 ^                 ^  ^              [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (75 ms)
GT  : P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
PRED: P<BRAFARIAS<DOS<SANTOS<<RODRIGO<<<<<<<<<<<<<
        ^^               ^    ^                      [4 mismatch]
GT  : AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
PRED: AA000261<7BRAU403162M2507053<<<<<<<<<<<<<<00
                   ^                                 [1 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (96 ms)
GT  : P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
PRED: P<BRAFARIAS<DOS<SANTOS<<RODRIGO<<<<<<<<<<<<<
        ^^               ^    ^                      [4 mismatch]
GT  : AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
PRED: AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=98% conf2=100%
- backend: indra  (50 ms)
GT  : P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
PRED: P<BRAFARIAS<DOS<SANTOS<<RODRIGO<<<<<<<<<<<<<
        ^^               ^    ^                      [4 mismatch]
GT  : AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
PRED: AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=98% conf2=99%
- backend: indra  (197 ms)
GT  : P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<
PRED: P<BRAFARIAS<DOS<SANTOS<<RODRIGO<<<<<<<<<<<<<
        ^^               ^    ^                      [4 mismatch]
GT  : AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
PRED: AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00
                                                     [0 mismatch]

## 72ef4ce48bf27731_0002.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=139
- 官方 line1: P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<
- 官方 line2: LV99031769LVA6411183F3311237327707<98976<<58
- 理由: PaddleOCR 识别并通过校验

### traditional  [FAIL]
- error: OCR 未输出完整两行

### cnn  [decoded] conf1=63% conf2=74%
- error: decode failed: check digit mismatch

### crnn  [decoded] conf1=90% conf2=87%
- backend: crnn  (12 ms)
- error: decode failed: check digit mismatch

### fuse  [decoded] conf1=67% conf2=84%
- error: decode failed: check digit mismatch

### tesseract  [DECODE-OK]
- backend: tesseract  (93 ms)

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (67 ms)

### indra  [DECODE-OK] conf1=92% conf2=99%
- backend: indra  (54 ms)

### indra_fullpage  [decoded] conf1=87% conf2=98%
- backend: indra  (271 ms)
- error: lines=55 edits=0 composite_edits=0 country=LVA

## 7626df394c39f314_0020.png
- 官方判定: PASS  verified=True  chosen=fuse  ms=182
- 官方 line1: P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUEIF<<<<<<<
- 官方 line2: B0000000<7BHM9001284M3212161000000000<<K<K04
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<`
- GT line2: `B0000000<7BHM9001284M3212161000000000<<K<K04`

### traditional  [decoded] conf1=68% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
PRED: PBBHWEBMAH1N<<NOHANNED<ABDLLLA<YCLG1F<<<<<<<
       ^  ^  ^  ^^  ^   ^^      ^     ^^^^           [13 mismatch]
GT  : B0000000<7BHM9001284M3212161000000000<<K<K04
PRED: ACCCCCCC<7AHA9CC428AH32424K4CCCCCCCCC<<<<<CA
      ^^^^^^^^  ^ ^ ^^^  ^^  ^ ^^^^^^^^^^^^  ^ ^^^   [32 mismatch]

### cnn  [DECODE-OK] conf1=93% conf2=94%
GT  : P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
PRED: P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUEEF<<<<<<<
                                        ^^           [2 mismatch]
GT  : B0000000<7BHM9001284M3212161000000000<<K<K04
PRED: B0000000<7BHM9001284M3212161000000000<<K<K04
                                                     [0 mismatch]

### crnn  [decoded] conf1=97% conf2=95%
- backend: crnn  (17 ms)
- error: decode failed: check digit mismatch
GT  : P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
PRED: PBBHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
       ^                                             [1 mismatch]
GT  : B0000000<7BHM9001284M3212161000000000<<K<K04
PRED: P0000000<7BHR9001284M3212161000000000<<<<<04
      ^           ^                          ^ ^     [4 mismatch]

### fuse  [DECODE-OK] conf1=94% conf2=98%
GT  : P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
PRED: P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUEIF<<<<<<<
                                        ^            [1 mismatch]
GT  : B0000000<7BHM9001284M3212161000000000<<K<K04
PRED: B0000000<7BHM9001284M3212161000000000<<K<K04
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (92 ms)
GT  : P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
PRED: PBBHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
       ^                                             [1 mismatch]
GT  : B0000000<7BHM9001284M3212161000000000<<K<K04
PRED: B0000000<7BHR9001284M3212161000000000<<<<<04
                  ^                          ^ ^     [3 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=91%
- backend: paddle-onnxruntime  (95 ms)
GT  : P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
PRED: PBBHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
       ^                                             [1 mismatch]
GT  : B0000000<7BHM9001284M3212161000000000<<K<K04
PRED: B0000000<7BHR9001284M321216100000000<<<<0428
                  ^                       ^  ^^^^^   [7 mismatch]

### indra  [DECODE-OK] conf1=97% conf2=99%
- backend: indra  (56 ms)
GT  : P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
PRED: PBBHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
       ^                                             [1 mismatch]
GT  : B0000000<7BHM9001284M3212161000000000<<K<K04
PRED: B0000000<7BHR9001284M3212161000000000<<<<<04
                  ^                          ^ ^     [3 mismatch]

### indra_fullpage  [DECODE-OK] conf1=98% conf2=100%
- backend: indra  (209 ms)
GT  : P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<
PRED: PBBHREBRAHIM<<MOHAMMED<ABDULLAKYOUSIF<<<<<<<
       ^                            ^                [2 mismatch]
GT  : B0000000<7BHM9001284M3212161000000000<<K<K04
PRED: B0000000<7BHR9001284M3212161000000000<<<<<04
                  ^                          ^ ^     [3 mismatch]

## 78f45152a658f728_0012.jpg
- 官方判定: PASS  verified=True  chosen=indra  ms=291
- 官方 line1: P<IDNLENGKAP<<NAMA<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: X00000O<<9IDN4508179S1601264<<<<<<<<<<<<<<00
- 理由: IndraOCR 识别并通过校验

### traditional  [decoded] conf1=74% conf2=70%
- error: decode failed: check digit mismatch

### cnn  [decoded] conf1=98% conf2=90%
- error: decode failed: check digit mismatch

### crnn  [decoded] conf1=98% conf2=83%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch

### fuse  [decoded] conf1=99% conf2=91%
- error: decode failed: check digit mismatch

### tesseract  [decoded]
- backend: tesseract  (122 ms)
- error: no MRZ candidate decoded

### paddle  [decoded] conf1=99% conf2=90%
- backend: paddle-onnxruntime  (187 ms)
- error: decode failed: check digit mismatch

### indra  [DECODE-OK] conf1=97% conf2=97%
- backend: indra  (56 ms)

### indra_fullpage  [DECODE-OK] conf1=86% conf2=74%
- backend: indra  (278 ms)

## 7c5899e1f86a07a0_0008.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=185
- 官方 line1: P<LTUBASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
- 官方 line2: 00000000<0LTU9003118F290102049003111045<<<86
- 理由: IndraOCR 识别并通过校验

- GT line1: `P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<`
- GT line2: `00000000<0LTU9003118F290102049003111045<<<86`

### traditional  [decoded] conf1=77% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
PRED: P<LILHA5ANAVICIENE<<B1WLIL<<<<<<<<<<<<<<<<<<
         ^^^ ^             ^^^^^                     [9 mismatch]
GT  : 00000000<0LTU9003118F290102049003111045<<<86
PRED: CCCCCCCC<CL1C9CC3448L29C4C2CA9LC1444LA5<<<8K
      ^^^^^^^^ ^ ^^ ^^ ^^ ^  ^^^ ^^ ^^^^^^^^     ^   [30 mismatch]

### cnn  [decoded] conf1=92% conf2=93%
- error: decode failed: check digit mismatch
GT  : P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
PRED: P<LTURASANAVIGIRHE<<BIRUTK<<<<<<<<<<<<<<<<<<
                   ^ ^^        ^                     [4 mismatch]
GT  : 00000000<0LTU9003118F290102049003111045<<<86
PRED: 0000000040ATU9003118F290102049003111045<5<86
              ^ ^                             ^      [3 mismatch]

### crnn  [decoded] conf1=97% conf2=91%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
PRED: P<LTUBASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : 00000000<0LTU9003118F290102049003111045<<<86
PRED: 00000000<0LTU9003118F290102049003111045<<<06
                                                ^    [1 mismatch]

### fuse  [decoded] conf1=96% conf2=93%
- error: decode failed: check digit mismatch
GT  : P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
PRED: P<LTURASANAVIGIRNE<<BIRUTE<<<<<<<<<<<<<<<<<<
                   ^ ^                               [2 mismatch]
GT  : 00000000<0LTU9003118F290102049003111045<<<86
PRED: 0000000040ATU9003118F290102049003111045<<<86
              ^ ^                                    [2 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (80 ms)
GT  : P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
PRED: P<LTUBASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : 00000000<0LTU9003118F290102049003111045<<<86
PRED: 00000000<0LTU9003118F290102049003111045<<<86
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=98%
- backend: paddle-onnxruntime  (86 ms)
GT  : P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
PRED: P<LTUBASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : 00000000<0LTU9003118F290102049003111045<<<86
PRED: 00000000<0LTU9003118F290102049003111045<<<86
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=98% conf2=98%
- backend: indra  (56 ms)
GT  : P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
PRED: P<LTUBASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : 00000000<0LTU9003118F290102049003111045<<<86
PRED: 00000000<0LTU9003118F290102049003111045<<<86
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=99% conf2=98%
- backend: indra  (209 ms)
GT  : P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
PRED: P<LTUBASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : 00000000<0LTU9003118F290102049003111045<<<86
PRED: 00000000<0LTU9003118F290102049003111045<<<86
                                                     [0 mismatch]

## 7c6f47bad47d70e9_0230.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=188
- 官方 line1: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: RA01234564AUS9105045F3208014<99999000T<<<<86
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `RA01234564AUS9105045F3208014<99999000T<<<<86`

### traditional  [decoded] conf1=75% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: F<AL9C1117GH<<3AKE<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^  ^^ ^^^^^^  ^ ^                              [11 mismatch]
GT  : RA01234564AUS9105045F3208014<99999000T<<<<86
PRED: HAC42345A4AC8V4C5C45L32C8C44<VV4VVCCC1<<<<8K
      ^ ^^    ^  ^^^^^ ^  ^  ^ ^^  ^^^^^^^^^     ^   [24 mismatch]

### cnn  [DECODE-OK] conf1=98% conf2=85%
GT  : P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCIYIZKN<<IANE<<<<<<<<<<<<<<<<<<<<<<<<<<
                ^   ^                                [2 mismatch]
GT  : RA01234564AUS9105045F3208014<99999000T<<<<86
PRED: M801234M64TBA9105045F32080144HNHNH0001444A86
      ^^     ^  ^^^               ^^^^^^   ^^^^^     [17 mismatch]

### crnn  [decoded] conf1=94% conf2=85%
- backend: crnn  (16 ms)
- error: decode failed: check digit mismatch
GT  : P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : RA01234564AUS9105045F3208014<99999000T<<<<86
PRED: PA01294866AU69105045F3208014<99999000T<<<<06
      ^    ^ ^ ^  ^                             ^    [6 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=92%
GT  : P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCIYIZEN<<IANE<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : RA01234564AUS9105045F3208014<99999000T<<<<86
PRED: M801234M64TBS9105045F32080144999N90001444486
      ^^     ^  ^^                ^   ^    ^^^^^     [12 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (77 ms)
GT  : P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : RA01234564AUS9105045F3208014<99999000T<<<<86
PRED: RA01234564AUS9105045F3208014<99999000T<<<<86
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=98%
- backend: paddle-onnxruntime  (95 ms)
GT  : P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : RA01234564AUS9105045F3208014<99999000T<<<<86
PRED: RA01234564AUS9105045F3208014<99999000T<<<<86
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=98% conf2=98%
- backend: indra  (57 ms)
GT  : P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : RA01234564AUS9105045F3208014<99999000T<<<<86
PRED: RA01234564AUS9105045F3208014<99999000T<<<<86
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=96% conf2=98%
- backend: indra  (176 ms)
GT  : P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUSCITIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : RA01234564AUS9105045F3208014<99999000T<<<<86
PRED: RA01234564AUS9105045F3208014<99999000T<<<<86
                                                     [0 mismatch]

## 86c6d990773daff2_0002.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=159
- 官方 line1: P<MLTMICALLEF<<GIOVANNI<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06`

### traditional  [decoded] conf1=72% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
PRED: P<NLJN1CALLEF<<G1OVANK1<<<<<<<<<<<<<<<<<<<<<
        ^ ^^^^   ^    ^ ^  ^^                        [10 mismatch]
GT  : 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
PRED: CV84A8V<<5NL1A844448W2OCA2AH<<<<<<<<<<<<<<CA
      ^^ ^^ ^   ^ ^^ ^^^^ ^ ^^^ ^^              ^^   [20 mismatch]

### cnn  [decoded] conf1=93% conf2=86%
- error: decode failed: check digit mismatch
GT  : P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
PRED: P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
PRED: 09UM489NN5MKM6811118M200626800K<4NNNNNNNNN06
        ^^   ^^  ^^               ^^^ ^^^^^^^^^^     [19 mismatch]

### crnn  [DECODE-OK] conf1=95% conf2=95%
- backend: crnn  (15 ms)
GT  : P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
PRED: P<MLTMICALLEF<<GIOVANNI<<<<<<<<<<<<<<<<<<<<<
             ^   ^      ^   ^                        [4 mismatch]
GT  : 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
PRED: 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### fuse  [DECODE-OK] conf1=97% conf2=94%
GT  : P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
PRED: P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
PRED: 09UM48QNN5MKM6811118M2006268KK<<<NGNNNNNNN06
        ^^  ^^^  ^^               ^^   ^^^^^^^^^     [18 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (78 ms)
GT  : P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
PRED: P<MLTMICALLEF<<GIOVANNI<<<<<<<<<<<<<<<<<<<<<
             ^   ^      ^   ^                        [4 mismatch]
GT  : 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
PRED: 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=96%
- backend: paddle-onnxruntime  (90 ms)
GT  : P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
PRED: P<MLTMICALLEF<<GIOVANNI<<<<<<<<<<<<<<<<<<<<<
             ^   ^      ^   ^                        [4 mismatch]
GT  : 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
PRED: 0981489<<5MLT6811118M20062680<<<<<<<<<<<<<06
                                  ^                  [1 mismatch]

### indra  [DECODE-OK] conf1=93% conf2=98%
- backend: indra  (49 ms)
GT  : P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
PRED: P<MLTMICALLEF<<GIOVANNI<<<<<<<<<<<<<<<<<<<<<
             ^   ^      ^   ^                        [4 mismatch]
GT  : 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
PRED: 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=98% conf2=97%
- backend: indra  (241 ms)
GT  : P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<
PRED: REASBBLIKATAMALTAREPUBLICOFMALTA<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^               [32 mismatch]
GT  : 0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06
PRED: PASSAPORT2ASBPORTPA2SEPO<<<7<<<<<<<<<<<<<<00
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^               ^   [29 mismatch]

## 8a5e3bc18a6ff7d2_0043.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=187
- 官方 line1: P<NLDDE<BRUIJN<<WILLEKE<LISELOTTE<<<<<<<<<<<
- 官方 line2: SPECI20142NLD6503101F2403096999999990<<<<<84
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<`
- GT line2: `18ECI20142NLS6503101F2403096999999990<<<<<84`

### traditional  [decoded] conf1=72% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
PRED: P<NLDDE<BML13H<<W1LLE<E<L19ELCTTE<<<<<<<<<<<
               ^^^^^   ^   ^   ^^  ^^^               [12 mismatch]
GT  : 18ECI20142NLS6503101F2403096999999990<<<<<84
PRED: GPCC12C4A2HLDA5C34C4P24C3C9K99999999C<<<<<H<
      ^^^ ^ ^^^ ^ ^^ ^ ^^^^  ^ ^ ^        ^     ^^   [21 mismatch]

### cnn  [DECODE-OK] conf1=97% conf2=94%
GT  : P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
PRED: P<HLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
        ^                                            [1 mismatch]
GT  : 18ECI20142NLS6503101F2403096999999990<<<<<84
PRED: 18ECI20142NLS6503101F2403096999999990<<<<<84
                                                     [0 mismatch]

### crnn  [decoded] conf1=97% conf2=86%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
PRED: P<NLDDE<BRUIJN<<WILLEKE<LISELOTTE<<<<<<<<<<<
                                    ^^               [2 mismatch]
GT  : 18ECI20142NLS6503101F2403096999999990<<<<<84
PRED: GPECI201<2NLB<50310112<03096999999990<<<<<8<
      ^^      ^   ^^      ^ ^                    ^   [8 mismatch]

### fuse  [DECODE-OK] conf1=98% conf2=96%
GT  : P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
PRED: P<HLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
        ^                                            [1 mismatch]
GT  : 18ECI20142NLS6503101F2403096999999990<<<<<84
PRED: 18ECI20142NLS6503101F2403096999999990<<<<<84
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (84 ms)
GT  : P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
PRED: P<NLDDE<BRUIJN<<WILLEKE<LISELOTTE<<<<<<<<<<<
                                    ^^               [2 mismatch]
GT  : 18ECI20142NLS6503101F2403096999999990<<<<<84
PRED: SPECI20142NLD6503101F2403096999999990<<<<<84
      ^^          ^                                  [3 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (88 ms)
GT  : P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
PRED: P<NLDDE<BRUIJN<<WILLEKE<LISELOTTE<<<<<<<<<<<
                                    ^^               [2 mismatch]
GT  : 18ECI20142NLS6503101F2403096999999990<<<<<84
PRED: SPECI20142NLD6503101F2403096999999990<<<<<84
      ^^          ^                                  [3 mismatch]

### indra  [DECODE-OK] conf1=98% conf2=99%
- backend: indra  (53 ms)
GT  : P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
PRED: P<NLDDE<BRUIJN<<WILLEKE<LISELOTTE<<<<<<<<<<<
                                    ^^               [2 mismatch]
GT  : 18ECI20142NLS6503101F2403096999999990<<<<<84
PRED: SPECI20142NLD6503101F2403096999999990<<<<<84
      ^^          ^                                  [3 mismatch]

### indra_fullpage  [DECODE-OK] conf1=56% conf2=96%
- backend: indra  (455 ms)
GT  : P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<
PRED: 42LCO17850112210144418482<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^              [33 mismatch]
GT  : 18ECI20142NLS6503101F2403096999999990<<<<<84
PRED: VALIDTORA1LCOUNTRIE5VALABLE7OURTOUSTESPA<<86
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^   [41 mismatch]

## 8de1f04907363a2d_0000.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=167
- 官方 line1: P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
- 官方 line2: 82040011<0ISL1212123F1106297121212<1239<<<66
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<`
- GT line2: `82040011<0ISL1212123F1106297121212<1239<<<66`

### traditional  [decoded] conf1=68% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
PRED: AA14LAL6AAAAM1Y1A<<1ALA1NLA<LAAA<Z<<<<<<Z<<<
      ^^^^^ ^^ ^^^^^ ^^  ^^^^^^^^ ^^^^ ^      ^      [28 mismatch]
GT  : 82040011<0ISL1212123F1106297121212<1239<<<66
PRED: A2C4C543<C15L3Z4242JF44CK2973Z4242<3P39<<<ZF
      ^ ^ ^^^^ ^^^ ^^^ ^ ^ ^^^^   ^^^ ^  ^^     ^^   [26 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=98% conf2=91%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
PRED: PAISLAEVARSDOTTIR<<THURIDUR<OESP<<<<<<<<<<<<
       ^ ^^  ^ ^  ^^^ ^  ^^ ^ ^ ^ ^                  [15 mismatch]
GT  : 82040011<0ISL1212123F1106297121212<1239<<<66
PRED: 2040611<0I6L1212123F1106297121212<1239<<<06 
      ^^^^^^ ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^  ^ ^   [39 mismatch]

### fuse  [DECODE-OK] conf1=98% conf2=97%
GT  : P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
PRED: P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 82040011<0ISL1212123F1106297121212<1239<<<66
PRED: 82040011<0ISL1212123F1106297121212<1239<<<66
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (79 ms)
GT  : P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
PRED: PAISLAEVARSDOTTIR<<THURIDUR<OESP<<<<<<<<<<<<
       ^ ^^  ^ ^  ^^^ ^  ^^ ^ ^ ^ ^                  [15 mismatch]
GT  : 82040011<0ISL1212123F1106297121212<1239<<<66
PRED: A2040611<0ISL1212123F1106297121212<1239<<<66
      ^    ^                                         [2 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (87 ms)
GT  : P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
PRED: PAISLAEVARSDOTTIR<<THURIDUR<OESP<<<<<<<<<<<<
       ^ ^^  ^ ^  ^^^ ^  ^^ ^ ^ ^ ^                  [15 mismatch]
GT  : 82040011<0ISL1212123F1106297121212<1239<<<66
PRED: A2040611<0ISL1212123F1106297121212<1239<<<66
      ^    ^                                         [2 mismatch]

### indra  [DECODE-OK] conf1=98% conf2=98%
- backend: indra  (53 ms)
GT  : P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
PRED: PAISLAEVARSDOTTIR<<THURIDUR<OESP<<<<<<<<<<<<
       ^ ^^  ^ ^  ^^^ ^  ^^ ^ ^ ^ ^                  [15 mismatch]
GT  : 82040011<0ISL1212123F1106297121212<1239<<<66
PRED: A2040611<0ISL1212123F1106297121212<1239<<<66
      ^    ^                                         [2 mismatch]

### indra_fullpage  [DECODE-OK] conf1=98% conf2=99%
- backend: indra  (174 ms)
GT  : P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<
PRED: PAISLAEVARSDOTTIR<<THURIDUR<OESP<<<<<<<<<<<<
       ^ ^^  ^ ^  ^^^ ^  ^^ ^ ^ ^ ^                  [15 mismatch]
GT  : 82040011<0ISL1212123F1106297121212<1239<<<66
PRED: A2040611<0ISL1212123F1106297121212<1239<<<66
      ^    ^                                         [2 mismatch]

## 90b464ea218a9909_0001.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=179
- 官方 line1: P<SYCSAMPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00`

### traditional  [decoded] conf1=75% conf2=71%
- error: decode failed: check digit mismatch
GT  : P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
PRED: M<9YL9ANDLE<<CWDIHAWY<<<<<<<<<<<<<<<<<<<<<<<
      ^ ^ ^^  ^    ^^  ^ ^                           [9 mismatch]
GT  : NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
PRED: A9CCCUCA<78YC8KC3337P32C34CM<<<<<<<<<<<<<<<C
      ^^^^^^^^  ^   ^^^^  ^  ^ ^^^              ^^   [20 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=95%
GT  : P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYCSANPLE<<ORDIHARY<<<<<<<<<<<<<<<<<<<<<<<
                       ^                             [1 mismatch]
GT  : NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
PRED: N8000004<7SYC8601137F320310155555555<<5<5550
       ^                          ^^^^^^^^  ^ ^^^    [13 mismatch]

### crnn  [decoded] conf1=97% conf2=93%
- backend: crnn  (15 ms)
- error: decode failed: bad MRZ length
GT  : P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYCSAMPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
PRED: 8000004<76YC8601137F3203101<<<<<<<<<<<<<<<0 
      ^^    ^^^^^^^^^^ ^^^^^^^^^^^               ^   [24 mismatch]

### fuse  [DECODE-OK] conf1=98% conf2=97%
GT  : P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
PRED: N8000004<7SYC8601137F320310155<55555<<<<<<50
       ^                          ^^ ^^^^^      ^    [9 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (79 ms)
GT  : P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYCSAMPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
PRED: NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=95%
- backend: paddle-onnxruntime  (82 ms)
GT  : P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYCSAMPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
PRED: NS000004<7SYC8601137F3203101<<<<<<<<<<<<0<00
                                              ^      [1 mismatch]

### indra  [DECODE-OK] conf1=95% conf2=98%
- backend: indra  (53 ms)
GT  : P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYCSAMPLEK<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
             ^   ^                                   [2 mismatch]
GT  : NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
PRED: NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=94% conf2=98%
- backend: indra  (184 ms)
GT  : P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYCSAMPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<
             ^                                       [1 mismatch]
GT  : NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
PRED: NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00
                                                     [0 mismatch]

## 927cd546b5a8111d_0108.jpg
- 官方判定: SUSPECT  verified=False  chosen=None  ms=1479
- 官方 line1: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 理由: PaddleOCR 识别到 MRZ，校验未通过

- GT line1: `P<CHNLOI<<HANG<MIO<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `MB00000007CHN8807191F291203910004067<<<<<<28`

### traditional  [decoded] conf1=73% conf2=49%
- error: decode failed: bad MRZ length
GT  : P<CHNLOI<<HANG<MIO<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<<<<<<<<<<<<<<<<<<<<<<<<<<<IY              
      ^ ^^^^^^  ^^^^ ^^^          ^^^^^^^^^^^^^^^^   [30 mismatch]
GT  : MB00000007CHN8807191F291203910004067<<<<<<28
PRED: <<<<<<<<<<<<<<<<<I<J<<<<<<<<A<<<<<LLILJ<<<IL
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^   [41 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=64% conf2=0%
- backend: crnn  (10 ms)
- error: OCR 未输出完整两行
GT  : P<CHNLOI<<HANG<MIO<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: 0003<<<<<<<<<<<<<<<<<<<<<<                  
      ^^^^^^^^  ^^^^ ^^^        ^^^^^^^^^^^^^^^^^^   [33 mismatch]
- PRED line2: <空>

### fuse  [decoded] conf1=75% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<CHNLOI<<HANG<MIO<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<UPYHYCY<H<<<Y<HYGYYH<H<<YGHUH<N<<<<Y<<<<N<
        ^^^^^^^  ^^^^^^^^^^^ ^  ^^^^^ ^    ^    ^    [27 mismatch]
GT  : MB00000007CHN8807191F291203910004067<<<<<<28
PRED: EEL00E04H0GGG8040010<100444114L<00CC010G0E94
      ^^^  ^ ^^^^^^ ^^^^^^^^^^^^^^ ^^^^ ^^^^^^^^^^   [38 mismatch]

### tesseract  [FAIL]
- backend: tesseract  (20 ms)
- error: no MRZ candidate decoded
- PRED line1: <空>
- PRED line2: <空>

### paddle  [decoded] conf1=62% conf2=0%
- backend: paddle-onnxruntime  (683 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNLOI<<HANG<MIO<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^ ^^^^^^  ^^^^ ^^^                             [14 mismatch]
GT  : MB00000007CHN8807191F291203910004067<<<<<<28
PRED: <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^   [38 mismatch]

### indra  [FAIL]
- backend: indra  (19 ms)
- error: indra 未找到 MRZ 候选行 (0 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [decoded] conf1=98% conf2=84%
- backend: indra  (289 ms)
- error: lines=63 edits=4 composite_edits=1 country=CAA
GT  : P<CHNLOI<<HANG<MIO<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: MACAASPECIALADMINISTRATIVEREGIONOFTHEPEOPLES
      ^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [43 mismatch]
GT  : MB00000007CHN8807191F291203910004067<<<<<<28
PRED: RCODIGODO1CAADOEMIS3ORICODE6FISSUNGSTATERE36
      ^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [43 mismatch]

## a495fb368b46ab35_0280.png
- 官方判定: PASS  verified=True  chosen=fuse  ms=149
- 官方 line1: P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: G622925996CHN8310291F19042204C0CMXNENBPJB984
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `G622925996CHN8310291F19042204C0CMXNENBPJB984`

### traditional  [decoded] conf1=72% conf2=62%
- error: decode failed: check digit mismatch
GT  : P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<LLALL<L1<<Y<WCDFA<<<<<<<<<<<<<<<<<<<<<<<<<
      ^ ^^^^^^^^   ^^^^^^                            [15 mismatch]
GT  : G622925996CHN8310291F19042204C0CMXNENBPJB984
PRED: L822V23V44C9H834C294L<4C<2JCLC<LAFALAAP3FV8<
      ^^  ^ ^^^^ ^^  ^^  ^^^^^^ ^^^ ^^^^^^^^ ^^^ ^   [32 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=95% conf2=89%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PTCHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : G622925996CHN8310291F19042204C0CMXNENBPJB984
PRED: 0622925996CHN8310291F1904220L00CHKNENBPJB984
      ^                           ^^  ^^             [5 mismatch]

### fuse  [DECODE-OK] conf1=97% conf2=98%
GT  : P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : G622925996CHN8310291F19042204C0CMXNENBPJB984
PRED: G622925996CHN8310291F19042204C0CMXNENBPJB984
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (69 ms)
GT  : P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PTCHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : G622925996CHN8310291F19042204C0CMXNENBPJB984
PRED: G622925996CHN8310291F1904220LCOCMKNENBPJB984
                                  ^ ^  ^             [3 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (84 ms)
GT  : P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PTCHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : G622925996CHN8310291F19042204C0CMXNENBPJB984
PRED: G622925996CHN8310291F1904220LCOCMKNENBPJB984
                                  ^ ^  ^             [3 mismatch]

### indra  [DECODE-OK] conf1=97% conf2=98%
- backend: indra  (63 ms)
GT  : P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PTCHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : G622925996CHN8310291F19042204C0CMXNENBPJB984
PRED: G622925996CHN8310291F1904220LCOCMKNENBPJB984
                                  ^ ^  ^             [3 mismatch]

### indra_fullpage  [decoded] conf1=98% conf2=99%
- backend: indra  (312 ms)
- error: lines=47 edits=4 composite_edits=1 country=ECI
GT  : P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: THECINISTRYOFFOREIGNAFFAIRSOFTHEPEOPLESREPUB
      ^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [43 mismatch]
GT  : G622925996CHN8310291F19042204C0CMXNENBPJB984
PRED: REQUESTSA4ECIVILAND4ILITARY5UTHORITIESOFFO20
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]

## a60aae34a06f1867_0016.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=168
- 官方 line1: P<UKRTKACHENKO<<MARIANA<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: XX000000<0UKR9108242F23092571234567890<<<<70
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `X30OO0OO<0UKW9108242F23092571234567890<<<<70`

### traditional  [decoded] conf1=76% conf2=70%
- error: decode failed: check digit mismatch
GT  : P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
PRED: P<L<HT<ACFEN<C<<PADIAHA<<<<<<<<<<<<<<<<<<<<<
        ^^^ ^  ^^ ^^  ^ ^^ ^                         [12 mismatch]
GT  : X30OO0OO<0UKW9108242F23092571234567890<<<<70
PRED: XXCCCCCC<CL<AV4C82A2423C9257423A5K789C<<<<7L
       ^^^^^^^ ^^^^^^^  ^ ^  ^    ^  ^ ^   ^     ^   [22 mismatch]

### cnn  [DECODE-OK] conf1=93% conf2=86%
GT  : P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKXYKACH<HKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
           ^     ^                                   [2 mismatch]
GT  : X30OO0OO<0UKW9108242F23092571234567890<<<<70
PRED: X30OO0OO<0UKW9108242F23092571234567890<<<<70
                                                     [0 mismatch]

### crnn  [DECODE-OK] conf1=97% conf2=98%
- backend: crnn  (18 ms)
GT  : P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRTKACHENKO<<MARIANA<<<<<<<<<<<<<<<<<<<<<
          ^     ^       ^^                           [4 mismatch]
GT  : X30OO0OO<0UKW9108242F23092571234567890<<<<70
PRED: XX000000<0UKR9108242F23092571234567890<<<<70
       ^ ^^ ^^    ^                                  [6 mismatch]

### fuse  [DECODE-OK] conf1=98% conf2=96%
GT  : P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKXYKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : X30OO0OO<0UKW9108242F23092571234567890<<<<70
PRED: X30OO0OO<0UKW9108242F23092571234567890<<<<70
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (87 ms)
GT  : P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRTKACHENKO<<MARIANA<<<<<<<<<<<<<<<<<<<<<
          ^     ^       ^^                           [4 mismatch]
GT  : X30OO0OO<0UKW9108242F23092571234567890<<<<70
PRED: XX000000<0UKR9108242F23092571234567890<<<<70
       ^ ^^ ^^    ^                                  [6 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=98%
- backend: paddle-onnxruntime  (83 ms)
GT  : P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRTKACHENKO<<MARIANA<<<<<<<<<<<<<<<<<<<<<
          ^     ^       ^^                           [4 mismatch]
GT  : X30OO0OO<0UKW9108242F23092571234567890<<<<70
PRED: XX000000<0UKR9108242F23092571234567890<<<<70
       ^ ^^ ^^    ^                                  [6 mismatch]

### indra  [DECODE-OK] conf1=97% conf2=100%
- backend: indra  (55 ms)
GT  : P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRTKACHENKO<<MARIANA<<<<<<<<<<<<<<<<<<<<<
          ^     ^       ^^                           [4 mismatch]
GT  : X30OO0OO<0UKW9108242F23092571234567890<<<<70
PRED: XX000000<0UKR9108242F23092571234567890<<<<70
       ^ ^^ ^^    ^                                  [6 mismatch]

### indra_fullpage  [DECODE-OK] conf1=96% conf2=100%
- backend: indra  (180 ms)
GT  : P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<
PRED: P<UKRTKACHENKO<<MARIANA<<<<<<<<<<<<<<<<<<<<<
          ^     ^       ^^                           [4 mismatch]
GT  : X30OO0OO<0UKW9108242F23092571234567890<<<<70
PRED: XX000000<0UKR9108242F23092571234567890<<<<70
       ^ ^^ ^^    ^                                  [6 mismatch]

## a955e21ae58d8c1d_0013.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=204
- 官方 line1: P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: N000000001SYH8801018M27082410104006915555<12
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `N000000001SYH8801018M27082410104006915555<12`

### traditional  [decoded] conf1=70% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: MMNGYRNAN9OLR<<MAN1<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^ ^^^^^                            [18 mismatch]
GT  : N000000001SYH8801018M27082410104006915555<12
PRED: MWCMMMCCOC49YAA8C4M48N27C82AMM4CAOCK9455<<<4
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^ ^^   [41 mismatch]

### cnn  [DECODE-OK] conf1=95% conf2=97%
GT  : P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYRNANSOUR<<BANI<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : N000000001SYH8801018M27082410104006915555<12
PRED: N000000001SYH8801018M27082410104006915555<12
                                                     [0 mismatch]

### crnn  [decoded] conf1=96% conf2=87%
- backend: crnn  (13 ms)
- error: decode failed: check digit mismatch
GT  : P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PNSYRMANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^                                         [2 mismatch]
GT  : N000000001SYH8801018M27082410104006915555<12
PRED: N000000001SYR8801018M270824101040069158<<<12
                  ^                         ^^^      [4 mismatch]

### fuse  [DECODE-OK] conf1=97% conf2=98%
GT  : P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : N000000001SYH8801018M27082410104006915555<12
PRED: N000000001SYH8801018M27082410104006915555<12
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (79 ms)
GT  : P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PNSYRMANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^                                         [2 mismatch]
GT  : N000000001SYH8801018M27082410104006915555<12
PRED: N000000001SYR8801018M270824101040069155<<<12
                  ^                          ^^      [3 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (91 ms)
GT  : P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PNSYRMANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^                                         [2 mismatch]
GT  : N000000001SYH8801018M27082410104006915555<12
PRED: N000000001SYR8801018M270824101040069155<<<12
                  ^                          ^^      [3 mismatch]

### indra  [DECODE-OK] conf1=95% conf2=99%
- backend: indra  (58 ms)
GT  : P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PNSYRMANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^                                         [2 mismatch]
GT  : N000000001SYH8801018M27082410104006915555<12
PRED: N000000001SYR8801018M270824101040069155<<<12
                  ^                          ^^      [3 mismatch]

### indra_fullpage  [DECODE-OK] conf1=91% conf2=98%
- backend: indra  (323 ms)
GT  : P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PNSYRMANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^                                         [2 mismatch]
GT  : N000000001SYH8801018M27082410104006915555<12
PRED: N000000001SYR8801018M270824101040069155<<<12
                  ^                          ^^      [3 mismatch]

## acd3e9fd1ce9bd0a_0016.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=158
- 官方 line1: P<GBRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
- 官方 line2: 5334755143GBR8812049F2509286<<<<<<<<<<<<<<04
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<`
- GT line2: `5334755143GBH8812049F25092865<<<<<<<SD<SDS78`

### traditional  [decoded] conf1=72% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
PRED: P<G9ML<<5PECINEH<<AHGELA<2CE<<<<<<<<<<<<<<<<
         ^^^^ ^    ^ ^   ^     ^^                    [10 mismatch]
GT  : 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
PRED: 93XG7554A3GERB942CGVA29ED28E<<<<<<<<<<<<<<LK
      ^ ^^   ^^  ^^^^^ ^^^^ ^^^  ^^       ^^ ^^^^^   [26 mismatch]

### cnn  [DECODE-OK] conf1=96% conf2=85%
GT  : P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
PRED: P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
PRED: 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
                                                     [0 mismatch]

### crnn  [decoded] conf1=97% conf2=93%
- backend: crnn  (13 ms)
- error: decode failed: bad MRZ length
GT  : P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
PRED: P<GBRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
         ^                                           [1 mismatch]
GT  : 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
PRED: 334755143GBR8812049F2509286<<<<<<<<<<<<<<04 
      ^ ^^^ ^^^^^^^ ^^^^^^^^^^^^^^^       ^^ ^^^^^   [33 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=97%
GT  : P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
PRED: P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
PRED: 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (74 ms)
GT  : P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
PRED: P<GBRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
         ^                                           [1 mismatch]
GT  : 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
PRED: 5334755143GBR8812049F2509286<<<<<<<<<<<<<<04
                  ^               ^       ^^ ^^^^^   [9 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=90%
- backend: paddle-onnxruntime  (94 ms)
GT  : P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
PRED: P<GBRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
         ^                                           [1 mismatch]
GT  : 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
PRED: 5334755143GBR8812049F2509286<<<<<<<<<<<04<80
                  ^               ^       ^^ ^^^^^   [9 mismatch]

### indra  [DECODE-OK] conf1=99% conf2=99%
- backend: indra  (49 ms)
GT  : P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
PRED: P<GBRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
         ^                                           [1 mismatch]
GT  : 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
PRED: 5334755143GBR8812049F2509286<<<<<<<<<<<<<<04
                  ^               ^       ^^ ^^^^^   [9 mismatch]

### indra_fullpage  [DECODE-OK] conf1=96% conf2=99%
- backend: indra  (165 ms)
GT  : P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<
PRED: P<GBRUK<SPECIMEN<<ANGELAKZOE<<<<<<<<<<<<<<<<
         ^                    ^                      [2 mismatch]
GT  : 5334755143GBH8812049F25092865<<<<<<<SD<SDS78
PRED: 5334755143GBR8812049F2509286<<<<<<<<<<<<<<04
                  ^               ^       ^^ ^^^^^   [9 mismatch]

## acf14e0c631ec935_0030.png
- 官方判定: REJECT  verified=False  chosen=None  ms=15
- 理由: 未在照片中定位到 MRZ 双行

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [DECODE-OK] conf1=100% conf2=96%
- backend: indra  (185 ms)

## ae5fdb36f7498291_0019.png
- 官方判定: PASS  verified=True  chosen=tesseract  ms=215
- 官方 line1: P<ARGVAZQUEZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: ZZZ0001105ARG1ZI1018F26O6T8399999999<<<<<<86
- 理由: PaddleOCR 识别并通过校验

### traditional  [decoded] conf1=72% conf2=69%
- error: decode failed: check digit mismatch

### cnn  [decoded] conf1=93% conf2=94%
- error: decode failed: check digit mismatch

### crnn  [decoded] conf1=98% conf2=95%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch

### fuse  [decoded] conf1=96% conf2=96%
- error: decode failed: check digit mismatch

### tesseract  [DECODE-OK]
- backend: tesseract  (158 ms)

### paddle  [DECODE-OK] conf1=99% conf2=96%
- backend: paddle-onnxruntime  (99 ms)

### indra  [DECODE-OK] conf1=92% conf2=96%
- backend: indra  (51 ms)

### indra_fullpage  [DECODE-OK] conf1=94% conf2=98%
- backend: indra  (176 ms)

## b1442018a8167bf5_0046.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=168
- 官方 line1: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E000077347USA6502056F3010149500100120<095838
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E000077347USA6502056F3010149500100120<095838`

### traditional  [decoded] conf1=74% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: M<L9A1AAVLLLM<<HAFPY<<<<<<<<<<<<<<<<<<<<<<<<
      ^ ^^ ^^  ^ ^^    ^                             [9 mismatch]
GT  : E000077347USA6502056F3010149500100120<095838
PRED: MCCCC77347C9AM5CPC5KF3C4C4495CCMCC42C<C49A3N
      ^^^^^     ^^ ^ ^^^ ^  ^^^^   ^^^^^^ ^ ^^^^ ^   [28 mismatch]

### cnn  [decoded] conf1=95% conf2=93%
- error: decode failed: check digit mismatch
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USBYRAUELRR<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
          ^^  ^  ^                                   [4 mismatch]
GT  : E000077347USA6502056F3010149500100120<095838
PRED: L000077347USA6502056F3010149500100120<095137
      ^                                        ^ ^   [3 mismatch]

### crnn  [decoded] conf1=95% conf2=81%
- backend: crnn  (15 ms)
- error: decode failed: bad MRZ length
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E000077347USA6502056F3010149500100120<095838
PRED: 00007734706A6502056M3010149500100120<09508  
      ^   ^ ^^^^^^^^^^^^^^^^^^^^^^^ ^^ ^^^^^^^^ ^^   [37 mismatch]

### fuse  [decoded] conf1=96% conf2=95%
- error: decode failed: check digit mismatch
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USBYRAUELRR<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
          ^^  ^  ^                                   [4 mismatch]
GT  : E000077347USA6502056F3010149500100120<095838
PRED: L000077347USA6502056F3010149500100120<095137
      ^                                        ^ ^   [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (86 ms)
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E000077347USA6502056F3010149500100120<095838
PRED: E000077347USA6502056F3010149500100120<095838
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=98%
- backend: paddle-onnxruntime  (74 ms)
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E000077347USA6502056F3010149500100120<095838
PRED: E000077347USA6502056F3010149500100120<095838
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=95% conf2=97%
- backend: indra  (53 ms)
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E000077347USA6502056F3010149500100120<095838
PRED: E000077347USA6502056F30101495001001Z0<095<88
                                         ^     ^^    [3 mismatch]

### indra_fullpage  [decoded] conf1=73% conf2=66%
- backend: indra  (340 ms)
- error: lines=45 edits=4 composite_edits=1 country=ISA
GT  : P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: SEISADOAMOSOUAQUEPERUSPEAHEMPORENVEDOPJOREPI
      ^^^  ^^ ^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^   [40 mismatch]
GT  : E000077347USA6502056F3010149500100120<095838
PRED: SAUORAMON4ISARIODSS0ONUARY13TUMISSIOP<<<<<68
      ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^    [40 mismatch]

## b284565582147454_0098.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=152
- 官方 line1: P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 5E75870985KCN8507022F2404152213471C117621884
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `5E75870985KCN8507022F2404152213471C117621884`

### traditional  [decoded] conf1=77% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: 2P<CWLEE<<9LYECK<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^ ^^  ^^^^^^                               [13 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: P7CK89C985<CA85CZC22L24C<4S2245<243V4ZK2ZR84
      ^^^^ ^^   ^ ^  ^^^  ^  ^^^^  ^^^^^^^^^^ ^^     [28 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=96% conf2=98%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PPMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^ ^^^^^^^                           ^   [16 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=90%
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: 5E75870985KCN8507022F2404152213471C117621884
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (74 ms)
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^ ^^    ^^^                                [9 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (88 ms)
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^ ^^    ^^^                                [9 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

### indra  [DECODE-OK] conf1=97% conf2=97%
- backend: indra  (59 ms)
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^ ^^    ^^^                                [9 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

### indra_fullpage  [DECODE-OK] conf1=97% conf2=97%
- backend: indra  (218 ms)
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^ ^^    ^^^                                [9 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

## b284565582147454_0115.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=155
- 官方 line1: P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 5E75870985KCN8507022F2404152213471C117621884
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `5E75870985KCN8507022F2404152213471C117621884`

### traditional  [decoded] conf1=77% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: 2P<CWLEE<<9LYECK<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^ ^^  ^^^^^^                               [13 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: P7CK89C985<CA85CZC22L24C<4S2245<243V4ZK2ZR84
      ^^^^ ^^   ^ ^  ^^^  ^  ^^^^  ^^^^^^^^^^ ^^     [28 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=96% conf2=98%
- backend: crnn  (17 ms)
- error: decode failed: check digit mismatch
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PPMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^ ^^^^^^^                           ^   [16 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=90%
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: 5E75870985KCN8507022F2404152213471C117621884
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (81 ms)
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^ ^^    ^^^                                [9 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (85 ms)
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^ ^^    ^^^                                [9 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

### indra  [DECODE-OK] conf1=97% conf2=97%
- backend: indra  (52 ms)
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^ ^^    ^^^                                [9 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

### indra_fullpage  [DECODE-OK] conf1=97% conf2=97%
- backend: indra  (193 ms)
GT  : P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMKORLEE<<SUYEON<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^ ^^    ^^^                                [9 mismatch]
GT  : 5E75870985KCN8507022F2404152213471C117621884
PRED: M706890985KOR8507022F24041522154710V17627884
      ^^^^ ^     ^^                 ^   ^^    ^      [11 mismatch]

## b2b5008f3839530d_0001.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=164
- 官方 line1: P<SVNVZOREC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: PB90062122SVN7811182F26121221811978505602<14
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `M0N0062122SVN7811182F261212218IIA785M565EK14`

### traditional  [decoded] conf1=77% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<9VAV2AAAC<<AHA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^ ^^^^^^^^    ^                                [10 mismatch]
GT  : M0N0062122SVN7811182F261212218IIA785M565EK14
PRED: P9VCCA24228VN784448Z126424224844978SD5KC2<44
      ^^^^^^ ^  ^    ^^^ ^^  ^ ^  ^ ^^^  ^^ ^^^^^    [26 mismatch]

### cnn  [skip]

### crnn  [DECODE-OK] conf1=98% conf2=94%
- backend: crnn  (14 ms)
GT  : P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SVNVZOREC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^ ^^  ^                                     [4 mismatch]
GT  : M0N0062122SVN7811182F261212218IIA785M565EK14
PRED: PB900621226VN7811182F26121221811978505602<14
      ^^^       ^                   ^^^   ^  ^^^     [11 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=94%
GT  : P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : M0N0062122SVN7811182F261212218IIA785M565EK14
PRED: M0N0062122SVN7811182F261212218IIA785M565EK14
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (76 ms)
GT  : P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SVNVZOREC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^ ^^  ^                                     [4 mismatch]
GT  : M0N0062122SVN7811182F261212218IIA785M565EK14
PRED: PB90062122SVN7811182F26121221811978505602<14
      ^^^                           ^^^   ^  ^^^     [10 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (87 ms)
GT  : P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SVNVZOREC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^ ^^  ^                                     [4 mismatch]
GT  : M0N0062122SVN7811182F261212218IIA785M565EK14
PRED: PB90062122SVN7811182F26121221811978505602<14
      ^^^                           ^^^   ^  ^^^     [10 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=88%
- backend: indra  (58 ms)
GT  : P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SVNVZOREC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^ ^^  ^                                     [4 mismatch]
GT  : M0N0062122SVN7811182F261212218IIA785M565EK14
PRED: PB90062122SVN7811182F26121221811978505602<14
      ^^^                           ^^^   ^  ^^^     [10 mismatch]

### indra_fullpage  [DECODE-OK] conf1=94% conf2=90%
- backend: indra  (266 ms)
GT  : P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SVNVZOREC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^ ^^  ^                                     [4 mismatch]
GT  : M0N0062122SVN7811182F261212218IIA785M565EK14
PRED: PB90062122SVN7811182F26121221811978505602<14
      ^^^                           ^^^   ^  ^^^     [10 mismatch]

## b30dc79ee9550e9e_0024.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=171
- 官方 line1: P<DWLBPEDDNDN<<YHARDD<BPBGINBN<<<<<<<<<<<<<<
- 官方 line2: K15M12M1V5ISK1212123F290204912M2M24424VNN462
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<`
- GT line2: `K15M12M1V5ISK1212123F290204912M2M24424VNN462`

### traditional  [decoded] conf1=70% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<
PRED: MPAI9L5PECJNEW<<JHAECE<5GEC1NEK<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^                [30 mismatch]
GT  : K15M12M1V5ISK1212123F290204912M2M24424VNN462
PRED: MA35C4294<519L4242429G29E2C49424242<4239<<<A
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]

### cnn  [DECODE-OK] conf1=89% conf2=85%
GT  : P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<
PRED: P<DWLBPEDDNDN<<YHARDD<BPBGINBN<<<<<<<<<<<<<<
              ^          ^                           [2 mismatch]
GT  : K15M12M1V5ISK1212123F290204912M2M24424VNN462
PRED: K15M12M1V5ISK1212123F290204912M2M24424VNN462
                                                     [0 mismatch]

### crnn  [decoded] conf1=97% conf2=94%
- backend: crnn  (12 ms)
- error: decode failed: check digit mismatch
GT  : P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<
PRED: PAISLSPECIMEN<<THAEOE<SPECIMEN<<<<<<<<<<<<<<
       ^^^ ^   ^^^   ^  ^ ^ ^ ^^ ^^                  [15 mismatch]
GT  : K15M12M1V5ISK1212123F290204912M2M24424VNN462
PRED: A3501291<5IGL1212123F2902049121212<1239<<<<2
      ^^ ^  ^ ^  ^^                 ^ ^ ^^ ^^^^^^    [17 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=92%
GT  : P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<
PRED: P<DWLBPEDDNDN<<YHARDD<BPBGINBN<<<<<<<<<<<<<<
              ^          ^                           [2 mismatch]
GT  : K15M12M1V5ISK1212123F290204912M2M24424VNN462
PRED: K15M12M1V5ISK1212123F290204912M2M24424VNN462
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (73 ms)
GT  : P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<
PRED: PAISLSPECIMEN<<THAEOE<SPECIMEN<<<<<<<<<<<<<<
       ^^^ ^   ^^^   ^  ^ ^ ^ ^^ ^^                  [15 mismatch]
GT  : K15M12M1V5ISK1212123F290204912M2M24424VNN462
PRED: A3501291<5ISL1212123F2902049121212<1239<<<62
      ^^ ^  ^ ^   ^                 ^ ^ ^^ ^^^^^     [15 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (86 ms)
GT  : P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<
PRED: PAISLSPECIMEN<<THAEOE<SPECIMEN<<<<<<<<<<<<<<
       ^^^ ^   ^^^   ^  ^ ^ ^ ^^ ^^                  [15 mismatch]
GT  : K15M12M1V5ISK1212123F290204912M2M24424VNN462
PRED: A3501291<5ISL1212123F2902049121212<1239<<<62
      ^^ ^  ^ ^   ^                 ^ ^ ^^ ^^^^^     [15 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=96%
- backend: indra  (54 ms)
GT  : P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<
PRED: PAISLSPECIMEN<<THAEDE<SPECIMENK<<<<<<<<<<<<<
       ^^^ ^   ^^^   ^  ^^^ ^ ^^ ^^ ^                [17 mismatch]
GT  : K15M12M1V5ISK1212123F290204912M2M24424VNN462
PRED: A3501291<5ISL1212123F2902049121212<1239<<<62
      ^^ ^  ^ ^   ^                 ^ ^ ^^ ^^^^^     [15 mismatch]

### indra_fullpage  [DECODE-OK] conf1=72% conf2=92%
- backend: indra  (218 ms)
GT  : P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<
PRED: ISAANDICELANDISLANDE<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^                 [29 mismatch]
GT  : K15M12M1V5ISK1212123F290204912M2M24424VNN462
PRED: VEGABRFNR6AANORTNON4DUPASSE9O<<<<<<<<<<<<<82
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^    [42 mismatch]

## b7a97ca0edede07b_0164.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=146
- 官方 line1: P<POLSAMPLE<<WILLIAM<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: ZS80000384POL8306122M1607303<<<<<<<<<<<<<<02
- 理由: PaddleOCR 识别并通过校验

### traditional  [decoded] conf1=74% conf2=68%
- error: decode failed: check digit mismatch

### cnn  [skip]

### crnn  [decoded] conf1=96% conf2=91%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch

### fuse  [DECODE-OK] conf1=99% conf2=87%

### tesseract  [DECODE-OK]
- backend: tesseract  (66 ms)

### paddle  [decoded] conf1=98% conf2=90%
- backend: paddle-onnxruntime  (153 ms)
- error: decode failed: check digit mismatch

### indra  [DECODE-OK] conf1=81% conf2=96%
- backend: indra  (52 ms)

### indra_fullpage  [DECODE-OK] conf1=86% conf2=97%
- backend: indra  (247 ms)

## bc1ddfd8c13b5ca9_0007.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=174
- 官方 line1: P<FRASPECIMEN<ABB<<NATACHA<<<<<<<<<<<<<<<<<<
- 官方 line2: 90RF023318FRA7307122F2903035<<<<<<<<<<<<<<02
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<`
- GT line2: `9MM1024318MMA7307122F29030355N5NNN554N44K902`

### traditional  [decoded] conf1=70% conf2=70%
- error: decode failed: check digit mismatch
GT  : P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
PRED: P<FRA5PEC1NEN<ABB<<WATACHA<<<<<<<<<<<<<<<<<<
        ^  ^ ^ ^ ^       ^ ^                         [7 mismatch]
GT  : 9MM1024318MMA7307122F29030355N5NNN554N44K902
PRED: 4OAAM2334NPHA73C7422P29C3C35<<<<<<<<<<<<<<C2
      ^^^^^ ^ ^^^^   ^ ^  ^  ^ ^  ^^^^^^^^^^^^^^^    [30 mismatch]

### cnn  [DECODE-OK] conf1=86% conf2=86%
GT  : P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
PRED: P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 9MM1024318MMA7307122F29030355N5NNN554N44K902
PRED: 9MM1024318MMA7307122F29030355N5NNN554N44K902
                                                     [0 mismatch]

### crnn  [DECODE-OK] conf1=96% conf2=90%
- backend: crnn  (14 ms)
GT  : P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
PRED: P<FRASPECIMEN<ABB<<NAIACHA<<<<<<<<<<<<<<<<<<
        ^    ^ ^^^         ^                         [6 mismatch]
GT  : 9MM1024318MMA7307122F29030355N5NNN554N44K902
PRED: 90RF023318FRA7307122F2903035<<<<<<<<<<<<<<02
       ^^^  ^   ^^                ^^^^^^^^^^^^^^     [20 mismatch]

### fuse  [DECODE-OK] conf1=98% conf2=94%
GT  : P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
PRED: P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 9MM1024318MMA7307122F29030355N5NNN554N44K902
PRED: 9MM1024318MMA7307122F29030355N5NNN554N44K902
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (69 ms)
GT  : P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
PRED: P<FRASPECIMEN<ABB<<NATACHA<<<<<<<<<<<<<<<<<<
        ^    ^ ^^^         ^                         [6 mismatch]
GT  : 9MM1024318MMA7307122F29030355N5NNN554N44K902
PRED: 90RF023318FRA7307122F2903035<<<<<<<<<<<<<<02
       ^^^  ^   ^^                ^^^^^^^^^^^^^^     [20 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (92 ms)
GT  : P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
PRED: P<FRASPECIMEN<ABB<<NATACHA<<<<<<<<<<<<<<<<<<
        ^    ^ ^^^         ^                         [6 mismatch]
GT  : 9MM1024318MMA7307122F29030355N5NNN554N44K902
PRED: 90RF023318FRA7307122F29030350<<<<<<<<<<<<<02
       ^^^  ^   ^^                ^^^^^^^^^^^^^^     [20 mismatch]

### indra  [DECODE-OK] conf1=84% conf2=99%
- backend: indra  (51 ms)
GT  : P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
PRED: PFFAAPECIMENKABBC<NATACHAKKKKKKKKKKKKKKKKKK<
       ^^^ ^^^^^^^^^^ ^ ^^^^^^^^^^^^^^^^^^^^^^^^^    [39 mismatch]
GT  : 9MM1024318MMA7307122F29030355N5NNN554N44K902
PRED: 90RF023318FAA7307122F2903035<<<<<<<<<<<<<<02
       ^^^  ^   ^^                ^^^^^^^^^^^^^^     [20 mismatch]

### indra_fullpage  [DECODE-OK] conf1=86% conf2=99%
- backend: indra  (187 ms)
GT  : P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<
PRED: PCFRASPECIMENKABBC<NATACHAKKKKKKKKKKKKKKKKKK
       ^^    ^ ^^^ ^   ^   ^    ^^^^^^^^^^^^^^^^^^   [27 mismatch]
GT  : 9MM1024318MMA7307122F29030355N5NNN554N44K902
PRED: 90RF023318FRA7307122F2903035<<<<<<<<<<<<<<02
       ^^^  ^   ^^                ^^^^^^^^^^^^^^     [20 mismatch]

## c0cbf7c176328f3d_0006.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=176
- 官方 line1: P<GRCELLINAS<<GEORGIOS<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06`

### traditional  [decoded] conf1=70% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<
PRED: A<AACALLXNA9<<GMARG1O9<<<<<<<<<<<<<<<<<<<<<<
      ^ ^^^^^^^ ^^  ^^^^^^^^                         [18 mismatch]
GT  : AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
PRED: AA42345A78AAM79D9452N33424MA<<<<<<<<<<<<<<MA
        ^    ^  ^^^ ^^^^ ^^  ^ ^^^              ^^   [17 mismatch]

### cnn  [decoded] conf1=83% conf2=60%
- error: decode failed: check digit mismatch
GT  : P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<
PRED: P<WRBKDDDNAD<<WDDBODDB<<<<<<<<<<<<<<<<<<<<<<
        ^^^^  ^ ^   ^^^^^^^^                         [14 mismatch]
GT  : AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
PRED: KK12M4M6P4WMC7404141M4412104NNNNNNNNNNN44404
      ^^  ^ ^ ^^^^  ^ ^ ^^ ^^    ^^^^^^^^^^^^^^^ ^   [30 mismatch]

### crnn  [DECODE-OK] conf1=96% conf2=95%
- backend: crnn  (14 ms)
GT  : P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<
PRED: P<GRCELLINAS<<GEORGIOS<<<<<<<<<<<<<<<<<<<<<<
        ^^^^^^^ ^^  ^^^^^^^^                         [17 mismatch]
GT  : AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
PRED: AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### fuse  [decoded] conf1=86% conf2=62%
- error: decode failed: check digit mismatch
GT  : P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<
PRED: P<WRBBDDDNAD<<WDDBODDB<<<<<<<<<<<<<<<<<<<<<<
        ^^^^  ^ ^   ^^^^^^^^                         [14 mismatch]
GT  : AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
PRED: KK12M4M6P4WMC7404140M4412104NNNNNNNNNNN44404
      ^^  ^ ^ ^^^^  ^ ^ ^^ ^^    ^^^^^^^^^^^^^^^ ^   [30 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (75 ms)
GT  : P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<
PRED: P<GRCELLINAS<<GEORGIOS<<<<<<<<<<<<<<<<<<<<<<
        ^^^^^^^ ^^  ^^^^^^^^                         [17 mismatch]
GT  : AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
PRED: AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=96%
- backend: paddle-onnxruntime  (86 ms)
GT  : P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<
PRED: P<GRCELLINAS<<GEORGIOS<<<<<<<<<<<<<<<<<<<<<<
        ^^^^^^^ ^^  ^^^^^^^^                         [17 mismatch]
GT  : AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
PRED: AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=89% conf2=98%
- backend: indra  (54 ms)
GT  : P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<
PRED: PGRCCLLINASKKGEORGIOSKKKKKKKKKKKKKKKKKEE<<<<
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       [39 mismatch]
GT  : AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
PRED: AA12345678RCC7505157M3312108<<<<<<<<<<<<<<06
                ^^                                   [2 mismatch]

### indra_fullpage  [DECODE-OK] conf1=92% conf2=98%
- backend: indra  (192 ms)
GT  : P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<
PRED: P<GRCELLINAS<<GEORGIOS<<<<<<<<<<<<<<<<<<<<<<
        ^^^^^^^ ^^  ^^^^^^^^                         [17 mismatch]
GT  : AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
PRED: AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06
                                                     [0 mismatch]

## c2aee55b9017575f_0028.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=177
- 官方 line1: P<RUSOBRAZETS<<TESTER<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 5200360198RUS5608087M2002156<<<<<<<<<<<<<<08
- 理由: IndraOCR 识别并通过校验

- GT line1: `P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08`

### traditional  [decoded] conf1=74% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
PRED: MD<M13LHEWA7ET9<<TE5TEW<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^^^                        [23 mismatch]
GT  : D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
PRED: M52LL3KC498AL95KC8C87P2LL243L<<<<<<<<<<<<<<L
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^             ^^   [31 mismatch]

### cnn  [DECODE-OK] conf1=94% conf2=93%
GT  : P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<NUEOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
          ^                                          [1 mismatch]
GT  : D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
PRED: D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
                                                     [0 mismatch]

### crnn  [decoded] conf1=98% conf2=85%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch
GT  : P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<RUSOBRAZETS<<TESTER<<<<<<<<<<<<<<<<<<<<<<<
        ^        ^   ^^ ^^                           [6 mismatch]
GT  : D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
PRED: 3200360198NU65608087M2002156<<<<<<<<<<<<<<00
      ^ ^^^   ^ ^^^                              ^   [9 mismatch]

### fuse  [DECODE-OK] conf1=97% conf2=97%
GT  : P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<NUEOBRAZEYS<<YESYER<<<<<<<<<<<<<<<<<<<<<<<
          ^           ^  ^                           [3 mismatch]
GT  : D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
PRED: 52D01601M8KBS5608087M2002156<<<<<<<<<<<<<<08
      ^  ^                                           [2 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (81 ms)
GT  : P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<RUSOBRAZETS<<TESTER<<<<<<<<<<<<<<<<<<<<<<<
        ^        ^   ^^ ^^                           [6 mismatch]
GT  : D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
PRED: 5200360198RUS5608087M2002156<<<<<<<<<<<<<<08
      ^ ^^^   ^ ^^                                   [7 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=97%
- backend: paddle-onnxruntime  (106 ms)
GT  : P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<RUSOBRAZETS<<TESTER<<<<<<<<<<<<<<<<<<<<<<<
        ^        ^   ^^ ^^                           [6 mismatch]
GT  : D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
PRED: 5200360198RUS5608087M20021560<<<<<<<<<<<<<08
      ^ ^^^   ^ ^^                ^                  [8 mismatch]

### indra  [DECODE-OK] conf1=98% conf2=99%
- backend: indra  (57 ms)
GT  : P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<RUSOBRAZETS<<TESTER<<<<<<<<<<<<<<<<<<<<<<<
        ^        ^   ^^ ^^                           [6 mismatch]
GT  : D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
PRED: 5200360198RUS5608087M2002156<<<<<<<<<<<<<<08
      ^ ^^^   ^ ^^                                   [7 mismatch]

### indra_fullpage  [DECODE-OK] conf1=97% conf2=100%
- backend: indra  (216 ms)
GT  : P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<RUSOBRAZETS<<TESTER<<<<<<<<<<<<<<<<<<<<<<<
        ^        ^   ^^ ^^                           [6 mismatch]
GT  : D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08
PRED: 5200360198RUS5608087M2002156<<<<<<<<<<<<<<08
      ^ ^^^   ^ ^^                                   [7 mismatch]

## cb64e780ab32f735_0004.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=187
- 官方 line1: P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 7B12318830SVK1111112F34091871111111111<<<<06
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `7B12318830SVK1111112F34091871111111111<<<<06`

### traditional  [decoded] conf1=75% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
PRED: PP9V<9PLCIHLN<<V2CW<A<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^ ^   ^   ^^^^^                           [12 mismatch]
GT  : 7B12318830SVK1111112F34091871111111111<<<<06
PRED: XH3234883C3V<3433442P34CV4874444444444<<<<CK
      ^^^  ^   ^^ ^^^^^^^ ^  ^^^  ^^^^^^^^^^    ^^   [29 mismatch]

### cnn  [DECODE-OK] conf1=93% conf2=98%
GT  : P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 7B12318830SVK1111112F34091871111111111<<<<06
PRED: 7B12318830SVK1111112F34091871111111111<<<<06
                                                     [0 mismatch]

### crnn  [decoded] conf1=98% conf2=88%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPSVKSPECIMEN<<VZORKA<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^      ^    ^^                              [5 mismatch]
GT  : 7B12318830SVK1111112F34091871111111111<<<<06
PRED: XA123188308VK1111112F34091871111111111<<<<06
      ^^        ^                                    [3 mismatch]

### fuse  [DECODE-OK] conf1=97% conf2=98%
GT  : P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 7B12318830SVK1111112F34091871111111111<<<<06
PRED: 7B12318830SVK1111112F34091871111111111<<<<06
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (84 ms)
GT  : P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPSVKSPECIMEN<<VZORKA<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^      ^    ^^                              [5 mismatch]
GT  : 7B12318830SVK1111112F34091871111111111<<<<06
PRED: XR12318830SVK1111112F34091871111111111<<<<06
      ^^                                             [2 mismatch]

### paddle  [decoded] conf1=98% conf2=93%
- backend: paddle-onnxruntime  (190 ms)
- error: decode failed: check digit mismatch
GT  : P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPSVKSPECIMEN<<VZORKA<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^      ^    ^^                              [5 mismatch]
GT  : 7B12318830SVK1111112F34091871111111111<<<<06
PRED: XR12318830SVK111112F234091871111111<<06<<<<<
      ^^                ^^^              ^^^^   ^^   [11 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=99%
- backend: indra  (54 ms)
GT  : P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPSVKSPECIMENK<VZORKAK<<<<<<<<<<<<<<<<<<<<<<
       ^ ^      ^  ^ ^^    ^                         [7 mismatch]
GT  : 7B12318830SVK1111112F34091871111111111<<<<06
PRED: XR12318830SVK1111112F34091871111111111<<<<06
      ^^                                             [2 mismatch]

### indra_fullpage  [decoded] conf1=92% conf2=98%
- backend: indra  (236 ms)
- error: lines=61 edits=4 composite_edits=1 country=PAU
GT  : P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<
PRED: 13PAUNZZNAMYPAGERESERVEDFORSSUINGAURHEITIESP
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]
GT  : 7B12318830SVK1111112F34091871111111111<<<<06
PRED: COMPTENTE6PAURDLIVR5RLEPASS6PO<<<<<<<<<<<<72
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^   [40 mismatch]

## cd83b43756aff32d_0013.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=155
- 官方 line1: P<HRVSPECIMEN<<SPECIMEN<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 0070070071HRV8212258F1407019<<<<<<<<<<<<<<06
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `0M10070071HAV8212258F140701955535SSKKKKKKK56`

### traditional  [decoded] conf1=69% conf2=65%
- error: decode failed: check digit mismatch
GT  : P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<
PRED: 7F<HHVSPECINEN<<9FECIYEY<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^ ^^ ^^^^^^^^^                       [22 mismatch]
GT  : 0M10070071HAV8212258F140701955535SSKKKKKKK56
PRED: 4<<<0M<C7C374HHV624223BPM437C49<<<<<<<<<<<<<
      ^^^^ ^^^ ^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^   [41 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=96% conf2=94%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch
GT  : P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<
PRED: P<HRVSPECIMEN<<SPECIMEN<<<<<<<<<<<<<<<<<<<<<
         ^^  ^  ^^     ^  ^^                         [8 mismatch]
GT  : 0M10070071HAV8212258F140701955535SSKKKKKKK56
PRED: 0070070071HRV8212258F1<07019<<<<<<<<<<<<<<06
       ^^        ^          ^     ^^^^^^^^^^^^^^^    [19 mismatch]

### fuse  [DECODE-OK] conf1=95% conf2=92%
GT  : P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<
PRED: P<HRYSYGCIYGN<<SPSGIMEN<<<<<<<<<<<<<<<<<<<<<
         ^^ ^   ^^     ^^ ^^                         [9 mismatch]
GT  : 0M10070071HAV8212258F140701955535SSKKKKKKK56
PRED: 0070070071HAV8212258F140701955<5<<<<<KKKKK56
       ^^                           ^^^^^^^          [9 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (77 ms)
GT  : P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<
PRED: P<HRVSPECIMEN<<SPECIMEN<<<<<<<<<<<<<<<<<<<<<
         ^^  ^  ^^     ^  ^^                         [8 mismatch]
GT  : 0M10070071HAV8212258F140701955535SSKKKKKKK56
PRED: 0070070071HRV8212258F1407019<<<<<<<<<<<<<<06
       ^^        ^                ^^^^^^^^^^^^^^^    [18 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=97%
- backend: paddle-onnxruntime  (91 ms)
GT  : P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<
PRED: P<HRVSPECIMEN<<SPECIMEN<<<<<<<<<<<<<<<<<<<<<
         ^^  ^  ^^     ^  ^^                         [8 mismatch]
GT  : 0M10070071HAV8212258F140701955535SSKKKKKKK56
PRED: 0070070071HRV8212258F1407019<<<<<<<<<<<<0682
       ^^        ^                ^^^^^^^^^^^^^^^^   [19 mismatch]

### indra  [DECODE-OK] conf1=92% conf2=97%
- backend: indra  (48 ms)
GT  : P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<
PRED: P<HRVSPECIMENK<SPECIMEN<<<<<<<<<<<<<<<<<<<<<
         ^^  ^  ^^ ^   ^  ^^                         [9 mismatch]
GT  : 0M10070071HAV8212258F140701955535SSKKKKKKK56
PRED: 00Z0070O71HRV8212258F1407019<<<<<<<<<<<<<<06
       ^^    ^   ^                ^^^^^^^^^^^^^^^    [19 mismatch]

### indra_fullpage  [DECODE-OK] conf1=91% conf2=98%
- backend: indra  (218 ms)
GT  : P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<
PRED: P<HRVSPECIMENK<SPECIMEN<<<<<<<<<<<<<<<<<<<<<
         ^^  ^  ^^ ^   ^  ^^                         [9 mismatch]
GT  : 0M10070071HAV8212258F140701955535SSKKKKKKK56
PRED: 0070070071HRV8212258F1407019<<<<<<<<<<<<<<06
       ^^        ^                ^^^^^^^^^^^^^^^    [18 mismatch]

## cdcc06ad360b7849_0019.png
- 官方判定: REJECT  verified=False  chosen=None  ms=26
- 理由: 未在照片中定位到 MRZ 双行

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [DECODE-OK] conf1=98% conf2=100%
- backend: indra  (258 ms)

## dd6a2ea3ce305a5d_0192.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=156
- 官方 line1: P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 6451331382PHK6609302F1003145Z<<<<<<<<<<<<<54
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `6451331382PRK6609302F1003145<<<<<<<<<<<<<<04`

### traditional  [decoded] conf1=70% conf2=71%
- error: decode failed: check digit mismatch
GT  : P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: ALAAPLAAWA<<NYCNG9LK<<<<<<<<<<<<<<Z<<<<<<<<<
      ^^^^^^ ^^^ ^^^^^^^^^              ^            [19 mismatch]
GT  : 6451331382PRK6609302F1003145<<<<<<<<<<<<<<04
PRED: A454334282PRWAAEV3C2GMCC34A9<<<<<<<<<<<<<<<A
      ^  ^  ^^    ^^^^^ ^ ^^^^ ^^^              ^^   [19 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=96% conf2=96%
- backend: crnn  (14 ms)
- error: decode failed: check digit mismatch
GT  : P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POPRKJANG<<MYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
       ^  ^      ^                                   [3 mismatch]
GT  : 6451331382PRK6609302F1003145<<<<<<<<<<<<<<04
PRED: 6451331382PRK6609302F1003145<<<<<<<<<<<<<<<4
                                                ^    [1 mismatch]

### fuse  [DECODE-OK] conf1=97% conf2=99%
GT  : P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : 6451331382PRK6609302F1003145<<<<<<<<<<<<<<04
PRED: 6451331382PHK6609302F1003145Z<<<<<<<<<<<<<54
                 ^                ^             ^    [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (76 ms)
GT  : P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<PRKJANG<<MYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
          ^      ^                                   [2 mismatch]
GT  : 6451331382PRK6609302F1003145<<<<<<<<<<<<<<04
PRED: 6451331382PRK6609302F1003145<<<<<<<<<<<<<<04
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=96%
- backend: paddle-onnxruntime  (86 ms)
GT  : P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POPRKJANG<<MYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
       ^  ^      ^                                   [3 mismatch]
GT  : 6451331382PRK6609302F1003145<<<<<<<<<<<<<<04
PRED: 6451331382PRK6609302F1003145<<<<<<<<<<<<<428
                                               ^^^   [3 mismatch]

### indra  [DECODE-OK] conf1=99% conf2=100%
- backend: indra  (60 ms)
GT  : P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POPRKJANG<<MYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<
       ^  ^      ^                                   [3 mismatch]
GT  : 6451331382PRK6609302F1003145<<<<<<<<<<<<<<04
PRED: 6451331382PRK6609302F1003145<<<<<<<<<<<<<<04
                                                     [0 mismatch]

### indra_fullpage  [FAIL]
- backend: indra  (221 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

## e350cea71e824345_0007.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=165
- 官方 line1: P<BGRIVANOVA<<MARITSA<RADNEVA<<<<<<<<<<<<<<<
- 官方 line2: 3000000001BGR8508018F15032098508010133<<<<54
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<`
- GT line2: `3000000001BGH8508018F150320945040M0M33NNNN54`

### traditional  [decoded] conf1=70% conf2=63%
- error: decode failed: check digit mismatch
GT  : P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<
PRED: MM<BGM1VAHCVTNAM1T9A<RADNEVA<<<<<<<<<<<<<<<M
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^              ^   [30 mismatch]
GT  : 3000000001BGH8508018F150320945040M0M33NNNN54
PRED: M3CCCCCCCC4DGM85C8CM8P45C32C985C8CMCM39<<<<5
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^   [43 mismatch]

### cnn  [DECODE-OK] conf1=95% conf2=83%
GT  : P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<
PRED: P<BGRDWANOWAV<NARDYBA<NADNDWA<<<<<<<<<<<<<<<
                       ^                             [1 mismatch]
GT  : 3000000001BGH8508018F150320945040M0M33NNNN54
PRED: 3000000001BGH8508018F150320945040M0M33NNNN54
                                                     [0 mismatch]

### crnn  [DECODE-OK] conf1=97% conf2=97%
- backend: crnn  (17 ms)
GT  : P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<
PRED: P<BGRIVANOVA<<HARITSA<RADNEVA<<<<<<<<<<<<<<<
           ^^   ^ ^ ^   ^^  ^   ^^                   [10 mismatch]
GT  : 3000000001BGH8508018F150320945040M0M33NNNN54
PRED: 3000000001BGR8508018F15032098508010133<<<<54
                  ^               ^  ^ ^ ^  ^^^^     [9 mismatch]

### fuse  [DECODE-OK] conf1=95% conf2=90%
GT  : P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<
PRED: P<BGRDWANOWAV<NARDYBA<RADNDWA<<<<<<<<<<<<<<<
                       ^    ^                        [2 mismatch]
GT  : 3000000001BGH8508018F150320945040M0M33NNNN54
PRED: 3000000001BGH8508018F150320985080M0M33NNR554
                                  ^  ^        ^^     [4 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (72 ms)
GT  : P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<
PRED: P<BGRIVANOVAS<MARITSA<RADNEVA<<<<<<<<<<<<<<<
           ^^   ^ ^ ^   ^^  ^   ^^                   [10 mismatch]
GT  : 3000000001BGH8508018F150320945040M0M33NNNN54
PRED: 3000000001BGR8508018F15032098508010133<<<<54
                  ^               ^  ^ ^ ^  ^^^^     [9 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (104 ms)
GT  : P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<
PRED: P<BGRIVANOVA<<MARITSA<RADNEVA<<<<<<<<<<<<<<<
           ^^   ^ ^ ^   ^^  ^   ^^                   [10 mismatch]
GT  : 3000000001BGH8508018F150320945040M0M33NNNN54
PRED: 3000000001BGR8508018F15032098508010133<<<<54
                  ^               ^  ^ ^ ^  ^^^^     [9 mismatch]

### indra  [DECODE-OK] conf1=95% conf2=99%
- backend: indra  (54 ms)
GT  : P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<
PRED: P<BGRIVANOVAKKMARITSAKRADNEVA<<<<<<<<<<<<<<<
           ^^   ^ ^^^   ^^ ^^   ^^                   [12 mismatch]
GT  : 3000000001BGH8508018F150320945040M0M33NNNN54
PRED: 3000000001BGR8508018F15032098508010133<<<<54
                  ^               ^  ^ ^ ^  ^^^^     [9 mismatch]

### indra_fullpage  [DECODE-OK] conf1=86% conf2=93%
- backend: indra  (213 ms)
GT  : P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<
PRED: YASAOREPUBLICOFBULGARIA<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                  [29 mismatch]
GT  : 3000000001BGH8508018F150320945040M0M33NNNN54
PRED: ACNOPRNPA5SAORTNOPA2SEPORT<3<<<<<<<<<<<<<<08
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]

## e564d4372e28b9ed_0018.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=174
- 官方 line1: P<SRBTEST<<MILICA<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: 0000000000SRB6802295F22101042902968000000<40
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<`
- GT line2: `0000000000SRB6802295F22101042902968000000<40`

### traditional  [decoded] conf1=72% conf2=57%
- error: decode failed: check digit mismatch
GT  : P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
PRED: MP<9<RBTE5T<<NILICA<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^              ^             [20 mismatch]
GT  : 0000000000SRB6802295F22101042902968000000<40
PRED: MCCCQCCCCOC9A<D<68O2295P<224C4<C429C29K8GCCO
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^   [43 mismatch]

### cnn  [DECODE-OK] conf1=89% conf2=86%
GT  : P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
PRED: P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
                                                     [0 mismatch]
GT  : 0000000000SRB6802295F22101042902968000000<40
PRED: 0000000000GHB6802295M22101042N02HB8000000N40
                ^^        ^        ^  ^^       ^     [7 mismatch]

### crnn  [decoded] conf1=97% conf2=91%
- backend: crnn  (13 ms)
- error: decode failed: check digit mismatch
GT  : P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
PRED: P<SRBTEST<<MILICA<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^  ^^^^  ^^ ^^                 ^             [10 mismatch]
GT  : 0000000000SRB6802295F22101042902968000000<40
PRED: 00000000006RB<802295F22101042902968000000<40
                ^  ^                                 [2 mismatch]

### fuse  [DECODE-OK] conf1=95% conf2=92%
GT  : P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
PRED: P<BRBTBBT<<NDLDCA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
           ^^ ^      ^                               [4 mismatch]
GT  : 0000000000SRB6802295F22101042902968000000<40
PRED: 0000000000GHB6802295M22101042N02HB8000000N40
                ^^        ^        ^  ^^       ^     [7 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (87 ms)
GT  : P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
PRED: P<SRBTEST<<MILICA<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^  ^^^^  ^^ ^^                 ^             [10 mismatch]
GT  : 0000000000SRB6802295F22101042902968000000<40
PRED: 0000000000SRB6802295F22101042902968000000<40
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=96%
- backend: paddle-onnxruntime  (82 ms)
GT  : P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
PRED: P<SRBTEST<<MILICA<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^  ^^^^  ^^ ^^                 ^             [10 mismatch]
GT  : 0000000000SRB6802295F22101042902968000000<40
PRED: 0000000000SRB6802295F22101042902968000000<40
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=88% conf2=97%
- backend: indra  (56 ms)
GT  : P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
PRED: PKSRBTESTMILICACKKKKKKKKK<<<<<<<<<<<<<<<<<<<
       ^^  ^^^^^^^^^^^^^^^^^^^^        ^             [23 mismatch]
GT  : 0000000000SRB6802295F22101042902968000000<40
PRED: 0000000000SRB6802295F22101042902968000000<40
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=97% conf2=91%
- backend: indra  (228 ms)
GT  : P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<
PRED: MUARSRBINEPUZAGRADBEOGRAD<<<<<<<<<<<<<<<<<<<
      ^^^ ^^^^^^^^^^^^ ^^^^^^^^        ^             [24 mismatch]
GT  : 0000000000SRB6802295F22101042902968000000<40
PRED: MORNUCSIG2ARSRESIGN6TU<<<<<0<<<<<<<<<<<<<<06
      ^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^   [42 mismatch]

## e726442c11085334_0090.png
- 官方判定: PASS  verified=True  chosen=crnn  ms=168
- 官方 line1: P<SWESPECIMEN<<SVEA<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: XA00000012SWE8208215F2701013198208212384<<22
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `XA00000012SWE8208215F2701013198208212384<<22`

### traditional  [decoded] conf1=71% conf2=70%
- error: decode failed: check digit mismatch
GT  : P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<9WEBPEC1NEN<<9VEA<<<<<<<<<<<<<<<<<<<<<<<<<
        ^ ^ ^^ ^ ^   ^^^                             [9 mismatch]
GT  : XA00000012SWE8208215F2701013198208212384<<22
PRED: XAOOCOCO429WM82C8249M27O4O434V82C8242384<<22
        ^^^^^^^ ^ ^  ^  ^^^  ^^^^ ^^  ^  ^           [21 mismatch]

### cnn  [decoded] conf1=79% conf2=82%
- error: decode failed: check digit mismatch
GT  : P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<BWBBPBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
            ^                                        [1 mismatch]
GT  : XA00000012SWE8208215F2701013198208212384<<22
PRED: NM00000012GWB4204214M27010141042042MMM44KN22
      ^^        ^ ^^  ^  ^^      ^ ^^  ^ ^^^^ ^^     [18 mismatch]

### crnn  [decoded] conf1=98% conf2=92%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SWESPECIMEN<<SVEA<<<<<<<<<<<<<<<<<<<<<<<<<
        ^ ^^^^  ^^   ^^^                             [10 mismatch]
GT  : XA00000012SWE8208215F2701013198208212384<<22
PRED: XA000000125WE8208215F2701013190200212384<<22
                ^                   ^  ^             [3 mismatch]

### fuse  [decoded] conf1=92% conf2=89%
- error: decode failed: check digit mismatch
GT  : P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<BWBBPBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
            ^                                        [1 mismatch]
GT  : XA00000012SWE8208215F2701013198208212384<<22
PRED: NA00000012GWB4204214M270101419M20M2M2M44KN22
      ^         ^ ^^  ^  ^^      ^  ^  ^ ^ ^^ ^^     [15 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (71 ms)
GT  : P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SWESPECIMEN<<SVEA<<<<<<<<<<<<<<<<<<<<<<<<<
        ^ ^^^^  ^^   ^^^                             [10 mismatch]
GT  : XA00000012SWE8208215F2701013198208212384<<22
PRED: XA00000012SWE8208215F2701013198208212384<<22
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (84 ms)
GT  : P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SWESPECIMEN<<SVEA<<<<<<<<<<<<<<<<<<<<<<<<<
        ^ ^^^^  ^^   ^^^                             [10 mismatch]
GT  : XA00000012SWE8208215F2701013198208212384<<22
PRED: XA00000012SWE8208215F2701013198208212384<<22
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=91% conf2=99%
- backend: indra  (57 ms)
GT  : P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<SWESPECIMEN<<SVEA<<<<<<<<<<<<<<<<<<<<<<<<<
        ^ ^^^^  ^^   ^^^                             [10 mismatch]
GT  : XA00000012SWE8208215F2701013198208212384<<22
PRED: XA00000012SWE8208215F2701013198208212384<<22
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=86% conf2=99%
- backend: indra  (190 ms)
GT  : P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: SVEAE<<<<<<<<<<<<<<<<<XXXXXXX<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^  ^^^^   ^^^^^^^                  [24 mismatch]
GT  : XA00000012SWE8208215F2701013198208212384<<22
PRED: XA00000012EAE8208215F2701013198208212384<<22
                ^^                                   [2 mismatch]

## f22471c46575ec04_0008.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=174
- 官方 line1: PPAUTSPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08`

### traditional  [decoded] conf1=73% conf2=70%
- error: decode failed: check digit mismatch
GT  : P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
PRED: MPPALT9PEEINEH<<NARIA<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^ ^^^^^^                          [20 mismatch]
GT  : AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
PRED: MAP42345K73AC1CC32335F33333C7<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^^ ^^ ^^^             ^^   [29 mismatch]

### cnn  [decoded] conf1=94% conf2=94%
- error: decode failed: check digit mismatch
GT  : P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
PRED: 8P12345673AUI0012315F3311307555555<<55<<<<<8
      ^           ^               ^^^^^^  ^^    ^    [11 mismatch]

### crnn  [decoded] conf1=95% conf2=94%
- backend: crnn  (17 ms)
- error: decode failed: check digit mismatch
GT  : P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPAUTSPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
       ^  ^^                                         [3 mismatch]
GT  : AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
PRED: AP12345673AUT0012315F3311307<<<<<<<<<<<<<<<0
                                                ^^   [2 mismatch]

### fuse  [decoded] conf1=96% conf2=96%
- error: decode failed: check digit mismatch
GT  : P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
PRED: AP12345673AUI0012315F3311307<<<<<<<<<<<<<<<8
                  ^                             ^    [2 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (73 ms)
GT  : P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPAUTSPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
       ^  ^^                                         [3 mismatch]
GT  : AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
PRED: AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=98%
- backend: paddle-onnxruntime  (98 ms)
GT  : P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPAUTSPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
       ^  ^^                                         [3 mismatch]
GT  : AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
PRED: AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=100%
- backend: indra  (53 ms)
GT  : P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPAUTSPECIMENK<MARIAK<<<<<X<<<<<<<<<<<<<<<<<
       ^  ^^       ^      ^     ^                    [6 mismatch]
GT  : AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
PRED: AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=92% conf2=100%
- backend: indra  (273 ms)
GT  : P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPAUTSPECIMENK<MARIA<<<<<<X<<<<<<<<<<<<<<<<<
       ^  ^^       ^            ^                    [5 mismatch]
GT  : AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
PRED: AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08
                                                     [0 mismatch]

## f385b0113d785dfd_0040.jpg
- 官方判定: REJECT  verified=False  chosen=None  ms=9
- 理由: 未在照片中定位到 MRZ 双行

- GT line1: `P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `ZF40004998P0L8101028M3203031<<<<<<<<<<<<<<00`

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [DECODE-OK] conf1=77% conf2=98%
- backend: indra  (189 ms)
GT  : P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PKPOLKOWALSKIKIANKKKKKX<<<<<<<<<<<<<<<<<<<<<
       ^           ^^^^^^^^^^                        [11 mismatch]
GT  : ZF40004998P0L8101028M3203031<<<<<<<<<<<<<<00
PRED: ZF40004998POL8101028M3203031<<<<<<<<<<<<<<00
                 ^                                   [1 mismatch]

## f6132d0c5422d26a_0026.jpg
- 官方判定: PASS  verified=True  chosen=crnn  ms=162
- 官方 line1: P<NOROESTENBYEN<<AASAMUND<SPECIMEN<<<<<<<<<<
- 官方 line2: CCC0022514NOR5604230M3004157<<<<<<<<<<<<<<04
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<`
- GT line2: `CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04`

### traditional  [decoded] conf1=70% conf2=73%
- error: decode failed: check digit mismatch
GT  : P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
PRED: M<HCMCL91EHBYEN<<AA5AHUHD<9PECIHEH<<<<<<<<<<
      ^ ^^^^^^^^^  ^     ^ ^ ^  ^    ^ ^             [17 mismatch]
GT  : CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
PRED: CCCCC22534WCA5KC423CH3CC4M57<<<<<<<<<<<<<<CA
         ^^   ^ ^^^ ^^   ^^ ^^ ^  ^   ^         ^^   [17 mismatch]

### cnn  [DECODE-OK] conf1=94% conf2=95%
GT  : P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
PRED: P<HORORSTRHBYRH<<AASAMUND<SPECINEN<<<<<<<<<<
        ^       ^   ^                                [3 mismatch]
GT  : CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
PRED: CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
                                                     [0 mismatch]

### crnn  [decoded] conf1=97% conf2=91%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
PRED: P<NOROESTENBYEN<<AASAMUND<SPECIMEN<<<<<<<<<<
            ^  ^   ^                 ^               [4 mismatch]
GT  : CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
PRED: CC00022514NOR5604230M3004157<<<<<<<<<<<<<<04
        ^       ^ ^               ^   ^              [5 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=96%
GT  : P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
PRED: P<NOROESTEHBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
            ^  ^^                                    [3 mismatch]
GT  : CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
PRED: CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (70 ms)
GT  : P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
PRED: P<NOROESTENBYEN<<AASAMUND<SPECIMEN<<<<<<<<<<
            ^  ^   ^                 ^               [4 mismatch]
GT  : CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
PRED: CCC0022514NOR5604230M3004157<<<<<<<<<<<<<<04
                ^ ^               ^   ^              [4 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=93%
- backend: paddle-onnxruntime  (87 ms)
GT  : P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
PRED: P<NOROESTENBYEN<<AASAMUND<SPECIMEN<<<<<<<<<<
            ^  ^   ^                 ^               [4 mismatch]
GT  : CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
PRED: CCC0022514NOR5604230M3004157<<<<<<<<<<<<0428
                ^ ^               ^   ^       ^^^^   [8 mismatch]

### indra  [DECODE-OK] conf1=97% conf2=97%
- backend: indra  (57 ms)
GT  : P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
PRED: P<NOROESTENBYEN<<AASAMUND<SPECIMEN<<<<<<<<<<
            ^  ^   ^                 ^               [4 mismatch]
GT  : CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
PRED: CCC0022514NOR5604230M3004157<<<<<<<<<<<<<<04
                ^ ^               ^   ^              [4 mismatch]

### indra_fullpage  [DECODE-OK] conf1=99% conf2=99%
- backend: indra  (216 ms)
GT  : P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<
PRED: NOPGENOREGNORGANORWAY<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^ ^^^^^^^^^^^^^^ ^^^^^^^^             [32 mismatch]
GT  : CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04
PRED: PASSNUMME2PGEANUMM<9<<<<<<<0<<<<<<<<<<<<<<04
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^              [30 mismatch]

## f6eeb483a66e8d94_0084.jpg
- 官方判定: PASS  verified=True  chosen=tesseract  ms=154
- 官方 line1: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E000093499USA6502056M3007099340006673<085950
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E000093499USA6502056M3007099340006673<085950`

### traditional  [decoded] conf1=55% conf2=55%
- error: decode failed: check digit mismatch
GT  : P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<J1LX<<AJCLAL<<<A<JX<<<<IL<TJ<T7<<<<<<T<<<<
      ^ ^^^^^^^^^^^^ ^^^^^^    ^^ ^^ ^^      ^       [26 mismatch]
GT  : E000093499USA6502056M3007099340006673<085950
PRED: CCCCCV3499C9AK5C2C5<6<N3G<C<7CV93<AGG<C<6K73
      ^^^^^^    ^^ ^ ^ ^ ^^^^^^^^^^^^^^^^^^ ^^^^^^   [35 mismatch]

### cnn  [decoded] conf1=75% conf2=75%
- error: decode failed: check digit mismatch
GT  : P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<KKKNWMHWEWWWJMWWW<MGHMGMMGWGDWWWWWWWWWWWWW
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [42 mismatch]
GT  : E000093499USA6502056M3007099340006673<085950
PRED: Q0000N04N9UBA6902096M3007098M400066X1N08NH40
      ^    ^^ ^  ^  ^   ^        ^^      ^^^  ^^^    [15 mismatch]

### crnn  [decoded] conf1=95% conf2=83%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
              ^                                      [1 mismatch]
GT  : E000093499USA6502056M3007099340006673<085950
PRED: E000093499U8A650205603007098340006673<065850
                 ^        ^      ^           ^ ^     [5 mismatch]

### fuse  [decoded] conf1=76% conf2=80%
- error: decode failed: check digit mismatch
GT  : P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<KKKNWMHWEWWWJMWWW<MGHMMMMGWGDWWWWWWWWWWWWW
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [42 mismatch]
GT  : E000093499USA6502056M3007099340006673<085950
PRED: 10000N04N9USA6902096M3007098M400066X1408NM40
      ^    ^^ ^     ^   ^        ^^      ^^^  ^^^    [14 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (70 ms)
GT  : P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
              ^                                      [1 mismatch]
GT  : E000093499USA6502056M3007099340006673<085950
PRED: E000093499USA6502056M3007099340006673<085950
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=98%
- backend: paddle-onnxruntime  (76 ms)
GT  : P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
              ^                                      [1 mismatch]
GT  : E000093499USA6502056M3007099340006673<085950
PRED: E000093499USA6502056M3007099340006673<085950
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=97%
- backend: indra  (53 ms)
GT  : P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
              ^                                      [1 mismatch]
GT  : E000093499USA6502056M3007099340006673<085950
PRED: E000093499USA6502056M3007099340006673<085950
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=100% conf2=91%
- backend: indra  (181 ms)
GT  : P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<
PRED: UNIEEDSTATESDEPARTMENTOFSTATE<<<<<<<<<<<<<<<
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^                  [29 mismatch]
GT  : E000093499USA6502056M3007099340006673<085950
PRED: P<USATRAV0IEE<<HAPP7<<<<<<<0<<<<<<<<<<<<<<00
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^    [42 mismatch]

## f80334b200240666_0034.png
- 官方判定: PASS  verified=True  chosen=tesseract  ms=198
- 官方 line1: P<ARGGONZALES<RODRIGUEZ<MARIANA<FLORENCIA<<<
- 官方 line2: AAMO000000ARG5702147M3303023000000000<<<<<04
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<`
- GT line2: `8880000008ABG5702147M3303023000000000<<<<<08`

### traditional  [decoded] conf1=70% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<
PRED: P<AMGGCH7ALC3<MDDM1GLE2<NAM1AHA<FLDALWL1A<<<
         ^  ^^^  ^^ ^^ ^^ ^ ^ ^ ^^ ^  ^ ^^^^^^       [23 mismatch]
GT  : 8880000008ABG5702147M3303023000000000<<<<<08
PRED: AAHCCCCCCCAAG57C24A7P3XC3C23CLCCCLMML<<<<<LA
      ^^^^^^^^^^ ^   ^ ^^ ^ ^^ ^  ^^^^^^^^^     ^^   [29 mismatch]

### cnn  [DECODE-OK] conf1=90% conf2=96%
GT  : P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<
PRED: P<AGGGOHIALES<YODRIGUEZ<MARIANA<FLDNENCLN<<<
         ^   ^^     ^                 ^ ^^^  ^^      [10 mismatch]
GT  : 8880000008ABG5702147M3303023000000000<<<<<08
PRED: 84M0000000AHG5702147M330302300000M0D0K<K<K16
       ^^      ^ ^                     ^ ^ ^ ^ ^^^   [11 mismatch]

### crnn  [decoded] conf1=99% conf2=91%
- backend: crnn  (15 ms)
- error: decode failed: check digit mismatch
GT  : P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<
PRED: P<ARGGONZALES<RODRIGUEZ<MARIANA<FLORENCIA<<<
         ^                            ^   ^          [3 mismatch]
GT  : 8880000008ABG5702147M3303023000000000<<<<<08
PRED: AAM0000000AB05702147M3303023000000000<<<<<06
      ^^^      ^  ^                              ^   [6 mismatch]

### fuse  [decoded] conf1=94% conf2=96%
- error: decode failed: check digit mismatch
GT  : P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<
PRED: P<ARGGONIALES<RODRIGUEZ<MARIANA<FLDNENCLN<<<
         ^    ^                       ^ ^^^  ^^      [8 mismatch]
GT  : 8880000008ABG5702147M3303023000000000<<<<<08
PRED: 8AM0000000AHG5702147M3303023000000000K<K<<06
       ^^      ^ ^                         ^ ^   ^   [7 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (131 ms)
GT  : P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<
PRED: P<ARGGONZALES<RODRIGUEZ<MARIANA<FLORENCIA<<<
         ^                            ^   ^          [3 mismatch]
GT  : 8880000008ABG5702147M3303023000000000<<<<<08
PRED: AAMO000000ARG5702147M3303023000000000<<<<<04
      ^^^^     ^ ^                               ^   [7 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=90%
- backend: paddle-onnxruntime  (83 ms)
GT  : P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<
PRED: P<ARGGONZALES<RODRIGUEZ<MARIANA<FLORENCIA<<<
         ^                            ^   ^          [3 mismatch]
GT  : 8880000008ABG5702147M3303023000000000<<<<<08
PRED: AAMO000000ARG5702147M330302300000000<<<06<28
      ^^^^     ^ ^                        ^  ^^ ^    [10 mismatch]

### indra  [DECODE-OK] conf1=86% conf2=99%
- backend: indra  (57 ms)
GT  : P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<
PRED: PKARGGONZALESKRODRIGUEZKMARIANA<FLORENCIA<<<
       ^ ^         ^         ^        ^   ^          [6 mismatch]
GT  : 8880000008ABG5702147M3303023000000000<<<<<08
PRED: AAM000O000ARG5702147M3303023000000000<<<<<04
      ^^^   ^  ^ ^                               ^   [7 mismatch]

### indra_fullpage  [DECODE-OK] conf1=99% conf2=94%
- backend: indra  (196 ms)
GT  : P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<
PRED: LUGANDENACIMIENTOPLACEOFBIRTH<<<<<<<<<<<<<<<
      ^^^^^^^ ^^^^^^^^^^^^^ ^^^^ ^^^^ ^^^^^^^^^      [37 mismatch]
GT  : 8880000008ABG5702147M3303023000000000<<<<<08
PRED: FOCHADEEM3GANDATEOF5SS<<<<<6<<<<<<<<<<<<<<06
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^   [38 mismatch]

## ffa1dc640314fa58_0089.png
- 官方判定: PASS  verified=True  chosen=tesseract  ms=171
- 官方 line1: P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
- 官方 line2: FX12345672FIN7112214F2201018211271<426U<<<00
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<`
- GT line2: `FX12345672FIN7112214F2201018211271<426U<<<00`

### traditional  [decoded] conf1=72% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
PRED: A<MXWWXWTANEN<<NAHXA<QLIVIA<<<<<<<<<<<<<<<<<
      ^ ^^^^^^       ^ ^^  ^                         [11 mismatch]
GT  : FX12345672FIN7112214F2201018211271<426U<<<00
PRED: AW423A9A72PXN7MMPP44P2PD3D4N744274<A2AC<<<MA
      ^^^  ^^^  ^^  ^^^^^ ^ ^^^^^^^^^  ^ ^ ^^   ^^   [29 mismatch]

### cnn  [decoded] conf1=88% conf2=76%
- error: decode failed: check digit mismatch
GT  : P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
PRED: P<FDNDDBYANBN<<NARJA<OLDNDA<<<<<<<<<<<<<<<<<
         ^ ^^^^  ^   ^  ^    ^^^                     [11 mismatch]
GT  : FX12345672FIN7112214F2201018211271<426U<<<00
PRED: 8BRPM4M6M2MMN7110214M22010182MM27MN428B<<800
      ^^^^^ ^ ^ ^^    ^   ^        ^^  ^^  ^^  ^     [18 mismatch]

### crnn  [DECODE-OK] conf1=96% conf2=94%
- backend: crnn  (15 ms)
GT  : P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
PRED: P<FINVIRIANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
              ^                                      [1 mismatch]
GT  : FX12345672FIN7112214F2201018211271<426U<<<00
PRED: PX12345672FIN7112214F2201018211271<4260<<<00
      ^                                     ^        [2 mismatch]

### fuse  [DECODE-OK] conf1=92% conf2=85%
GT  : P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
PRED: P<FDNDDBYANBN<<NARJA<OLDNDA<<<<<<<<<<<<<<<<<
         ^ ^^^^  ^   ^  ^    ^^^                     [11 mismatch]
GT  : FX12345672FIN7112214F2201018211271<426U<<<00
PRED: OBRPM4M672MMN7112214M220101821127NN426B<<800
      ^^^^^ ^   ^^        ^            ^^   ^  ^     [13 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (89 ms)
GT  : P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
PRED: P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : FX12345672FIN7112214F2201018211271<426U<<<00
PRED: FX12345672FIN7112214F2201018211271<426U<<<00
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (84 ms)
GT  : P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
PRED: P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : FX12345672FIN7112214F2201018211271<426U<<<00
PRED: FX12345672FIN7112214F2201018211271<426U<<<00
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=93% conf2=98%
- backend: indra  (55 ms)
GT  : P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
PRED: P<FINVIRTANEN<<MARIAKOLIVIA<<<<<<<<<<<<<<<<<
                          ^                          [1 mismatch]
GT  : FX12345672FIN7112214F2201018211271<426U<<<00
PRED: FX12345672FIN7112214F2201018211271<426U<<<00
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=93% conf2=54%
- backend: indra  (227 ms)
GT  : P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<
PRED: SUOMIFINLANDFINLANDFINLANDE<<<<<<<<<<<<<<<<<
      ^^^^^^ ^^  ^^^^^ ^^^^^ ^^^^                    [22 mismatch]
GT  : FX12345672FIN7112214F2201018211271<426U<<<00
PRED: PASSA1D045OMIALOPOR1HOPASSI1ORT<<<<<<<<<<<84
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^   ^^   [40 mismatch]

## 2026-9-29-A.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=442
- 官方 line1: P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E0132B7855CHN0110141M3105179GB682050708G0B88
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E0132B7855CHN0110141M3105179GB682050708G0B88`

### traditional  [decoded] conf1=69% conf2=65%
- error: decode failed: check digit mismatch
GT  : P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: F<<<AAFLA<<F<D<D4SS7<<<<<<<<TSSF4<<<<<<<<<<<
      ^ ^^^^^^^ ^^^^^^^^^^        ^^^^^              [23 mismatch]
GT  : E0132B7855CHN0110141M3105179GB682050708G0B88
PRED: 2<JSSL<Z<<L<<A<44343NT3VS37V4AFASLLL<LLLLLLK
      ^^^^^^^^^^^^^^^^^^ ^^^^^^^ ^^^^^^^^^^^^^^^^^   [42 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=97%
GT  : P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E0132B7855CHN0110141M3105179GB682050708G0B88
PRED: E0132B7855CHN0110141M3105179GB682050708G0B88
                                                     [0 mismatch]

### crnn  [decoded] conf1=89% conf2=84%
- backend: crnn  (17 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PP<C<AYOA<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^ ^^^^^^                            ^   [15 mismatch]
GT  : E0132B7855CHN0110141M3105179GB682050708G0B88
PRED: E0132B7655CAN0110141M3105179GB682O5O108GQ880
             ^   ^                     ^ ^^   ^^ ^   [8 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E0132B7855CHN0110141M3105179GB682050708G0B88
PRED: E0132B7855CHN0110141M3105179GB682050708G0B88
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (372 ms)
GT  : P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHMNYUN<<YAOYU<<<<S<S<<<S<<<<<<<<<<<<<<<<<
          ^^^^^ ^^^^^^    ^ ^   ^                    [14 mismatch]
GT  : E0132B7855CHN0110141M3105179GB682050708G0B88
PRED: E0Z3Z87855CHN0110141M3105179G6B682050108G014
        ^ ^^                       ^^^^^^^^^^^^^^^   [18 mismatch]

### paddle  [DECODE-OK] conf1=95% conf2=99%
- backend: paddle-onnxruntime  (129 ms)
GT  : P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E0132B7855CHN0110141M3105179GB682050708G0B88
PRED: E0132B7855CHN0110141M3105179GB682O5O108G0B88
                                       ^ ^^          [3 mismatch]

### indra  [DECODE-OK] conf1=75% conf2=81%
- backend: indra  (66 ms)
GT  : P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PRCHNUNYADYUKKKKKKKKKKKKKKKKK<<<<<<<<<<<<<<<
       ^   ^^^^^ ^^^^^^^^^^^^^^^^^^                  [24 mismatch]
GT  : E0132B7855CHN0110141M3105179GB682050708G0B88
PRED: E043287855CHN0110141M3105179686820501086O088
        ^  ^                      ^^      ^  ^^^     [8 mismatch]

### indra_fullpage  [DECODE-OK] conf1=65% conf2=77%
- backend: indra  (227 ms)
GT  : P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <CCHNUNYADYUKKKKKKSSSSSSKKKKKKK<<<<<<<<<<<<<
      ^^   ^^^^^ ^^^^^^^^^^^^^^^^^^^^                [27 mismatch]
GT  : E0132B7855CHN0110141M3105179GB682050708G0B88
PRED: EO43267855CHN0110141M31051796868205010860002
       ^^  ^                      ^^      ^  ^ ^^^   [10 mismatch]

## 2026-9-29-B.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=262
- 官方 line1: P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E216923764CHN6703222M3510203H01JE6IP966FG406
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E216923764CHN6703222M3510203H01JE6IP966FG406`

### traditional  [decoded] conf1=70% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PA<CAA2AXC<<YCNG<<<<<ZZ<<<<44444444444444<4<
       ^^^^^^ ^^ ^^^^^     ^^    ^^^^^^^^^^^^^^ ^    [30 mismatch]
GT  : E216923764CHN6703222M3510203H01JE6IP966FG406
PRED: L27KV23Z6<CLK6ZC3222W394C2C3GC33L51L9EEYLGQL
      ^ ^^^  ^ ^ ^^ ^^    ^ ^^^ ^ ^^^^^^^^ ^^^^^^^   [30 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E216923764CHN6703222M3510203H01JE6IP966FG406
PRED: E216923764CHN6703222M3510203H01JE6IP966FG406
                                                     [0 mismatch]

### crnn  [decoded] conf1=90% conf2=87%
- backend: crnn  (20 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PP<CAAZAAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^ ^^ ^^^^^                            ^   [14 mismatch]
GT  : E216923764CHN6703222M3510203H01JE6IP966FG406
PRED: E276923764CHN6703222M3510203H01JE6IP96GYGQS6
        ^                                   ^^ ^^    [5 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E216923764CHN6703222M3510203H01JE6IP966FG406
PRED: E216923764CHN6703222M3510203H01JE6IP966FG406
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (223 ms)
GT  : P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E216923764CHN6703222M3510203H01JE6IP966FG406
PRED: E216923764CHN6703222M3510203H014E61P966FTG32
                                     ^  ^     ^^^^   [6 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (129 ms)
GT  : P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E216923764CHN6703222M3510203H01JE6IP966FG406
PRED: E216923764CHN6703222M3510203H01JE6IP966FG406
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=86%
- backend: indra  (64 ms)
GT  : P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHA0<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              ^                                      [1 mismatch]
GT  : E216923764CHN6703222M3510203H01JE6IP966FG406
PRED: E216923764CHN6703222M3510203H01JE6IP966EGL06
                                             ^ ^     [2 mismatch]

### indra_fullpage  [DECODE-OK] conf1=94% conf2=87%
- backend: indra  (261 ms)
GT  : P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHA0<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
              ^                                      [1 mismatch]
GT  : E216923764CHN6703222M3510203H01JE6IP966FG406
PRED: E216923764CHN6703222M3510203H01JE618966FG406
                                        ^^           [2 mismatch]

## 2026-9-29-C.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=263
- 官方 line1: P<CHNZHOU<<YUANSU<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E639O21058CHN7005059F3110153050268SZ0B110074
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E639021O58CHN7005059F31101530S0Z68SZ0B110074`

### traditional  [decoded] conf1=67% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P4<CAA2ACL<<LJI3L4L<444444444444444444444444
       ^^^^^^^^^ ^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^   [41 mismatch]
GT  : E639021O58CHN7005059F31101530S0Z68SZ0B110074
PRED: L<JVC2VDSALAA2AC9CS9P334C453D5Q1K891LA41L43L
      ^^^^^ ^^^^^^^^^^^^^ ^ ^^^^  ^^^^^ ^^^^^ ^^^^   [37 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=93%
GT  : P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHOU<<YUANSU<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : E639021O58CHN7005059F31101530S0Z68SZ0B110074
PRED: E639021O58CHN7005059F31101530S0Z68SZ0B110074
                                                     [0 mismatch]

### crnn  [decoded] conf1=85% conf2=86%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<AZH00<<Y0ANS0<<<<<<<<<<<<<<<<<<<<<<<<<<<  
        ^^^^^^^^^^^^^^^                         ^^   [17 mismatch]
GT  : E639021O58CHN7005059F31101530S0Z68SZ0B110074
PRED: E639027958CHA7005059F3110153O50Z68SZS811SS7E
            ^^    ^               ^^      ^^  ^^ ^   [10 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHOU<<YUANSU<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : E639021O58CHN7005059F31101530S0Z68SZ0B110074
PRED: E639O21058CHN7005059F3110153050268SZ0B110074
          ^  ^                     ^ ^               [4 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (204 ms)
GT  : P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHOU<<YUANSU<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : E639021O58CHN7005059F31101530S0Z68SZ0B110074
PRED: 63902710S8CHN7005059F31101530S0Z68SZ0B110074
      ^^^^^^ ^^                                      [8 mismatch]

### paddle  [DECODE-OK] conf1=66% conf2=99%
- backend: paddle-onnxruntime  (128 ms)
GT  : P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHOU<<YUANSU<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : E639021O58CHN7005059F31101530S0Z68SZ0B110074
PRED: E639O21058CHN7005059F3110153050Z68SZ0B1I0O74
          ^  ^                     ^         ^ ^     [5 mismatch]

### indra  [DECODE-OK] conf1=88% conf2=83%
- backend: indra  (58 ms)
GT  : P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHOU<<YUANSU<<KK<<<<<<<<<<<<<<<<<<<<<<<
                    ^    ^^                          [3 mismatch]
GT  : E639021O58CHN7005059F31101530S0Z68SZ0B110074
PRED: E639021O58CHN7005059F311015305026B9208110074
                                   ^ ^ ^^^ ^         [6 mismatch]

### indra_fullpage  [DECODE-OK] conf1=92% conf2=85%
- backend: indra  (235 ms)
GT  : P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHOU<<YUANSU<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : E639021O58CHN7005059F31101530S0Z68SZ0B110074
PRED: E639021O58CHN7005059F311015305026BSL08110074
                                   ^ ^ ^ ^ ^         [5 mismatch]

## 2026-9-29-D.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=252
- 官方 line1: P<CHNBAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E797044113CHN9004300F3405226GE0660ZR06BA1310
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E797044113CHN9004300F3405226GE0660IR06BAI310`

### traditional  [decoded] conf1=70% conf2=69%
- error: decode failed: bad MRZ length
GT  : P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <ZZAAAXO<<HUAN<<<<<<444444444444<<<<<<<<<4<<
      ^^^^^^^             ^^^^^^^^^^^^         ^     [20 mismatch]
GT  : E797044113CHN9004300F3405226GE0660IR06BAI310
PRED: <2V2D<<JJ3CHAVCC43CCP3AC52UGCQKKQ1AQKAL114M 
      ^^^^^^^^^   ^^^^  ^^^ ^^  ^^^^^^^^^^^^^^^^^^   [36 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E797044113CHN9004300F3405226GE0660IR06BAI310
PRED: E797044113CHN9004300F3405226GE0660IR06BAI310
                                                     [0 mismatch]

### crnn  [decoded] conf1=91% conf2=88%
- backend: crnn  (18 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PP<CAABAO<<A0AN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^ ^^^^^                             ^   [14 mismatch]
GT  : E797044113CHN9004300F3405226GE0660IR06BAI310
PRED: 497O44773CHM9004300F3405226GE066O1RQ6BRI31Q 
      ^^^^^ ^^^^^^^^ ^^^ ^^^^^^ ^^^^^ ^^^^^^^^^^^^   [39 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=97%
GT  : P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E797044113CHN9004300F3405226GE0660IR06BAI310
PRED: E797044113CHN9004300F3405226GE0660ZR06BA1310
                                        ^     ^      [2 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (130 ms)
GT  : P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E797044113CHN9004300F3405226GE0660IR06BAI310
PRED: E797044113CHN9004300F3405226GE6601IR06BA1310
                                    ^ ^^      ^      [4 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=98%
- backend: paddle-onnxruntime  (147 ms)
GT  : P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E797044113CHN9004300F3405226GE0660IR06BAI310
PRED: E797044113CHN9004300F3405226GE0660IR06BAI310
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=86%
- backend: indra  (68 ms)
GT  : P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E797044113CHN9004300F3405226GE0660IR06BAI310
PRED: E797044113CHN9004300F3405226GE06601R06RA1310
                                        ^   ^ ^      [3 mismatch]

### indra_fullpage  [DECODE-OK] conf1=95% conf2=87%
- backend: indra  (253 ms)
GT  : P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E797044113CHN9004300F3405226GE0660IR06BAI310
PRED: E797044113CHN9004300F3405226GE06601R06RA1310
                                        ^   ^ ^      [3 mismatch]

## 2026-9-29-E.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=494
- 官方 line1: P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E454243381CHN0703013F4211172S6310BB626637626
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E454243381CHN0703013F4211172S6310BB626637626`

### traditional  [decoded] conf1=70% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAACAD<<YCUYCU<44444444444444444444444<4<<
      ^  ^^ ^^   ^  ^  ^^^^^^^^^^^^^^^^^^^^^^^ ^     [31 mismatch]
GT  : E454243381CHN0703013F4211172S6310BB626637626
PRED: <<J<2<33AJCANCZC3O43P42343728634QDA61KK13G1L
      ^^^^ ^  ^^ ^ ^^^ ^^ ^  ^^^  ^  ^^^^ ^^^^^^^^   [30 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=95%
GT  : P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E454243381CHN0703013F4211172S6310BB626637626
PRED: E454243381CHN0703013F4211172S6310BB626637626
                                                     [0 mismatch]

### crnn  [decoded] conf1=96% conf2=86%
- backend: crnn  (20 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CAC0<<YO0YOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
         ^^^^^^^^^^^^^                          ^^   [15 mismatch]
GT  : E454243381CHN0703013F4211172S6310BB626637626
PRED: E454243387CHM0703013F4211172S631O8B6Z6G37S26
               ^  ^                   ^^  ^ ^  ^     [7 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E454243381CHN0703013F4211172S6310BB626637626
PRED: E454243381CHN0703013F4211172S6310BB626637626
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (398 ms)
GT  : P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E454243381CHN0703013F4211172S6310BB626637626
PRED: 4542433812HNO0703013F4211172S63I08B671663762
      ^^^^^^ ^^^^^^                  ^ ^  ^^ ^^^^^   [21 mismatch]

### paddle  [DECODE-OK] conf1=94% conf2=98%
- backend: paddle-onnxruntime  (144 ms)
GT  : P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E454243381CHN0703013F4211172S6310BB626637626
PRED: E454243381CHN0703013F4211172S63108B6Z6637626
                                       ^  ^          [2 mismatch]

### indra  [DECODE-OK] conf1=92% conf2=90%
- backend: indra  (66 ms)
GT  : P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E454243381CHN0703013F4211172S6310BB626637626
PRED: E454243381CHN0703013F4211172S63108862663T626
                                       ^^     ^      [3 mismatch]

### indra_fullpage  [DECODE-OK] conf1=94% conf2=85%
- backend: indra  (288 ms)
GT  : P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E454243381CHN0703013F4211172S6310BB626637626
PRED: E454243381CHN0703013F4211172S63I088676637626
                                     ^ ^^ ^          [4 mismatch]

## 2026-9-29-F.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=292
- 官方 line1: P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E410968B47CHN9311103F311203262G49S6626Z92252
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E410968B47CHN9311103F311203262G49S6626Z92252`

### traditional  [decoded] conf1=70% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAADEAG<<PENGWEN<<4444444444444<<<<<<<<<<<
      ^  ^^  ^            ^^^^^^^^^^^^^              [17 mismatch]
GT  : E410968B47CHN9311103F311203262G49S6626Z92252
PRED: Z<7DA<DA<ZCHNV3444DJPJ33PLQ3242444AAA1G14114
      ^^^^^^^^^^   ^ ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^   [39 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=95%
GT  : P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E410968B47CHN9311103F311203262G49S6626Z92252
PRED: E410968B47CHN9311103F311203262G49S6626Z92252
                                                     [0 mismatch]

### crnn  [decoded] conf1=94% conf2=90%
- backend: crnn  (19 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CADEMG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<< 
         ^^^^^^ ^^^^^^^^                         ^   [15 mismatch]
GT  : E410968B47CHN9311103F311203262G49S6626Z92252
PRED: E<70968B47CHN9311103F311203262G49S662G292Z52
       ^^                                  ^^  ^     [5 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E410968B47CHN9311103F311203262G49S6626Z92252
PRED: E410968B47CHN9311103F311203262G49S6626Z92252
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (231 ms)
GT  : P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E410968B47CHN9311103F311203262G49S6626Z92252
PRED: EW70968B47CHN9311103F31120326ZG49S66Z6292152
       ^^                          ^      ^ ^  ^     [6 mismatch]

### paddle  [DECODE-OK] conf1=89% conf2=99%
- backend: paddle-onnxruntime  (132 ms)
GT  : P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E410968B47CHN9311103F311203262G49S6626Z92252
PRED: E410968B47CHN9311103F31120326ZG49S6626292252
                                   ^        ^        [2 mismatch]

### indra  [DECODE-OK] conf1=95% conf2=84%
- backend: indra  (70 ms)
GT  : P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E410968B47CHN9311103F311203262G49S6626Z92252
PRED: E410968B47CHN9311103F311203262649966Z6292752
                                    ^  ^  ^ ^  ^     [5 mismatch]

### indra_fullpage  [DECODE-OK] conf1=95% conf2=85%
- backend: indra  (293 ms)
GT  : P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E410968B47CHN9311103F311203262G49S6626Z92252
PRED: E410968B47CHN9311103F31120326264996628292<52
                                    ^  ^   ^^  ^     [5 mismatch]

## 2026-9-29-G.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=259
- 官方 line1: P<CHNDING<<PIN<MO<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E505029416CHN6806178M2604114L2Y8F2Z01K4IF010
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E505029416CHN6806178M2604114L2YBF2201K4IF010`

### traditional  [decoded] conf1=70% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AZZAAAING<<PIN6NO444444444444444<<<<<<<<<<<<
      ^^^^^^        ^^ ^^^^^^^^^^^^^^^               [23 mismatch]
GT  : E505029416CHN6806178M2604114L2YBF2201K4IF010
PRED: Z<DJD2A<J6CHN68CK478NPKD4434C2SDV21Q4<G1VA46
      ^^^^^ ^^^      ^^^  ^^^^ ^^ ^ ^^^ ^^^^^^^^^^   [31 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E505029416CHN6806178M2604114L2YBF2201K4IF010
PRED: E505029416CHN6806178M2604114L2YBF2201K4IF010
                                                     [0 mismatch]

### crnn  [decoded] conf1=91% conf2=84%
- backend: crnn  (17 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CADIAG<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<< 
         ^^^^^^ ^^^^^^^                          ^   [14 mismatch]
GT  : E505029416CHN6806178M2604114L2YBF2201K4IF010
PRED: C505029476CHM6806178M2604114L2Y8F2ZQ1K4IEO1O
      ^       ^   ^                  ^  ^^    ^^ ^   [9 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E505029416CHN6806178M2604114L2YBF2201K4IF010
PRED: E505029416CHN6806178M2604114L2Y8F2Z01K4IF010
                                     ^  ^            [2 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (118 ms)
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E505029416CHN6806178M2604114L2YBF2201K4IF010
PRED: E505029416CHN6806178M2604114L2Y8F2Z0IK41F010
                                     ^  ^ ^  ^       [4 mismatch]

### paddle  [DECODE-OK] conf1=88% conf2=99%
- backend: paddle-onnxruntime  (130 ms)
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E505029416CHN6806178M2604114L2YBF2201K4IF010
PRED: E505029416CHN6806178M2604114L2Y8F2Z01K4IF010
                                     ^  ^            [2 mismatch]

### indra  [DECODE-OK] conf1=92% conf2=83%
- backend: indra  (64 ms)
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDRN6<<PINGM0<<<<<<<<<<<<<<<<<<<<<<<<<<<
            ^ ^       ^                              [3 mismatch]
GT  : E505029416CHN6806178M2604114L2YBF2201K4IF010
PRED: E505029416CHN6806178M260411412Y8F2201K4T5010
                                  ^  ^       ^^      [4 mismatch]

### indra_fullpage  [DECODE-OK] conf1=93% conf2=84%
- backend: indra  (230 ms)
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGM0<<<<<<<<<<<<<<<<<<<<<<<<<<<
                      ^                              [1 mismatch]
GT  : E505029416CHN6806178M2604114L2YBF2201K4IF010
PRED: E505029416CHN6806178M2604114T2Y8F22<<<<<<<08
                                  ^  ^   ^^^^^^^^^   [11 mismatch]

## 2026-9-29-H.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=286
- 官方 line1: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E505029416CHN6806178M2604114L2Y8F2Z01K4IFO34
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034`

### traditional  [decoded] conf1=71% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <ZZAAA1AA<<M1NGNO4<444444444444444444<<<<<<<
      ^^^^^^^^^  ^^  ^ ^ ^^^^^^^^^^^^^^^^^^          [31 mismatch]
GT  : EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034
PRED: <<<4A<VX4ALAA68C6478N24C4334L230V21M4<G1VQ1L
      ^^^^^^^^^^^^^  ^ ^  ^ ^^ ^^   ^^^ ^^^^^ ^^^^   [32 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=97%
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034
PRED: E505029416CHN6806178M2604114L2Y8F2Z01K4IFO34
       ^                             ^      ^^ ^     [5 mismatch]

### crnn  [decoded] conf1=92% conf2=86%
- backend: crnn  (18 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PP<CAADIAG<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^^ ^^^^^^^                          ^   [17 mismatch]
GT  : EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034
PRED: E595029476CHN6806178M2604114L2Y8F2ZO1K4IYS34
       ^^     ^                      ^   ^  ^^^^     [9 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034
PRED: E505029416CHN6806178M2604114L2Y8F2101K4IF034
       ^                             ^  ^   ^^       [5 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (210 ms)
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034
PRED: F5050Z9416CHN6806178M2604114L2Y8F2Z0O1K41F22
      ^^   ^                         ^    ^^^^^^^^   [12 mismatch]

### paddle  [DECODE-OK] conf1=90% conf2=99%
- backend: paddle-onnxruntime  (124 ms)
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034
PRED: E505029416CHN6806178M2604114L2Y8F2ZO1K4IF034
       ^                             ^   ^  ^^       [5 mismatch]

### indra  [FAIL]
- backend: indra  (42 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=88% conf2=82%
- backend: indra  (264 ms)
GT  : P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PECHNDING<<FINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^         ^                                   [2 mismatch]
GT  : EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034
PRED: 5050294162CHN80617882604714221B2201K41T0<<34
      ^^^^^^^^^^   ^^^^^^ ^^^^^ ^^^^^^^^^^^^^^^^     [37 mismatch]

## 2026-9-29-I.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=260
- 官方 line1: P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E592235158CHN6905163M30012716G0B612258S2B846
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E592235158CHN6905163M30012716G0B612258S2B846`

### traditional  [decoded] conf1=73% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAAGE<<NDHUA<<<<<<<<<<<<4<4444<<<<<<<<<<<<
      ^  ^^    ^^               ^ ^^^^               [10 mismatch]
GT  : E592235158CHN6905163M30012716G0B612258S2B846
PRED: ZJ422JSJS8CHN6VO54K3N3QC42746GD8K1225A41AALL
      ^^^  ^^^^     ^^ ^^ ^ ^^^  ^  ^^^    ^^^^^^^   [26 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=95%
GT  : P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E592235158CHN6905163M30012716G0B612258S2B846
PRED: E592235158CHN6905163M30012716G0B612258S2B846
                                                     [0 mismatch]

### crnn  [decoded] conf1=90% conf2=90%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<C4GE<<MOAUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
         ^^^^ ^^^^^^                             ^   [11 mismatch]
GT  : E592235158CHN6905163M30012716G0B612258S2B846
PRED: E592235758CHM6905163M30012716GOB6I2258S28846
             ^    ^                 ^  ^      ^      [5 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E592235158CHN6905163M30012716G0B612258S2B846
PRED: E592235158CHN6905163M30012716G0B612258S2B846
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (126 ms)
GT  : P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E592235158CHN6905163M30012716G0B612258S2B846
PRED: E592235158CHN6905163M300127106G0B612258S2384
                                  ^^^^^^^ ^^^^^^^^   [15 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (130 ms)
GT  : P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E592235158CHN6905163M30012716G0B612258S2B846
PRED: E592235158CHN6905163M30012716G0B61Z258S28846
                                        ^     ^      [2 mismatch]

### indra  [FAIL]
- backend: indra  (38 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=95% conf2=85%
- backend: indra  (253 ms)
GT  : P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E592235158CHN6905163M30012716G0B612258S2B846
PRED: E592235158CHN6905163M30012716608612258928B46
                                   ^ ^      ^ ^^     [5 mismatch]

## 2026-9-29-J.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=280
- 官方 line1: P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E062480684CHN8201113F251225501S8IBBG51B52Z66
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E062480684CHN8201113F251225501S8IBBG51B52Z66`

### traditional  [decoded] conf1=71% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<CANGE<<5ODMNG<<<4444444444<444<<44444444<<
      ^  ^     ^^ ^     ^^^^^^^^^^ ^^^  ^^^^^^^^     [26 mismatch]
GT  : E062480684CHN8201113F251225501S8IBBG51B52Z66
PRED: ZD<2<DDAA<CHA82C4433P753P255Q38B1884548511KK
      ^^^ ^^^^^^  ^  ^^^^ ^^ ^^   ^^^^^^^^ ^^ ^^^^   [32 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E062480684CHN8201113F251225501S8IBBG51B52Z66
PRED: E062480684CHN8201113F251225501S8IBBG51B52Z66
                                                     [0 mismatch]

### crnn  [decoded] conf1=93% conf2=87%
- backend: crnn  (19 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CAAGE<<SUDONC<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^^         ^                                [3 mismatch]
GT  : E062480684CHN8201113F251225501S8IBBG51B52Z66
PRED: EO6248O684CHN8201113F251225501S8IBBG51852ZGG
       ^    ^                               ^   ^^   [5 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E062480684CHN8201113F251225501S8IBBG51B52Z66
PRED: E062480684CHN8201113F251225501S8IBBG51B52Z66
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (211 ms)
GT  : P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E062480684CHN8201113F251225501S8IBBG51B52Z66
PRED: F0624B0684CHN8201113F251225501S81BBGS1B52166
      ^    ^                          ^   ^    ^     [5 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (127 ms)
GT  : P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E062480684CHN8201113F251225501S8IBBG51B52Z66
PRED: E062480684CHN8201113F251225501S8IBBG51B52Z66
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (49 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=94% conf2=80%
- backend: indra  (268 ms)
GT  : P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E062480684CHN8201113F251225501S8IBBG51B52Z66
PRED: E062480684CHN8201113F251225501881BB651852Z66
                                    ^ ^  ^  ^        [4 mismatch]

## 2026-9-29-K.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=273
- 官方 line1: P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E916769I46CHN6008248M240122126QZG26600S16132
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E916769I46CHN6008248M240122126QZG26600S16132`

### traditional  [decoded] conf1=72% conf2=67%
- error: decode failed: bad MRZ length
GT  : P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAAMOM<<LUMU<<<<<4444444444444444444<<<<<<
      ^  ^^^^^    ^      ^^^^^^^^^^^^^^^^^^^         [26 mismatch]
GT  : E916769I46CHN6008248M240122126QZG26600S16132
PRED: ZAI84AI<0CAK6CC8248N24C422424D1A2KKM444L411 
      ^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^ ^^   [42 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=95%
GT  : P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E916769I46CHN6008248M240122126QZG26600S16132
PRED: E916769I46CHN6008248M240122126QZG26600S16132
                                                     [0 mismatch]

### crnn  [decoded] conf1=91% conf2=86%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CAAGO<<LORU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
         ^^ ^^ ^^^^^                             ^   [10 mismatch]
GT  : E916769I46CHN6008248M240122126QZG26600S16132
PRED: Z976769746CHA6008248M240122126O2G266QOS1G432
      ^ ^    ^    ^                 ^^    ^^  ^^     [10 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E916769I46CHN6008248M240122126QZG26600S16132
PRED: E916769I46CHN6008248M240122126QZG26600S16132
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (205 ms)
GT  : P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E916769I46CHN6008248M240122126QZG26600S16132
PRED: E916769I46CHNG008248M24012212602G26600SI6132
                   ^                ^^       ^       [4 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=98%
- backend: paddle-onnxruntime  (119 ms)
GT  : P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E916769I46CHN6008248M240122126QZG26600S16132
PRED: E916769I46CHN6008248M240122126QZG26600S16132
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (45 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=94% conf2=89%
- backend: indra  (257 ms)
GT  : P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E916769I46CHN6008248M240122126QZG26600S16132
PRED: E916769I46CHN6008248M24012212602626600SI6132
                                    ^^^      ^       [4 mismatch]

## 2026-9-29-L.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=273
- 官方 line1: P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E665282843CHN8807054M28102160012032119B08660
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E665282843CHN8807054M28102160012032119B08660`

### traditional  [decoded] conf1=73% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAAAD<<P1AGYCU<<<<4444444<<<<<<<<<<<<<<<<<
      ^  ^^^^  ^^^  ^     ^^^^^^^                    [16 mismatch]
GT  : E665282843CHN8807054M28102160012032119B08660
PRED: <<<42A2A<3CAN88DZQ54NPN3VP36QQ31QJ7114AQAGG4
      ^^^^ ^ ^^  ^   ^^^  ^^^^^^^ ^^^^^^^  ^^^^^^^   [32 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E665282843CHN8807054M28102160012032119B08660
PRED: E665282843CHN8807054M28102160012032119B08660
                                                     [0 mismatch]

### crnn  [decoded] conf1=94% conf2=86%
- backend: crnn  (19 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CAAGO<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^^ ^                                        [3 mismatch]
GT  : E665282843CHN8807054M28102160012032119B08660
PRED: E665282843CHM8807054M2810216001ZO321198Q866Q
                  ^                  ^^     ^^   ^   [6 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E665282843CHN8807054M28102160012032119B08660
PRED: E665282843CHN8807054M28102160012032119B08660
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (204 ms)
GT  : P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E665282843CHN8807054M28102160012032119B08660
PRED: F6652B2843CHN8807054M2810216O01Z032119808660
      ^    ^                      ^  ^      ^        [5 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=97%
- backend: paddle-onnxruntime  (129 ms)
GT  : P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E665282843CHN8807054M28102160012032119B08660
PRED: E665282843CHN8807054M28102160012032119B08660
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=88% conf2=86%
- backend: indra  (71 ms)
GT  : P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<CHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^    ^^  ^^^^^^^                               [10 mismatch]
GT  : E665282843CHN8807054M28102160012032119B08660
PRED: E665282843CHN8807054M28102160012032119B08660
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=89% conf2=84%
- backend: indra  (265 ms)
GT  : P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PECHNEUCKYINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^ ^^                                      [4 mismatch]
GT  : E665282843CHN8807054M28102160012032119B08660
PRED: E665282843CHN88070<9<<<<<<<0<<<<<<<<<<<<<<04
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^   [26 mismatch]

## 2026-9-29-M.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=271
- 官方 line1: P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E096730195CHN8410021M300821488S61022S7Z59328
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E096730195CHN8410021M300821488S61022S7Z59328`

### traditional  [decoded] conf1=71% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <ZLAAAL<<POPO<<44<444444444<<<444444<<<<<<<<
      ^^^^^^^  ^^^^  ^^ ^^^^^^^^^   ^^^^^^           [28 mismatch]
GT  : E096730195CHN8410021M300821488S61022S7Z59328
PRED: <DA<23AJAACAK8<VDO23NJQC024A88GK4Q22A314411A
      ^^^^^ ^^^^ ^^ ^^^^ ^^^^^^ ^^  ^^^^  ^^^^^^^^   [35 mismatch]

### cnn  [DECODE-OK] conf1=98% conf2=96%
GT  : P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E096730195CHN8410021M300821488S61022S7Z59328
PRED: E096730195CHN8410021M300821488S61V22S2Z59128
                                       ^   ^   ^     [3 mismatch]

### crnn  [decoded] conf1=87% conf2=89%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PP<CAAG<<Y0YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^   ^                                    [7 mismatch]
GT  : E096730195CHN8410021M300821488S61022S7Z59328
PRED: E096730795CHN8410021M300821488S64922S759328 
             ^                        ^^    ^^^^^^   [9 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=97%
GT  : P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E096730195CHN8410021M300821488S61022S7Z59328
PRED: E096730195CHN8410021M300821488S61022S7Z59328
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (203 ms)
GT  : P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E096730195CHN8410021M300821488S61022S7Z59328
PRED: E096730195CHN8410021M300821488S6102ZS71S9328
                                         ^  ^^       [3 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=99%
- backend: paddle-onnxruntime  (122 ms)
GT  : P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E096730195CHN8410021M300821488S61022S7Z59328
PRED: E096730195CHN8410021M300821488S6102ZS7259328
                                         ^  ^        [2 mismatch]

### indra  [FAIL]
- backend: indra  (44 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=78% conf2=80%
- backend: indra  (283 ms)
GT  : P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: UYCHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^   ^^  ^^^^                                  [8 mismatch]
GT  : E096730195CHN8410021M300821488S61022S7Z59328
PRED: 6095730I93CHN8410021M30082148356102257259322
      ^  ^   ^ ^                   ^^     ^ ^    ^   [9 mismatch]

## 2026-9-29-N.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=288
- 官方 line1: P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E653505611CHN8409285F3001167B0BS1G1886G2O864
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E653505611CHN8409285F3001167B0BS1G1886G2O864`

### traditional  [decoded] conf1=73% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAA9U<<5HUYUE<<<<<444444444444<<<<<<<<<<<<
      ^  ^^^   ^          ^^^^^^^^^^^^               [17 mismatch]
GT  : E653505611CHN8409285F3001167B0BS1G1886G2O864
PRED: Z8J39DKZJJCHN8<CV205P3QC446Y8QHA4A4AAK416ALL
      ^^^ ^^^^^^    ^^^ ^ ^ ^^^^ ^^^^^^^^^^^^^^^^^   [35 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E653505611CHN8409285F3001167B0BS1G1886G2O864
PRED: E653505611CHN8409285F3001167B0BS1G1886G2O864
                                                     [0 mismatch]

### crnn  [decoded] conf1=93% conf2=84%
- backend: crnn  (19 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CAAS<<SAUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
         ^^ ^ ^^^^^^^                            ^   [11 mismatch]
GT  : E653505611CHN8409285F3001167B0BS1G1886G2O864
PRED: E6535O5671CAN8409285F3001167B0BS1G1886G2Q8G4
           ^  ^  ^                            ^ ^    [5 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=97%
GT  : P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E653505611CHN8409285F3001167B0BS1G1886G2O864
PRED: E653505611CHN8409285F3001167B0BS1G1886G2O864
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (224 ms)
GT  : P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHMNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<4<<
          ^^^^ ^^^^^^^                         ^     [12 mismatch]
GT  : E653505611CHN8409285F3001167B0BS1G1886G2O864
PRED: F6S3505611CHN8409285F3001167BQOBS1G1886G2088
      ^ ^                          ^^^^^^^ ^^^^^^^   [16 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (127 ms)
GT  : P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E653505611CHN8409285F3001167B0BS1G1886G2O864
PRED: E653505611CHN8409285F3001167B0BS1G1886G2O864
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (42 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=94% conf2=84%
- backend: indra  (278 ms)
GT  : P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E653505611CHN8409285F3001167B0BS1G1886G2O864
PRED: E653505611CHN8409285F30011678O8S161886620864
                                  ^^^  ^    ^ ^      [6 mismatch]

## 2026-9-29-O.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=281
- 官方 line1: P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E421853076CHN7906282F3506190JJ6M1T1JVTBGC610
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E421853076CHN7906282F3506190JJ6MVT1JVTBGC610`

### traditional  [decoded] conf1=72% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAAWAA<<YUEYUN<444444444444444<<<<<<<<<<<<
      ^  ^^ ^^         ^^^^^^^^^^^^^^^               [20 mismatch]
GT  : E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
PRED: Z<278K3D2NCHK2VCK202PJ5DK4VQJ36A<44J<4A4LG46
      ^^ ^ ^ ^^^  ^^^^^ ^ ^^ ^^^^^ ^ ^^^^ ^^^^^^^^   [33 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
PRED: E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
                                                     [0 mismatch]

### crnn  [decoded] conf1=85% conf2=85%
- backend: crnn  (19 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<GAAZEA<<YOEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^^^ ^   ^                                   [6 mismatch]
GT  : E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
PRED: 4427853076CHM7906282F3506190JJ6MVT13VT8GC619
      ^  ^        ^                      ^  ^    ^   [6 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
PRED: E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (215 ms)
GT  : P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
PRED: E421853076CHN7906282F3506190JJ6MN11JNTBGC610
                                      ^^  ^          [3 mismatch]

### paddle  [DECODE-OK] conf1=90% conf2=99%
- backend: paddle-onnxruntime  (134 ms)
GT  : P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
PRED: E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=85%
- backend: indra  (69 ms)
GT  : P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
PRED: E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=95% conf2=86%
- backend: indra  (294 ms)
GT  : P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E421853076CHN7906282F3506190JJ6MVT1JVTBGC610
PRED: E421853076CHN7906282F3506190JJ6MVT1JMTBGCG22
                                          ^    ^^^   [4 mismatch]

## 2026-9-29-P.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=276
- 官方 line1: P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E766622355CHN0011293F2604181Y861DEUSDLKZG542
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E766622355CHN0011293F2604181Y861DEUSDLKZG542`

### traditional  [decoded] conf1=72% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAAWO<<AAAAYAC<<<44444444<<<44<<4<<<<<<<<<
      ^  ^^ ^  ^ ^^  ^   ^^^^^^^^   ^^  ^            [19 mismatch]
GT  : E766622355CHN0011293F2604181Y861DEUSDLKZG542
PRED: <Z<<<22399LAADCV42V3PP4VA304SD64MKD46L<1A4L1
      ^^^^^   ^^^^^^^^^ ^ ^^^^^^^^^^ ^^^^^^ ^^^^^^   [37 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E766622355CHN0011293F2604181Y861DEUSDLKZG542
PRED: E766622355CHN0011293F2604181Y861DEUSDLKZG542
                                                     [0 mismatch]

### crnn  [decoded] conf1=90% conf2=85%
- backend: crnn  (19 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CAAA0<<F4NGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^^^^   ^                                    [5 mismatch]
GT  : E766622355CHN0011293F2604181Y861DEUSDLKZG542
PRED: E76622355CHA0011293F2604181Y861DEVSDLKZG5G2 
          ^ ^^ ^^^^ ^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [36 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E766622355CHN0011293F2604181Y861DEUSDLKZG542
PRED: E766622355CHN0011293F2604181Y861DEUSDLKZG542
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (211 ms)
GT  : P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E766622355CHN0011293F2604181Y861DEUSDLKZG542
PRED: F7666223S5CHN0011293F2604181Y861DEUSDLKZG542
      ^       ^                                      [2 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (121 ms)
GT  : P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E766622355CHN0011293F2604181Y861DEUSDLKZG542
PRED: E766622355CHN0011293F2604181Y861DEUSDLKZG542
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=84% conf2=82%
- backend: indra  (57 ms)
GT  : P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<FANGYAOK<<K<<<<<<<<<<<<<<<<<<<<<<<<
                      ^  ^                           [2 mismatch]
GT  : E766622355CHN0011293F2604181Y861DEUSDLKZG542
PRED: E766622355CHN0011293F2604181Y861DEUSDLX28542
                                            ^^^      [3 mismatch]

### indra_fullpage  [DECODE-OK] conf1=89% conf2=84%
- backend: indra  (267 ms)
GT  : P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: YACHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^   ^^  ^^^^^^^                               [11 mismatch]
GT  : E766622355CHN0011293F2604181Y861DEUSDLKZG542
PRED: E766622355CHN0011293F2604181Y861DEUS0LKIG542
                                          ^  ^       [2 mismatch]

## 2026-9-29-Q.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=241
- 官方 line1: P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E436102633CHN8211158F32082106GIB15B16Z6BB264
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E436102633CHN8211158F32082106GIB15B16Z6BB264`

### traditional  [decoded] conf1=73% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAAWO<<AUZHU<<<<44444444444444<<<<<<<<<<<<
      ^  ^^ ^  ^        ^^^^^^^^^^^^^^               [19 mismatch]
GT  : E436102633CHN8211158F32082106GIB15B16Z6BB264
PRED: L<J87D2<13CHN8244458P32QH24C441A15A4A1KAA1LL
      ^^^^^^ ^^      ^^^  ^  ^^ ^^^^^^  ^^^^^^^^^^   [30 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=94%
GT  : P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E436102633CHN8211158F32082106GIB15B16Z6BB264
PRED: E436102633CHN8211158F32082106GIB15B16Z6BB264
                                                     [0 mismatch]

### crnn  [decoded] conf1=91% conf2=88%
- backend: crnn  (19 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<6AAAO<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^^^^                                        [5 mismatch]
GT  : E436102633CHN8211158F32082106GIB15B16Z6BB264
PRED: E436702633CHN8211158F32082106GIBI5B16Z688Z64
          ^                           ^      ^^^     [5 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=95%
GT  : P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E436102633CHN8211158F32082106GIB15B16Z6BB264
PRED: E436102633CHN8211158F32082106GIB15B16Z6BB264
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (127 ms)
GT  : P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E436102633CHN8211158F32082106GIB15B16Z6BB264
PRED: E436102633CHN8211158F328082106GIBI5B1616BB16
                             ^^^^^^^^^^^^^^^^^ ^^^   [20 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=98%
- backend: paddle-onnxruntime  (131 ms)
GT  : P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E436102633CHN8211158F32082106GIB15B16Z6BB264
PRED: E436102633CHN8211158F32082106GIBI5B16Z6BBZ64
                                      ^        ^     [2 mismatch]

### indra  [DECODE-OK] conf1=92% conf2=82%
- backend: indra  (65 ms)
GT  : P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E436102633CHN8211158F32082106GIB15B16Z6BB264
PRED: E436102633CHN8211158F3208210661815B762688764
                                   ^^^   ^ ^ ^^^     [8 mismatch]

### indra_fullpage  [DECODE-OK] conf1=94% conf2=81%
- backend: indra  (281 ms)
GT  : P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E436102633CHN8211158F32082106GIB15B16Z6BB264
PRED: E436102633CHN8211158F3208210661815B762688764
                                   ^^^   ^ ^ ^^^     [8 mismatch]

## 2026-9-29-R.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=465
- 官方 line1: P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E123241791CHN6208185M36010409K94WG6UGCK6FR64
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E123241791CHN6208185M36010409K94WG6UGCK6FR64`

### traditional  [decoded] conf1=72% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAAY1<<PU<<<<<<<4444444444444444<4<4<<<4<<
      ^  ^^^^  ^        ^^^^^^^^^^^^^^^^ ^ ^   ^     [25 mismatch]
GT  : E123241791CHN6208185M36010409K94WG6UGCK6FR64
PRED: <YJ32XJ2V4CHA62O8485N3KC4CAC9<94VGKQAL<6VAGL
      ^^^  ^^^^^  ^  ^ ^  ^ ^^^^^^ ^  ^ ^^^^^ ^^^^   [29 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E123241791CHN6208185M36010409K94WG6UGCK6FR64
PRED: E123241791CHN6208185M36010409K94WG6UGCK6FR64
                                                     [0 mismatch]

### crnn  [decoded] conf1=92% conf2=82%
- backend: crnn  (18 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<6AAX1<<Y0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^^ ^   ^                                    [5 mismatch]
GT  : E123241791CHN6208185M36010409K94WG6UGCK6FR64
PRED: E723247791CAN6208185M36010409K94XG6UGCK6FR6L
       ^    ^    ^                    ^          ^   [5 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E123241791CHN6208185M36010409K94WG6UGCK6FR64
PRED: E123241791CHN6208185M36010409K94WG6UGCK6FR64
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (378 ms)
GT  : P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHANMXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^^^^^^^^^                                  [9 mismatch]
GT  : E123241791CHN6208185M36010409K94WG6UGCK6FR64
PRED: F723Z4T791CHN6208185M36010409K94HG6UGCK6EB64
      ^^  ^ ^                         ^       ^^     [7 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (131 ms)
GT  : P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E123241791CHN6208185M36010409K94WG6UGCK6FR64
PRED: E123241791CHN6208185M36010409K94WG6UGCK6FR64
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (45 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=85% conf2=83%
- backend: indra  (258 ms)
GT  : P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PECHNX<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^    ^ ^^^                                    [5 mismatch]
GT  : E123241791CHN6208185M36010409K94WG6UGCK6FR64
PRED: E123241791CHN6208185M36010409K944G6UGCK67R64
                                      ^       ^      [2 mismatch]

## 2026-9-29-S.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=268
- 官方 line1: P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: FA25881562CHN9505209F3503269GIZI52520121S504
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E62588I562CHN9505209F3503269G1Z152520121S504`

### traditional  [decoded] conf1=72% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAAPUXA<<PCL9LI<<44444444<<444<<<<<<<<<<<<
      ^  ^^^ ^^  ^^^^^   ^^^^^^^^  ^^^               [22 mismatch]
GT  : E62588I562CHN9505209F3503269G1Z152520121S504
PRED: <<24AAJ4A2CAAV9C52C7FJVVJZ4VD1115157M41149MG
      ^^ ^^^^^^  ^^^^^  ^^ ^^^^^^^^ ^  ^ ^^^^ ^^^^   [33 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=95%
GT  : P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E62588I562CHN9505209F3503269G1Z152520121S504
PRED: E62588I562CHN9505209F3503269G1Z152520121S504
                                                     [0 mismatch]

### crnn  [decoded] conf1=92% conf2=85%
- backend: crnn  (18 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<GAAYUAA<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^^   ^                                      [4 mismatch]
GT  : E62588I562CHN9505209F3503269G1Z152520121S504
PRED: Z<25881562CHN9505209F3503269O1ZI5Z52Q121S5Q4
      ^^    ^                     ^  ^ ^  ^     ^    [8 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E62588I562CHN9505209F3503269G1Z152520121S504
PRED: E62588I562CHN9505209F3503269G1Z152520121S504
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (124 ms)
GT  : P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E62588I562CHN9505209F3503269G1Z152520121S504
PRED: E62588I562CHN9505209F350326901Z152520121S580
                                  ^             ^^   [3 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=96%
- backend: paddle-onnxruntime  (146 ms)
GT  : P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E62588I562CHN9505209F3503269G1Z152520121S504
PRED: E625881S62CHN9505209F350326901ZI5252Q12IS504
            ^^                    ^  ^    ^  ^       [6 mismatch]

### indra  [DECODE-OK] conf1=92% conf2=86%
- backend: indra  (68 ms)
GT  : P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E62588I562CHN9505209F3503269G1Z152520121S504
PRED: E625881S62CHN9505209F350326901715752012T9504
            ^^                    ^ ^  ^     ^^      [7 mismatch]

### indra_fullpage  [DECODE-OK] conf1=90% conf2=86%
- backend: indra  (253 ms)
GT  : P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E62588I562CHN9505209F3503269G1Z152520121S504
PRED: E625881S62CHN9505209F3503269012152520127SS04
            ^^                    ^ ^        ^ ^     [6 mismatch]

## 2026-9-29-T.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=274
- 官方 line1: P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E0755268S1CHN9004012M27090195GI5BB15B85B1026
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E0755268S1CHN9004012M27090195GI5BBISB85B1O26`

### traditional  [decoded] conf1=72% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <ZLAAPDXA<<YUE5UI<<<4444444444444444<<<<<<<<
      ^^^^^^^^^     ^     ^^^^^^^^^^^^^^^^           [26 mismatch]
GT  : E0755268S1CHN9004012M27090195GI5BBISB85B1O26
PRED: <<2A42<AAYCAAVCD4C42NP7DVV3V5G15AQ13AA4A4L1L
      ^^^^^ ^^^^ ^^^^^ ^^ ^^ ^^^^^  ^ ^^^^^^^^^^^^   [36 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E0755268S1CHN9004012M27090195GI5BBISB85B1O26
PRED: E0755268S1CHN9004012M27090195GI5BBISB85B1O26
                                                     [0 mismatch]

### crnn  [decoded] conf1=91% conf2=84%
- backend: crnn  (17 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PP<CAAYO<A<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^^ ^^^^^^^                          ^   [17 mismatch]
GT  : E0755268S1CHN9004012M27090195GI5BBISB85B1O26
PRED: E0755266S7CHN9004012M27090195GI58B1S88584S2C
             ^ ^                      ^ ^ ^  ^^^ ^   [9 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E0755268S1CHN9004012M27090195GI5BBISB85B1O26
PRED: E0755268S1CHN9004012M27090195GI5BB15B85B1026
                                        ^^     ^     [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (204 ms)
GT  : P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<4
                                                 ^   [1 mismatch]
GT  : E0755268S1CHN9004012M27090195GI5BBISB85B1O26
PRED: F0755Z68S1CHN9004012M2709019SG15BB1SB8581026
      ^    ^                      ^ ^   ^    ^ ^     [7 mismatch]

### paddle  [DECODE-OK] conf1=89% conf2=99%
- backend: paddle-onnxruntime  (131 ms)
GT  : P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E0755268S1CHN9004012M27090195GI5BBISB85B1O26
PRED: E0755268S1CHN9004012M27090195GI5BBISB85B1O26
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=86% conf2=80%
- backend: indra  (62 ms)
GT  : P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUANK<YUESUI<<<<555<<<<<<<<<<<<<<<<<<<<
               ^           ^^^                       [4 mismatch]
GT  : E0755268S1CHN9004012M27090195GI5BBISB85B1O26
PRED: E9755Z68S1CHN9004012M27090195615BB1SB853T026
       ^   ^                       ^^   ^    ^^^     [8 mismatch]

### indra_fullpage  [DECODE-OK] conf1=80% conf2=79%
- backend: indra  (283 ms)
GT  : P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PECHNYDANKYUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^    ^  ^^^^^^^^                              [10 mismatch]
GT  : E0755268S1CHN9004012M27090195GI5BBISB85B1O26
PRED: E675526351CHN9004012M27090195615BB1SB853T026
       ^     ^^                    ^^   ^    ^^^     [9 mismatch]

## 2026-9-29-U.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=550
- 官方 line1: P<CHNYUE<<YUFCHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E111191503<WU7807091M2505233356G5111B0748734
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E158193223CHN7807091M2505233356G5111BO7S6134`

### traditional  [decoded] conf1=69% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<<<L<LLA<<FVPVNV44TSSSSSSSSSSSF4444<<<<<<<<
        ^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^           [32 mismatch]
GT  : E158193223CHN7807091M2505233356G5111BO7S6134
PRED: ZSLG<6<<54LAA28QZV73NPV<<FSPTTT544544465S<<<
      ^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [43 mismatch]

### cnn  [decoded] conf1=99% conf2=84%
- error: decode failed: check digit mismatch
GT  : P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUE<<YUESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                ^^                                   [2 mismatch]
GT  : E158193223CHN7807091M2505233356G5111BO7S6134
PRED: <19E793523CHN7807991M7305233356GS1116O7S67<4
      ^ ^^^  ^         ^   ^^         ^   ^    ^^    [12 mismatch]

### crnn  [decoded] conf1=93% conf2=88%
- backend: crnn  (18 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<CAAYOE<<YDESH0<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^  ^^ ^   ^^   ^                               [7 mismatch]
GT  : E158193223CHN7807091M2505233356G5111BO7S6134
PRED: P<SRZ93223CHN7807091M2505233356G5111BO7S613Q
      ^^^^^                                      ^   [6 mismatch]

### fuse  [decoded] conf1=99% conf2=90%
- error: decode failed: check digit mismatch
GT  : P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUE<<YUESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                ^^                                   [2 mismatch]
GT  : E158193223CHN7807091M2505233356G5111BO7S6134
PRED: E198793523CHN7807991M7505233356GS1116O7S6734
        ^ ^  ^         ^   ^          ^   ^    ^     [8 mismatch]

### tesseract  [FAIL]
- backend: tesseract  (215 ms)
- error: no MRZ candidate decoded
- PRED line1: <空>
- PRED line2: <空>

### paddle  [DECODE-OK] conf1=95% conf2=99%
- backend: paddle-onnxruntime  (142 ms)
GT  : P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUE<<YUESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                ^^                                   [2 mismatch]
GT  : E158193223CHN7807091M2505233356G5111BO7S6134
PRED: 51S8193223CHN7807091M2505233356G5111BO7S6134
      ^ ^                                            [2 mismatch]

### indra  [DECODE-OK] conf1=74% conf2=82%
- backend: indra  (65 ms)
GT  : P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PENYU<<YUESHUK<<<<<<<<<<<<<<<<<<<<<<<<CE<<<<
       ^^^^^^^^^^^^^^^                      ^^       [17 mismatch]
GT  : E158193223CHN7807091M2505233356G5111BO7S6134
PRED: 27932230H1NYU7097M2505233359651T1807561<<<36
      ^^^^^^ ^^^^^^ ^^ ^^^^^^^^^^^^ ^^^^^^^^^^^^ ^   [39 mismatch]

### indra_fullpage  [DECODE-OK] conf1=77% conf2=80%
- backend: indra  (259 ms)
GT  : P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PECHN<<ESHUK<<<<<<<<<<<<<<<<<<<<<<<<C<<<<<<<
       ^   ^^ ^^^^^^^^                    ^          [12 mismatch]
GT  : E158193223CHN7807091M2505233356G5111BO7S6134
PRED: Z759793223CHN7807091125052333566511180756T34
      ^^ ^^               ^          ^    ^^ ^ ^     [10 mismatch]

## 2026-9-29-W.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=281
- 官方 line1: P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E048156G62CHN6209285F3512252O850ZIGZG2581G50
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E048156G62CHN6209285F35122520X50Z1GZG2581G50`

### traditional  [decoded] conf1=70% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<<CHNPVN<<ALVIAL<<<<<<<<<<<<<<<<<<4<<<<<<<<
        ^^^^^^^ ^^^^^^^                  ^           [15 mismatch]
GT  : E048156G62CHN6209285F35122520X50Z1GZG2581G50
PRED: <4<V74VKK2CAA<ADA2H5L3K42252DV5DT3V1475K145L
      ^^^^^^^^^  ^^^^^^ ^ ^ ^^    ^^ ^^^^^^^ ^ ^ ^   [31 mismatch]

### cnn  [DECODE-OK] conf1=97% conf2=92%
GT  : P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHHYUN<<PUYYNG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^        ^                                 [2 mismatch]
GT  : E048156G62CHN6209285F35122520X50Z1GZG2581G50
PRED: E048156662CHN6209285F35122520850Z1BZG2581G50
             ^                     ^    ^            [3 mismatch]

### crnn  [decoded] conf1=91% conf2=89%
- backend: crnn  (21 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CAAYOA<<P0YING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^^ ^^   ^                                   [5 mismatch]
GT  : E048156G62CHN6209285F35122520X50Z1GZG2581G50
PRED: A948756G62CHN6209285F3512252O85OZ1GZG2581G5Q
      ^^  ^                       ^^ ^           ^   [7 mismatch]

### fuse  [DECODE-OK] conf1=98% conf2=94%
GT  : P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHHYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^                                          [1 mismatch]
GT  : E048156G62CHN6209285F35122520X50Z1GZG2581G50
PRED: E048156662CHN6209285F35122520850Z1BZG2581G50
             ^                     ^    ^            [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (233 ms)
GT  : P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E048156G62CHN6209285F35122520X50Z1GZG2581G50
PRED: F0481S6G62CHN6209285F351225208S0Z1GZG62S8162
      ^    ^                       ^^      ^^^^^^^   [11 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=99%
- backend: paddle-onnxruntime  (147 ms)
GT  : P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E048156G62CHN6209285F35122520X50Z1GZG2581G50
PRED: E048156G62CHN6209285F3512252O85OZ1GZG258IG50
                                  ^^ ^        ^      [4 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=90%
- backend: indra  (62 ms)
GT  : P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E048156G62CHN6209285F35122520X50Z1GZG2581G50
PRED: E048156662CHN6209285F351225208502162625B1650
             ^                     ^  ^ ^^^  ^ ^     [8 mismatch]

### indra_fullpage  [DECODE-OK] conf1=69% conf2=88%
- backend: indra  (262 ms)
GT  : P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PUCHNG<<<<<<<<<<<<KKKKKKKKKKKK<<<<<<<<<<<<<<
       ^   ^^^  ^^^^^^  ^^^^^^^^^^^^                 [22 mismatch]
GT  : E048156G62CHN6209285F35122520X50Z1GZG2581G50
PRED: 2048T56662CHN62092851351225208502162625B1650
      ^   ^  ^            ^        ^  ^ ^^^  ^ ^     [11 mismatch]

## 2026-9-30-A.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=248
- 官方 line1: POCHNRAO<<MEINA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E821973372CHN8612261F2607078MIMEMDMEMEMIA944
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E821973372CHN8612261F2607078MIMEMDMEMEMIA944`

### traditional  [decoded] conf1=70% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <A<<A<A<C<<NL1NA4444444T4444<4<<<<<<<<<<<<<<
      ^^^^^^ ^^ ^^^^ ^^^^^^^^^^^^^ ^                 [26 mismatch]
GT  : E821973372CHN8612261F2607078MIMEMDMEMEMIA944
PRED: <A<7V23322CHA8K32243FZ4C7C78P1PCHDPLALA4G4CC
      ^^^^^^  ^   ^ ^^  ^^ ^^^ ^  ^^^^^ ^^^^^^^^^^   [31 mismatch]

### cnn  [DECODE-OK] conf1=92% conf2=99%
GT  : P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNHAO<<MEINA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : E821973372CHN8612261F2607078MIMEMDMEMEMIA944
PRED: E821973372CHN8612261F2607078MIMEMDMEMEMIA944
                                                     [0 mismatch]

### crnn  [decoded] conf1=92% conf2=81%
- backend: crnn  (15 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PP<CAR4O<<MEINA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^       ^                                [7 mismatch]
GT  : E821973372CHN8612261F2607078MIMEMDMEMEMIA944
PRED: P62793372CHM8612261F2607078MIMEMDMEACMIR944 
      ^^ ^ ^ ^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^ ^   [39 mismatch]

### fuse  [DECODE-OK] conf1=98% conf2=99%
GT  : P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNHAO<<MEINA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                    ^                                [1 mismatch]
GT  : E821973372CHN8612261F2607078MIMEMDMEMEMIA944
PRED: E821973372CHN8612261F2607078MIMEMDMEMEMIA944
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (187 ms)
GT  : P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNRAO<<MEINA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^        ^                                [2 mismatch]
GT  : E821973372CHN8612261F2607078MIMEMDMEMEMIA944
PRED: E821973372CHN8612261F2607078MINEMDMEMEWNIA94
                                    ^       ^^^^^    [6 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (102 ms)
GT  : P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNRAO<<MEINA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^        ^                                [3 mismatch]
GT  : E821973372CHN8612261F2607078MIMEMDMEMEMIA944
PRED: E821973372CHN8612261F2607078MIMEMDMEMEMIA944
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=89%
- backend: indra  (69 ms)
GT  : P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNRAO<<MEINA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^        ^                                [3 mismatch]
GT  : E821973372CHN8612261F2607078MIMEMDMEMEMIA944
PRED: E821973372CHN8612261F2607078MIMEMDMEM<<<<<56
                                           ^^^^^^^   [7 mismatch]

### indra_fullpage  [DECODE-OK] conf1=90% conf2=56%
- backend: indra  (310 ms)
GT  : P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: E8CHA73372CHN8612261F2607078MIMEMDMEMENTAGL<
      ^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    [41 mismatch]
GT  : E821973372CHN8612261F2607078MIMEMDMEMEMIA944
PRED: EASISSERS1CHASTISSE5ESTOSOS3RTENOWEREATVO<46
       ^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^   [40 mismatch]

## 2026-9-30-AA.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=310
- 官方 line1: P<CHNLIN<<YUNLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E940780938CHN8410179F3401077KJYROL12DY2WM670
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E940780938CHN8410179F3401077KJYROL12DY2WW670`

### traditional  [decoded] conf1=70% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P4CANLJA<<PLALAN<<<<<<<<<<<<<<<<4441F4444444
       ^ ^  ^^  ^^^                   ^^^^^^^^^^^^   [19 mismatch]
GT  : E940780938CHN8410179F3401077KJYROL12DY2WW670
PRED: PP<VZ8DV3ACAV8<4C42VA9AO4DYY<39A4C47VS7VP63M
      ^^^^^ ^^ ^ ^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^   [39 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=94% conf2=86%
- backend: crnn  (19 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PP<CAAZ7A<<YUNLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^ ^^^^^^^                           ^   [16 mismatch]
GT  : E940780938CHN8410179F3401077KJYROL12DY2WW670
PRED: E94O78O938CAN8410179F3401077KJYROL12DY2VX676
         ^  ^    ^                           ^^  ^   [6 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNLIN<<YUNLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                  ^                                  [1 mismatch]
GT  : E940780938CHN8410179F3401077KJYROL12DY2WW670
PRED: E940780938CHN8410179F3401077KJYROL12DY2WM670
                                              ^      [1 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (229 ms)
GT  : P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNLIN<<YUNLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                  ^                                  [1 mismatch]
GT  : E940780938CHN8410179F3401077KJYROL12DY2WW670
PRED: E940780938CHN8410179F3401077KJYRDL12DX2WW670
                                      ^    ^         [2 mismatch]

### paddle  [DECODE-OK] conf1=93% conf2=99%
- backend: paddle-onnxruntime  (145 ms)
GT  : P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNLIN<<YUNLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                  ^                                  [1 mismatch]
GT  : E940780938CHN8410179F3401077KJYROL12DY2WW670
PRED: E940780938CHN8410179F3401077KJYROL12DY2WM670
                                              ^      [1 mismatch]

### indra  [DECODE-OK] conf1=80% conf2=83%
- backend: indra  (71 ms)
GT  : P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNLINKKYUNLANK<<<<<<<<<<<<<<<<SSSSSSSS<<<
              ^^  ^   ^                ^^^^^^^^      [12 mismatch]
GT  : E940780938CHN8410179F3401077KJYROL12DY2WW670
PRED: E940780938CHN8410179F3401077KJYROL12DY2WN<68
                                              ^^^^   [4 mismatch]

### indra_fullpage  [DECODE-OK] conf1=82% conf2=84%
- backend: indra  (273 ms)
GT  : P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNLINKKYUNLAN<<<<<<<<<<<<<<<<<SSSSSSS<<<<
              ^^  ^                    ^^^^^^^       [10 mismatch]
GT  : E940780938CHN8410179F3401077KJYROL12DY2WW670
PRED: E940780938CHN8410179F3401077KJYROL12DY2MN<68
                                             ^^^^^   [5 mismatch]

## 2026-9-30-AB.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=247
- 官方 line1: P<CHNBAO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42`

### traditional  [decoded] conf1=72% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<LNANAD<<PLAVA<<<<<<<<<<<<<<<<<<444444444<<
        ^^^^ ^  ^^^^                   ^^^^^^^^^     [18 mismatch]
GT  : E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
PRED: PNQZZV77SVCHA<VD4AA<A2AMA222AY5DVHC9H9<AJ1L1
      ^^^^^^^^^^  ^^^^^^^^^ ^^^   ^^ ^^^   ^^^ ^^^   [33 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
PRED: E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
                                                     [0 mismatch]

### crnn  [decoded] conf1=90% conf2=84%
- backend: crnn  (17 ms)
- error: decode failed: check digit mismatch
GT  : P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PP<CAAB4O<<YOEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^ ^^^^^^                            ^   [15 mismatch]
GT  : E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
PRED: C602707756CAA6901084F2808222Q75OWNC9XSVE1Z42
      ^     ^^   ^^                       ^   ^      [7 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
PRED: E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (119 ms)
GT  : P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
PRED: E602701156CHN6901084F2808222Q750HNC9USVEJZ42
                                     ^^   ^          [3 mismatch]

### paddle  [DECODE-OK] conf1=95% conf2=99%
- backend: paddle-onnxruntime  (125 ms)
GT  : P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
PRED: E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=87% conf2=82%
- backend: indra  (60 ms)
GT  : P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
PRED: E602707756CHN6901084F28082229750WNC9NSMEJL42
            ^^                    ^  ^    ^ ^  ^     [7 mismatch]

### indra_fullpage  [DECODE-OK] conf1=96% conf2=83%
- backend: indra  (242 ms)
GT  : P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBAO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42
PRED: E602707756CHN6901084F28082220750WNC945NEJL42
            ^^                    ^  ^    ^^^  ^     [8 mismatch]

## 2026-9-30-AC.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=277
- 官方 line1: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E141543482CHN6011103M32062610MU9FH60NPJJ1G24
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E141543482CHN6011103M32062610MU9FH60NPJJ1G24`

### traditional  [decoded] conf1=72% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CKADD<<<W1YLXH<<<<44444444444444<<<<<<<<<<
         ^^^^  ^^^ ^^^    ^^^^^^^^^^^^^^             [24 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: P7<7S<J<82LAAAM444M3N32C6Z44DHC9VAKMAVJJ4LVC
      ^^^^^^^^  ^^^^^^^^^ ^  ^ ^^^^^^ ^^^^^^  ^^^^   [35 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M32062610MU9FH60NPJJ1G24
                                                     [0 mismatch]

### crnn  [decoded] conf1=89% conf2=83%
- backend: crnn  (21 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<2TABO<<KAIY0AN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^^ ^      ^                                 [5 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: Z<47543482CHA6011103M3206261OMU9FH6ONP337G24
      ^^ ^        ^               ^      ^  ^^^      [9 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M32062610MU9FH60NPJJ1G24
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (206 ms)
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: 7471SW3482CHN6011103M3206261OMU9FH60NP33TG24
      ^^^ ^^                      ^         ^^^      [9 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (117 ms)
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M3206261OMU9FH60NPJJ7G24
                                  ^           ^      [2 mismatch]

### indra  [DECODE-OK] conf1=83% conf2=82%
- backend: indra  (66 ms)
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M3206261OMU9FN60NP331624
                                  ^    ^    ^^ ^     [5 mismatch]

### indra_fullpage  [DECODE-OK] conf1=90% conf2=76%
- backend: indra  (278 ms)
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<AA<
                                               ^^    [2 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M32062610MU9FH6ONP337624
                                         ^  ^^^^     [5 mismatch]

## 2026-9-30-AC_R.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=277
- 官方 line1: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E141543482CHN6011103M3206261OMU9FH60NPJJ7G24
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E141543482CHN6011103M32062610MU9FH60NPJJ1G24`

### traditional  [decoded] conf1=72% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CKADD<<<W1YLXH<<<<44444444444444<<<<<<<<<<
         ^^^^  ^^^ ^^^    ^^^^^^^^^^^^^^             [24 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: P7<7S<J<82LAAAM444M3N32C6Z44DHC9VAKMAVJJ4LVC
      ^^^^^^^^  ^^^^^^^^^ ^  ^ ^^^^^^ ^^^^^^  ^^^^   [35 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=96%
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M32062610MU9FH60NPJJ1G24
                                                     [0 mismatch]

### crnn  [decoded] conf1=89% conf2=83%
- backend: crnn  (20 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<2TABO<<KAIY0AN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^^ ^      ^                                 [5 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: Z<47543482CHA6011103M3206261OMU9FH6ONP337G24
      ^^ ^        ^               ^      ^  ^^^      [9 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M32062610MU9FH60NPJJ1G24
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (214 ms)
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: 7471SW3482CHN6011103M3206261OMU9FH60NP33TG24
      ^^^ ^^                      ^         ^^^      [9 mismatch]

### paddle  [DECODE-OK] conf1=99% conf2=99%
- backend: paddle-onnxruntime  (119 ms)
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M3206261OMU9FH60NPJJ7G24
                                  ^           ^      [2 mismatch]

### indra  [DECODE-OK] conf1=83% conf2=82%
- backend: indra  (65 ms)
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M3206261OMU9FN60NP331624
                                  ^    ^    ^^ ^     [5 mismatch]

### indra_fullpage  [DECODE-OK] conf1=90% conf2=76%
- backend: indra  (248 ms)
GT  : P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<AA<
                                               ^^    [2 mismatch]
GT  : E141543482CHN6011103M32062610MU9FH60NPJJ1G24
PRED: E141543482CHN6011103M32062610MU9FH6ONP337624
                                         ^  ^^^^     [5 mismatch]

## 2026-9-30-AD.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=362
- 官方 line1: P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E277194572CHN9601I80530052246MN1TZC2X4MC4X60
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E277194572CHN9601189F30052246MN1TZC2X4MC4X60`

### traditional  [decoded] conf1=68% conf2=65%
- error: decode failed: check digit mismatch
GT  : P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: FSFPPF<7<<P<AL<F<LILIII<<<<<<<44FS1SS<<<<<<<
      ^^^^^^^^  ^^^^^^^^^^^^^       ^^^^^^^          [28 mismatch]
GT  : E277194572CHN9601189F30052246MN1TZC2X4MC4X60
PRED: 4<<T2P<4<A<<<L<L<<ALL<AL<<<<LAA4<I<2Y4P<F<49
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^ ^^^^^^   [42 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E277194572CHN9601189F30052246MN1TZC2X4MC4X60
PRED: E277194572CHN9601189F30052246MN1TZC2X4MC4X60
                                                     [0 mismatch]

### crnn  [decoded] conf1=93% conf2=91%
- backend: crnn  (18 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CAAC47<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^^ ^^                                       [4 mismatch]
GT  : E277194572CHN9601189F30052246MN1TZC2X4MC4X60
PRED: E<77794572CAA9601189F30052246MN1IZC2X4WC4X66
       ^  ^      ^^                   ^     ^    ^   [7 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E277194572CHN9601189F30052246MN1TZC2X4MC4X60
PRED: E277194572CHN9601189F30052246MN1TZC2X4MC4X60
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (278 ms)
GT  : P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHANCAI<<FANGMENG<<<<<ZX<<<<<<L<<<L<<L<4A4
          ^^^^^ ^^^^^^^^^     ^^      ^   ^  ^ ^^^   [22 mismatch]
GT  : E277194572CHN9601189F30052246MN1TZC2X4MC4X60
PRED: E277194572CHN9601189F30052246MN1TZC2X4UNMC46
                                            ^^^^^^   [6 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (156 ms)
GT  : P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E277194572CHN9601189F30052246MN1TZC2X4MC4X60
PRED: E277194572CHN9601189F30052246MN1TZC2X4MC4X60
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (28 ms)
- error: indra 未找到 MRZ 候选行 (0 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=90% conf2=62%
- backend: indra  (245 ms)
GT  : P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: NAFANNETIMMIGRATIONADMINISTRATION<<<<<<<<<<<
      ^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^              [32 mismatch]
GT  : E277194572CHN9601189F30052246MN1TZC2X4MC4X60
PRED: PECHNCAIS1FANGMENGE5CCCCC<<6<<<<<<<<<<<<<<04
      ^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [43 mismatch]

## 2026-9-30-AE.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=298
- 官方 line1: P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E309178198CHN9612202F2702261G3JPA90J2I<K6Z66`

### traditional  [decoded] conf1=68% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PP<CAACAWAA<<AI<<<<<<44444444<<<<<<<<<<<<<22
       ^^^^^^ ^^^ ^^^      ^^^^^^^^             ^^   [22 mismatch]
GT  : E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
PRED: PJD4YZ8JVALAAVA422C2P77C2264G33AK4M3V4<VL5<<
      ^^^^^^ ^^^^^^^^^  ^ ^^ ^   ^  ^^^^^^^^ ^^^^^   [33 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
PRED: E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
                                                     [0 mismatch]

### crnn  [decoded] conf1=92% conf2=86%
- backend: crnn  (21 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: 7P<CA24AG<<M1<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
      ^^^^^^^ ^^ ^^^                             ^   [13 mismatch]
GT  : E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
PRED: 4309778798CAA9612202F2702261G3JPR9OJ2I<X6Z6G
      ^   ^  ^   ^^                   ^ ^    ^   ^   [9 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
PRED: E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (223 ms)
GT  : P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHANG<<NJI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                   ^^                                [2 mismatch]
GT  : E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
PRED: 30977R7198CHN9612202F2702261G3JPA9032T<X6166
      ^^^^^^^                            ^ ^ ^ ^     [11 mismatch]

### paddle  [DECODE-OK] conf1=94% conf2=99%
- backend: paddle-onnxruntime  (122 ms)
GT  : P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
PRED: E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (52 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=94% conf2=81%
- backend: indra  (226 ms)
GT  : P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E309178198CHN9612202F2702261G3JPA90J2I<K6Z66
PRED: E3097T8198CHN9612202F2702261633PA90321<X6Z66
          ^^                      ^ ^    ^ ^ ^       [7 mismatch]

## 2026-9-30-AF.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=309
- 官方 line1: P<CHNCHEN<<DAJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E646452438CHN0211196M3203075FIJJWA3L90ZU0054
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E646452438CHN0211196M3203075FIJJWA3L90ZU0054`

### traditional  [decoded] conf1=68% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<LAALALA<Z6<3LA<<<444TT4T44444444444<<<<<<2
      ^ ^^^^^^^ ^^^^^^   ^^^^^^^^^^^^^^^^^^      ^   [33 mismatch]
GT  : E646452438CHN0211196M3203075FIJJWA3L90ZU0054
PRED: <<<<<<2<AALAAA2444VKNM7VJV75P103VK<C9QILL6<L
      ^^^^^^ ^^^^^^^ ^^^^^^^^^^^  ^^^^^^^^ ^^^^^^^   [39 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHEN<<DAJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                 ^                                   [1 mismatch]
GT  : E646452438CHN0211196M3203075FIJJWA3L90ZU0054
PRED: E646452438CHN0211196M3203075FIJJWA3L90ZU0054
                                                     [0 mismatch]

### crnn  [decoded] conf1=87% conf2=86%
- backend: crnn  (21 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: P<C<ACAEA<<DAJ0N<<<<<<<<<<<<<<<<<<<<<<<<<<<<1
         ^^ ^ ^  ^  ^                             ^   [7 mismatch]
GT  : E646452438CHN0211196M3203075FIJJWA3L90ZU0054
PRED: C646452438CAA0211196M3203075FIJJWA3L9OZOO547
      ^          ^^                        ^ ^^^^^   [9 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHEN<<DAJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                 ^                                   [1 mismatch]
GT  : E646452438CHN0211196M3203075FIJJWA3L90ZU0054
PRED: E646452438CHN0211196M3203075FIJJWA3L90ZU0054
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (250 ms)
GT  : P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHEN<<DAJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                 ^                                   [1 mismatch]
GT  : E646452438CHN0211196M3203075FIJJWA3L90ZU0054
PRED: G646452432CHN0211196M3203075F1J3J4WMA3L90Z70
      ^        ^                   ^ ^^^^^^^^^ ^^^   [15 mismatch]

### paddle  [DECODE-OK] conf1=90% conf2=99%
- backend: paddle-onnxruntime  (137 ms)
GT  : P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCHEN<<DAJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                 ^                                   [1 mismatch]
GT  : E646452438CHN0211196M3203075FIJJWA3L90ZU0054
PRED: E646452438CHN0211196M3203075FIJJWA3L90ZU0054
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=60% conf2=82%
- backend: indra  (67 ms)
GT  : P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: DACHN<KKKKKKKKKKKKKKKKKK<<<<<<<<<<<<<<<<<<<<
      ^^   ^^^^^^^^^^^^^^^^^^^                       [21 mismatch]
GT  : E646452438CHN0211196M3203075FIJJWA3L90ZU0054
PRED: E646452438CHN0211196M3203075FIJJWA3L902VOU54
                                            ^^^^     [4 mismatch]

### indra_fullpage  [DECODE-OK] conf1=76% conf2=83%
- backend: indra  (233 ms)
GT  : P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: DACHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^   ^^^^   ^^^^                               [10 mismatch]
GT  : E646452438CHN0211196M3203075FIJJWA3L90ZU0054
PRED: E646452438CHN0211196M3203075FIJJWA3L90LU0030
                                            ^   ^^   [3 mismatch]

## 2026-9-30-AG.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=256
- 官方 line1: P<CHNCONG<<YINGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E066908744CHN9303298F34112631WW9ZQLBES70S062
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E066908744CHN9303298F34112631WW9ZQLBES70S062`

### traditional  [decoded] conf1=66% conf2=65%
- error: decode failed: check digit mismatch
GT  : P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<IZ<A<AAA<<<JNV<I4VT44T4TT<4<<<<<<<<<<<<<<I
        ^^^^^^^^  ^^^^^^^^^^^^^^^ ^              ^   [25 mismatch]
GT  : E066908744CHN9303298F34112631WW9ZQLBES70S062
PRED: JAA2ADAZ<<CHN7JV7770FJ434P4TJVVV<4CAAAJ6<6<<
      ^^^^^^^^^^   ^^^^^^^ ^ ^^^^^^^^^^^^^^^^^^^^^   [39 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=97%
GT  : P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCONG<<YINGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
                 ^                                   [1 mismatch]
GT  : E066908744CHN9303298F34112631WW9ZQLBES70S062
PRED: E066908744CHN9303298F34112631WW9ZQLBES70S062
                                                     [0 mismatch]

### crnn  [decoded] conf1=89% conf2=86%
- backend: crnn  (18 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: PP<<ACOWG<<YIAGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<1
       ^^^^  ^   ^ ^                              ^   [8 mismatch]
GT  : E066908744CHN9303298F34112631WW9ZQLBES70S062
PRED: Z06908744CHA9303298F3411263TWW9ZQLBES7QSSG2V
      ^  ^^^^^ ^^^^^^^^^^^^^^ ^^^^^ ^^^^^^^^^^ ^^^   [38 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCONG<<YINGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
                 ^                                   [1 mismatch]
GT  : E066908744CHN9303298F34112631WW9ZQLBES70S062
PRED: E066908744CHN9303298F34112631WW9ZQLBES70S062
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (132 ms)
GT  : P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCONG<<YINGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
                 ^                                   [1 mismatch]
GT  : E066908744CHN9303298F34112631WW9ZQLBES70S062
PRED: O066908744CHN9303298F3411263THW9ZQLBES70SD62
      ^                           ^^           ^     [4 mismatch]

### paddle  [DECODE-OK] conf1=95% conf2=99%
- backend: paddle-onnxruntime  (123 ms)
GT  : P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCONG<<YINGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
                 ^                                   [1 mismatch]
GT  : E066908744CHN9303298F34112631WW9ZQLBES70S062
PRED: E066908744CHN9303298F3411263TWW9ZQLBES7OS062
                                  ^          ^       [2 mismatch]

### indra  [DECODE-OK] conf1=89% conf2=84%
- backend: indra  (64 ms)
GT  : P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: DECHNCONG<<YINGXIAOC<<<<<<<<<<<<<<<<<<<<<<<<
      ^^         ^       ^                           [4 mismatch]
GT  : E066908744CHN9303298F34112631WW9ZQLBES70S062
PRED: 2650O87442CHN303298334112637WW92OLBESTOS0<62
      ^^^^^^^^ ^   ^^^^^^^^^^ ^^^^^ ^^^^^^^^^^^^     [36 mismatch]

### indra_fullpage  [DECODE-OK] conf1=88% conf2=76%
- backend: indra  (259 ms)
GT  : P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: IACHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^   ^^^^   ^^^^^^^                            [13 mismatch]
GT  : E066908744CHN9303298F34112631WW9ZQLBES70S062
PRED: 506690R744CHN9303298F3411263TWW92QLBES7OS362
      ^     ^                     ^   ^      ^ ^     [6 mismatch]

## 2026-9-30-AH.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=276
- 官方 line1: P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56`

### traditional  [decoded] conf1=73% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<<AACDI<<L1Y1<<<<444444<<<<<<<<<<<<<<<<<<Z2
      ^ ^^^ ^    ^ ^    ^^^^^^                  ^^   [15 mismatch]
GT  : E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
PRED: <JA7D<J<V4CANZQ37Z77P333Q2C4G4CA32A<16<4<J<L
      ^^^^^^^^^  ^ ^^^^^^^^ ^^^ ^^^^^^^ ^ ^^^^^ ^^   [36 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
PRED: E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
                                                     [0 mismatch]

### crnn  [decoded] conf1=89% conf2=85%
- backend: crnn  (17 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<QAC0I<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
        ^^^^^^ ^^^^^                             ^   [12 mismatch]
GT  : E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
PRED: E567043494CAA7012299F3110201S1UBJ2E<3D3PC356
        ^^       ^^                           ^^     [6 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
PRED: E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (208 ms)
GT  : P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: CHANCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<A
      ^^^^^^^^ ^^^^^                             ^   [14 mismatch]
GT  : E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
PRED: E581043494CHN7012299F3110201S7UBJ2E<3D3PL356
                                   ^           ^     [2 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (118 ms)
GT  : P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
PRED: E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (55 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=94% conf2=80%
- backend: indra  (267 ms)
GT  : P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56
PRED: E581043494CHN7012299F3110201810B32F<30303<56
                                  ^ ^ ^ ^  ^ ^^^     [8 mismatch]

## 2026-9-30-AI.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=283
- 官方 line1: P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E996968603CHN0205089M3208184OK1U4WCP4K4E7L42
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E996968603CHN0205089M32081840K1U4WCPWK4E7L42`

### traditional  [decoded] conf1=72% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: A<CAADXI<<X1H<<<<<44<44<<<<<<<<<<<<<<<<<<<<<
      ^  ^^ ^    ^^     ^^ ^^                        [10 mismatch]
GT  : E996968603CHN0205089M32081840K1U4WCPWK4E7L42
PRED: ZVA8V086D3CAAC2C5O8VN32C84AAA<4OAWLAL<LV4<LS
      ^^^^^^  ^  ^^^ ^ ^ ^^  ^ ^^^^^^^^ ^^^^^^^^^^   [33 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=93%
GT  : P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E996968603CHN0205089M32081840K1U4WCPWK4E7L42
PRED: E996968603CHN0205089M32081840K1U4WCPWK4E7L42
                                                     [0 mismatch]

### crnn  [decoded] conf1=90% conf2=86%
- backend: crnn  (18 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: P<CAADA7<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         ^^  ^                                    ^   [4 mismatch]
GT  : E996968603CHN0205089M32081840K1U4WCPWK4E7L42
PRED: C996968603CHN0205089M3208184OK1U4XCP6K4E7VQ2
      ^                           ^    ^  ^    ^^    [6 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=93%
GT  : P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E996968603CHN0205089M32081840K1U4WCPWK4E7L42
PRED: E996968603CHN0205089M32081840K1U4WCPWK4E7L42
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (205 ms)
GT  : P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<2
                                                 ^   [1 mismatch]
GT  : E996968603CHN0205089M32081840K1U4WCPWK4E7L42
PRED: F99696R603CHN0205089M32081840K1UAMCP4KLETL92
      ^     ^                         ^^  ^ ^ ^ ^    [8 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (124 ms)
GT  : P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E996968603CHN0205089M32081840K1U4WCPWK4E7L42
PRED: E996968603CHN0205089M3208184OK1U4WCP4K4E7L42
                                  ^       ^          [2 mismatch]

### indra  [FAIL]
- backend: indra  (57 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=95% conf2=85%
- backend: indra  (249 ms)
GT  : P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E996968603CHN0205089M32081840K1U4WCPWK4E7L42
PRED: E996968603CHN0205089M32081840K1044CP4KLE1<42
                                     ^ ^  ^ ^ ^^     [6 mismatch]

## 2026-9-30-AJ.jpg
- 官方判定: PASS  verified=True  chosen=paddle  ms=284
- 官方 line1: P<CHNDING<<HANGYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E818572393CHN6603171M2702157FWS91S8GQ4H16188
- 理由: PaddleOCR 识别并通过校验

- GT line1: `P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E818572393CHN6603171M2702157EBS91S800WHI6I88`

### traditional  [decoded] conf1=72% conf2=66%
- error: decode failed: check digit mismatch
GT  : P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: FL<2<ADIAG<<AXAGYAH<444<44<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^ ^^^^^^^^ ^^^ ^^                     [23 mismatch]
GT  : E818572393CHN6603171M2702157EBS91S800WHI6I88
PRED: L<IA<Z2JPJCHNVVQJ7Z3N77QP4K7FOA91A8AAVA4L4A4
      ^^^^^^ ^^^   ^^^^^^^^^ ^^^^ ^^^  ^ ^^^^^^^^^   [35 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=93%
GT  : P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<HANGYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
            ^     ^ ^                                [3 mismatch]
GT  : E818572393CHN6603171M2702157EBS91S800WHI6I88
PRED: E818572393CHN6603171M2702157EBS91S8GAWH16188
                                         ^^  ^ ^     [4 mismatch]

### crnn  [decoded] conf1=91% conf2=82%
- backend: crnn  (20 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PP<CAZ7AG<<HAMGYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^    ^^^                                [10 mismatch]
GT  : E818572393CHN6603171M2702157EBS91S800WHI6I88
PRED: Z876572393CAA6603171M2702157FHS9IS8GQXXIGI88
      ^ ^^       ^^               ^^  ^  ^^^^ ^      [13 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=95%
GT  : P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<HANGYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
            ^     ^ ^                                [3 mismatch]
GT  : E818572393CHN6603171M2702157EBS91S800WHI6I88
PRED: E818572393CHN6603171M2702157EBS91S8G0WH16188
                                         ^   ^ ^     [3 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (211 ms)
GT  : P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<HANGYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
            ^     ^ ^                                [3 mismatch]
GT  : E818572393CHN6603171M2702157EBS91S800WHI6I88
PRED: FR78572393CHN6603171M2702157FNS9IS8GQWH16188
      ^^^                         ^^  ^  ^^  ^ ^     [10 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (139 ms)
GT  : P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<HANGYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
            ^     ^ ^                                [3 mismatch]
GT  : E818572393CHN6603171M2702157EBS91S800WHI6I88
PRED: E818572393CHN6603171M2702157FHS9IS8GQWHI6I88
                                  ^^  ^  ^^          [5 mismatch]

### indra  [DECODE-OK] conf1=88% conf2=84%
- backend: indra  (71 ms)
GT  : P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: DECNDING<<HANGYANK<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^ ^^^^^^ ^^ ^^^^^                             [15 mismatch]
GT  : E818572393CHN6603171M2702157EBS91S800WHI6I88
PRED: 9857Z393C3CND03111M2702157F8S9ISSGQURI6I<<88
      ^ ^^^^^ ^  ^^^^^^ ^^^^^^^^^^^^^^^^^^^^^ ^^     [36 mismatch]

### indra_fullpage  [DECODE-OK] conf1=88% conf2=76%
- backend: indra  (225 ms)
GT  : P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNDING<<HANGYAN<<<<<<<<<<<<<<<<<<<<<C<CA<
            ^     ^ ^                        ^ ^^    [6 mismatch]
GT  : E818572393CHN6603171M2702157EBS91S800WHI6I88
PRED: E818572393CHN603171427021570HS9IS86QV363<<38
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    [29 mismatch]

## 2026-9-30-B.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=248
- 官方 line1: P<CHNHU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E187424516CHN8706216M2406086LKPKOFLA<<<<A936
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E187424516CHN8706216M2406086LKPKOFLA<<<<A936`

### traditional  [decoded] conf1=73% conf2=69%
- error: decode failed: check digit mismatch
GT  : P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: LAMCAAAO<<AXC<<<4444444444<<<<<<<<<<<<<<<<<<
      ^^^^^^^^ ^ ^^   ^^^^^^^^^^                     [21 mismatch]
GT  : E187424516CHN8706216M2406086LKPKOFLA<<<<A936
PRED: Z7X2Z2<3JZLGA8ZCK244NZ4C4C8KL<A<AVLF<<<<F41A
      ^^^^^ ^^^^^^^ ^^^ ^^^^ ^^^ ^ ^^^^^ ^    ^^^^   [33 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNHU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E187424516CHN8706216M2406086LKPKOFLA<<<<A936
PRED: E187424516CHN8706216M2406086LKPKOFLA<<<<A936
                                                     [0 mismatch]

### crnn  [decoded] conf1=88% conf2=87%
- backend: crnn  (20 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPCAAAO<<AAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^^^^  ^                                     [6 mismatch]
GT  : E187424516CHN8706216M2406086LKPKOFLA<<<<A936
PRED: E<7<516CHN8706216M2406086LKPKOELR<<<<R936   
       ^^^^^^^^^^^^^^^ ^^^^^^^ ^^^^^^^^^^^ ^^^^^^^   [40 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=98%
GT  : P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNHU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E187424516CHN8706216M2406086LKPKOFLA<<<<A936
PRED: E187424516CHN8706216M2406086LKPKOFLA<<<<A936
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (184 ms)
GT  : P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNHU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E187424516CHN8706216M2406086LKPKOFLA<<<<A936
PRED: E7874Z4S16CHN8706216M2406086LKPKQFLA<<<<BA98
       ^   ^ ^                        ^       ^^^^   [8 mismatch]

### paddle  [DECODE-OK] conf1=95% conf2=99%
- backend: paddle-onnxruntime  (110 ms)
GT  : P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P0CHNHU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^                                         [2 mismatch]
GT  : E187424516CHN8706216M2406086LKPKOFLA<<<<A936
PRED: E187424516CHN8706216M2406086LKPKOFLA<<<<A936
                                                     [0 mismatch]

### indra  [FAIL]
- backend: indra  (50 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=96% conf2=96%
- backend: indra  (282 ms)
GT  : P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNHU<<HA0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^   ^     ^                                   [3 mismatch]
GT  : E187424516CHN8706216M2406086LKPKOFLA<<<<A936
PRED: E187424516CHN8706216M2406086LKPKOFLA<<<<A936
                                                     [0 mismatch]

## 2026-9-30-C.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=205
- 官方 line1: P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
- 官方 line2: N7368155<2AUS8703226F2306186<12078730W<<<<68
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<`
- GT line2: `N7368155<2AUS8703226F2306186<12078730W<<<<68`

### traditional  [decoded] conf1=72% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
PRED: AZ3OXAICAXHD5<<XEJRA<3AYNE<<<<<<<<<<<<<<Z<<Z
      ^^^^^^  ^^^ ^  ^ ^   ^                  ^  ^   [15 mismatch]
GT  : N7368155<2AUS8703226F2306186<12078730W<<<<68
PRED: <OZA7K1<2XDK8ZD322KP2JC430A<42DY8Y1LA<<<<AA4
      ^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^   ^^^   [40 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
PRED: P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : N7368155<2AUS8703226F2306186<12078730W<<<<68
PRED: N7368155<2AUS8703226F2306186<12078730W<<<<68
                                                     [0 mismatch]

### crnn  [decoded] conf1=90% conf2=81%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
PRED: P<4<SR7CA4ADS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
        ^^  ^ ^^^                                    [6 mismatch]
GT  : N7368155<2AUS8703226F2306186<12078730W<<<<68
PRED: 6755<2AD587O9226F2306186<4207873OV<<<<68    
      ^ ^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^   [42 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
PRED: P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : N7368155<2AUS8703226F2306186<12078730W<<<<68
PRED: N7368155<2AUS8703226F2306186<12078730W<<<<68
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (134 ms)
GT  : P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
PRED: P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : N7368155<2AUS8703226F2306186<12078730W<<<<68
PRED: 73687155<2AUS8703226F2306186<12078730W<<<<68
      ^^^^^                                          [5 mismatch]

### paddle  [DECODE-OK] conf1=92% conf2=99%
- backend: paddle-onnxruntime  (100 ms)
GT  : P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
PRED: P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : N7368155<2AUS8703226F2306186<12078730W<<<<68
PRED: N7368155<2AUS8703226F2306186<12078730W<<<<68
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=100% conf2=97%
- backend: indra  (60 ms)
GT  : P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
PRED: P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : N7368155<2AUS8703226F2306186<12078730W<<<<68
PRED: N7368155<2AUS8703226F2306186<12078730W<<<<68
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=100% conf2=97%
- backend: indra  (207 ms)
GT  : P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
PRED: P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : N7368155<2AUS8703226F2306186<12078730W<<<<68
PRED: N7368155<2AUS8703226F2306186<12078730W<<<<68
                                                     [0 mismatch]

## 2026-9-30-D.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=214
- 官方 line1: P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: KE00345952CHN9207156M2912095A911<611158<<<14
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `KE00345952CHN9207156M2912095A911<611158<<<14`

### traditional  [decoded] conf1=70% conf2=67%
- error: decode failed: check digit mismatch
GT  : P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: 3ZCAA2ADD<<QJANVTTTT44444<<<<<<<<<<<<<<<<<<<
      ^^ ^^^^^^   ^  ^^^^^^^^^^                      [19 mismatch]
GT  : KE00345952CHN9207156M2912095A911<611158<<<14
PRED: L<<G3ZKV92CHN7ZDZ45KN2V42QV44V44<A4444A<<<4L
      ^^^^ ^^^^    ^^^^^ ^^ ^^ ^^^^^^^ ^^^^^^   ^^   [32 mismatch]

### cnn  [DECODE-OK] conf1=94% conf2=99%
GT  : P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : KE00345952CHN9207156M2912095A911<611158<<<14
PRED: KE00345952CHN9207156M2912095A911<611158<<<14
                                                     [0 mismatch]

### crnn  [decoded] conf1=93% conf2=89%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: 4K2<O0<<Q1ANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<   
      ^^^^^^^^^^^^^^^^                         ^^^   [19 mismatch]
GT  : KE00345952CHN9207156M2912095A911<611158<<<14
PRED: Z00345952CHN9207156M2912095A911<611158<<<141
      ^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^  ^^^  ^^^   [38 mismatch]

### fuse  [DECODE-OK] conf1=94% conf2=99%
GT  : P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : KE00345952CHN9207156M2912095A911<611158<<<14
PRED: KE00345952CHN9207156M2912095A911<611158<<<14
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (124 ms)
GT  : P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AKCHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^                                             [2 mismatch]
GT  : KE00345952CHN9207156M2912095A911<611158<<<14
PRED: KE00345952CHN9207156M2912095A911<611158<<<14
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=97% conf2=99%
- backend: paddle-onnxruntime  (100 ms)
GT  : P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AKCHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^                                             [2 mismatch]
GT  : KE00345952CHN9207156M2912095A911<611158<<<14
PRED: KE00345952CHN9207156M2912095A911<611158<<<14
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=94% conf2=95%
- backend: indra  (60 ms)
GT  : P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AKCHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^                                             [2 mismatch]
GT  : KE00345952CHN9207156M2912095A911<611158<<<14
PRED: KE00345952CHN9207156M2912095A911<611158<<<14
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=91% conf2=95%
- backend: indra  (221 ms)
GT  : P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AKCHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<A
      ^^                                         ^   [3 mismatch]
GT  : KE00345952CHN9207156M2912095A911<611158<<<14
PRED: KE00345952CHN9207156M2912095A911<611158<<<14
                                                     [0 mismatch]

## 2026-9-30-E.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=208
- 官方 line1: P<CHNPHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: E662323994CHN9009121F2601113NFMFMNPBMOMEA976
- 理由: 轻量CNN 识别并通过校验

- GT line1: `P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `E662323994CHN9009121F2601113NFMFMNPBMOMEA976`

### traditional  [decoded] conf1=74% conf2=65%
- error: decode failed: check digit mismatch
GT  : P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <0LAA2A<AG<<WANWLN<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^       ^                              [10 mismatch]
GT  : E662323994CHN9009121F2601113NFMFMNPBMOMEA976
PRED: <8Z2323VV<CHAVDOV423F2KC4443HPNPHHAAAAAAG44A
      ^^^    ^^^  ^^^^^^ ^  ^^^^^ ^^^^^^^^^^^^^^^^   [34 mismatch]

### cnn  [DECODE-OK] conf1=96% conf2=99%
GT  : P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNPHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E662323994CHN9009121F2601113NFMFMNPBMOMEA976
PRED: E662323994CHN9009121F2601113NFMFMNPBMOMEA976
                                                     [0 mismatch]

### crnn  [decoded] conf1=90% conf2=78%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POGAZA4AG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<< 
       ^^^^^^ ^^ ^^^^^^^                         ^   [16 mismatch]
GT  : E662323994CHN9009121F2601113NFMFMNPBMOMEA976
PRED: 6<4523994CN9009121F2601113NFMEMNPBRONER976  
      ^^^^^^^ ^^^^^^^^^ ^^^^^^ ^^^^^ ^^^^^^^^^^^^^   [40 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=99%
GT  : P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNPHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : E662323994CHN9009121F2601113NFMFMNPBMOMEA976
PRED: E662323994CHN9009121F2601113NFMFMNPBMOMEA976
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (121 ms)
GT  : P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : E662323994CHN9009121F2601113NFMFMNPBMOMEA976
PRED: E662323994CHN9009121F2601113NFMFMNPBMOMEA976
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (107 ms)
GT  : P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : E662323994CHN9009121F2601113NFMFMNPBMOMEA976
PRED: E662323994CHN9009121F2601113NFMFMNPBMOMEA976
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=86% conf2=92%
- backend: indra  (62 ms)
GT  : P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNZHANG<<WANWENK<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                ^                            [2 mismatch]
GT  : E662323994CHN9009121F2601113NFMFMNPBMOMEA976
PRED: E662323994CHN9009121F2601113NFMFMNPBNOMER976
                                          ^   ^      [2 mismatch]

### indra_fullpage  [DECODE-OK] conf1=98% conf2=92%
- backend: indra  (321 ms)
GT  : P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : E662323994CHN9009121F2601113NFMFMNPBMOMEA976
PRED: E662323994CHN9009121F2601113NFMFMNPBMOMEA976
                                                     [0 mismatch]

## 2026-9-30-F.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=287
- 官方 line1: P<CNNYAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: A000000000CHN9801060M24061900DMGNFNM<<<<4998
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `A000000000CHN9801060M24061900DMGNFNM<<<<4998`

### traditional  [decoded] conf1=75% conf2=65%
- error: decode failed: check digit mismatch
GT  : P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AALAAP<A<<2HL<<<<<<<<<<<<<<<<<<<<<<<<<ZZ<ZZ2
      ^^^^^^^^  ^ ^                         ^^ ^^^   [15 mismatch]
GT  : A000000000CHN9801060M24061900DMGNFNM<<<<4998
PRED: WDDDDDDDDDCHNV8O4C6CN24C649CD6AAAVWW<<<<G44A
      ^^^^^^^^^^   ^ ^^^ ^^  ^ ^ ^^^^^^^^^    ^^^^   [31 mismatch]

### cnn  [DECODE-OK] conf1=96% conf2=96%
GT  : P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : A000000000CHN9801060M24061900DMGNFNM<<<<4998
PRED: A000000000CHN9801060M24061900DMGNFNM<<<<4998
                                                     [0 mismatch]

### crnn  [decoded] conf1=89% conf2=81%
- backend: crnn  (18 ms)
- error: decode failed: bad MRZ length
GT  : P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PPCA74A<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
       ^ ^^^ ^ ^^^^                              ^   [10 mismatch]
GT  : A000000000CHN9801060M24061900DMGNFNM<<<<4998
PRED: 420000000CMM9801060M2406190ODMGNFNM<<<<K998 
      ^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^ ^^   [33 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=98%
GT  : P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CNNYAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^                                         [1 mismatch]
GT  : A000000000CHN9801060M24061900DMGNFNM<<<<4998
PRED: A000000000CHN9801060M24061900DMGNFNM<<<<4998
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (218 ms)
GT  : P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMCHNYAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^ ^                                         [3 mismatch]
GT  : A000000000CHN9801060M24061900DMGNFNM<<<<4998
PRED: 0O0O0OO0O6CHN9801060M2406190ODMGNFNM<<<<A996
      ^^ ^ ^^ ^^                  ^           ^  ^   [10 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=99%
- backend: paddle-onnxruntime  (95 ms)
GT  : P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMCHNYAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^ ^                                         [3 mismatch]
GT  : A000000000CHN9801060M24061900DMGNFNM<<<<4998
PRED: A000000000CHN9801060M2406190ODMGNFNM<<<<A998
                                  ^           ^      [2 mismatch]

### indra  [DECODE-OK] conf1=96% conf2=91%
- backend: indra  (67 ms)
GT  : P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMCHNYAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^ ^ ^                                         [3 mismatch]
GT  : A000000000CHN9801060M24061900DMGNFNM<<<<4998
PRED: 4000000O00CHN9801060M24061900DMGNFHM<<<<A998
      ^      ^                          ^     ^      [4 mismatch]

### indra_fullpage  [DECODE-OK] conf1=90% conf2=85%
- backend: indra  (294 ms)
GT  : P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PMCHNYAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<AAAA
       ^ ^ ^                                  ^^^^   [7 mismatch]
GT  : A000000000CHN9801060M24061900DMGNFNM<<<<4998
PRED: 4000000O00CHN9801060M240619000MNHM<<<<1<<<98
      ^      ^                     ^ ^^^^^  ^ ^^     [11 mismatch]

## 2026-9-30-G.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=209
- 官方 line1: P<CHNSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: H224724969CHN0907246M2808185<S3054553<<<<<90
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `H224724969CHN0907246M2808185<S3054553<<<<<90`

### traditional  [decoded] conf1=70% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: PA<CAAXDA<<HAI4HUA<<4<4<<<<<<<<<<<<<<<<<<<<<
       ^^^^^^^^ ^^^^^^^^  ^ ^                        [18 mismatch]
GT  : H224724969CHN0907246M2808185<S3054553<<<<<90
PRED: A22<22<VAVCAADVCZ246H28CB4B9<A3M1A441<<<<<46
      ^  ^^ ^^^^ ^^^^^^   ^  ^^^^^ ^ ^^^^^^     ^^   [28 mismatch]

### cnn  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^                                          [1 mismatch]
GT  : H224724969CHN0907246M2808185<S3054553<<<<<90
PRED: H224724969CHN0907246M2808185<S3054553<<<<<90
                                                     [0 mismatch]

### crnn  [decoded] conf1=91% conf2=81%
- backend: crnn  (19 ms)
- error: decode failed: check digit mismatch
GT  : P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: 7P<CAASDA<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^ ^^^^^^^^                          ^   [18 mismatch]
GT  : H224724969CHN0907246M2808185<S3054553<<<<<90
PRED: <2<Z2<969CMM0907246M2808185<G3O54553<<<<<9S 
      ^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^    ^^^   [38 mismatch]

### fuse  [DECODE-OK] conf1=99% conf2=99%
GT  : P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^                                          [1 mismatch]
GT  : H224724969CHN0907246M2808185<S3054553<<<<<90
PRED: H224724969CHN0907246M2808185<S3054553<<<<<90
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (125 ms)
GT  : P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^                                          [1 mismatch]
GT  : H224724969CHN0907246M2808185<S3054553<<<<<90
PRED: H224724969CHN0907246M2808185<S3054553<<<<<90
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=88% conf2=98%
- backend: paddle-onnxruntime  (105 ms)
GT  : P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^                                          [1 mismatch]
GT  : H224724969CHN0907246M2808185<S3054553<<<<<90
PRED: H224724969CHN0907246M2808185<S3054553<<<<<90
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=85% conf2=93%
- backend: indra  (59 ms)
GT  : P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^                                          [1 mismatch]
GT  : H224724969CHN0907246M2808185<S3054553<<<<<90
PRED: 4Z24724989CHN0907246M2808185<S3054553<<<<<90
      ^^      ^                                      [3 mismatch]

### indra_fullpage  [DECODE-OK] conf1=91% conf2=68%
- backend: indra  (371 ms)
GT  : P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: SPECIALADMHNISTRATTVE<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^^ ^ ^^^ ^^^^                          [18 mismatch]
GT  : H224724969CHN0907246M2808185<S3054553<<<<<90
PRED: C45CODEOF5ECINGSTA<0<<<<<<<0<<<<<<<<<<<<<<08
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^     ^^   [38 mismatch]

## 2026-9-30-H.jpg
- 官方判定: PASS  verified=True  chosen=cnn  ms=209
- 官方 line1: P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: KE90003518CHN9707014F2912268A911<121101<<<72
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `KE90003518CHN9707014F2912268A911<121101<<<72`

### traditional  [decoded] conf1=72% conf2=68%
- error: decode failed: check digit mismatch
GT  : P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<<CAAPXAG<<2HEK<<<<<<<<<<<<<<<<<<<<<Z222222
        ^^^^^^^^ ^^^^^                     ^^^^^^^   [20 mismatch]
GT  : KE90003518CHN9707014F2912268A911<121101<<<72
PRED: <LF66SJS78CHAFZCZC34P29422A8A944<4244A4222SS
      ^^^^^^^^^   ^^^^^^^ ^  ^  ^   ^^ ^ ^^^^^^^^^   [31 mismatch]

### cnn  [DECODE-OK] conf1=94% conf2=99%
GT  : P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : KE90003518CHN9707014F2912268A911<121101<<<72
PRED: KE90003518CHN9707014F2912268A911<121101<<<72
                                                     [0 mismatch]

### crnn  [decoded] conf1=92% conf2=85%
- backend: crnn  (19 ms)
- error: decode failed: check digit mismatch
GT  : P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
PRED: 7<4<CAY4AG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^ ^^^^^^^^ ^^^^^                            ^   [15 mismatch]
GT  : KE90003518CHN9707014F2912268A911<121101<<<72
PRED: KF90O3518CMA9707014F2912268A941<124104<<<72 
       ^  ^^^^^^^^^^^^^^^^^^^^ ^^^^^ ^^^^ ^^^  ^^^   [36 mismatch]

### fuse  [DECODE-OK] conf1=94% conf2=99%
GT  : P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : KE90003518CHN9707014F2912268A911<121101<<<72
PRED: KE90003518CHN9707014F2912268A911<121101<<<72
                                                     [0 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (128 ms)
GT  : P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AKCHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^                                             [2 mismatch]
GT  : KE90003518CHN9707014F2912268A911<121101<<<72
PRED: KE90003518CHN9707014F2912268A911<121101<<<72
                                                     [0 mismatch]

### paddle  [DECODE-OK] conf1=98% conf2=99%
- backend: paddle-onnxruntime  (100 ms)
GT  : P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AKCHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^                                             [2 mismatch]
GT  : KE90003518CHN9707014F2912268A911<121101<<<72
PRED: KE90003518CHN9707014F2912268A911<121101<<<72
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=89% conf2=96%
- backend: indra  (51 ms)
GT  : P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AKCHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^                                             [2 mismatch]
GT  : KE90003518CHN9707014F2912268A911<121101<<<72
PRED: KE90003518CHN9707014F2912268A911<121101<<<72
                                                     [0 mismatch]

### indra_fullpage  [DECODE-OK] conf1=96% conf2=56%
- backend: indra  (236 ms)
GT  : P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: KEASO03518CHN9707014F2912268A911<121101<<<72
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^   ^^   [40 mismatch]
GT  : KE90003518CHN9707014F2912268A911<121101<<<72
PRED: KTOBEASCA2ASOSOLVES6TSSESV<3<<<<<<<<<<<<<<06
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^   ^^   [39 mismatch]

## 2026-9-30-I.jpg
- 官方判定: REJECT  verified=False  chosen=None  ms=14
- 理由: 未在照片中定位到 MRZ 双行

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [FAIL]
- backend: indra  (125 ms)
- error: indra 未找到 MRZ 候选行 (1 行)

## 2026-9-30-J.jpg
- 官方判定: REJECT  verified=False  chosen=None  ms=9
- 理由: 未在照片中定位到 MRZ 双行

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [FAIL]
- backend: indra  (129 ms)
- error: indra 未找到 MRZ 候选行 (1 行)

## 2026-9-30-K.jpg
- 官方判定: REJECT  verified=False  chosen=None  ms=8
- 理由: 未在照片中定位到 MRZ 双行

### traditional  [no_pair]

### cnn  [no_pair]

### crnn  [no_pair]

### fuse  [no_pair]

### tesseract  [no_pair]

### paddle  [no_pair]

### indra  [no_pair]

### indra_fullpage  [FAIL]
- backend: indra  (130 ms)
- error: indra 未找到 MRZ 候选行 (1 行)

## IMG_1792133000_003.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=322
- 官方 line1: P<SNNSJN<<LINEN<<<PPPPPPPP<<<<<<<<<<<<<<<<<<
- 官方 line2: EC20426530CHN7410024M2808303LMN0MKKEMG9BA934
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<`
- GT line2: `EE20426536CHN7410024M2808303LMNGMBKENG9BA970`

### traditional  [decoded] conf1=76% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: MPMC9A3IX<<L1N1H<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^ ^^^^^^  ^^^^^^^^        ^            [24 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: MFFZQ<2<S36CHHZ4JQQP4NP8C8<C<LAHGAA<LA6LAF43
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^   [43 mismatch]

### cnn  [DECODE-OK] conf1=86% conf2=91%
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
                                                     [0 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKENG9BA970
                                                     [0 mismatch]

### crnn  [decoded] conf1=88% conf2=81%
- backend: crnn  (19 ms)
- error: decode failed: bad MRZ length
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCA1A<<E7MIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
       ^^^^^^^^^^  ^^   ^^^^^^^^        ^       ^^   [23 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EC0426536CAN7410024M2808303LMNGW8KCND88RS1S 
       ^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^   [42 mismatch]

### fuse  [DECODE-OK] conf1=92% conf2=92%
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<SNNSJN<<LINEN<<<PPPPPPPP<<<<<<<<<<<<<<<<<<
                ^         ^             ^            [3 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EC20426530CHN7410024M2808303LMN0MKKEMG9BA934
       ^       ^                     ^ ^  ^     ^^   [7 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (234 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^ ^^^  ^ ^^    ^^^^^^^^        ^            [17 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCWDPBA970
                                         ^^^^        [4 mismatch]

### paddle  [DECODE-OK] conf1=93% conf2=99%
- backend: paddle-onnxruntime  (122 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^    ^^^^^^^^        ^            [18 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
                                         ^^^^        [4 mismatch]

### indra  [FAIL]
- backend: indra  (50 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=97% conf2=87%
- backend: indra  (292 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^    ^^^^^^^^        ^            [18 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCNPSB9<70
                                         ^ ^^ ^^     [5 mismatch]

## IMG_1792133002_004.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=323
- 官方 line1: P<SNNSJN<<LINEN<<<PPPPPPPP<<<<<<<<<<<<<<<<<<
- 官方 line2: EC20426530CHN7410024M2808303LMN0MKKEMG9BA934
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<`
- GT line2: `EE20426536CHN7410024M2808303LMNGMBKENG9BA970`

### traditional  [decoded] conf1=76% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: MPMC9A3IX<<L1N1H<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^ ^^^^^^  ^^^^^^^^        ^            [24 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: MFFZQ<2<S36CHHZ4JQQP4NP8C8<C<LAHGAA<LA6LAF43
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^   [43 mismatch]

### cnn  [DECODE-OK] conf1=86% conf2=91%
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
                                                     [0 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKENG9BA970
                                                     [0 mismatch]

### crnn  [decoded] conf1=88% conf2=81%
- backend: crnn  (20 ms)
- error: decode failed: bad MRZ length
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCA1A<<E7MIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
       ^^^^^^^^^^  ^^   ^^^^^^^^        ^       ^^   [23 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EC0426536CAN7410024M2808303LMNGW8KCND88RS1S 
       ^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^   [42 mismatch]

### fuse  [DECODE-OK] conf1=92% conf2=92%
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<SNNSJN<<LINEN<<<PPPPPPPP<<<<<<<<<<<<<<<<<<
                ^         ^             ^            [3 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EC20426530CHN7410024M2808303LMN0MKKEMG9BA934
       ^       ^                     ^ ^  ^     ^^   [7 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (233 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^ ^^^  ^ ^^    ^^^^^^^^        ^            [17 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCWDPBA970
                                         ^^^^        [4 mismatch]

### paddle  [DECODE-OK] conf1=93% conf2=99%
- backend: paddle-onnxruntime  (108 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^    ^^^^^^^^        ^            [18 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
                                         ^^^^        [4 mismatch]

### indra  [FAIL]
- backend: indra  (47 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=97% conf2=87%
- backend: indra  (288 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^    ^^^^^^^^        ^            [18 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCNPSB9<70
                                         ^ ^^ ^^     [5 mismatch]

## IMG_1792133232_005.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=337
- 官方 line1: P<SNNSJN<<LINEN<<<PPPPPPPP<<<<<<<<<<<<<<<<<<
- 官方 line2: EC20426530CHN7410024M2808303LMN0MKKEMG9BA934
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<`
- GT line2: `EE20426536CHN7410024M2808303LMNGMBKENG9BA970`

### traditional  [decoded] conf1=76% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: MPMC9A3IX<<L1N1H<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^ ^^^^^^  ^^^^^^^^        ^            [24 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: MFFZQ<2<S36CHHZ4JQQP4NP8C8<C<LAHGAA<LA6LAF43
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^   [43 mismatch]

### cnn  [DECODE-OK] conf1=86% conf2=91%
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
                                                     [0 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKENG9BA970
                                                     [0 mismatch]

### crnn  [decoded] conf1=88% conf2=81%
- backend: crnn  (23 ms)
- error: decode failed: bad MRZ length
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCA1A<<E7MIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
       ^^^^^^^^^^  ^^   ^^^^^^^^        ^       ^^   [23 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EC0426536CAN7410024M2808303LMNGW8KCND88RS1S 
       ^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^   [42 mismatch]

### fuse  [DECODE-OK] conf1=92% conf2=92%
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<SNNSJN<<LINEN<<<PPPPPPPP<<<<<<<<<<<<<<<<<<
                ^         ^             ^            [3 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EC20426530CHN7410024M2808303LMN0MKKEMG9BA934
       ^       ^                     ^ ^  ^     ^^   [7 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (239 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^ ^^^  ^ ^^    ^^^^^^^^        ^            [17 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCWDPBA970
                                         ^^^^        [4 mismatch]

### paddle  [DECODE-OK] conf1=93% conf2=99%
- backend: paddle-onnxruntime  (107 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^    ^^^^^^^^        ^            [18 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
                                         ^^^^        [4 mismatch]

### indra  [FAIL]
- backend: indra  (47 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=97% conf2=87%
- backend: indra  (287 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^    ^^^^^^^^        ^            [18 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCNPSB9<70
                                         ^ ^^ ^^     [5 mismatch]

## IMG_1792133581_006.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=313
- 官方 line1: P<SNNSJN<<LINEN<<<PPPPPPPP<<<<<<<<<<<<<<<<<<
- 官方 line2: EC20426530CHN7410024M2808303LMN0MKKEMG9BA934
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<`
- GT line2: `EE20426536CHN7410024M2808303LMNGMBKENG9BA970`

### traditional  [decoded] conf1=76% conf2=64%
- error: decode failed: check digit mismatch
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: MPMC9A3IX<<L1N1H<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^^ ^^^^^^  ^^^^^^^^        ^            [24 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: MFFZQ<2<S36CHHZ4JQQP4NP8C8<C<LAHGAA<LA6LAF43
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^   [43 mismatch]

### cnn  [DECODE-OK] conf1=86% conf2=91%
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
                                                     [0 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKENG9BA970
                                                     [0 mismatch]

### crnn  [decoded] conf1=88% conf2=81%
- backend: crnn  (20 ms)
- error: decode failed: bad MRZ length
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCA1A<<E7MIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
       ^^^^^^^^^^  ^^   ^^^^^^^^        ^       ^^   [23 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EC0426536CAN7410024M2808303LMNGW8KCND88RS1S 
       ^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^   [42 mismatch]

### fuse  [DECODE-OK] conf1=92% conf2=92%
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<SNNSJN<<LINEN<<<PPPPPPPP<<<<<<<<<<<<<<<<<<
                ^         ^             ^            [3 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EC20426530CHN7410024M2808303LMN0MKKEMG9BA934
       ^       ^                     ^ ^  ^     ^^   [7 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (227 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^ ^^^  ^ ^^    ^^^^^^^^        ^            [17 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCWDPBA970
                                         ^^^^        [4 mismatch]

### paddle  [DECODE-OK] conf1=93% conf2=99%
- backend: paddle-onnxruntime  (111 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^    ^^^^^^^^        ^            [18 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
                                         ^^^^        [4 mismatch]

### indra  [FAIL]
- backend: indra  (48 ms)
- error: indra 未找到 MRZ 候选行 (1 行)
- PRED line1: <空>
- PRED line2: <空>

### indra_fullpage  [DECODE-OK] conf1=97% conf2=87%
- backend: indra  (283 ms)
GT  : P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^    ^^^^^^^^        ^            [18 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKENG9BA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCNPSB9<70
                                         ^ ^^ ^^     [5 mismatch]

## IMG_1792133649_007.jpg
- 官方判定: PASS  verified=True  chosen=fuse  ms=472
- 官方 line1: P<CHNHIA<<LININ<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: EE20426536CHN7410024M2808303LMNGMBKENNBNN956
- 理由: 融合CNN+CRNN 识别并通过校验

- GT line1: `P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `EE20426536CHN7410024M28083034MNGMNKFH6BB8994`

### traditional  [decoded] conf1=72% conf2=65%
- error: decode failed: check digit mismatch
GT  : P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: <<<A<L7<<<<J1A1N<<<<<<<<<<<<<<<<<<<<<<2I2IJI
      ^ ^^^^^^  ^^^^^^                      ^^^^^^   [19 mismatch]
GT  : EE20426536CHN7410024M28083034MNGMNKFH6BB8994
PRED: <<2K<2<S3<LA4Z43FVPAPPHCH3D3LPWGPA<LA566<<<L
      ^^ ^^ ^^ ^^^^^ ^^^^^^^^^^ ^ ^^^ ^^^^^^^^^^^^   [37 mismatch]

### cnn  [skip]

### crnn  [decoded] conf1=92% conf2=50%
- backend: crnn  (15 ms)
- error: decode failed: bad MRZ length
GT  : P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<<4<<LIIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<     
        ^^^^^^^^^^^^^                        ^^^^^   [18 mismatch]
GT  : EE20426536CHN7410024M28083034MNGMNKFH6BB8994
PRED: <<<<5<CA400QM2COA9OSLMNCMB<QMP9K976         
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   [44 mismatch]

### fuse  [DECODE-OK] conf1=96% conf2=97%
GT  : P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNHIA<<LININ<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           ^      ^                                  [2 mismatch]
GT  : EE20426536CHN7410024M28083034MNGMNKFH6BB8994
PRED: EE20426536CHN7410024M2808303LMNGMBKENNBNN956
                                  ^    ^ ^^^ ^^ ^^   [9 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (449 ms)
GT  : P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                                                     [0 mismatch]
GT  : EE20426536CHN7410024M28083034MNGMNKFH6BB8994
PRED: EE20426536CHN7410024M2808303LMNGMBKCMDPSA944
                                  ^    ^ ^^^^^^ ^    [9 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=99%
- backend: paddle-onnxruntime  (74 ms)
GT  : P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^                                             [1 mismatch]
GT  : EE20426536CHN7410024M28083034MNGMNKFH6BB8994
PRED: EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
                                  ^    ^ ^^^^ ^ ^^   [9 mismatch]

### indra  [DECODE-OK] conf1=86% conf2=88%
- backend: indra  (61 ms)
GT  : P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<CACEA<
       ^                                    ^^^^^    [6 mismatch]
GT  : EE20426536CHN7410024M28083034MNGMNKFH6BB8994
PRED: EE20426536CHN7410024M2808303LMNGMBKCNS3<<<94
                                  ^    ^ ^^^^^^^     [9 mismatch]

### indra_fullpage  [DECODE-OK] conf1=83% conf2=89%
- backend: indra  (275 ms)
GT  : P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNJIAKLIMINKKKKK<<<<<<<<<<<<<<<<CCACEAE<<
       ^      ^^^^^^^^^^^                ^^^^^^^     [19 mismatch]
GT  : EE20426536CHN7410024M28083034MNGMNKFH6BB8994
PRED: EE20426536CHN7410024M2808303LMNGMBKCM5PSR970
                                  ^    ^ ^^^^^^ ^^   [10 mismatch]

## IMG_1792133682_008.jpg
- 官方判定: PASS  verified=True  chosen=indra  ms=222
- 官方 line1: <<CHN<<<<<<<<<<<<<<A<<<<<<<<<<<<<<<<<<<<<<<<
- 官方 line2: EE20426536CHN7410024M2808303LMNGMSK<<<<<<<20
- 理由: IndraOCR 识别并通过校验

- GT line1: `P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
- GT line2: `EE20426536CHN7410024M2808303LMNGMBKCMDPBA970`

### traditional  [decoded] conf1=71% conf2=60%
- error: decode failed: check digit mismatch
GT  : P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: AFALAAJ7<<<L7NJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ^^^^^^^^  ^^^^^^                               [14 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
PRED: PFP24<ZNSJ<CHNZ4YQOZ4NZ0Q8JC3LHHLAA<LA6A6F4<
      ^^^^ ^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^   [42 mismatch]

### cnn  [decoded] conf1=82% conf2=63%
- error: decode failed: check digit mismatch
GT  : P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<QSV<<<<<SKYJN<<<SS<<<<<<<<<<<<<<<<<<<<<<<<
          ^      ^      ^^                           [4 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
PRED: <827K5G<33BNE7004582M75300140M502UUHO49U1C19
      ^^ ^^^^^ ^^^^ ^^^^^^ ^^^^^^^^ ^^^^^^^^^^^^^^   [39 mismatch]

### crnn  [decoded] conf1=86% conf2=83%
- backend: crnn  (21 ms)
- error: decode failed: bad MRZ length
GT  : P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POC74<<EIM7N<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   
       ^^^^  ^^^^^^^^                          ^^^   [15 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
PRED: 7<E426536CAA7470024M2808303LMNGMBKCNDP8R97Q 
      ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^   [43 mismatch]

### fuse  [decoded] conf1=87% conf2=71%
- error: decode failed: check digit mismatch
GT  : P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<QSV<<<<<IJYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
          ^     ^^                                   [3 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
PRED: 8827K5G033BNE7000582M7590014GM502UO484YU1110
      ^^ ^^^^^ ^^^^ ^^ ^^^ ^^^^^^^^ ^^^^^^^^^^^^^    [37 mismatch]

### tesseract  [DECODE-OK]
- backend: tesseract  (107 ms)
GT  : P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        ^^ ^^^  ^ ^^                                 [8 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCNDPBR970
                                          ^   ^      [2 mismatch]

### paddle  [DECODE-OK] conf1=96% conf2=99%
- backend: paddle-onnxruntime  (111 ms)
GT  : P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^                                 [9 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
PRED: EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
                                                     [0 mismatch]

### indra  [DECODE-OK] conf1=99% conf2=84%
- backend: indra  (70 ms)
GT  : P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: POCHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
       ^^^ ^^^  ^ ^^                                 [9 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
PRED: EE20426536CHN741002428083038MNGMBKCMOP<<<<58
                          ^^^^^^^^^^^^^^^^^^^^^^^^   [24 mismatch]

### indra_fullpage  [DECODE-OK] conf1=86% conf2=58%
- backend: indra  (460 ms)
GT  : P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
PRED: EEUSE26536CHN7410024M2808303LMNGMBKCMOPAG79<
      ^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    [42 mismatch]
GT  : EE20426536CHN7410024M2808303LMNGMBKCMDPBA970
PRED: SCHASEIOO9USEOSSOET9MENEKEN6ARVA<<<<<<<<<<28
      ^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^   [43 mismatch]

## 引擎汇总

| engine | ran | decode_ok | exact(GT) | char acc |
|---|---|---|---|---|
| traditional | 106 | 0 | 0 | 41.9% |
| cnn | 89 | 67 | 37 | 92.7% |
| crnn | 106 | 13 | 1 | 74.3% |
| fuse | 106 | 84 | 37 | 92.6% |
| tesseract | 106 | 99 | 5 | 86.0% |
| paddle | 106 | 99 | 17 | 91.6% |
| indra | 106 | 88 | 2 | 70.0% |
| indra_fullpage | 115 | 103 | 1 | 65.5% |

## 需人工核对的照片（98 张，GT 已缺失或全引擎不匹配）

- [08806d29a6268e1f_0003.jpg] (无引擎完全匹配)
    l1=`P<CHEDE<MAIENFELD<<PIERRE<<<<<<<<<<<<<<<<<<<`
    l2=`S2100023<6CHE7108019M2002293<<<<<<<<<<<<<<06`
- [0f9780b8e1e3cc21_0008.jpg] (无引擎完全匹配)
    l1=`P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`3400072370USA6707046F1608078910000193<113538`
- [16c07a99a0fb81e4_0013.jpg] (无引擎完全匹配)
    l1=`P<CZEWPECIMEN<<NGOH<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`R900R054<4CZL6906229F16072996956220612<KKK74`
- [1cce0f3fc387c893_0001.jpg] (无引擎完全匹配)
    l1=`P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`5C39917886POL8101028M2909185<<<<<<<<<<<<<<04`
- [239e042aba0c3ab2_0102.jpg] (2)
    l1=`P<CHNYANG<<CHAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`5B00000008CHN8509037F1505018192000015555KK92`
- [23f16a7d9b6ab9da_0062.jpg] (无引擎完全匹配)
    l1=`P<UKRGRYTSENKO<<IVAN<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`4400000147UKH8307255M19080333052125257S5S500`
- [28a42952ba25584a_0078.png] (无引擎完全匹配)
    l1=`P<CHNNGKELNPONBPJLBLO<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`Q100547988CHN8410124F1907173192044155<<5<<46`
- [28ccd101a752acef_0016.jpg] (1)
    l1=`P<PHLDELA<CRUZ<<MARIA<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`P0000000A0PHL8003166F2106269<<<<<<<<<<<<<<02`
- [2c793efe12276f9b_0258.jpg] (2)
    l1=`P<LVAOZOLKALNA<<LIENE<MARA<<<<<<<<<<<<<<<<<<`
    l2=`LV90008223LVA8612010F2501121111111<11410<<70`
- [31611582cbf14d7f_0018.jpg] (无引擎完全匹配)
    l1=`P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`ZF40004998P0L8101028M3203031<<<<<<<<<<<<<<00`
- [4086f51e94e9c5c7_0007.jpg] (2)
    l1=`P<GUSSHIRNONA<<NALENYINA<<<<<<<<<<<<<<<<<<<<`
    l2=`1100000000HUS8008046F0902274555<55555<5K5K08`
- [41ab45d4bde758dc_0306.jpg] (2)
    l1=`P<AGRHUSEYNLI<<ORKHAN<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`3110003442AZE7503153M230801030KJU52555<<<<86`
- [43ecb49033fb30c1_0073.jpg] (无引擎完全匹配)
    l1=`P<UIKYAUNZO<<ONDINA<MA<HSLLE<<<<<<<<<<<<<<<<`
    l2=`0418276532USA3309209F9707014<<<<<<<<<<<<<<06`
- [4dbd1b9e24b069f4_0023.png] (无引擎完全匹配)
    l1=`P<DSYNUSYERNANN<<ENIKA<<<<<<<<<<<<<<<<<<<<<<`
    l2=`C01X00T478D<<6408125F2702283<<<<<<<<<<<<<<04`
- [526d2e38b6b977bc_0014.jpg] (2)
    l1=`PEROUPOPESCU<<MIHAELA<STEFANIA<<<<<<<<<<<<<<`
    l2=`0400210397ROU8001014F34090284800101017005<70`
- [52f421c92ccf6ba4_0136.jpg] (1)
    l1=`P<AUBCITIZEN<<JANK<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`KCG01505B6U<O9105045F2509013N999990001<<5<82`
- [6b0e5919dd6f3ac8_0006.jpg] (无引擎完全匹配)
    l1=`P<SSPRSPANOLA<ESPANDLK<<EKNNEN<<<<<<<<<<<<<<`
    l2=`ZAB0002549ESP8001014F2501017A9999999900<<<44`
- [6b91117492dc24f3_0022.jpg] (1)
    l1=`P<UHNAL<SIYARI<<ISSA<ARBULLAH<SAEEH<<<<<<<<<`
    l2=`JG20705515OMN7803048M2411185300036145<<<5<50`
- [7294ad26ea8d95e0_0010.jpg] (无引擎完全匹配)
    l1=`P<GGAFARIAS<DOS<SANYOS<<KODRIGO<<<<<<<<<<<<<`
    l2=`AA000261<7BRA0403162M2507053<<<<<<<<<<<<<<00`
- [7626df394c39f314_0020.png] (无引擎完全匹配)
    l1=`P<BHREBRAHIM<<MOHAMMED<ABDULLA<YOUSIF<<<<<<<`
    l2=`B0000000<7BHM9001284M3212161000000000<<K<K04`
- [7c5899e1f86a07a0_0008.jpg] (无引擎完全匹配)
    l1=`P<LTURASANAVICIENE<<BIRUTE<<<<<<<<<<<<<<<<<<`
    l2=`00000000<0LTU9003118F290102049003111045<<<86`
- [7c6f47bad47d70e9_0230.jpg] (无引擎完全匹配)
    l1=`P<AUSCIYIZEN<<JANE<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`RA01234564AUS9105045F3208014<99999000T<<<<86`
- [86c6d990773daff2_0002.jpg] (无引擎完全匹配)
    l1=`P<MLTMIGALLBF<<GIOWANND<<<<<<<<<<<<<<<<<<<<<`
    l2=`0981489<<5MLT6811118M2006268<<<<<<<<<<<<<<06`
- [8a5e3bc18a6ff7d2_0043.jpg] (无引擎完全匹配)
    l1=`P<NLDDE<BRUIJN<<WILLEKE<LISELOYYE<<<<<<<<<<<`
    l2=`18ECI20142NLS6503101F2403096999999990<<<<<84`
- [8de1f04907363a2d_0000.jpg] (1)
    l1=`P<INJAE<ASSDGYYIG<<YYUGIGUG<GESP<<<<<<<<<<<<`
    l2=`82040011<0ISL1212123F1106297121212<1239<<<66`
- [90b464ea218a9909_0001.jpg] (无引擎完全匹配)
    l1=`P<SYCSANPLE<<ORDINARY<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`NS000004<7SYC8601137F3203101<<<<<<<<<<<<<<00`
- [927cd546b5a8111d_0108.jpg] (无引擎完全匹配)
    l1=`P<CHNLOI<<HANG<MIO<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`MB00000007CHN8807191F291203910004067<<<<<<28`
- [a495fb368b46ab35_0280.png] (1)
    l1=`P<CHNCESHI<<YANGBEN<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`G622925996CHN8310291F19042204C0CMXNENBPJB984`
- [a60aae34a06f1867_0016.jpg] (无引擎完全匹配)
    l1=`P<UKXTKACH<NKO<<MA<<ANA<<<<<<<<<<<<<<<<<<<<<`
    l2=`X30OO0OO<0UKW9108242F23092571234567890<<<<70`
- [a955e21ae58d8c1d_0013.jpg] (1)
    l1=`P<SYRNANSOUR<<DANI<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`N000000001SYH8801018M27082410104006915555<12`
- [acd3e9fd1ce9bd0a_0016.jpg] (2)
    l1=`P<GRRUK<SPECIMEN<<ANGELA<ZOE<<<<<<<<<<<<<<<<`
    l2=`5334755143GBH8812049F25092865<<<<<<<SD<SDS78`
- [b1442018a8167bf5_0046.jpg] (2)
    l1=`P<USATRAVELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E000077347USA6502056F3010149500100120<095838`
- [b284565582147454_0098.jpg] (1)
    l1=`P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`5E75870985KCN8507022F2404152213471C117621884`
- [b284565582147454_0115.jpg] (1)
    l1=`P<NPALFF<<SUAS<N<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`5E75870985KCN8507022F2404152213471C117621884`
- [b2b5008f3839530d_0001.jpg] (1)
    l1=`P<SNNNIORDC<<ANA<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`M0N0062122SVN7811182F261212218IIA785M565EK14`
- [b30dc79ee9550e9e_0024.jpg] (无引擎完全匹配)
    l1=`P<DWLBPECDNDN<<YHAROD<BPBGINBN<<<<<<<<<<<<<<`
    l2=`K15M12M1V5ISK1212123F290204912M2M24424VNN462`
- [bc1ddfd8c13b5ca9_0007.jpg] (2)
    l1=`P<YRASPBCSNDN<ABB<<NAYACHA<<<<<<<<<<<<<<<<<<`
    l2=`9MM1024318MMA7307122F29030355N5NNN554N44K902`
- [c0cbf7c176328f3d_0006.jpg] (无引擎完全匹配)
    l1=`P<NNNNDDANND<<QNNNQWNW<<<<<<<<<<<<<<<<<<<<<<`
    l2=`AA12345678GRC7505157M3312108<<<<<<<<<<<<<<06`
- [c2aee55b9017575f_0028.jpg] (无引擎完全匹配)
    l1=`P<NUSOBRAZEYS<<YKSYKR<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`D2DM1601M8KBS5608087M2002156<<<<<<<<<<<<<<08`
- [cb64e780ab32f735_0004.jpg] (2)
    l1=`P<SNKSPECIHEN<<NIORKA<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`7B12318830SVK1111112F34091871111111111<<<<06`
- [cd83b43756aff32d_0013.jpg] (无引擎完全匹配)
    l1=`P<HAQSPGCINNN<<SPNCINGN<<<<<<<<<<<<<<<<<<<<<`
    l2=`0M10070071HAV8212258F140701955535SSKKKKKKK56`
- [dd6a2ea3ce305a5d_0192.jpg] (无引擎完全匹配)
    l1=`P<PREJANG<<NYONGSUN<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`6451331382PRK6609302F1003145<<<<<<<<<<<<<<04`
- [e350cea71e824345_0007.jpg] (无引擎完全匹配)
    l1=`P<BGRDWANOWAV<NARIYBA<NADNDWA<<<<<<<<<<<<<<<`
    l2=`3000000001BGH8508018F150320945040M0M33NNNN54`
- [e564d4372e28b9ed_0018.jpg] (无引擎完全匹配)
    l1=`P<BRBYRBY<<NDLDGA<<<<<<<<<<<<<<<<Q<<<<<<<<<<`
    l2=`0000000000SRB6802295F22101042902968000000<40`
- [e726442c11085334_0090.png] (无引擎完全匹配)
    l1=`P<BWBBHBCINBN<<BNBA<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`XA00000012SWE8208215F2701013198208212384<<22`
- [f22471c46575ec04_0008.jpg] (无引擎完全匹配)
    l1=`P<AUYJPECIMEN<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`AP12345673AUT0012315F3311307<<<<<<<<<<<<<<08`
- [f385b0113d785dfd_0040.jpg] (无引擎完全匹配)
    l1=`P<POLKOWALSKI<<JAN<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`ZF40004998P0L8101028M3203031<<<<<<<<<<<<<<00`
- [f6132d0c5422d26a_0026.jpg] (无引擎完全匹配)
    l1=`P<NORORSTRNBYRN<<AASAMUND<SPECINEN<<<<<<<<<<`
    l2=`CCC0022514HOH5604230M30041575<<<5<<<<<<<<<04`
- [f6eeb483a66e8d94_0084.jpg] (无引擎完全匹配)
    l1=`P<USATRANELER<<HAPPY<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E000093499USA6502056M3007099340006673<085950`
- [f80334b200240666_0034.png] (无引擎完全匹配)
    l1=`P<ANGGONZALES<RODRIGUEZ<MARIANA<SLORRNCIA<<<`
    l2=`8880000008ABG5702147M3303023000000000<<<<<08`
- [ffa1dc640314fa58_0089.png] (2)
    l1=`P<FINVIRTANEN<<MARIA<OLIVIA<<<<<<<<<<<<<<<<<`
    l2=`FX12345672FIN7112214F2201018211271<426U<<<00`
- [2026-9-29-A.jpg] (2)
    l1=`P<CHNYUN<<YAOYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E0132B7855CHN0110141M3105179GB682050708G0B88`
- [2026-9-29-B.jpg] (3)
    l1=`P<CHNZHAO<<YONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E216923764CHN6703222M3510203H01JE6IP966FG406`
- [2026-9-29-C.jpg] (无引擎完全匹配)
    l1=`P<CHNZHOU<<YUAASU<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E639021O58CHN7005059F31101530S0Z68SZ0B110074`
- [2026-9-29-D.jpg] (无引擎完全匹配)
    l1=`P<CHNNAO<<HUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E797044113CHN9004300F3405226GE0660IR06BAI310`
- [2026-9-29-E.jpg] (2)
    l1=`P<CHNCHU<<YOUYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E454243381CHN0703013F4211172S6310BB626637626`
- [2026-9-29-F.jpg] (2)
    l1=`P<CHNDENG<<PENGWEN<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E410968B47CHN9311103F311203262G49S6626Z92252`
- [2026-9-29-G.jpg] (1)
    l1=`P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E505029416CHN6806178M2604114L2YBF2201K4IF010`
- [2026-9-29-H.jpg] (无引擎完全匹配)
    l1=`P<CHNDING<<PINGMO<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`EZ05029416CHN6806178M2604114L2YBF2Z01KA1F034`
- [2026-9-29-I.jpg] (2)
    l1=`P<CHNGE<<MOHUA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E592235158CHN6905163M30012716G0B612258S2B846`
- [2026-9-29-J.jpg] (3)
    l1=`P<CHNGE<<SUDONG<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E062480684CHN8201113F251225501S8IBBG51B52Z66`
- [2026-9-29-K.jpg] (3)
    l1=`P<CHNGUO<<LURU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E916769I46CHN6008248M240122126QZG26600S16132`
- [2026-9-29-L.jpg] (3)
    l1=`P<CHNGU<<YINGYOU<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E665282843CHN8807054M28102160012032119B08660`
- [2026-9-29-M.jpg] (1)
    l1=`P<CHNGU<<YUYU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E096730195CHN8410021M300821488S61022S7Z59328`
- [2026-9-29-N.jpg] (3)
    l1=`P<CHNSU<<SHUYUE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E653505611CHN8409285F3001167B0BS1G1886G2O864`
- [2026-9-29-O.jpg] (4)
    l1=`P<CHNWEN<<YUEYUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E421853076CHN7906282F3506190JJ6MVT1JVTBGC610`
- [2026-9-29-P.jpg] (3)
    l1=`P<CHNWU<<FANGYAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E766622355CHN0011293F2604181Y861DEUSDLKZG542`
- [2026-9-29-Q.jpg] (2)
    l1=`P<CHNWU<<RUZHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E436102633CHN8211158F32082106GIB15B16Z6BB264`
- [2026-9-29-R.jpg] (3)
    l1=`P<CHNXI<<YU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E123241791CHN6208185M36010409K94WG6UGCK6FR64`
- [2026-9-29-S.jpg] (2)
    l1=`P<CHNYUAN<<YOUSUI<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E62588I562CHN9505209F3503269G1Z152520121S504`
- [2026-9-29-T.jpg] (2)
    l1=`P<CHNYUAN<<YUESUI<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E0755268S1CHN9004012M27090195GI5BBISB85B1O26`
- [2026-9-29-U.jpg] (无引擎完全匹配)
    l1=`P<CHNYUE<<<<ESHU<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E158193223CHN7807091M2505233356G5111BO7S6134`
- [2026-9-29-W.jpg] (无引擎完全匹配)
    l1=`P<CHNYUN<<PUYING<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E048156G62CHN6209285F35122520X50Z1GZG2581G50`
- [2026-9-30-A.jpg] (无引擎完全匹配)
    l1=`P<CHNHAO<<MEINN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E821973372CHN8612261F2607078MIMEMDMEMEMIA944`
- [2026-9-30-AA.jpg] (无引擎完全匹配)
    l1=`P<CHNLIN<<YUHLAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E940780938CHN8410179F3401077KJYROL12DY2WW670`
- [2026-9-30-AB.jpg] (无引擎完全匹配)
    l1=`P<CHN0AO<<YUEYA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E602701156CHN6901084F2808222Q75OWNC9HSVEJZ42`
- [2026-9-30-AC.jpg] (2)
    l1=`P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E141543482CHN6011103M32062610MU9FH60NPJJ1G24`
- [2026-9-30-AC_R.jpg] (2)
    l1=`P<CHNBU<<KAIYUAN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E141543482CHN6011103M32062610MU9FH60NPJJ1G24`
- [2026-9-30-AD.jpg] (3)
    l1=`P<CHNCAI<<FANGMENG<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E277194572CHN9601189F30052246MN1TZC2X4MC4X60`
- [2026-9-30-AE.jpg] (3)
    l1=`P<CHNCHANG<<NI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E309178198CHN9612202F2702261G3JPA90J2I<K6Z66`
- [2026-9-30-AF.jpg] (无引擎完全匹配)
    l1=`P<CHNCHEN<<<AJUN<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E646452438CHN0211196M3203075FIJJWA3L90ZU0054`
- [2026-9-30-AG.jpg] (无引擎完全匹配)
    l1=`P<CHNCONG<<<INGXIAO<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E066908744CHN9303298F34112631WW9ZQLBES70S062`
- [2026-9-30-AH.jpg] (3)
    l1=`P<CHNCUI<<LIYI<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E581043494CHN7012299F3110201S1UBJ2E<3D3PLJ56`
- [2026-9-30-AI.jpg] (2)
    l1=`P<CHNDAI<<XIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E996968603CHN0205089M32081840K1U4WCPWK4E7L42`
- [2026-9-30-AJ.jpg] (无引擎完全匹配)
    l1=`P<CHNDCNG<<HNNEYAN<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E818572393CHN6603171M2702157EBS91S800WHI6I88`
- [2026-9-30-B.jpg] (无引擎完全匹配)
    l1=`P<CHNNU<<HAO<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E187424516CHN8706216M2406086LKPKOFLA<<<<A936`
- [2026-9-30-C.jpg] (5)
    l1=`P<AUSRICHARDS<<KEIRA<JAYNE<<<<<<<<<<<<<<<<<<`
    l2=`N7368155<2AUS8703226F2306186<12078730W<<<<68`
- [2026-9-30-D.jpg] (2)
    l1=`P<CHNZHOU<<QIANG<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`KE00345952CHN9207156M2912095A911<611158<<<14`
- [2026-9-30-E.jpg] (1)
    l1=`P<CHNZHANG<<WANWEN<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`E662323994CHN9009121F2601113NFMFMNPBMOMEA976`
- [2026-9-30-F.jpg] (1)
    l1=`P<CNNXAN<<ZHE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`A000000000CHN9801060M24061900DMGNFNM<<<<4998`
- [2026-9-30-G.jpg] (无引擎完全匹配)
    l1=`P<CHHSUN<<HAI<HUA<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`H224724969CHN0907246M2808185<S3054553<<<<<90`
- [2026-9-30-H.jpg] (2)
    l1=`P<CHNYANG<<ZHEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`KE90003518CHN9707014F2912268A911<121101<<<72`
- [IMG_1792133000_003.jpg] (1)
    l1=`P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<`
    l2=`EE20426536CHN7410024M2808303LMNGMBKENG9BA970`
- [IMG_1792133002_004.jpg] (1)
    l1=`P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<`
    l2=`EE20426536CHN7410024M2808303LMNGMBKENG9BA970`
- [IMG_1792133232_005.jpg] (1)
    l1=`P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<`
    l2=`EE20426536CHN7410024M2808303LMNGMBKENG9BA970`
- [IMG_1792133581_006.jpg] (1)
    l1=`P<SNNSJN<<EINEN<<<PPHPPPPP<<<<<<<<Z<<<<<<<<<`
    l2=`EE20426536CHN7410024M2808303LMNGMBKENG9BA970`
- [IMG_1792133649_007.jpg] (无引擎完全匹配)
    l1=`P<CHNJIA<<LIMIN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`EE20426536CHN7410024M28083034MNGMNKFH6BB8994`
- [IMG_1792133682_008.jpg] (无引擎完全匹配)
    l1=`P<QSN<<<<<SIYJN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<`
    l2=`EE20426536CHN7410024M2808303LMNGMBKCMDPBA970`
