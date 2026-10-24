# 真实护照照片评测

> ../tmp/omtest pic_scan{1,2,3} n=41; 弱标签来自板端 fixmrz.

| file | gt_cksum | cnn l1/l2/full | trad l1/l2/full |
| pic_scan1/2026-9-29-A.jpg | False | 0.136/0.023/N | 0.477/0.045/N |
| pic_scan1/2026-9-29-B.jpg | True | 0.659/0.045/N | 0.318/0.318/N |
| pic_scan1/2026-9-29-C.jpg | False | 0.159/0.045/N | 0.068/0.182/N |
| pic_scan1/2026-9-29-D.jpg | False | 0.75/0.0/N | 0.523/0.205/N |
| pic_scan1/2026-9-29-E.jpg | False | 0.614/0.114/N | 0.295/0.318/N |
| pic_scan1/2026-9-29-F.jpg | False | 0.273/0.114/N | 0.614/0.114/N |
| pic_scan1/2026-9-29-G.jpg | False | 0.477/0.205/N | 0.477/0.295/N |
| pic_scan1/2026-9-29-H.jpg | False | 0.659/0.023/N | 0.273/0.25/N |
| pic_scan1/2026-9-29-I.jpg | False | 0.727/0.023/N | 0.773/0.409/N |
| pic_scan1/2026-9-29-J.jpg | True | 0.682/0.091/N | 0.432/0.273/N |
| pic_scan1/2026-9-29-K.jpg | False | 0.25/0.114/N | 0.318/0.045/N |
| pic_scan1/2026-9-29-L.jpg | False | 0.682/0.068/N | 0.636/0.273/N |
| pic_scan1/2026-9-29-M.jpg | False | 0.455/0.091/N | 0.364/0.205/N |
| pic_scan1/2026-9-29-N.jpg | False | 0.727/0.091/N | 0.614/0.205/N |
| pic_scan1/2026-9-29-O.jpg | True | 0.659/0.091/N | 0.545/0.25/N |
| pic_scan1/2026-9-29-P.jpg | True | 0.341/0.136/N | 0.568/0.159/N |
| pic_scan1/2026-9-29-Q.jpg | False | 0.705/0.091/N | 0.568/0.295/N |
| pic_scan1/2026-9-29-R.jpg | True | 0.591/0.136/N | 0.364/0.341/N |
| pic_scan1/2026-9-29-S.jpg | False | 0.182/0.068/N | 0.5/0.25/N |
| pic_scan1/2026-9-29-T.jpg | False | 0.659/0.159/N | 0.409/0.182/N |
| pic_scan1/2026-9-29-U.jpg | False | 0.091/0.0/N | 0.227/0.023/N |
| pic_scan1/2026-9-29-W.jpg | False | 0.727/0.045/N | 0.659/0.295/N |
| pic_scan2/2026-9-30-A.jpg | True | 0.341/0.0/N | 0.5/0.341/N |
| pic_scan2/2026-9-30-B.jpg | False | 0.705/0.114/N | 0.523/0.25/N |
| pic_scan2/2026-9-30-C.jpg | True | 0.432/0.205/N | 0.659/0.091/N |
| pic_scan2/2026-9-30-D.jpg | True | 0.614/0.273/N | 0.568/0.273/N |
| pic_scan2/2026-9-30-E.jpg | True | 0.591/0.227/N | 0.795/0.227/N |
| pic_scan2/2026-9-30-F.jpg | False | 0.75/0.25/N | 0.659/0.273/N |
| pic_scan2/2026-9-30-G.jpg | True | 0.364/0.136/N | 0.568/0.364/N |
| pic_scan2/2026-9-30-H.jpg | True | 0.659/0.318/N | 0.5/0.295/N |
| pic_scan3/2026-9-30-AA.jpg | False | 0.568/0.114/N | 0.568/0.114/N |
| pic_scan3/2026-9-30-AB.jpg | False | 0.727/0.068/N | 0.591/0.25/N |
| pic_scan3/2026-9-30-AC.jpg | False | 0.705/0.091/N | 0.455/0.205/N |
| pic_scan3/2026-9-30-AC_R.jpg | False | 0.705/0.091/N | 0.455/0.205/N |
| pic_scan3/2026-9-30-AD.jpg | True | 0.477/0.0/N | 0.364/0.045/N |
| pic_scan3/2026-9-30-AE.jpg | False | 0.705/0.023/N | 0.5/0.25/N |
| pic_scan3/2026-9-30-AF.jpg | True | 0.182/0.023/N | 0.25/0.114/N |
| pic_scan3/2026-9-30-AG.jpg | False | 0.659/0.045/N | 0.409/0.114/N |
| pic_scan3/2026-9-30-AH.jpg | True | 0.591/0.068/N | 0.659/0.182/N |
| pic_scan3/2026-9-30-AI.jpg | False | 0.659/0.091/N | 0.773/0.25/N |
| pic_scan3/2026-9-30-AJ.jpg | False | 0.591/0.045/N | 0.5/0.205/N |

- **cnn**: l1_acc=0.542 l2_acc=0.096 full=0/41 cksum=0/41
- **trad**: l1_acc=0.496 l2_acc=0.219 full=0/41 cksum=0/41
