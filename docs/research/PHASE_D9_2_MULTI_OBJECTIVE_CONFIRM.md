# D9.2b Multi-Objective Confirm Report

Verdict: `D9_2_FULL_GENERALIST_CONFIRMED`

## Conclusion

D9.2b confirms that the D9.2a microprobe was not seed luck. The top three
multi-objective endpoints all retain the D9 smooth/accuracy gains, remain
echo-safe, and flip unigram from a regression into a positive baseline-relative
delta on fresh seeds at both eval lengths.

Recommended promotion candidate: `top_01.ckpt`.

## Gate Policy

- `smooth`: lower95 >= +0.0120
- `accuracy`: lower95 >= +0.0020
- `echo`: abs(mean) <= 0.0010 and CI remains near zero
- `unigram`: lower95 >= 0.0 for strict full-generalist confirm

## Endpoint Results

- `top_01.ckpt`: strict_all=True, moderate_all=True
- `top_02.ckpt`: strict_all=True, moderate_all=True
- `top_03.ckpt`: strict_all=True, moderate_all=True

## Metric Summary

| endpoint | eval_len | metric | mean | std | lower95 | upper95 | positive_rate |
|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | 4000 | accuracy | 0.004850000 | 0.001871520 | 0.004180285 | 0.005519715 | 0.967 |
| 1 | 4000 | echo | -0.000133333 | 0.000875267 | -0.000446544 | 0.000179877 | 0.100 |
| 1 | 4000 | smooth | 0.016886946 | 0.001193826 | 0.016459741 | 0.017314151 | 1.000 |
| 1 | 4000 | unigram | 0.004835775 | 0.002912927 | 0.003793397 | 0.005878152 | 0.967 |
| 1 | 16000 | accuracy | 0.004304167 | 0.001534072 | 0.003755206 | 0.004853127 | 1.000 |
| 1 | 16000 | echo | 0.000002083 | 0.000011411 | -0.000002000 | 0.000006167 | 0.033 |
| 1 | 16000 | smooth | 0.016997820 | 0.000226826 | 0.016916652 | 0.017078989 | 1.000 |
| 1 | 16000 | unigram | 0.005412439 | 0.000730792 | 0.005150928 | 0.005673949 | 1.000 |
| 2 | 4000 | accuracy | 0.004850000 | 0.001870368 | 0.004180697 | 0.005519303 | 0.967 |
| 2 | 4000 | echo | -0.000133333 | 0.000875267 | -0.000446544 | 0.000179877 | 0.100 |
| 2 | 4000 | smooth | 0.016888343 | 0.001194102 | 0.016461039 | 0.017315647 | 1.000 |
| 2 | 4000 | unigram | 0.003470525 | 0.002885554 | 0.002437943 | 0.004503108 | 0.967 |
| 2 | 16000 | accuracy | 0.004310417 | 0.001541058 | 0.003758956 | 0.004861877 | 1.000 |
| 2 | 16000 | echo | 0.000002083 | 0.000011411 | -0.000002000 | 0.000006167 | 0.033 |
| 2 | 16000 | smooth | 0.016998007 | 0.000226446 | 0.016916975 | 0.017079040 | 1.000 |
| 2 | 16000 | unigram | 0.003994207 | 0.000700337 | 0.003743594 | 0.004244819 | 1.000 |
| 3 | 4000 | accuracy | 0.004841667 | 0.001870233 | 0.004172412 | 0.005510921 | 0.967 |
| 3 | 4000 | echo | -0.000133333 | 0.000875267 | -0.000446544 | 0.000179877 | 0.100 |
| 3 | 4000 | smooth | 0.017068666 | 0.001250436 | 0.016621203 | 0.017516129 | 1.000 |
| 3 | 4000 | unigram | 0.003141362 | 0.002899339 | 0.002103846 | 0.004178877 | 0.967 |
| 3 | 16000 | accuracy | 0.004279167 | 0.001572060 | 0.003716612 | 0.004841721 | 1.000 |
| 3 | 16000 | echo | 0.000002083 | 0.000011411 | -0.000002000 | 0.000006167 | 0.033 |
| 3 | 16000 | smooth | 0.017236361 | 0.000247870 | 0.017147662 | 0.017325060 | 1.000 |
| 3 | 16000 | unigram | 0.003664577 | 0.000771657 | 0.003388443 | 0.003940711 | 1.000 |
