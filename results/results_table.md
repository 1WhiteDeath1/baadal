| scenario | budget (PKR m) | exact plan | exact score | damage avoided (PKR m) | people protected | greedy score | QAOA-unb plan score | QAOA-unb = exact? | QAOA-unb + local search = exact? | random 512 + local search = exact? | QAOA-unb P(opt) | QAOA-unb P(find in 512) | QAOA-slack P(opt) | QAOA-slack P(find in 512) | random P(find in 512) | QAOA-unb P(feasible) | random P(feasible) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exceptional | 100 | BREACH + EVAC_S + EWS | 890 | 292 | 10500 | 890 | 890 | yes | yes | yes | 0.0119 | 1.00 | 0.0106 | 1.00 | 0.39 | 0.35 | 0.03 |
| exceptional | 160 | BREACH + EVAC_S + PUMPS + GENSETS + EWS | 1146 | 439 | 12075 | 1146 | 995 | no | no | yes | 0.0011 | 0.44 | 0.0002 | 0.08 | 0.39 | 0.56 | 0.09 |
| exceptional | 220 | EMB_TOWN + BREACH + EVAC_S + EWS | 1488 | 670 | 14975 | 1488 | 1488 | yes | yes | no | 0.0096 | 0.99 | 0.0034 | 0.83 | 0.39 | 0.58 | 0.20 |
| exceptional | 280 | EMB_TOWN + BREACH + EVAC_S + PUMPS + GENSETS + EWS | 1702 | 773 | 16156 | 1702 | 1702 | no | no | yes | 0.0042 | 0.89 | 0.0028 | 0.76 | 0.39 | 0.59 | 0.33 |
| exceptional | 340 | EMB_TOWN + SPURS + BREACH + EVAC_S + PUMPS + GENSETS + EWS | 1910 | 877 | 16844 | 1910 | 1910 | yes | yes | no | 0.0049 | 0.92 | 0.0011 | 0.44 | 0.39 | 0.48 | 0.46 |
| exceptional | 400 | EMB_TOWN + SPURS + BUND_HW + BREACH + EVAC_S + PUMPS + EWS | 1982 | 1041 | 16359 | 1982 | 1982 | yes | yes | yes | 0.0089 | 0.99 | 0.0007 | 0.31 | 0.39 | 0.39 | 0.55 |
| very_high | 100 | PUMPS + GENSETS + EWS | 421 | 125 | 5665 | 421 | 421 | yes | yes | yes | 0.0478 | 1.00 | 0.0157 | 1.00 | 0.39 | 0.58 | 0.03 |
| very_high | 160 | EMB_TOWN + EWS | 757 | 312 | 8888 | 757 | 757 | yes | yes | yes | 0.0323 | 1.00 | 0.0044 | 0.90 | 0.63 | 0.60 | 0.09 |
| very_high | 220 | EMB_TOWN + PUMPS + GENSETS + EWS | 942 | 395 | 10200 | 926 | 942 | yes | yes | no | 0.0054 | 0.94 | 0.0110 | 1.00 | 0.39 | 0.25 | 0.20 |
| very_high | 280 | EMB_TOWN + SPURS + PUMPS + GENSETS + EWS | 1108 | 475 | 10880 | 1108 | 1108 | yes | yes | yes | 0.0206 | 1.00 | 0.0001 | 0.04 | 0.39 | 0.52 | 0.33 |
| very_high | 340 | EMB_TOWN + SPURS + PUMPS + GENSETS + WETLAND + EWS | 1204 | 534 | 11574 | 1204 | 1204 | yes | yes | yes | 0.0037 | 0.85 | 0.0005 | 0.23 | 0.39 | 0.71 | 0.46 |
| very_high | 400 | EMB_TOWN + SPURS + EVAC_S + PUMPS + GENSETS + WETLAND + EWS | 1245 | 538 | 12337 | 1245 | 1245 | yes | yes | yes | 0.0306 | 1.00 | 0.0006 | 0.27 | 0.39 | 0.46 | 0.55 |
