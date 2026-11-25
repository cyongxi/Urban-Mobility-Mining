# Final RQ3/RQ4 key numbers (Test, Nov–Dec 2025)

- M1′ MAE 3.5753, RMSE 10.2132; M2′ MAE 3.5555, RMSE 10.2008
- M1′→M2′ MAE reduction 0.0198 (0.55%), 95% daily-block CI [0.0050, 0.0358]; RMSE reduction 0.12%
- M2′→M3′ (degree) -0.0041, CI [-0.0205, 0.0121]
- M3′→M4′ (HHI/flags) +0.0018, CI [-0.0064, 0.0097]
- Best iterations: {'m1': 8669, 'm2': 11840, 'm3': 8498, 'm4': 12776}; any model at cap: False
- Seasonal naive Test MAE: daily 6.7510, weekly 6.9064
- Zones improved (Test): 146/263; in both periods: 87
- Gain by demand group: Low +0.0007 (1% of net); Medium -0.0039 (-6% of net); High +0.0622 (105% of net)
- Gain by RQ2 cluster: C0 Commuter (weekday AM peak) -0.0000; C1 Core activity (afternoon-evening) +0.0388; C2 Nightlife (weekend late night) +0.1069; Unclustered (low volume) +0.0000
- Gain by RQ1 community: Community 0 (Manhattan 97%, 34 zones) +0.1243; Community 1 (Queens 36%, 194 zones) -0.0168; Community 2 (Manhattan 94%, 33 zones) +0.1283; No OD trips +0.0000