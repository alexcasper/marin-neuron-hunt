# Encoder-attribution candidate wave

Provisional dossiers and evidence-grounded working reads for the Marin-32B MLP-neuron candidates nominated by the 64k-SAE encoder-attribution pass. The historical input filename is `sae/candidates-68.json`, but the full 64-layer wave contains **137 candidates**.

**Status:** evidence summaries and model-authored working reads only. These are not confirmed neuron explanations.

## Counts

- Candidates: **137**
- Working reads: **137 / 137**
- Priority: high 18, medium 42, low 77
- Families: broad-contextual 18, broad/high-frequency 3, mixed-contextual 36, multi-latent contextual 4, punctuation/layout 27, sharp-token/context 49

## Selection and evidence

- Nomination filter: per contributing latent `top1_share >= 0.5` and `fired_tokens >= 20`.
- Attribution source: 64k SAE final checkpoints.
- Atlas source: byte-verified `data/pilot/fineweb-10m-sparse` copy.
- Selectivity source: `data/derived/token_selectivity_full/layer=NN.parquet`.
- The `nominate-enc` v1 shuffle controls are invalid and are intentionally not used.

## Candidate index

| candidate | bead | priority | family | working read | read confidence | latents | fired range | max top1_share |
|---|---|---|---|---|---|---:|---:|---:|
| [L27-N1187](L27-N1187.md) | `sh-8wt.3` | high | sharp-token/context | yes | moderate | 15 | 68–159 | 0.655 |
| [L55-N10278](L55-N10278.md) | `sh-8wt.13` | high | broad/high-frequency | yes | low | 5 | 3,593–13,814 | 0.597 |
| [L62-N9554](L62-N9554.md) | `sh-8wt.6` | high | sharp-token/context | yes | medium (token-identity is clear; role unclear) | 4 | 31–48 | 0.642 |
| [L22-N22450](L22-N22450.md) | `sh-8wt.7` | high | sharp-token/context | yes | high | 4 | 21–59 | 0.640 |
| [L17-N14375](L17-N14375.md) | `sh-8wt.18` | high | punctuation/layout | yes | low | 4 | 25–38 | 0.585 |
| [L27-N22697](L27-N22697.md) | `sh-8wt.34` | high | sharp-token/context | yes | medium | 4 | 81–893 | 0.566 |
| [L11-N15351](L11-N15351.md) | `sh-8wt.33` | high | sharp-token/context | yes | high | 3 | 77–112 | 0.567 |
| [L22-N18816](L22-N18816.md) | `sh-8wt.77` | high | punctuation/layout | yes | medium | 3 | 20–24 | 0.527 |
| [L33-N2459](L33-N2459.md) | `sh-8wt.21` | high | punctuation/layout | yes | medium-high | 2 | 20–21 | 0.584 |
| [L36-N5138](L36-N5138.md) | `sh-8wt.32` | high | punctuation/layout | yes | high | 2 | 39–46 | 0.567 |
| [L7-N20520](L7-N20520.md) | `sh-8wt.35` | high | sharp-token/context | yes | high | 2 | 390–395 | 0.563 |
| [L45-N21163](L45-N21163.md) | `sh-8wt.36` | high | punctuation/layout | yes | medium | 2 | 276–315 | 0.563 |
| [L27-N9808](L27-N9808.md) | `sh-8wt.43` | high | punctuation/layout | yes | moderate | 2 | 20–55 | 0.553 |
| [L21-N19899](L21-N19899.md) | `sh-8wt.57` | high | sharp-token/context | yes | medium | 2 | 35–50 | 0.539 |
| [L15-N17223](L15-N17223.md) | `sh-8wt.64` | high | punctuation/layout | yes | low | 2 | 127–143 | 0.536 |
| [L8-N9298](L8-N9298.md) | `sh-8wt.81` | high | sharp-token/context | yes | medium | 2 | 25–42 | 0.523 |
| [L35-N9217](L35-N9217.md) | `sh-8wt.95` | high | sharp-token/context | yes | high | 2 | 29–77 | 0.517 |
| [L16-N15747](L16-N15747.md) | `sh-8wt.134` | high | sharp-token/context | yes | medium | 2 | 65–69 | 0.501 |
| [L44-N14525](L44-N14525.md) | `sh-8wt.49` | medium | multi-latent contextual | yes | medium | 5 | 393–455 | 0.549 |
| [L11-N8533](L11-N8533.md) | `sh-8wt.1` | medium | multi-latent contextual | yes | medium | 4 | 43–72 | 0.687 |
| [L11-N5811](L11-N5811.md) | `sh-8wt.8` | medium | multi-latent contextual | yes | medium | 3 | 81–331 | 0.623 |
| [L34-N26836](L34-N26836.md) | `sh-8wt.20` | medium | multi-latent contextual | yes | high | 3 | 39–42 | 0.584 |
| [L62-N4855](L62-N4855.md) | `sh-8wt.9` | medium | mixed-contextual | yes | low | 2 | 23–109 | 0.622 |
| [L49-N895](L49-N895.md) | `sh-8wt.12` | medium | broad/high-frequency | yes | medium | 2 | 10,199–11,924 | 0.605 |
| [L10-N15969](L10-N15969.md) | `sh-8wt.22` | medium | mixed-contextual | yes | medium | 2 | 20–30 | 0.581 |
| [L16-N6007](L16-N6007.md) | `sh-8wt.23` | medium | broad-contextual | yes | medium | 2 | 23–49 | 0.580 |
| [L25-N20093](L25-N20093.md) | `sh-8wt.50` | medium | broad-contextual | yes | medium | 2 | 88–115 | 0.548 |
| [L44-N16258](L44-N16258.md) | `sh-8wt.53` | medium | mixed-contextual | yes | moderate | 2 | 44–88 | 0.546 |
| [L11-N24556](L11-N24556.md) | `sh-8wt.72` | medium | broad-contextual | yes | medium | 2 | 126–127 | 0.531 |
| [L29-N23842](L29-N23842.md) | `sh-8wt.17` | medium | mixed-contextual | yes | high | 1 | 104–104 | 0.587 |
| [L35-N10673](L35-N10673.md) | `sh-8wt.37` | medium | sharp-token/context | yes | high | 1 | 447–447 | 0.562 |
| [L47-N26202](L47-N26202.md) | `sh-8wt.38` | medium | sharp-token/context | yes | moderate | 1 | 172–172 | 0.562 |
| [L12-N7309](L12-N7309.md) | `sh-8wt.51` | medium | broad-contextual | yes | medium | 1 | 275–275 | 0.548 |
| [L19-N21740](L19-N21740.md) | `sh-8wt.69` | medium | mixed-contextual | yes | medium | 1 | 110–110 | 0.534 |
| [L13-N11930](L13-N11930.md) | `sh-8wt.71` | medium | broad-contextual | yes | low | 1 | 122–122 | 0.532 |
| [L10-N21821](L10-N21821.md) | `sh-8wt.74` | medium | broad-contextual | yes | low | 1 | 111–111 | 0.529 |
| [L27-N23150](L27-N23150.md) | `sh-8wt.75` | medium | punctuation/layout | yes | high | 1 | 230–230 | 0.528 |
| [L15-N8204](L15-N8204.md) | `sh-8wt.76` | medium | broad-contextual | yes | medium | 1 | 132–132 | 0.528 |
| [L46-N5406](L46-N5406.md) | `sh-8wt.80` | medium | mixed-contextual | yes | high | 1 | 116–116 | 0.527 |
| [L11-N134](L11-N134.md) | `sh-8wt.83` | medium | punctuation/layout | yes | medium | 1 | 106–106 | 0.523 |
| [L59-N11005](L59-N11005.md) | `sh-8wt.90` | medium | punctuation/layout | yes | high | 1 | 138–138 | 0.520 |
| [L36-N14337](L36-N14337.md) | `sh-8wt.91` | medium | sharp-token/context | yes | high (token identities), medium (context framing) | 1 | 134–134 | 0.519 |
| [L45-N4166](L45-N4166.md) | `sh-8wt.96` | medium | sharp-token/context | yes | medium-high (positive), medium (negative) | 1 | 115–115 | 0.517 |
| [L59-N11448](L59-N11448.md) | `sh-8wt.101` | medium | punctuation/layout | yes | high (positive token identity), medium (negative) | 1 | 210–210 | 0.514 |
| [L9-N12281](L9-N12281.md) | `sh-8wt.106` | medium | mixed-contextual | yes | medium-low | 1 | 147–147 | 0.510 |
| [L46-N23521](L46-N23521.md) | `sh-8wt.107` | medium | punctuation/layout | yes | high | 1 | 564–564 | 0.510 |
| [L26-N11025](L26-N11025.md) | `sh-8wt.108` | medium | mixed-contextual | yes | moderate | 1 | 114–114 | 0.509 |
| [L63-N11857](L63-N11857.md) | `sh-8wt.111` | medium | sharp-token/context | yes | high (statistics), low (semantic reading) | 1 | 648–648 | 0.508 |
| [L16-N21335](L16-N21335.md) | `sh-8wt.114` | medium | sharp-token/context | yes | medium | 1 | 2,856–2,856 | 0.507 |
| [L42-N16424](L42-N16424.md) | `sh-8wt.117` | medium | mixed-contextual | yes | medium | 1 | 996–996 | 0.506 |
| [L11-N13535](L11-N13535.md) | `sh-8wt.120` | medium | mixed-contextual | yes | medium | 1 | 124–124 | 0.505 |
| [L8-N13193](L8-N13193.md) | `sh-8wt.119` | medium | sharp-token/context | yes | medium | 1 | 822–822 | 0.505 |
| [L11-N13518](L11-N13518.md) | `sh-8wt.121` | medium | punctuation/layout | yes | low | 1 | 108–108 | 0.504 |
| [L57-N26033](L57-N26033.md) | `sh-8wt.125` | medium | sharp-token/context | yes | high | 1 | 119–119 | 0.504 |
| [L28-N20626](L28-N20626.md) | `sh-8wt.127` | medium | mixed-contextual | yes | weak | 1 | 206–206 | 0.503 |
| [L37-N4754](L37-N4754.md) | `sh-8wt.130` | medium | sharp-token/context | yes | high | 1 | 385–385 | 0.503 |
| [L10-N6998](L10-N6998.md) | `sh-8wt.132` | medium | mixed-contextual | yes | medium | 1 | 136–136 | 0.502 |
| [L35-N2465](L35-N2465.md) | `sh-8wt.133` | medium | sharp-token/context | yes | moderate | 1 | 106–106 | 0.502 |
| [L7-N19201](L7-N19201.md) | `sh-8wt.135` | medium | mixed-contextual | yes | medium | 1 | 444–444 | 0.501 |
| [L9-N24830](L9-N24830.md) | `sh-8wt.136` | medium | mixed-contextual | yes | medium-high (positive), low (negative) | 1 | 258–258 | 0.500 |
| [L10-N20671](L10-N20671.md) | `sh-8wt.2` | low | punctuation/layout | yes | low | 1 | 47–47 | 0.677 |
| [L4-N25503](L4-N25503.md) | `sh-8wt.4` | low | sharp-token/context | yes | low | 1 | 70–70 | 0.651 |
| [L6-N5374](L6-N5374.md) | `sh-8wt.5` | low | sharp-token/context | yes | medium | 1 | 67–67 | 0.645 |
| [L38-N9134](L38-N9134.md) | `sh-8wt.10` | low | sharp-token/context | yes | high | 1 | 65–65 | 0.621 |
| [L40-N4441](L40-N4441.md) | `sh-8wt.11` | low | mixed-contextual | yes | medium-high | 1 | 31–31 | 0.617 |
| [L9-N23934](L9-N23934.md) | `sh-8wt.14` | low | sharp-token/context | yes | low | 1 | 40–40 | 0.596 |
| [L35-N5309](L35-N5309.md) | `sh-8wt.15` | low | mixed-contextual | yes | high | 1 | 28–28 | 0.596 |
| [L12-N26213](L12-N26213.md) | `sh-8wt.16` | low | punctuation/layout | yes | medium | 1 | 26–26 | 0.587 |
| [L58-N3443](L58-N3443.md) | `sh-8wt.19` | low | punctuation/layout | yes | low | 1 | 36–36 | 0.584 |
| [L45-N21413](L45-N21413.md) | `sh-8wt.24` | low | sharp-token/context | yes | medium | 1 | 42–42 | 0.577 |
| [L59-N19776](L59-N19776.md) | `sh-8wt.25` | low | sharp-token/context | yes | medium | 1 | 25–25 | 0.576 |
| [L35-N24020](L35-N24020.md) | `sh-8wt.26` | low | sharp-token/context | yes | high (negative side), low (positive side) | 1 | 26–26 | 0.575 |
| [L27-N22312](L27-N22312.md) | `sh-8wt.27` | low | punctuation/layout | yes | medium | 1 | 29–29 | 0.574 |
| [L29-N12980](L29-N12980.md) | `sh-8wt.28` | low | punctuation/layout | yes | low | 1 | 53–53 | 0.574 |
| [L60-N26229](L60-N26229.md) | `sh-8wt.29` | low | mixed-contextual | yes | low | 1 | 27–27 | 0.574 |
| [L20-N27076](L20-N27076.md) | `sh-8wt.30` | low | punctuation/layout | yes | low | 1 | 50–50 | 0.569 |
| [L36-N25354](L36-N25354.md) | `sh-8wt.31` | low | sharp-token/context | yes | medium-high (positive), low (negative) | 1 | 51–51 | 0.569 |
| [L60-N10309](L60-N10309.md) | `sh-8wt.39` | low | punctuation/layout | yes | low | 1 | 34–34 | 0.562 |
| [L37-N17388](L37-N17388.md) | `sh-8wt.40` | low | sharp-token/context | yes | high | 1 | 26–26 | 0.561 |
| [L12-N4205](L12-N4205.md) | `sh-8wt.41` | low | mixed-contextual | yes | low | 1 | 39–39 | 0.559 |
| [L44-N22828](L44-N22828.md) | `sh-8wt.42` | low | sharp-token/context | yes | high | 1 | 32–32 | 0.556 |
| [L37-N17833](L37-N17833.md) | `sh-8wt.44` | low | broad-contextual | yes | medium | 1 | 20–20 | 0.552 |
| [L34-N8043](L34-N8043.md) | `sh-8wt.45` | low | mixed-contextual | yes | high | 1 | 56–56 | 0.551 |
| [L62-N16811](L62-N16811.md) | `sh-8wt.46` | low | sharp-token/context | yes | high | 1 | 21–21 | 0.551 |
| [L29-N24910](L29-N24910.md) | `sh-8wt.47` | low | sharp-token/context | yes | medium | 1 | 65–65 | 0.551 |
| [L30-N5450](L30-N5450.md) | `sh-8wt.48` | low | punctuation/layout | yes | moderate | 1 | 53–53 | 0.550 |
| [L35-N20498](L35-N20498.md) | `sh-8wt.52` | low | mixed-contextual | yes | medium | 1 | 27–27 | 0.546 |
| [L25-N19258](L25-N19258.md) | `sh-8wt.54` | low | mixed-contextual | yes | low | 1 | 26–26 | 0.545 |
| [L6-N5700](L6-N5700.md) | `sh-8wt.55` | low | mixed-contextual | yes | medium | 1 | 58–58 | 0.544 |
| [L37-N5043](L37-N5043.md) | `sh-8wt.56` | low | sharp-token/context | yes | high (token identity), low (semantic role) | 1 | 38–38 | 0.541 |
| [L36-N27357](L36-N27357.md) | `sh-8wt.58` | low | sharp-token/context | yes | high | 1 | 31–31 | 0.539 |
| [L37-N6076](L37-N6076.md) | `sh-8wt.59` | low | sharp-token/context | yes | medium | 1 | 36–36 | 0.538 |
| [L12-N14813](L12-N14813.md) | `sh-8wt.60` | low | broad-contextual | yes | medium | 1 | 27–27 | 0.537 |
| [L59-N557](L59-N557.md) | `sh-8wt.61` | low | broad-contextual | yes | medium | 1 | 25–25 | 0.537 |
| [L63-N17168](L63-N17168.md) | `sh-8wt.62` | low | mixed-contextual | yes | medium | 1 | 20–20 | 0.536 |
| [L26-N17855](L26-N17855.md) | `sh-8wt.63` | low | mixed-contextual | yes | moderate | 1 | 33–33 | 0.536 |
| [L36-N4765](L36-N4765.md) | `sh-8wt.65` | low | sharp-token/context | yes | high | 1 | 37–37 | 0.535 |
| [L10-N12917](L10-N12917.md) | `sh-8wt.66` | low | broad-contextual | yes | medium | 1 | 24–24 | 0.535 |
| [L60-N17154](L60-N17154.md) | `sh-8wt.67` | low | punctuation/layout | yes | high | 1 | 84–84 | 0.534 |
| [L11-N2883](L11-N2883.md) | `sh-8wt.68` | low | mixed-contextual | yes | medium | 1 | 22–22 | 0.534 |
| [L36-N16585](L36-N16585.md) | `sh-8wt.70` | low | broad-contextual | yes | medium | 1 | 20–20 | 0.532 |
| [L18-N12330](L18-N12330.md) | `sh-8wt.73` | low | sharp-token/context | yes | medium | 1 | 80–80 | 0.530 |
| [L27-N27558](L27-N27558.md) | `sh-8wt.78` | low | broad-contextual | yes | low | 1 | 42–42 | 0.527 |
| [L39-N15151](L39-N15151.md) | `sh-8wt.79` | low | sharp-token/context | yes | medium | 1 | 24–24 | 0.527 |
| [L8-N10861](L8-N10861.md) | `sh-8wt.82` | low | punctuation/layout | yes | weak | 1 | 23–23 | 0.523 |
| [L12-N8027](L12-N8027.md) | `sh-8wt.84` | low | sharp-token/context | yes | medium | 1 | 65–65 | 0.523 |
| [L29-N24862](L29-N24862.md) | `sh-8wt.85` | low | broad-contextual | yes | medium | 1 | 63–63 | 0.523 |
| [L35-N7530](L35-N7530.md) | `sh-8wt.86` | low | mixed-contextual | yes | medium | 1 | 46–46 | 0.523 |
| [L36-N16965](L36-N16965.md) | `sh-8wt.87` | low | sharp-token/context | yes | high | 1 | 42–42 | 0.523 |
| [L42-N9154](L42-N9154.md) | `sh-8wt.88` | low | mixed-contextual | yes | moderate | 1 | 49–49 | 0.521 |
| [L33-N18287](L33-N18287.md) | `sh-8wt.89` | low | mixed-contextual | yes | medium | 1 | 87–87 | 0.520 |
| [L22-N26066](L22-N26066.md) | `sh-8wt.92` | low | mixed-contextual | yes | weak | 1 | 39–39 | 0.518 |
| [L37-N7052](L37-N7052.md) | `sh-8wt.93` | low | sharp-token/context | yes | high | 1 | 57–57 | 0.517 |
| [L48-N18063](L48-N18063.md) | `sh-8wt.94` | low | broad/high-frequency | yes | low | 1 | 6,685–6,685 | 0.517 |
| [L24-N8156](L24-N8156.md) | `sh-8wt.97` | low | punctuation/layout | yes | medium | 1 | 24–24 | 0.516 |
| [L42-N873](L42-N873.md) | `sh-8wt.98` | low | mixed-contextual | yes | high | 1 | 31–31 | 0.515 |
| [L35-N2631](L35-N2631.md) | `sh-8wt.99` | low | mixed-contextual | yes | medium | 1 | 56–56 | 0.515 |
| [L17-N9960](L17-N9960.md) | `sh-8wt.100` | low | sharp-token/context | yes | low | 1 | 49–49 | 0.514 |
| [L16-N3305](L16-N3305.md) | `sh-8wt.102` | low | sharp-token/context | yes | low | 1 | 34–34 | 0.513 |
| [L12-N6048](L12-N6048.md) | `sh-8wt.103` | low | broad-contextual | yes | low | 1 | 45–45 | 0.511 |
| [L22-N5747](L22-N5747.md) | `sh-8wt.104` | low | punctuation/layout | yes | low | 1 | 49–49 | 0.511 |
| [L45-N14989](L45-N14989.md) | `sh-8wt.105` | low | mixed-contextual | yes | medium | 1 | 24–24 | 0.511 |
| [L28-N22988](L28-N22988.md) | `sh-8wt.109` | low | punctuation/layout | yes | low | 1 | 26–26 | 0.509 |
| [L27-N15105](L27-N15105.md) | `sh-8wt.110` | low | sharp-token/context | yes | high | 1 | 93–93 | 0.508 |
| [L35-N8512](L35-N8512.md) | `sh-8wt.112` | low | mixed-contextual | yes | medium | 1 | 84–84 | 0.507 |
| [L30-N10652](L30-N10652.md) | `sh-8wt.113` | low | punctuation/layout | yes | low | 1 | 28–28 | 0.507 |
| [L62-N6036](L62-N6036.md) | `sh-8wt.115` | low | mixed-contextual | yes | low | 1 | 65–65 | 0.506 |
| [L16-N22118](L16-N22118.md) | `sh-8wt.116` | low | broad-contextual | yes | medium | 1 | 25–25 | 0.506 |
| [L29-N25622](L29-N25622.md) | `sh-8wt.118` | low | sharp-token/context | yes | moderate | 1 | 21–21 | 0.505 |
| [L12-N7352](L12-N7352.md) | `sh-8wt.122` | low | broad-contextual | yes | low | 1 | 39–39 | 0.504 |
| [L13-N3314](L13-N3314.md) | `sh-8wt.123` | low | mixed-contextual | yes | medium | 1 | 20–20 | 0.504 |
| [L50-N18816](L50-N18816.md) | `sh-8wt.124` | low | sharp-token/context | yes | medium | 1 | 43–43 | 0.504 |
| [L10-N9319](L10-N9319.md) | `sh-8wt.126` | low | broad-contextual | yes | low | 1 | 85–85 | 0.503 |
| [L59-N21614](L59-N21614.md) | `sh-8wt.128` | low | mixed-contextual | yes | low | 1 | 35–35 | 0.503 |
| [L36-N9381](L36-N9381.md) | `sh-8wt.129` | low | sharp-token/context | yes | low | 1 | 62–62 | 0.503 |
| [L10-N10572](L10-N10572.md) | `sh-8wt.131` | low | sharp-token/context | yes | medium | 1 | 20–20 | 0.502 |
| [L37-N24889](L37-N24889.md) | `sh-8wt.137` | low | sharp-token/context | yes | high | 1 | 21–21 | 0.500 |

## Uniform next steps

For each candidate, first run its dossier-specific **Next checks**, then close its bead only after wrong-context controls, random/same-layer comparisons, and a verdict of `promote`, `reject`, or `needs-more-evidence`, with evidence paths recorded.
