# StatsBomb open data: terms for the xg-vendor study

Read at open-data commit `4b73468fc5b0f1950f9f66fada70ad3a4f9327cb` (master on 2026-10-11, committed 2026-09-07), the commit the
data version `statsbomb_open_v1` is pinned to. Both files are stored unchanged in `statsbomb_open_v1/raw/terms/`.

| file | sha256 |
|---|---|
| `README.md` | `98a757466c4d931dba4b6c4290c55c2fe74e4550d7455bbe05fd884dc831aff3` |
| `LICENSE.pdf` (StatsBomb Public Data User Agreement) | `a5462e69aeb71a39268b760b110c5c2190a2e0cee3015ba976f66bd71f6c2bb4` |

## What the terms say

README, "Terms & Conditions": if you publish, share or distribute any research, analysis or insights based on this data, state the
data source as StatsBomb and use their logo (media pack at statsbomb.com/media-pack).

User Agreement (extract):

- 1.1 Access is for analysis, research and shared understanding of the data.
- 1.2.1 The user may not "edit, distort, distribute, reproduce, sell or in any way provide the data to any external or third party".
- 1.2.2 The user may not "commercially exploit the data or any analysis derived from the use of the Service".
- 1.4 Any publication of analysis formed from StatsBomb data must carry the StatsBomb brand logo.
- 2.2 Users are asked to register name and email at statsbomb.com/resource-centre before accessing the data.
- 3.2 and 3.4: provided as is; no warranty on accuracy or completeness. Section 5: no liability.
- 6.1 StatsBomb may suspend access if it believes the data is used or distributed outside the agreement.
- 7 All data is StatsBomb's property. The user shall not "modify, translate, transfer, distribute, license, sell or otherwise exploit"
  it without StatsBomb's prior written consent.

## What follows for the study

- Publishing results, figures and tables built from the data is allowed with the StatsBomb name and logo (README, 1.4). Every figure and
  table in the report and the paper carries "Data: StatsBomb" and the logo.
- Decision D1 allows releasing the match map, the pair table and derived features. The agreement does not say that in so many words.
  Clauses 1.2.1 and 7 forbid giving the data to third parties and transferring or distributing it without written consent. A release is
  safe where it holds only our own computations plus StatsBomb ids: match ids, event ids, our pairing, our timing residuals, our
  comparator and G-combo predictions. It is doubtful where it holds StatsBomb's own values or raw event content: vendor xG per shot,
  outcomes, locations, freeze frames. For that second kind the owner should ask StatsBomb for written consent before the release
  (ticket 16), or release scripts that rebuild the table from the public repository at the pinned commit.
- The raw event JSON, the shots table and the freeze frames stay in the private data version and are never released. No SoccerNet frames
  are released (SoccerNet NDA).
- Commercial use is excluded by 1.2.2; the study is academic.
