\# Data acquisition record: HAI 21.03



| Item | Value |

|---|---|

| Dataset | HAI 21.03 (HIL-based Augmented ICS security dataset) |

| Publisher | The Affiliated Institute of ETRI |

| Source | https://github.com/icsdataset/hai (folder hai-21.03) |

| Source commit | 2a814cebc9a66b06c9e5cd545e2d72e65d383737 |

| Source commit date | 7 June 2023, 08:32:27 (UTC+09:00) |

| Downloaded by | Saheed Oyewo |

| Download date | 18 November 2026 |

| Licence | CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/) |

| Ethics approval | granted 18 November 2026, University of Essex |

| Storage | University of Essex OneDrive; not committed to Git |

| File hashes | docs/hai\_manifest.json (SHA-256 of each file as downloaded, still compressed) |



\## Files



| File | Segment | Use |

|---|---|---|

| train1 to train3 | Normal training | Model fitting |

| test1, test2 | Validation | Tuning, threshold calibration, pilot |

| test3 to test5 | Held-out test | Sealed; not decompressed or opened until week 20 |



\## Notes



\- Eight files downloaded, all compressed (.csv.gz). No other files were in the folder.

\- Files were hashed and verified (8 ok, 0 changed, 0 missing) immediately after download, before any file was opened.

\- Exploratory checks on 18 November 2026 used the training files only.

