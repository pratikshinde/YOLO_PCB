# Cross-Validation Split Audit (v2)

Date: 2026-10-10  
Audit Method: Exhaustive file-by-file, board-by-board, and byte-level MD5 collision verification.

## Summary of Audit Rules
1. **Split strictly by physical board**: photos of one physical board are never divided between train and val.
2. **0 shared photos** between train and val in any fold.
3. **0 byte-identical crops** between train and val in any fold.
4. **0 synthetic or replicated crops** in any validation set (validation is 100% real crops).
5. **Test boards isolated**: `data/presence/test/red/` is never in any fold or final training set.

---

## Fold-by-Fold Audit Results

### fold_1
- **Validation Removal Boards:** `['m_01', 'm_07']`
- **Validation Good Boards:** `['G01', 'P1', 'P2', 'P3']`
- **Train Photos:** 145 | **Val Photos:** 34
- **Train Crops:** 17993 | **Val Crops:** 2924
- **Shared Physical Boards:** `0` (Requirement: 0) -> PASS
- **Shared Photos:** `0` (Requirement: 0) -> PASS
- **Byte-Identical MD5 Collisions:** `0` (Requirement: 0) -> PASS
- **Val Synthetic/Replicated Crops:** `0` (Requirement: 0) -> PASS
- **Held-Out Missing Part Slots:** `['C16', 'Q2', 'R10', 'R22', 'R23', 'R26', 'R36', 'R42']`
- **Previously Seen Missing Slots in Train:** `['Q2']`
- **Unseen Held-Out Missing Slots (Novel to Val):** `['C16', 'R10', 'R22', 'R23', 'R26', 'R36', 'R42']`
- **Status:** **VERIFIED LEAK-FREE**

### fold_2
- **Validation Removal Boards:** `['m_02', 'm_08']`
- **Validation Good Boards:** `['G02', 'P4', 'P5', 'P6']`
- **Train Photos:** 142 | **Val Photos:** 37
- **Train Crops:** 17621 | **Val Crops:** 3182
- **Shared Physical Boards:** `0` (Requirement: 0) -> PASS
- **Shared Photos:** `0` (Requirement: 0) -> PASS
- **Byte-Identical MD5 Collisions:** `0` (Requirement: 0) -> PASS
- **Val Synthetic/Replicated Crops:** `0` (Requirement: 0) -> PASS
- **Held-Out Missing Part Slots:** `['R12', 'R19', 'R28', 'R34', 'R38', 'R54']`
- **Previously Seen Missing Slots in Train:** `[]`
- **Unseen Held-Out Missing Slots (Novel to Val):** `['R12', 'R19', 'R28', 'R34', 'R38', 'R54']`
- **Status:** **VERIFIED LEAK-FREE**

### fold_3
- **Validation Removal Boards:** `['m_03', 'm_09']`
- **Validation Good Boards:** `['G03', 'P7', 'P8', 'P9']`
- **Train Photos:** 146 | **Val Photos:** 33
- **Train Crops:** 18109 | **Val Crops:** 2838
- **Shared Physical Boards:** `0` (Requirement: 0) -> PASS
- **Shared Photos:** `0` (Requirement: 0) -> PASS
- **Byte-Identical MD5 Collisions:** `0` (Requirement: 0) -> PASS
- **Val Synthetic/Replicated Crops:** `0` (Requirement: 0) -> PASS
- **Held-Out Missing Part Slots:** `['Q2', 'R27', 'R39', 'R45']`
- **Previously Seen Missing Slots in Train:** `['Q2', 'R39']`
- **Unseen Held-Out Missing Slots (Novel to Val):** `['R27', 'R45']`
- **Status:** **VERIFIED LEAK-FREE**

### fold_4
- **Validation Removal Boards:** `['m_04', 'm_06']`
- **Validation Good Boards:** `['G04', 'G05', 'P10', 'P11']`
- **Train Photos:** 139 | **Val Photos:** 40
- **Train Crops:** 17244 | **Val Crops:** 3440
- **Shared Physical Boards:** `0` (Requirement: 0) -> PASS
- **Shared Photos:** `0` (Requirement: 0) -> PASS
- **Byte-Identical MD5 Collisions:** `0` (Requirement: 0) -> PASS
- **Val Synthetic/Replicated Crops:** `0` (Requirement: 0) -> PASS
- **Held-Out Missing Part Slots:** `['D11', 'Q9', 'R21', 'R25', 'R29', 'R51']`
- **Previously Seen Missing Slots in Train:** `['D11']`
- **Unseen Held-Out Missing Slots (Novel to Val):** `['Q9', 'R21', 'R25', 'R29', 'R51']`
- **Status:** **VERIFIED LEAK-FREE**

### fold_5
- **Validation Removal Boards:** `['m_05', 'm_10']`
- **Validation Good Boards:** `['G06', 'P12', 'P13', 'P14', 'P15']`
- **Train Photos:** 144 | **Val Photos:** 35
- **Train Crops:** 17858 | **Val Crops:** 3010
- **Shared Physical Boards:** `0` (Requirement: 0) -> PASS
- **Shared Photos:** `0` (Requirement: 0) -> PASS
- **Byte-Identical MD5 Collisions:** `0` (Requirement: 0) -> PASS
- **Val Synthetic/Replicated Crops:** `0` (Requirement: 0) -> PASS
- **Held-Out Missing Part Slots:** `['D1', 'D11', 'Q6', 'R11', 'R30', 'R37', 'R39', 'R48', 'R53', 'R60']`
- **Previously Seen Missing Slots in Train:** `['D11', 'R39']`
- **Unseen Held-Out Missing Slots (Novel to Val):** `['D1', 'Q6', 'R11', 'R30', 'R37', 'R48', 'R53', 'R60']`
- **Status:** **VERIFIED LEAK-FREE**

---

## Overall Isolation Conclusion
**ALL 5 CROSS-VALIDATION FOLDS ARE VERIFIED 100% LEAK-FREE.** No physical boards, photos, or crops are shared between training and validation.