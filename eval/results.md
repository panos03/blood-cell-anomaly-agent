# Evaluation Results

Questions: 4 (4 answerable via tools, 0 intentionally unsupported)
NOTE: only these 4 were run due to free API rate limits

## Headline metrics

- Accuracy **with tools** (agent): 4/4 (100%)
- Accuracy **without tools** (plain Gemini): 0/4 (0%)
- Correct tool call rate: 4/4 (100%)
- On unsupported questions: 0/0 declined appropriately, 0/0 hallucinated an answer

## Failure modes (supported questions the agent got wrong)

- None

## Per-category accuracy (with tools)

- flagged_cells: 4/4 (100%)

## Per-question detail

### fc-01 (flagged_cells)

**Question:** How many true positives did the distance detector get on the basophil holdout?

**Expected answer:** `1042`  
**Expected tools:** get_flagged_cells

**With tools** - verdict: `correct` - tools called: get_flagged_cells

> The distance detector got 1042 true positives on the basophil holdout.

**Without tools** - verdict: `incorrect`

> I do not know the specific numbers for the true positives obtained by the distance detector on the basophil holdout, as I don't have access to the internal data or specific experimental results of that particular research project.

---

### fc-02 (flagged_cells)

**Question:** What was the MSP baseline's F1 score on the eosinophil holdout?

**Expected answer:** `0.7027`  
**Expected tools:** get_flagged_cells

**With tools** - verdict: `correct` - tools called: get_flagged_cells

> The MSP baseline's F1 score on the eosinophil holdout was 0.7027 (with 1753 true positives, 119 false positives, 1364 false negatives, 1978 true negatives, 0.9364 precision, and 0.5624 recall).

**Without tools** - verdict: `incorrect`

> I do not know the specific numbers for the MSP baseline's F1 score on the eosinophil holdout, as I don't have access to the specific experimental data or paper you are referencing.

---

### fc-03 (flagged_cells)

**Question:** How many false negatives did the distance detector have on the erythroblast holdout?

**Expected answer:** `33`  
**Expected tools:** get_flagged_cells

**With tools** - verdict: `correct` - tools called: get_flagged_cells

> The distance detector had 33 false negatives on the erythroblast holdout.

**Without tools** - verdict: `incorrect`

> I do not know the specific numbers for the false negatives of the distance detector on the erythroblast holdout, as that requires access to the exact experimental data and evaluation metrics of the specific research project you are referencing.

---

### fc-04 (flagged_cells)

**Question:** What is the distance detector's recall on the monocyte holdout?

**Expected answer:** `0.5732`  
**Expected tools:** get_flagged_cells

**With tools** - verdict: `correct` - tools called: get_flagged_cells

> The distance detector's recall on the monocyte holdout is 0.5732 (or 57.32%), with 814 true positives and 606 false negatives.

**Without tools** - verdict: `incorrect`

> I do not know the specific numbers for the distance detector's recall on the monocyte holdout, as I do not have access to the proprietary data or specific experimental results of that particular research project.

---
