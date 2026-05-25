# ================================================================
# src/anomaly_detector.py
# ================================================================
# CONTEXT:
#   The InsuranceEDAEngine found patterns. Now we need to find EXCEPTIONS.
#   Which specific claims have amounts that are statistically unusual?
#   Which confirmed fraudulent claims look completely normal by amount?
#
# THE BUSINESS QUESTION:
#   "We know the average claim is £94k. But are there any claims whose
#    amounts are so far from normal that we should investigate them?
#    Either the amount is a data entry error, or it is a suspicious claim
#    that needs manual review before approval."
#
# THE ANALOGY:
#   Imagine plotting all claim amounts on a number line.
#   Most cluster in the middle. A few sit far to the left or right.
#   The AnomalyDetector finds those outliers using statistics.
#   It uses TWO methods and only flags something as confirmed if BOTH agree.
#   This "consensus" approach reduces false alarms.
#
# TWO DETECTION METHODS:
#   Method 1 — IQR (Interquartile Range):
#     Works on any distribution. No assumptions about shape.
#     Uses Q1 - 1.5×IQR as the lower fence and Q3 + 1.5×IQR as the upper fence.
#     Preferred for insurance claim data which is right-skewed.
#
#   Method 2 — Z-score:
#     Assumes normally distributed data.
#     Flags values more than 3 standard deviations from the mean.
#
# WHY USE TWO METHODS?
#   Each method has blind spots. IQR is robust to skewed data but can
#   miss outliers that cluster near the fence. Z-score catches extreme
#   values precisely but is sensitive to the mean being pulled by outliers.
#   Consensus (flagged by BOTH) gives us the most reliable results.
#
# INSURANCE-SPECIFIC ADDITION — flag_fraud_patterns():
#   Beyond standard anomaly detection, this module profiles fraudulent vs
#   legitimate claims and flags EVASIVE FRAUD — confirmed fraudulent claims
#   with normal amounts that bypass automated screening.
# ================================================================

import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
from scipy import stats   # scipy.stats: statistical functions

from config import INDUSTRY, REPORTS_DIR, logger


class AnomalyDetector:
    """
    Detects statistical anomalies in insurance claims using IQR and Z-score methods.

    Uses consensus: a row is a CONFIRMED anomaly only if flagged by BOTH methods.
    Consensus reduces false positives compared to using either method alone.

    Also provides flag_fraud_patterns() — an insurance-specific method that
    profiles fraudulent vs legitimate claims and isolates evasive fraud:
    confirmed fraudulent claims with normal amounts that blend into
    legitimate activity and are most likely to bypass automated rules engines.

    Attributes
    ──────────
    df            pd.DataFrame   the input DataFrame to scan
    results       dict           anomaly findings per column
    confirmed     pd.DataFrame   rows confirmed as anomalies (flagged by both methods)
    fraud_profile dict           statistical profile of fraudulent vs legitimate claims
    _n_checked    int            number of columns that were checked
    """

    # Z-score threshold: values more than this many standard deviations
    # from the mean are flagged as outliers.
    # 3.0 is the standard (covers 99.73% of normal distribution inside the fence)
    ZSCORE_THRESHOLD = 3.0

    def __init__(self, df: pd.DataFrame):
        """
        Initialise with the DataFrame to scan.

        Args:
            df   the processed DataFrame from InsuranceEDAEngine
        """
        self.df           = df.copy()         # always work on a copy
        self.results      = {}                # per-column anomaly statistics
        self.confirmed    = pd.DataFrame()    # rows flagged by both methods
        self.fraud_profile = {}               # fraud vs legitimate profile
        self._n_checked   = 0

        logger.info(f"AnomalyDetector initialised — {len(df):,} rows to scan")

    # ── DETECTION METHODS ─────────────────────────────────────────────

    def _detect_iqr(self, col: str) -> pd.Series:
        """
        IQR method: flag values outside Tukey's fences.

        HOW THE IQR METHOD WORKS:
        ──────────────────────────
        Picture all claim amounts sorted from smallest to largest.
        Q1 = the value at position 25% (first quartile)
        Q3 = the value at position 75% (third quartile)
        IQR = Q3 - Q1  (the range of the middle 50% of the data)

        Tukey's fences (the standard outlier definition since 1977):
          Lower fence = Q1 - 1.5 × IQR
          Upper fence = Q3 + 1.5 × IQR

        Any value BELOW the lower fence OR ABOVE the upper fence is an outlier.

        WHY IQR IS PREFERRED FOR INSURANCE DATA:
        ──────────────────────────────────────────
        Claim amounts are right-skewed — a few catastrophic claims (fire,
        major surgery, liability settlements) drag the distribution far right.
        IQR uses percentiles, not the mean, so it is unaffected by this skew.
        Z-score would undercount outliers on skewed data because the mean
        and std are themselves pulled upward by extreme claims.

        WHY 1.5?
        ────────
        John Tukey chose 1.5 in his 1977 book "Exploratory Data Analysis".
        Under a normal distribution, 1.5 × IQR fences contain 99.3% of values.
        This has been the standard in box plots ever since.

        Args:
            col   the numeric column to check

        Returns:
            Boolean pd.Series — True where the value is an outlier
        """
        Q1  = self.df[col].quantile(0.25)   # 25th percentile
        Q3  = self.df[col].quantile(0.75)   # 75th percentile
        IQR = Q3 - Q1                       # interquartile range

        lower = Q1 - 1.5 * IQR   # Tukey lower fence
        upper = Q3 + 1.5 * IQR   # Tukey upper fence

        # (self.df[col] < lower) → True where value is below lower fence
        # (self.df[col] > upper) → True where value is above upper fence
        # The | operator is OR — True if EITHER condition is True
        return (self.df[col] < lower) | (self.df[col] > upper)

    def _detect_zscore(self, col: str) -> pd.Series:
        """
        Z-score method: flag values far from the mean.

        HOW THE Z-SCORE WORKS:
        ───────────────────────
        Z = (value - mean) / standard_deviation

        The Z-score tells us how many standard deviations a value is from the mean.
          Z = 0    → the value equals the mean (perfectly typical)
          Z = 1    → one standard deviation above the mean
          Z = -2   → two standard deviations below the mean
          Z = 3.5  → three and a half standard deviations above — likely an outlier

        For a normal distribution:
          68.3% of values fall within |Z| < 1
          95.4% of values fall within |Z| < 2
          99.7% of values fall within |Z| < 3
          Only 0.3% of values have |Z| > 3 — these are our outliers.

        LIMITATION FOR INSURANCE DATA:
        ────────────────────────────────
        If claim amounts are highly skewed (common in insurance), the mean
        and std are themselves pulled by catastrophic claims, making Z-scores
        less reliable. This is why we combine Z-score with IQR (which has no
        such assumption) and only confirm anomalies where BOTH methods agree.

        Args:
            col   the numeric column to check

        Returns:
            Boolean pd.Series — True where |Z| > ZSCORE_THRESHOLD
        """
        mean = self.df[col].mean()   # arithmetic average
        std  = self.df[col].std()    # standard deviation

        # Guard: if std == 0, all values are identical — no outliers possible
        if std == 0:
            return pd.Series(False, index=self.df.index)

        # Compute Z-score for each value
        # (value - mean) / std → how many std deviations from the mean
        z_scores = (self.df[col] - mean).abs() / std   # .abs() for absolute value

        # Flag rows where the absolute Z-score exceeds our threshold
        return z_scores > self.ZSCORE_THRESHOLD

    def run(self, columns: list = None) -> "AnomalyDetector":
        """
        Run both detection methods on the specified columns and find consensus.

        CONSENSUS LOGIC:
        ─────────────────
        A row is flagged as a CONFIRMED anomaly only if it is flagged by
        BOTH the IQR method AND the Z-score method.

        Why consensus reduces false positives:
          IQR alone might flag a value just barely outside the fence
          Z-score alone might flag a value because the mean was itself skewed
          BOTH agreeing means the value is an outlier by two independent definitions

        For insurance claims, confirmed anomalies are candidates for:
          - Manual review before approval
          - Escalation to the fraud investigation team
          - Data quality investigation (possible entry error)

        Args:
            columns   list of numeric columns to analyse.
                      If not provided, analyses first 3 numeric columns.
                      Recommended: ['amount_claimed', 'amount_approved', 'coverage_amount']

        Returns self.
        """
        # Identify which columns to analyse
        num_cols    = self.df.select_dtypes(include=["number"]).columns.tolist()
        target_cols = columns if columns else num_cols[:3]   # default: first 3

        # Track anomaly flags across all analysed columns
        all_flags = pd.DataFrame(index=self.df.index)

        for col in target_cols:
            if col not in self.df.columns:
                logger.warning(f"[ANOMALY] Column '{col}' not found — skipping")
                continue

            # Run both methods independently
            iqr_flags = self._detect_iqr(col)     # IQR method → boolean Series
            z_flags   = self._detect_zscore(col)  # Z-score method → boolean Series

            # Consensus: True only where BOTH methods flag the row
            # The & operator is AND — True only if BOTH conditions are True
            consensus = iqr_flags & z_flags

            # Count flagged rows for reporting
            iqr_count  = int(iqr_flags.sum())
            z_count    = int(z_flags.sum())
            conf_count = int(consensus.sum())

            # Store per-column statistics
            self.results[col] = {
                "iqr_flagged":         iqr_count,
                "zscore_flagged":      z_count,
                "confirmed_anomalies": conf_count,
                "anomaly_pct":         round(conf_count / len(self.df) * 100, 2),
            }

            # Store flags for each method so we can identify confirmed rows
            all_flags[f"{col}_iqr"]       = iqr_flags
            all_flags[f"{col}_z"]         = z_flags
            all_flags[f"{col}_confirmed"] = consensus

            logger.info(
                f"[ANOMALY] {col}: "
                f"IQR={iqr_count} | Z-score={z_count} | "
                f"Confirmed={conf_count}"
            )

        # Find rows confirmed as anomalies in AT LEAST ONE column
        # .any(axis=1) checks across columns (axis=1 = row direction)
        # Returns True for rows where any confirmed flag is True
        confirmed_cols = [c for c in all_flags.columns if c.endswith("_confirmed")]
        if confirmed_cols:
            confirmed_mask  = all_flags[confirmed_cols].any(axis=1)
            self.confirmed  = self.df[confirmed_mask].copy()

        self._n_checked = len(target_cols)
        logger.info(
            f"[ANOMALY] Scan complete: "
            f"{len(self.confirmed):,} confirmed anomaly rows found"
        )

        return self

    # ── INSURANCE-SPECIFIC ────────────────────────────────────────────

    def flag_fraud_patterns(self,
                            fraud_col:  str = "is_fraudulent",
                            amount_col: str = "amount_claimed") -> "AnomalyDetector":
        """
        Build a statistical profile of fraudulent vs legitimate claims.
        Also flags evasive fraud: fraudulent claims with NORMAL amounts.

        TWO FRAUD CATEGORIES:
        ──────────────────────
        1. HIGH-VALUE FRAUD
           is_fraudulent = True AND amount_claimed is a statistical outlier
           (outside IQR fences)
           → Easy to detect — automated rules engines catch these

        2. EVASIVE FRAUD (the hard problem)
           is_fraudulent = True AND amount_claimed is WITHIN normal IQR bounds
           → Blends into legitimate claim patterns
           → Most likely to pass automated screening undetected
           → Highest priority for manual adjuster review

        WHY THIS MATTERS:
        ──────────────────
        A claims team reviewing thousands of records cannot manually check
        every one. Flagging evasive fraud isolates the highest-priority subset:
        claims that look completely legitimate but are confirmed fraudulent.
        Standard anomaly detection misses these entirely.

        PROFILE COMPUTED:
        ──────────────────
        For both fraudulent and legitimate groups:
          - Mean and median amount_claimed
          - Fraud rate by claim_type
          - Fraud rate by risk_grade
        This gives the actuarial team a full picture of where fraud concentrates.

        Results saved to reports/anomalies.csv with anomaly_type column
        distinguishing HIGH_VALUE_FRAUD from EVASIVE_FRAUD.

        Args:
            fraud_col   boolean fraud flag column (default: 'is_fraudulent')
            amount_col  numeric claim amount column (default: 'amount_claimed')

        Returns self.
        """
        for col in [fraud_col, amount_col]:
            if col not in self.df.columns:
                logger.warning(
                    f"[ANOMALY] Column '{col}' not found — "
                    f"skipping flag_fraud_patterns"
                )
                return self

        logger.info("[ANOMALY] Profiling fraud patterns...")

        # ── Compute IQR bounds for amount_claimed ─────────────────────
        Q1  = self.df[amount_col].quantile(0.25)
        Q3  = self.df[amount_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        fraudulent = self.df[self.df[fraud_col] == True].copy()
        legit      = self.df[self.df[fraud_col] == False].copy()

        # ── Flag 1: high-value fraud ───────────────────────────────────
        # Fraudulent claims that are also statistical outliers
        # These are the obvious cases — automated rules engines already catch them
        high_value_fraud = fraudulent[
            (fraudulent[amount_col] < lower) | (fraudulent[amount_col] > upper)
        ].copy()
        high_value_fraud["anomaly_type"] = "HIGH_VALUE_FRAUD"

        # ── Flag 2: evasive fraud ──────────────────────────────────────
        # Fraudulent claims with amounts WITHIN normal IQR bounds
        # These blend into legitimate activity — the hardest cases to detect
        evasive_fraud = fraudulent[
            (fraudulent[amount_col] >= lower) & (fraudulent[amount_col] <= upper)
        ].copy()
        evasive_fraud["anomaly_type"] = "EVASIVE_FRAUD"

        # ── Statistical profile ────────────────────────────────────────
        profile = {
            "total_claims":        len(self.df),
            "fraudulent_claims":   len(fraudulent),
            "legitimate_claims":   len(legit),
            "fraud_rate_pct":      round(len(fraudulent) / len(self.df) * 100, 2),
            "high_value_fraud":    len(high_value_fraud),
            "evasive_fraud":       len(evasive_fraud),
            "fraud_mean_amount":   round(float(fraudulent[amount_col].mean()),   2),
            "legit_mean_amount":   round(float(legit[amount_col].mean()),        2),
            "fraud_median_amount": round(float(fraudulent[amount_col].median()), 2),
            "legit_median_amount": round(float(legit[amount_col].median()),      2),
            "iqr_lower_bound":     round(lower, 2),
            "iqr_upper_bound":     round(upper, 2),
        }

        # ── Fraud rate by claim_type ───────────────────────────────────
        if "claim_type" in self.df.columns:
            fraud_by_type = (
                self.df.groupby("claim_type")[fraud_col]
                .agg(fraud_count="sum", total="count")
                .reset_index()
            )
            fraud_by_type["fraud_rate_pct"] = (
                fraud_by_type["fraud_count"] / fraud_by_type["total"] * 100
            ).round(2)
            profile["fraud_by_claim_type"] = (
                fraud_by_type
                .sort_values("fraud_rate_pct", ascending=False)
                .to_dict(orient="records")
            )

        # ── Fraud rate by risk_grade ───────────────────────────────────
        if "risk_grade" in self.df.columns:
            fraud_by_grade = (
                self.df.groupby("risk_grade")[fraud_col]
                .agg(fraud_count="sum", total="count")
                .reset_index()
            )
            fraud_by_grade["fraud_rate_pct"] = (
                fraud_by_grade["fraud_count"] / fraud_by_grade["total"] * 100
            ).round(2)
            profile["fraud_by_risk_grade"] = (
                fraud_by_grade
                .sort_values("fraud_rate_pct", ascending=False)
                .to_dict(orient="records")
            )

        # ── Combine all flagged records ────────────────────────────────
        all_flagged      = pd.concat([high_value_fraud, evasive_fraud], ignore_index=True)
        self.confirmed   = all_flagged
        self.fraud_profile = profile

        logger.info(
            f"[ANOMALY] Fraud profile complete — "
            f"{len(fraudulent):,} fraudulent | "
            f"{len(high_value_fraud):,} high-value | "
            f"{len(evasive_fraud):,} evasive"
        )

        return self

    def save_anomalies(self) -> "AnomalyDetector":
        """
        Save the confirmed anomaly rows to reports/anomalies.csv.

        WHY SAVE ANOMALIES SEPARATELY?
        ────────────────────────────────
        Anomaly rows need to be reviewed by a human:
          - Is this a data entry error? → fix in the source system
          - Is this a legitimate extreme claim (e.g. major fire)? → keep, flag for modelling
          - Is this evasive fraud that passed automated screening? → escalate immediately

        The CSV allows the claims team to review without needing Python.
        It can be opened in Excel and investigated by adjusters directly.
        The anomaly_type column tells the reviewer which category each record is.

        Returns self.
        """
        if len(self.confirmed) == 0:
            logger.info("[ANOMALY] No anomalies to save")
            return self

        path = REPORTS_DIR / "anomalies.csv"
        self.confirmed.to_csv(path, index=False)
        logger.info(f"[ANOMALY] {len(self.confirmed):,} anomaly rows saved: {path}")

        return self

    def summary(self) -> pd.DataFrame:
        """
        Return anomaly findings as a formatted DataFrame.

        Useful for displaying results in a notebook or adding to a report.
        If flag_fraud_patterns() was run, returns fraud profile summary instead.
        """
        # If fraud profile exists, return that as the primary summary
        if self.fraud_profile:
            profile = self.fraud_profile
            rows = [
                {"metric": "Total claims",        "value": profile["total_claims"]},
                {"metric": "Fraudulent claims",   "value": profile["fraudulent_claims"]},
                {"metric": "Legitimate claims",   "value": profile["legitimate_claims"]},
                {"metric": "Fraud rate (%)",      "value": profile["fraud_rate_pct"]},
                {"metric": "High-value fraud",    "value": profile["high_value_fraud"]},
                {"metric": "Evasive fraud",       "value": profile["evasive_fraud"]},
                {"metric": "Fraud mean amount",   "value": profile["fraud_mean_amount"]},
                {"metric": "Legit mean amount",   "value": profile["legit_mean_amount"]},
                {"metric": "Fraud median amount", "value": profile["fraud_median_amount"]},
                {"metric": "Legit median amount", "value": profile["legit_median_amount"]},
            ]
            return pd.DataFrame(rows)

        # Otherwise return the standard IQR/Z-score column-level summary
        if not self.results:
            return pd.DataFrame(columns=[
                "column", "iqr_flagged", "zscore_flagged",
                "confirmed_anomalies", "anomaly_pct"
            ])

        rows = []
        for col, result in self.results.items():
            rows.append({
                "column":               col,
                "iqr_flagged":          result["iqr_flagged"],
                "zscore_flagged":       result["zscore_flagged"],
                "confirmed_anomalies":  result["confirmed_anomalies"],
                "anomaly_pct":          result["anomaly_pct"],
            })

        return pd.DataFrame(rows)

    def __str__(self) -> str:
        return (
            f"AnomalyDetector("
            f"{self._n_checked} columns checked | "
            f"{len(self.confirmed):,} confirmed anomalies)"
        )

    def __repr__(self) -> str:
        return (
            f"AnomalyDetector("
            f"columns={self._n_checked}, "
            f"anomalies={len(self.confirmed)})"
        )