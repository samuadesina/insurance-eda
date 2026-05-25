# ================================================================
# src/eda_engine.py
# ================================================================
# CONTEXT:
#   We have processed-data.csv — clean, typed, enriched by the ETL pipeline.
#   Now we need to UNDERSTAND what is in it.
#
# THE BUSINESS QUESTION:
#   The Head of Actuarial Science at ShieldGuard Insurance Group wants to know:
#     - Which claim types and risk grades drive the highest losses?
#     - Are claim amounts statistically different across risk grades?
#     - Which features correlate most strongly with amount_approved?
#     - Is claim volume trending upward — and what is next period's forecast?
#     - Which claims are anomalous, and which fraudulent claims are hardest
#       to detect because they look completely normal?
#
# THE ANALOGY:
#   Imagine you just received a claims report from every regional office.
#   Before presenting to the board, you need to read it, find the patterns,
#   and summarise the key findings.
#   InsuranceEDAEngine reads the claims data, finds the patterns, summarises them.
#
# WHY A CLASS AND NOT JUST FUNCTIONS?
#   Because we need to run multiple types of analysis and keep ALL results.
#   A class stores everything in self.results so any other module can access:
#     engine.results["group_analysis"]  → group stats
#     engine.results["correlation"]     → correlation pairs
#     engine.results["anova"]           → ANOVA test results
#     engine.results["forecast"]        → next period estimate
#   Functions would run and throw away results. The class remembers.
#
# DESIGN PRINCIPLE: READ-ONLY
#   InsuranceEDAEngine never modifies the DataFrame. It only reads and summarises.
#   (Same as DataValidator in the ETL pipeline — analysts inspect, they do not edit.)
# ================================================================

# ── IMPORTS ───────────────────────────────────────────────────────
import sys        # sys: for manipulating Python's module search path
import pathlib    # pathlib: cross-platform file paths

# Walk up from this file's directory until we find config.py
_root = pathlib.Path(__file__).resolve().parent
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd                              # pandas: the core Python data library
import numpy as np                               # numpy: numerical operations
from scipy.stats import f_oneway, kruskal        # ANOVA and Kruskal-Wallis tests
from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # exponential smoothing

# Import our settings from config.py
from config import (
    INDUSTRY,              # which industry schema
    DATA_PATH,             # where processed-data.csv lives
    REPORTS_DIR,           # where to save the report
    TOP_N_GROUPS,          # how many top groups to show (8)
    CORRELATION_THRESHOLD, # minimum r to include (0.3)
    logger                 # shared logger
)


# ================================================================
# CLASS 1: InsuranceEDAEngine
# ================================================================

class InsuranceEDAEngine:
    """
    Runs Exploratory Data Analysis on the processed insurance dataset.

    WHAT IS EDA?
    ─────────────
    EDA (Exploratory Data Analysis) is the process of examining a dataset
    to discover patterns, relationships, and anomalies before building models.
    It was formalised by statistician John Tukey in 1977 and is now standard
    practice at every data-driven company.

    Every data scientist and analyst runs EDA as their FIRST step after
    receiving clean data. It answers the question: "What is in here?"

    WHAT THIS CLASS DOES:
    ──────────────────────
    Eight methods, each answering a different business question:
      1. load()                  → How many rows/columns? What types?
      2. profile()               → What are the distributions and completeness?
      3. group_analysis()        → How do claim amounts vary by claim type and risk grade?
      4. correlation()           → Which numeric variables move together?
      5. time_trends()           → How do claim volumes change over time?
      6. anova_test()            → Are claim amounts significantly different across risk grades?
      7. kruskal_test()          → Non-parametric alternative to ANOVA
      8. feature_importance()    → Which features correlate most with amount_approved?
      9. forecast_next_period()  → What is next period's expected claim volume?

    METHOD CHAIN PATTERN:
    ──────────────────────
    engine.load().profile().group_analysis().correlation().time_trends()
          .anova_test().kruskal_test().feature_importance().forecast_next_period()
          .report()

    Each method returns self so they can be chained like this.

    Attributes
    ──────────
    df         pd.DataFrame  the loaded processed data
    results    dict          all analysis outputs (keyed by analysis name)
    num_cols   list[str]     numeric column names (set by load())
    cat_cols   list[str]     categorical column names (set by load())
    _status    str           lifecycle state
    """

    def __init__(self):
        """
        Initialise the EDA engine.

        We do NOT load data here — that is load()'s job.
        This separation allows:
          - Object creation without any I/O
          - Testing without needing a real CSV file
          - Clear lifecycle: ready → loaded → analysed → reported
        """
        self.df       = None    # will hold the DataFrame after load()
        self.results  = {}      # will hold all analysis outputs
        self.num_cols = []      # numeric columns (identified in load())
        self.cat_cols = []      # categorical columns (identified in load())
        self._status  = "ready"

        logger.info(f"InsuranceEDAEngine initialised — industry: {INDUSTRY}")

    def load(self) -> "InsuranceEDAEngine":
        """
        Load processed-data.csv and identify column types.

        WHY DO WE REMOVE METADATA COLUMNS?
        ─────────────────────────────────────
        The ETL pipeline added three columns starting with _:
          _industry, _processed_at, _pipeline_version
        These describe the pipeline run — NOT the business data.
        Including them in groupby or correlation analysis would add noise.
        We exclude them for analysis but keep the full DataFrame for saving.

        WHY select_dtypes?
        ─────────────────
        select_dtypes(include=["number"]) returns a subset of the DataFrame
        containing ONLY numeric columns (int64, float64).
        select_dtypes(include=["object"]) returns only text/categorical columns.
        This is how pandas separates column types automatically.

        Returns self for method chaining.
        """
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"processed-data.csv not found at: {DATA_PATH}\n"
                "Run the ETL pipeline first:\n"
                "  python run.py\n"
                "Then copy processed-data.csv to data/"
            )

        logger.info(f"[EDA] Loading: {DATA_PATH.name}")

        # pd.read_csv() loads a CSV file from disk into a pandas DataFrame
        # low_memory=False reads the entire file before inferring column types
        self.df = pd.read_csv(DATA_PATH, low_memory=False)

        logger.info(f"[EDA] Loaded {len(self.df):,} rows × {self.df.shape[1]} columns")

        # Remove pipeline metadata columns (start with _) from analysis
        analysis_df = self.df.drop(
            columns=[c for c in self.df.columns if c.startswith("_")],
            errors="ignore"
        )

        # Identify column types using pandas type detection
        self.num_cols = analysis_df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = analysis_df.select_dtypes(include=["object"]).columns.tolist()

        self._status = "loaded"

        logger.info(
            f"[EDA] Column types: "
            f"{len(self.num_cols)} numeric, "
            f"{len(self.cat_cols)} categorical"
        )

        return self

    def profile(self) -> "InsuranceEDAEngine":
        """
        Compute a complete statistical profile of the dataset.

        WHY PROFILE FIRST?
        ────────────────────
        Before asking "which claim type has the highest loss ratio?"
        you need to know: "Do we have claim_type for all records,
        or is 5% missing?" The profile gives you confidence in — or warnings
        about — the data before you draw any conclusions.

        WHAT pd.DataFrame.describe() DOES:
        ─────────────────────────────────
        For each numeric column, it computes:
          count  → how many non-null values
          mean   → arithmetic average
          std    → standard deviation (how spread out values are)
          min    → smallest value
          25%    → 25th percentile (first quartile)
          50%    → median (middle value)
          75%    → 75th percentile (third quartile)
          max    → largest value

        WHY IS MEDIAN (50%) OFTEN MORE USEFUL THAN MEAN?
        ──────────────────────────────────────────────────
        Claim amounts are right-skewed — a few catastrophic claims
        drag the mean upward. The median gives the genuine middle value.

        Returns self.
        """
        logger.info("[EDA] Computing dataset profile...")

        profile = {
            "rows":             len(self.df),
            "columns":          len(self.df.columns),
            "numeric_cols":     len(self.num_cols),
            "categorical_cols": len(self.cat_cols),

            # Grand total of null cells across the entire DataFrame
            "total_nulls":      int(self.df.isna().sum().sum()),

            # Null as a percentage of all cells
            "null_pct":         round(
                                    self.df.isna().sum().sum() / self.df.size * 100, 2
                                ),

            # Memory consumed by this DataFrame in megabytes
            "memory_mb":        round(
                                    self.df.memory_usage(deep=True).sum() / 1024**2, 2
                                ),

            # Rows that are exact copies of another row
            "duplicates":       int(self.df.duplicated().sum()),
        }

        # Descriptive statistics for numeric columns
        if self.num_cols:
            desc = self.df[self.num_cols].describe().round(3)
            profile["descriptive_stats"] = desc.to_dict()

        # Value counts for categorical columns (top values and their frequencies)
        cat_profiles = {}
        for col in self.cat_cols[:8]:
            vc = self.df[col].value_counts()
            cat_profiles[col] = {
                "unique_count": int(self.df[col].nunique()),
                "top_5":        vc.head(5).to_dict(),
                "null_count":   int(self.df[col].isna().sum()),
            }
        profile["categorical_profiles"] = cat_profiles

        self.results["profile"] = profile

        logger.info(
            f"[EDA] Profile complete — "
            f"{profile['rows']:,} rows | "
            f"{profile['null_pct']}% nulls"
        )

        return self

    def group_analysis(self) -> "InsuranceEDAEngine":
        """
        Group numeric metrics by categorical columns and compute aggregates.

        WHY THIS IS THE MOST IMPORTANT EDA STEP FOR INSURANCE:
        ────────────────────────────────────────────────────────
        "What is the average claim amount?" is a weak question.
        "What is the average claim amount per claim type and risk grade?" is
        a strong question that drives actuarial pricing decisions.

        For ShieldGuard the Head of Actuarial Science cares about:
          - Average amount_claimed by claim_type
          - Fraud rate by risk_grade
          - Claim count by policy_type

        HOW pd.DataFrame.groupby() WORKS:
        ───────────────────────────────────
        df.groupby("claim_type")["amount_claimed"].agg(["mean", "median", "std"])
          → splits the DataFrame into one group per unique claim_type
          → takes the amount_claimed column from each group
          → computes mean, median, std for each group
          → returns a new DataFrame with one row per claim_type

        Returns self.
        """
        logger.info("[EDA] Running group analysis...")

        group_results = {}

        for cat in self.cat_cols[:2]:
            col_results = {}
            for num in self.num_cols[:3]:

                agg = (
                    self.df.groupby(cat)[num]
                    .agg(["count", "mean", "median", "std", "min", "max"])
                    .round(2)
                    .reset_index()
                )

                # rank(ascending=False) gives rank 1 to the highest mean
                agg["rank"] = agg["mean"].rank(ascending=False).astype(int)

                col_results[num] = (
                    agg.sort_values("rank")
                       .head(TOP_N_GROUPS)
                       .to_dict(orient="records")
                )

            group_results[cat] = col_results

        # ── Insurance-specific: fraud rate by claim_type and risk_grade ──
        fraud_col = "is_fraudulent"
        if fraud_col in self.df.columns:
            fraud_groups = {}
            for cat in ["claim_type", "risk_grade", "policy_type"]:
                if cat not in self.df.columns:
                    continue
                fraud_rate = (
                    self.df.groupby(cat)[fraud_col]
                    .agg(
                        fraud_count="sum",
                        total_claims="count",
                    )
                    .reset_index()
                )
                fraud_rate["fraud_rate_pct"] = (
                    fraud_rate["fraud_count"] / fraud_rate["total_claims"] * 100
                ).round(2)
                fraud_rate = fraud_rate.sort_values("fraud_rate_pct", ascending=False)
                fraud_groups[cat] = fraud_rate.to_dict(orient="records")

            group_results["fraud_by_group"] = fraud_groups

        self.results["group_analysis"] = group_results

        logger.info(f"[EDA] Group analysis: {len(group_results)} grouping variables")

        return self

    def correlation(self) -> "InsuranceEDAEngine":
        """
        Compute Pearson correlation between all numeric column pairs.

        WHAT IS PEARSON CORRELATION?
        ─────────────────────────────
        The Pearson correlation coefficient (r) measures the LINEAR relationship
        between two numeric variables.

        r ranges from -1 to +1:
          +1.0 → perfect positive relationship
           0.0 → no linear relationship
          -1.0 → perfect negative relationship

        BUSINESS INTERPRETATION FOR INSURANCE:
        ────────────────────────────────────────
        We are especially interested in:
          - correlation between amount_claimed and amount_approved
            (do adjusters approve proportionally to what is claimed?)
          - correlation between credit_score and is_fraudulent
            (does credit quality predict fraud exposure?)
          - correlation between coverage_amount and amount_approved
            (are payouts bounded by policy coverage?)

        |r| > 0.7 → STRONG
        |r| > 0.5 → MODERATE
        |r| > 0.3 → WEAK
        |r| < 0.3 → NEGLIGIBLE (excluded from report)

        Returns self.
        """
        if len(self.num_cols) < 2:
            logger.warning("[EDA] Not enough numeric columns for correlation")
            return self

        logger.info("[EDA] Computing correlation matrix...")

        corr_matrix = self.df[self.num_cols].corr(numeric_only=True).round(3)

        corr_pairs = []
        for i, col_a in enumerate(self.num_cols):
            for j, col_b in enumerate(self.num_cols):
                if i >= j:
                    continue

                val = corr_matrix.loc[col_a, col_b]

                if abs(val) < CORRELATION_THRESHOLD:
                    continue

                if abs(val) > 0.7:
                    strength = "STRONG"
                elif abs(val) > 0.5:
                    strength = "MODERATE"
                else:
                    strength = "WEAK"

                corr_pairs.append({
                    "col_a":       col_a,
                    "col_b":       col_b,
                    "correlation": float(val),
                    "strength":    strength,
                    "direction":   "positive" if val > 0 else "negative",
                })

        corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        self.results["correlation"] = {
            "matrix":       corr_matrix.to_dict(),
            "strong_pairs": corr_pairs[:10],
        }

        logger.info(
            f"[EDA] Correlation: {len(corr_pairs)} meaningful pairs "
            f"(threshold |r| > {CORRELATION_THRESHOLD})"
        )

        return self

    def time_trends(self) -> "InsuranceEDAEngine":
        """
        Detect time/period columns and compute month-over-month trends.

        WHY TIME TRENDS MATTER FOR INSURANCE:
        ───────────────────────────────────────
        A single claim count is a snapshot. A monthly claim count trend is a movie.
        "We processed 1,200 claims this quarter" — is that growing or shrinking?
        Are certain months higher risk (e.g. winter weather events driving property claims)?

        METRICS WE COMPUTE FOR EACH TIME PERIOD:
        ──────────────────────────────────────────
          sum          → total claim value for this period
          mean         → average claim amount for this period
          mom_change   → mean minus previous period mean (absolute change)
          mom_pct      → percentage change from previous period
          rolling_3m   → 3-period rolling average (smooths short-term noise)

        Returns self.
        """
        time_keywords = ["month", "period", "quarter", "year", "date"]
        time_cols = [
            c for c in self.df.columns
            if any(kw in c.lower() for kw in time_keywords)
            and "date" not in c.lower()
        ]

        if not time_cols:
            # Try to extract month from claim_date if it exists
            if "claim_date" in self.df.columns:
                self.df["_month"] = pd.to_datetime(
                    self.df["claim_date"], errors="coerce"
                ).dt.to_period("M").astype(str)
                time_cols = ["_month"]
            else:
                logger.info("[EDA] No time column found — skipping time trends")
                return self

        time_col = time_cols[0]

        if self.df[time_col].nunique() > 200:
            logger.info(f"[EDA] {time_col} has too many unique values — skipping")
            return self

        logger.info(f"[EDA] Computing time trends on '{time_col}'...")

        trend_results = {}
        for num_col in ["amount_claimed", "amount_approved"]:
            if num_col not in self.df.columns:
                continue

            monthly = (
                self.df.groupby(time_col)[num_col]
                .agg(["count", "sum", "mean"])
                .round(2)
                .reset_index()
                .sort_values(time_col)
            )

            monthly["mom_change"] = monthly["mean"].diff()
            monthly["mom_pct"]    = monthly["mean"].pct_change().mul(100).round(1)
            monthly["rolling_3m"] = monthly["mean"].rolling(3).mean().round(2)

            trend_results[num_col] = monthly.to_dict(orient="records")

        # ── Insurance-specific: fraud count per month ─────────────────
        if "is_fraudulent" in self.df.columns:
            fraud_trend = (
                self.df.groupby(time_col)["is_fraudulent"]
                .agg(fraud_count="sum", total="count")
                .reset_index()
                .sort_values(time_col)
            )
            fraud_trend["fraud_rate_pct"] = (
                fraud_trend["fraud_count"] / fraud_trend["total"] * 100
            ).round(2)
            trend_results["fraud_rate"] = fraud_trend.to_dict(orient="records")

        self.results["time_trends"] = trend_results
        logger.info(f"[EDA] Time trends: {len(trend_results)} metrics analysed")

        return self

    def anova_test(self,
                   group_col: str = "risk_grade",
                   value_col: str = "amount_claimed") -> "InsuranceEDAEngine":
        """
        One-way ANOVA test to determine if claim amounts differ significantly
        across groups (e.g. risk grades).

        WHAT IS ANOVA?
        ───────────────
        Analysis of Variance (ANOVA) tests whether the MEANS of three or more
        groups are statistically different from each other.

        H₀ (null hypothesis):      mean claim amount is EQUAL across all risk grades
        H₁ (alternative):          at least one risk grade has a different mean

        DECISION RULE:
          p < 0.05  → REJECT H₀  — group membership significantly affects claim amount
          p ≥ 0.05  → FAIL TO REJECT H₀ — differences may be noise

        WHY ANOVA AND NOT MULTIPLE T-TESTS?
        ──────────────────────────────────────
        If you run a t-test between every pair of groups (A vs B, A vs C, B vs C...),
        your chance of a false positive compounds with each test.
        ANOVA tests ALL groups simultaneously with a single p-value.

        ASSUMPTION: ANOVA assumes normally distributed groups with similar variance.
        If violated, use kruskal_test() instead (see below).

        Args:
            group_col   categorical column defining groups (default: 'risk_grade')
            value_col   numeric column to test (default: 'amount_claimed')

        Returns self.
        """
        for col in [group_col, value_col]:
            if col not in self.df.columns:
                logger.warning(f"[EDA] ANOVA skipped — column '{col}' not found")
                return self

        logger.info(f"[EDA] ANOVA test: '{value_col}' grouped by '{group_col}'")

        # Build one array per group — f_oneway() takes *args (one per group)
        groups = [
            group[value_col].dropna().values
            for _, group in self.df.groupby(group_col)
        ]

        group_labels = sorted(self.df[group_col].dropna().unique().tolist())

        # scipy.stats.f_oneway() returns the F-statistic and p-value
        # F-statistic: ratio of between-group variance to within-group variance
        # Large F → groups differ more than expected by chance
        f_stat, p_value = f_oneway(*groups)

        result = {
            "test":         "One-Way ANOVA",
            "group_col":    group_col,
            "value_col":    value_col,
            "groups":       group_labels,
            "n_groups":     len(groups),
            "f_statistic":  round(float(f_stat), 4),
            "p_value":      round(float(p_value), 6),
            "significant":  bool(p_value < 0.05),
            "conclusion": (
                f"REJECT H₀ — claim amounts differ significantly across "
                f"'{group_col}' groups (p={p_value:.4f}). "
                f"Risk grade is a statistically significant driver of claim size."
                if p_value < 0.05 else
                f"FAIL TO REJECT H₀ — no significant difference in '{value_col}' "
                f"across '{group_col}' groups (p={p_value:.4f}). "
                f"Consider kruskal_test() if distributions are non-normal."
            ),
        }

        self.results["anova"] = result

        logger.info(
            f"[EDA] ANOVA: F={f_stat:.2f} | p={p_value:.4f} | "
            f"significant={p_value < 0.05}"
        )

        return self

    def kruskal_test(self,
                     group_col: str = "risk_grade",
                     value_col: str = "amount_claimed") -> "InsuranceEDAEngine":
        """
        Kruskal-Wallis test — non-parametric alternative to one-way ANOVA.

        WHEN TO USE KRUSKAL-WALLIS INSTEAD OF ANOVA:
        ──────────────────────────────────────────────
        ANOVA assumes:
          1. Groups are normally distributed
          2. Groups have similar variance (homoscedasticity)

        Insurance claim amounts are almost always RIGHT-SKEWED — a few
        catastrophic claims drag the distribution. This violates assumption 1.

        Kruskal-Wallis:
          - Makes NO assumptions about distribution shape
          - Works on RANKS rather than raw values
          - Is the standard choice for financial/insurance claim data

        RULE OF THUMB:
          Run ANOVA first. If p is borderline OR you suspect non-normality,
          run Kruskal-Wallis. If both agree → conclusion is robust.
          If they disagree → trust Kruskal-Wallis for skewed data.

        H₀: the distributions of all groups are identical
        H₁: at least one group has a different distribution

        Args:
            group_col   categorical column defining groups (default: 'risk_grade')
            value_col   numeric column to test (default: 'amount_claimed')

        Returns self.
        """
        for col in [group_col, value_col]:
            if col not in self.df.columns:
                logger.warning(f"[EDA] Kruskal skipped — column '{col}' not found")
                return self

        logger.info(f"[EDA] Kruskal-Wallis test: '{value_col}' grouped by '{group_col}'")

        groups = [
            group[value_col].dropna().values
            for _, group in self.df.groupby(group_col)
        ]

        group_labels = sorted(self.df[group_col].dropna().unique().tolist())

        # scipy.stats.kruskal() returns the H-statistic and p-value
        # H-statistic: based on ranks, not raw values
        h_stat, p_value = kruskal(*groups)

        result = {
            "test":         "Kruskal-Wallis",
            "group_col":    group_col,
            "value_col":    value_col,
            "groups":       group_labels,
            "n_groups":     len(groups),
            "h_statistic":  round(float(h_stat), 4),
            "p_value":      round(float(p_value), 6),
            "significant":  bool(p_value < 0.05),
            "conclusion": (
                f"REJECT H₀ — distributions of '{value_col}' differ significantly "
                f"across '{group_col}' groups (p={p_value:.4f}). "
                f"Result is robust to non-normality — recommended over ANOVA for skewed claim data."
                if p_value < 0.05 else
                f"FAIL TO REJECT H₀ — no significant difference in '{value_col}' "
                f"distributions across '{group_col}' groups (p={p_value:.4f})."
            ),
        }

        self.results["kruskal"] = result

        logger.info(
            f"[EDA] Kruskal-Wallis: H={h_stat:.2f} | p={p_value:.4f} | "
            f"significant={p_value < 0.05}"
        )

        return self

    def feature_importance(self,
                           target_col: str = "amount_approved") -> "InsuranceEDAEngine":
        """
        Rank all numeric features by Pearson correlation with a target column.

        WHY PEARSON CORRELATION AS FEATURE IMPORTANCE?
        ─────────────────────────────────────────────────
        Before building a predictive model (e.g. claim approval amount predictor),
        you need to know which input features are worth including.

        Pearson correlation gives a fast, interpretable ranking:
          High |r| → feature moves strongly with the target → likely useful
          Low  |r| → feature has little linear relationship → candidate for removal

        LIMITATION:
        ────────────
        Pearson only captures LINEAR relationships.
        A feature with r=0.05 might still be useful in a non-linear model.
        This ranking is a starting point, not a final answer.

        BUSINESS INTERPRETATION FOR INSURANCE:
        ────────────────────────────────────────
        If coverage_amount has r=0.85 with amount_approved → payouts are
        strongly bounded by policy limits (expected, validates data integrity).
        If credit_score has r=-0.30 with amount_approved → lower credit score
        customers receive higher approvals (risk pricing signal).

        Args:
            target_col   column to rank features against (default: 'amount_approved')

        Returns self.
        """
        if target_col not in self.df.columns:
            logger.warning(f"[EDA] feature_importance skipped — '{target_col}' not found")
            return self

        logger.info(f"[EDA] Computing feature importance for '{target_col}'...")

        # Compute correlation of every numeric column with the target
        correlations = (
            self.df[self.num_cols]
            .corr()[target_col]
            .drop(labels=[target_col], errors="ignore")  # exclude target vs itself
            .dropna()
        )

        # Build ranked list — absolute value for ranking, keep sign for direction
        importance = []
        for col, r in correlations.items():
            importance.append({
                "feature":      col,
                "correlation":  round(float(r), 4),
                "abs_corr":     round(abs(float(r)), 4),
                "direction":    "positive" if r > 0 else "negative",
                "strength": (
                    "STRONG"   if abs(r) > 0.7 else
                    "MODERATE" if abs(r) > 0.5 else
                    "WEAK"     if abs(r) > 0.3 else
                    "NEGLIGIBLE"
                ),
            })

        # Sort by absolute correlation descending — most important feature first
        importance.sort(key=lambda x: x["abs_corr"], reverse=True)

        self.results["feature_importance"] = {
            "target":   target_col,
            "features": importance,
        }

        logger.info(
            f"[EDA] Feature importance: {len(importance)} features ranked | "
            f"top feature: {importance[0]['feature']} "
            f"(r={importance[0]['correlation']})"
            if importance else "[EDA] No features found"
        )

        return self

    def forecast_next_period(self,
                             col: str = "claim_date") -> "InsuranceEDAEngine":
        """
        Forecast next period's claim volume using exponential smoothing.

        WHAT IS EXPONENTIAL SMOOTHING?
        ────────────────────────────────
        Exponential smoothing forecasts the next value in a time series
        by computing a weighted average of past observations — where more
        recent observations receive exponentially more weight than older ones.

        For claim volume forecasting:
          - We count claims per month from claim_date
          - We fit SimpleExpSmoothing to the monthly series
          - We forecast one step ahead (next month's expected claim count)

        WHY EXPONENTIAL SMOOTHING FOR INSURANCE?
        ──────────────────────────────────────────
        Claim volumes have trend and seasonality (winter weather, flooding).
        Exponential smoothing is the actuarial industry standard for
        short-term volume projection — it is fast, interpretable, and
        does not require stationarity like ARIMA.

        BUSINESS USE:
        ──────────────
        If current quarter claim count is 5,400 and the forecast is 5,900,
        that is a +9% increase — the Head of Actuarial Science needs to
        communicate this to the board as a reserve adequacy warning.

        Args:
            col   date column to extract monthly volumes from (default: 'claim_date')

        Returns self.
        """
        if col not in self.df.columns:
            logger.warning(f"[EDA] forecast_next_period skipped — '{col}' not found")
            return self

        logger.info(f"[EDA] Forecasting next period claim volume from '{col}'...")

        # Parse claim_date to datetime, extract year-month period
        dates = pd.to_datetime(self.df[col], errors="coerce")
        monthly_counts = (
            dates.dt.to_period("M")
            .value_counts()
            .sort_index()
        )

        if len(monthly_counts) < 3:
            logger.warning("[EDA] Forecast skipped — fewer than 3 monthly periods available")
            return self

        series = monthly_counts.values.astype(float)

        # SimpleExpSmoothing from statsmodels fits an exponential smoothing model
        # optimized=True lets statsmodels find the best smoothing parameter (alpha)
        model  = SimpleExpSmoothing(series).fit(optimized=True)
        forecast_value = float(model.forecast(1)[0])

        last_period  = str(monthly_counts.index[-1])
        last_value   = float(series[-1])
        pct_change   = round((forecast_value - last_value) / last_value * 100, 2)

        result = {
            "method":          "SimpleExpSmoothing",
            "col":             col,
            "n_periods":       len(series),
            "last_period":     last_period,
            "last_value":      round(last_value, 0),
            "forecast_value":  round(forecast_value, 0),
            "pct_change":      pct_change,
            "smoothing_level": round(float(model.params["smoothing_level"]), 4),
            "monthly_series":  monthly_counts.reset_index()
                                             .rename(columns={col: "period", "count": "claim_count"})
                                             .to_dict(orient="records"),
            "interpretation": (
                f"Claim volume is forecast to {'increase' if pct_change > 0 else 'decrease'} "
                f"by {abs(pct_change)}% next period "
                f"(from {last_value:.0f} to {forecast_value:.0f} claims). "
                + (
                    "Rising claim volume without a corresponding premium increase "
                    "signals further loss ratio deterioration."
                    if pct_change > 0 else
                    "Declining claim volume may reflect seasonal patterns "
                    "or improved underwriting controls."
                )
            ),
        }

        self.results["forecast"] = result

        logger.info(
            f"[EDA] Forecast: {last_value:.0f} → {forecast_value:.0f} claims "
            f"({pct_change:+.1f}%)"
        )

        return self

    def report(self, save: bool = True) -> None:
        """
        Print and optionally save the structured analysis report.

        This is the final step — it turns numbers into language the
        Head of Actuarial Science can present to the board.
        A good EDA report:
          - States findings as sentences, not just tables
          - Ranks items (highest to lowest) so the most important comes first
          - Flags anomalies and exceptions explicitly
          - Recommends next steps

        Args:
            save   if True, saves the report to reports/insurance_eda_report.html
        """
        lines = []

        lines += [
            "═" * 65,
            f"  SHIELDGUARD EDA REPORT  |  INDUSTRY: {INDUSTRY.upper()}",
            "═" * 65,
        ]

        # ── Dataset profile ────────────────────────────────────────────
        if "profile" in self.results:
            p = self.results["profile"]
            lines += [
                "",
                "  DATASET OVERVIEW",
                f"    Records:              {p['rows']:,}",
                f"    Columns:              {p['columns']}",
                f"    Numeric columns:      {p['numeric_cols']}",
                f"    Categorical columns:  {p['categorical_cols']}",
                f"    Missing values:       {p['total_nulls']:,} ({p['null_pct']}%)",
                f"    Duplicate rows:       {p['duplicates']:,}",
                f"    Memory usage:         {p['memory_mb']} MB",
            ]

            if "descriptive_stats" in p:
                lines += ["", "  DESCRIPTIVE STATISTICS"]
                lines.append(
                    f"    {'Metric':<30} {'Mean':>12} {'Median':>12} {'Std Dev':>10}"
                )
                lines.append("    " + "-" * 65)
                for col in list(p["descriptive_stats"].get("mean", {}).keys())[:6]:
                    mean = p["descriptive_stats"].get("mean", {}).get(col)
                    med  = p["descriptive_stats"].get("50%",  {}).get(col)
                    std  = p["descriptive_stats"].get("std",  {}).get(col)
                    m_s  = f"{mean:>12,.2f}" if isinstance(mean, float) else f"{mean:>12}"
                    md_s = f"{med:>12,.2f}"  if isinstance(med,  float) else f"{med:>12}"
                    st_s = f"{std:>10,.2f}"  if isinstance(std,  float) else f"{std:>10}"
                    lines.append(f"    {col:<30} {m_s} {md_s} {st_s}")

        # ── Fraud by group ─────────────────────────────────────────────
        if "group_analysis" in self.results:
            fraud_groups = self.results["group_analysis"].get("fraud_by_group", {})
            if fraud_groups:
                lines += ["", "  FRAUD RATES BY GROUP"]
                for group_col, rows in fraud_groups.items():
                    lines += [
                        "",
                        f"    {group_col.upper()}",
                        f"    {'Group':<28} {'Fraud Count':>12} {'Total':>10} {'Fraud %':>9}",
                        "    " + "-" * 62,
                    ]
                    for row in rows[:TOP_N_GROUPS]:
                        g  = str(row.get(group_col, ""))[:27]
                        fc = row.get("fraud_count", 0)
                        tt = row.get("total_claims", 0)
                        fr = row.get("fraud_rate_pct", 0.0)
                        lines.append(f"    {g:<28} {fc:>12,} {tt:>10,} {fr:>8.2f}%")

        # ── ANOVA ──────────────────────────────────────────────────────
        if "anova" in self.results:
            av = self.results["anova"]
            lines += [
                "",
                "  ONE-WAY ANOVA",
                f"    Groups:       {av['group_col']} ({av['n_groups']} groups)",
                f"    Target:       {av['value_col']}",
                f"    F-statistic:  {av['f_statistic']}",
                f"    p-value:      {av['p_value']}",
                f"    Significant:  {'YES — reject H₀' if av['significant'] else 'NO — fail to reject H₀'}",
                f"    Conclusion:   {av['conclusion']}",
            ]

        # ── Kruskal-Wallis ─────────────────────────────────────────────
        if "kruskal" in self.results:
            kw = self.results["kruskal"]
            lines += [
                "",
                "  KRUSKAL-WALLIS TEST",
                f"    Groups:       {kw['group_col']} ({kw['n_groups']} groups)",
                f"    Target:       {kw['value_col']}",
                f"    H-statistic:  {kw['h_statistic']}",
                f"    p-value:      {kw['p_value']}",
                f"    Significant:  {'YES — reject H₀' if kw['significant'] else 'NO — fail to reject H₀'}",
                f"    Conclusion:   {kw['conclusion']}",
            ]

        # ── Feature importance ─────────────────────────────────────────
        if "feature_importance" in self.results:
            fi = self.results["feature_importance"]
            lines += [
                "",
                f"  FEATURE IMPORTANCE → '{fi['target']}'",
                f"    {'Feature':<30} {'Correlation':>12}  Strength",
                "    " + "-" * 55,
            ]
            for feat in fi["features"][:8]:
                lines.append(
                    f"    {feat['feature']:<30} {feat['correlation']:>12.4f}  "
                    f"{feat['strength']} {feat['direction']}"
                )

        # ── Correlation ────────────────────────────────────────────────
        if "correlation" in self.results:
            pairs = self.results["correlation"]["strong_pairs"]
            if pairs:
                lines += [
                    "",
                    f"  TOP CORRELATIONS (|r| > {CORRELATION_THRESHOLD})",
                    f"    {'Column A':<28} {'Column B':<28} {'r':>8}  Strength",
                    "    " + "-" * 70,
                ]
                for pair in pairs[:8]:
                    lines.append(
                        f"    {pair['col_a']:<28} {pair['col_b']:<28} "
                        f"{pair['correlation']:>8.3f}  {pair['strength']} "
                        f"{pair['direction']}"
                    )

        # ── Forecast ───────────────────────────────────────────────────
        if "forecast" in self.results:
            fc = self.results["forecast"]
            lines += [
                "",
                "  CLAIM VOLUME FORECAST (Exponential Smoothing)",
                f"    Method:           {fc['method']}",
                f"    Periods used:     {fc['n_periods']}",
                f"    Last period:      {fc['last_period']} ({fc['last_value']:.0f} claims)",
                f"    Next period:      {fc['forecast_value']:.0f} claims ({fc['pct_change']:+.1f}%)",
                f"    Smoothing alpha:  {fc['smoothing_level']}",
                f"    Interpretation:   {fc['interpretation']}",
            ]

        # ── Time trends ────────────────────────────────────────────────
        if "time_trends" in self.results:
            lines += ["", "  TIME TRENDS (last 6 periods)"]
            for metric, rows in self.results["time_trends"].items():
                if not rows:
                    continue
                period_key = list(rows[0].keys())[0]
                lines += [
                    "",
                    f"    {metric.upper()}",
                    f"    {'Period':<15} {'Mean':>12} {'MoM %':>9} {'Rolling 3':>12}",
                    "    " + "-" * 52,
                ]
                for row in rows[-6:]:
                    period  = str(row.get(period_key, ""))
                    mean_v  = row.get("mean", 0) or row.get("fraud_rate_pct", 0)
                    mom_p   = row.get("mom_pct") or 0.0
                    rolling = row.get("rolling_3m") or 0.0
                    lines.append(
                        f"    {period:<15} {mean_v:>12,.2f} {mom_p:>8.1f}% "
                        f"{rolling:>12,.2f}"
                    )

        # ── Next steps ─────────────────────────────────────────────────
        lines += [
            "",
            "  NEXT STEPS:",
            "    → Use ANOVA/Kruskal findings to reprice high-risk grade policies",
            "    → Pass feature importance rankings to actuarial modelling team",
            "    → Use forecast to stress-test reserve adequacy for next quarter",
            "    → Use correlation findings for ML feature selection",
            "    → Use profile as drift detection baseline",
            "",
            "═" * 65,
        ]

        report_text = "\n".join(lines)
        print(report_text)

        if save:
            report_path = REPORTS_DIR / "insurance_eda_report.txt"
            report_path.write_text(report_text, encoding="utf-8")
            logger.info(f"[EDA] Report saved: {report_path}")

    def __str__(self) -> str:
        rows = len(self.df) if self.df is not None else 0
        return (
            f"InsuranceEDAEngine("
            f"industry={INDUSTRY!r}, "
            f"rows={rows:,}, "
            f"analyses={list(self.results.keys())})"
        )

    def __repr__(self) -> str:
        return (
            f"InsuranceEDAEngine("
            f"industry={INDUSTRY!r}, "
            f"status={self._status!r})"
        )


# ================================================================
# CLASS 2: AnomalyDetector
# ================================================================

class AnomalyDetector:
    """
    Detects anomalous claims and profiles fraudulent vs legitimate claims.

    BUSINESS CONTEXT:
    ──────────────────
    ShieldGuard processes thousands of claims monthly. Two problems:
      1. HIGH-VALUE anomalies → obvious outliers, easy to catch
      2. EVASIVE fraud → fraudulent claims with NORMAL amounts
         that blend into legitimate activity and bypass automated rules

    This class addresses both.

    DESIGN PRINCIPLE: SINGLE RESPONSIBILITY
    ─────────────────────────────────────────
    AnomalyDetector does ONE thing: flag and profile anomalous claims.
    It does not load data, run statistical tests, or produce charts.

    Attributes
    ──────────
    df        pd.DataFrame   the processed dataset (copy — never mutated)
    results   list[dict]     flagged anomalous records
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df   the processed DataFrame (from InsuranceEDAEngine.df or read directly)
        """
        # df.copy() protects the caller's DataFrame from any accidental mutation
        self.df      = df.copy()
        self.results = []

        logger.info(
            f"AnomalyDetector initialised — "
            f"{len(self.df):,} rows"
        )

    def flag_anomalies(self,
                       amount_col: str = "amount_claimed") -> "AnomalyDetector":
        """
        Flag claims with anomalous amounts using IQR bounds.

        WHAT IS IQR-BASED OUTLIER DETECTION?
        ──────────────────────────────────────
        IQR (Interquartile Range) = Q3 - Q1
        Lower bound = Q1 - 1.5 × IQR
        Upper bound = Q3 + 1.5 × IQR

        Any value outside these bounds is flagged as a statistical outlier.
        This is the Tukey fence method — robust to skewed distributions
        because it uses percentiles rather than mean/std.

        WHY NOT JUST USE Z-SCORE?
        ─────────────────────────
        Z-score assumes normality. Claim amounts are right-skewed.
        IQR is the preferred method for financial and insurance data.

        Args:
            amount_col   column to check for outliers (default: 'amount_claimed')

        Returns self.
        """
        if amount_col not in self.df.columns:
            logger.warning(f"[ANOMALY] Column '{amount_col}' not found — skipping")
            return self

        logger.info(f"[ANOMALY] Flagging outliers in '{amount_col}'...")

        q1  = self.df[amount_col].quantile(0.25)
        q3  = self.df[amount_col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = self.df[
            (self.df[amount_col] < lower) | (self.df[amount_col] > upper)
        ].copy()

        outliers["anomaly_type"]  = "HIGH_VALUE_OUTLIER"
        outliers["anomaly_score"] = (
            (outliers[amount_col] - self.df[amount_col].mean())
            / self.df[amount_col].std()
        ).round(3)

        self.results = outliers.to_dict(orient="records")

        logger.info(
            f"[ANOMALY] {len(outliers):,} outliers flagged "
            f"(bounds: {lower:,.2f} — {upper:,.2f})"
        )

        return self

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
           → Easy to detect — automated rules engines catch these

        2. EVASIVE FRAUD (the hard problem)
           is_fraudulent = True AND amount_claimed is WITHIN normal IQR bounds
           → Blends into legitimate claim patterns
           → Most likely to pass automated screening undetected
           → Highest priority for manual review

        WHY THIS MATTERS:
        ──────────────────
        A claims adjuster reviewing 5,000 claims cannot manually check every one.
        Flagging evasive fraud isolates the highest-priority subset:
        claims that look legitimate but are confirmed fraudulent.

        Results saved to reports/anomalies.csv with anomaly_type column.

        Args:
            fraud_col   boolean fraud flag column (default: 'is_fraudulent')
            amount_col  numeric claim amount column (default: 'amount_claimed')

        Returns self.
        """
        for col in [fraud_col, amount_col]:
            if col not in self.df.columns:
                logger.warning(f"[ANOMALY] Column '{col}' not found — skipping fraud patterns")
                return self

        logger.info("[ANOMALY] Profiling fraud patterns...")

        # ── Compute IQR bounds for amount_claimed ─────────────────────
        q1  = self.df[amount_col].quantile(0.25)
        q3  = self.df[amount_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        fraudulent = self.df[self.df[fraud_col] == True].copy()

        # ── Flag 1: high-value fraud ───────────────────────────────────
        high_value_fraud = fraudulent[
            (fraudulent[amount_col] < lower) | (fraudulent[amount_col] > upper)
        ].copy()
        high_value_fraud["anomaly_type"] = "HIGH_VALUE_FRAUD"

        # ── Flag 2: evasive fraud — normal amount but confirmed fraudulent
        evasive_fraud = fraudulent[
            (fraudulent[amount_col] >= lower) & (fraudulent[amount_col] <= upper)
        ].copy()
        evasive_fraud["anomaly_type"] = "EVASIVE_FRAUD"

        # ── Statistical profile comparison ────────────────────────────
        legit = self.df[self.df[fraud_col] == False]

        profile = {
            "total_claims":          len(self.df),
            "fraudulent_claims":     len(fraudulent),
            "fraud_rate_pct":        round(len(fraudulent) / len(self.df) * 100, 2),
            "high_value_fraud":      len(high_value_fraud),
            "evasive_fraud":         len(evasive_fraud),
            "fraud_mean_amount":     round(float(fraudulent[amount_col].mean()),  2),
            "legit_mean_amount":     round(float(legit[amount_col].mean()),       2),
            "fraud_median_amount":   round(float(fraudulent[amount_col].median()), 2),
            "legit_median_amount":   round(float(legit[amount_col].median()),      2),
        }

        # Fraud rate by claim_type if available
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
                fraud_by_type.sort_values("fraud_rate_pct", ascending=False)
                             .to_dict(orient="records")
            )

        # Fraud rate by risk_grade if available
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
                fraud_by_grade.sort_values("fraud_rate_pct", ascending=False)
                              .to_dict(orient="records")
            )

        # Combine all flagged records
        all_flagged = pd.concat([high_value_fraud, evasive_fraud], ignore_index=True)
        self.results = all_flagged.to_dict(orient="records")
        self.fraud_profile = profile

        logger.info(
            f"[ANOMALY] Fraud profile complete — "
            f"{len(fraudulent):,} fraudulent claims | "
            f"{len(high_value_fraud):,} high-value | "
            f"{len(evasive_fraud):,} evasive"
        )

        return self

    def save(self, path=None) -> "AnomalyDetector":
        """
        Save flagged anomalies to reports/anomalies.csv.

        Args:
            path   file path to save to. Defaults to reports/anomalies.csv

        Returns self for chaining.
        """
        if not self.results:
            logger.warning("[ANOMALY] No results to save — run flag_anomalies() or flag_fraud_patterns() first")
            return self

        save_path = path or (REPORTS_DIR / "anomalies.csv")
        pd.DataFrame(self.results).to_csv(save_path, index=False)
        logger.info(f"[ANOMALY] Saved: {save_path} ({len(self.results):,} records)")

        return self

    def summary(self) -> str:
        """
        Return a plain-English summary of anomaly findings.
        """
        if not self.results:
            return "No anomalies detected yet. Run flag_anomalies() or flag_fraud_patterns() first."

        profile = getattr(self, "fraud_profile", {})
        if profile:
            return (
                f"Fraud Pattern Summary\n"
                f"  Total claims:       {profile['total_claims']:,}\n"
                f"  Fraudulent claims:  {profile['fraudulent_claims']:,} "
                f"({profile['fraud_rate_pct']}%)\n"
                f"  High-value fraud:   {profile['high_value_fraud']:,} "
                f"(easy to detect)\n"
                f"  Evasive fraud:      {profile['evasive_fraud']:,} "
                f"(blends into normal — priority for manual review)\n"
                f"  Fraud mean amount:  £{profile['fraud_mean_amount']:,.2f}\n"
                f"  Legit mean amount:  £{profile['legit_mean_amount']:,.2f}"
            )

        return f"Anomalies flagged: {len(self.results):,} records. Run save() to export."

    def __str__(self) -> str:
        return (
            f"AnomalyDetector("
            f"flagged={len(self.results)}, "
            f"profiled={'yes' if hasattr(self, 'fraud_profile') else 'no'})"
        )

    def __repr__(self) -> str:
        return f"AnomalyDetector(rows={len(self.df):,}, flagged={len(self.results)})"


# ================================================================
# QUICK SELF-TEST
# ================================================================
# This block only runs when you execute this file DIRECTLY:
#   python src/eda_engine.py
# It does NOT run when another file imports InsuranceEDAEngine or AnomalyDetector.
# ================================================================

if __name__ == "__main__":
    print("Running InsuranceEDAEngine + AnomalyDetector self-test...")
    print("=" * 55)

    # Minimal synthetic insurance DataFrame
    test_df = pd.DataFrame({
        "claim_id":        range(1, 101),
        "amount_claimed":  np.random.uniform(1000, 500000, 100).round(2),
        "amount_approved": np.random.uniform(500,  400000, 100).round(2),
        "coverage_amount": np.random.uniform(50000, 1000000, 100).round(2),
        "premium_monthly": np.random.uniform(200, 5000, 100).round(2),
        "credit_score":    np.random.randint(300, 850, 100),
        "risk_score":      np.random.uniform(1, 100, 100).round(1),
        "is_fraudulent":   np.random.choice([True, False], 100, p=[0.08, 0.92]),
        "claim_type":      np.random.choice(["property", "medical", "liability", "auto"], 100),
        "risk_grade":      np.random.choice(["A", "B", "C", "D", "E"], 100),
        "policy_type":     np.random.choice(["home", "health", "life", "auto"], 100),
        "claim_date":      pd.date_range("2024-01-01", periods=100, freq="3D").astype(str),
    })

    # Test AnomalyDetector
    detector = AnomalyDetector(test_df)
    detector.flag_fraud_patterns()
    print(detector.summary())
    print()

    print("Self-test complete.")